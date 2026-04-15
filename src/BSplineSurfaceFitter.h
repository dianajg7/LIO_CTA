#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "RationalBezier.h"

struct PointUVZ {
    double u; // Forward axis [meters]
    double v; // Left/Lateral axis [meters]
    double z; // Elevation [meters]
};

class BSplineSurfaceFitter {
public:
    static constexpr int DEGREE = 3; 

    // Uniform 1D transformation for mapping 4 global B-Spline basis functions to 4 local Bernstein polynomials
    const Eigen::Matrix4d C_uniform = (Eigen::Matrix4d() << 
        1.0,     0.0,       0.0,       0.0,
        0.5,     0.5,       0.0,       0.0,
        0.25,    7.0/12.0,  1.0/6.0,   0.0,
        1.0/6.0, 2.0/3.0,   1.0/6.0,   0.0
    ).finished();

    struct GridConfig {
        double u_min, u_max;
        double v_min, v_max;
        double element_size_u;
        double element_size_v;
        int num_elements_u;
        int num_elements_v;
        int num_cpts_u;
        int num_cpts_v;
    };

    GridConfig config;
    Eigen::VectorXd global_control_points_z;

    BSplineSurfaceFitter(double u_min, double u_max, double v_min, double v_max, double element_u_len = 5.0, double element_v_len = 1.0) {
        config.u_min = u_min;
        config.u_max = u_max;
        config.v_min = v_min;
        config.v_max = v_max;
        config.element_size_u = element_u_len;
        config.element_size_v = element_v_len;

        config.num_elements_u = std::ceil((u_max - u_min) / element_u_len);
        config.num_elements_v = std::ceil((v_max - v_min) / element_v_len);

        config.num_cpts_u = config.num_elements_u + DEGREE;
        config.num_cpts_v = config.num_elements_v + DEGREE;
    }

    bool fitSurface(const std::vector<PointUVZ>& cloud) {
        int num_pts = cloud.size();
        int total_cpts = config.num_cpts_u * config.num_cpts_v;

        std::cout << "\n--- Initializing B-Spline Global Fit ---" << std::endl;
        std::cout << "Grid Elements: " << config.num_elements_u << " x " << config.num_elements_v << std::endl;
        std::cout << "Total Control Points to Solve: " << total_cpts << std::endl;

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(num_pts * 16);
        Eigen::VectorXd Z(num_pts);

        Bezier::BernsteinBasisCache<2, DEGREE> basis_cache;

        for (int k = 0; k < num_pts; ++k) {
            const auto& pt = cloud[k];
            Z(k) = pt.z;

            double u_norm = (pt.u - config.u_min) / config.element_size_u;
            double v_norm = (pt.v - config.v_min) / config.element_size_v;

            int elem_u = std::clamp(static_cast<int>(std::floor(u_norm)), 0, config.num_elements_u - 1);
            int elem_v = std::clamp(static_cast<int>(std::floor(v_norm)), 0, config.num_elements_v - 1);

            double xi  = u_norm - elem_u;
            double eta = v_norm - elem_v;

            basis_cache.compute(xi, eta);
            auto arr_xi = basis_cache.getNXi();
            auto arr_eta = basis_cache.getNEta();

            Eigen::Vector4d B_xi(arr_xi[0], arr_xi[1], arr_xi[2], arr_xi[3]);
            Eigen::Vector4d B_eta(arr_eta[0], arr_eta[1], arr_eta[2], arr_eta[3]);

            Eigen::RowVector4d N_u_global = B_xi.transpose() * C_uniform;
            Eigen::RowVector4d N_v_global = B_eta.transpose() * C_uniform;

            for (int i = 0; i <= DEGREE; ++i) {
                for (int j = 0; j <= DEGREE; ++j) {
                    
                    int global_u_idx = elem_u + i;
                    int global_v_idx = elem_v + j;
                    int flattened_cpt_idx = global_u_idx * config.num_cpts_v + global_v_idx;

                    double weight = N_u_global(i) * N_v_global(j);

                    triplets.push_back(Eigen::Triplet<double>(k, flattened_cpt_idx, weight));
                }
            }
        }

        std::cout << "Assembling Sparse Matrix..." << std::endl;
        Eigen::SparseMatrix<double> A(num_pts, total_cpts);
        A.setFromTriplets(triplets.begin(), triplets.end());

        std::cout << "Solving Normal Equations via SimplicialLDLT..." << std::endl;
        Eigen::SparseMatrix<double> AtA = A.transpose() * A;
        Eigen::VectorXd AtZ = A.transpose() * Z;

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(AtA);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed! The grid might be severely under-sampled." << std::endl;
            return false;
        }

        global_control_points_z = solver.solve(AtZ);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solving failed!" << std::endl;
            return false;
        }

        std::cout << "B-Spline global fit completed successfully!" << std::endl;
        return true;
    }

    double getControlPointZ(int u_idx, int v_idx) const {
        if (u_idx < 0 || u_idx >= config.num_cpts_u || v_idx < 0 || v_idx >= config.num_cpts_v) {
            return 0.0; 
        }
        return global_control_points_z(u_idx * config.num_cpts_v + v_idx);
    }
};