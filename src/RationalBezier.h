/**
 * @file RationalBezier.h
 * @brief Bernstein basis evaluation and rational Bezier function computation for 2D and 3D elements.
 *
 * Provides compile-time-sized structures and evaluators for NURBS-based FEM shape functions.
 * The central class is BezierFunctionEvaluator, which computes rational shape function values
 * and their first and second parametric derivatives at a point.
 */
#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>
#include <Eigen/Core>

/**
 * @brief Bernstein basis evaluation and rational Bezier function computation.
 */
namespace Bezier {
    /**
     * @brief Compile-time integer exponentiation.
     * @tparam Base The base.
     * @tparam Exp  The exponent (non-negative).
     */
    template<int Base, int Exp>
    struct Pow {
        static constexpr int value = Base * Pow<Base, Exp - 1>::value;
    };
    template<int Base>
    struct Pow<Base, 0> {
        static constexpr int value = 1;
    };
    // === Pow === End

    /**
     * @brief Storage for Bernstein basis values and their first and second derivatives at a parametric point.
     *
     * Specialised for 2D (xi, eta) and 3D (xi, eta, zeta). The `_lower` arrays hold
     * degree-(DEG-1) basis values needed to compute first derivatives; `_lower2` holds
     * degree-(DEG-2) values needed for second derivatives.
     *
     * @tparam DIM Parametric dimension (2 or 3).
     * @tparam DEG Polynomial degree.
     */
    template<int DIM, int DEG> struct LocalPointNFields;
    template<int DEG> struct LocalPointNFields<2, DEG> {
        std::array<double, DEG + 1> xi, eta;
        std::array<double, DEG + 1> xi_diff, eta_diff;
        std::array<double, DEG + 1> xi_diff2, eta_diff2;
        std::array<double, DEG> xi_lower, eta_lower;
        std::array<double, (DEG > 1 ? DEG - 1 : 1)> xi_lower2, eta_lower2;
    };
    template<int DEG> struct LocalPointNFields<3, DEG> {
        std::array<double, DEG + 1> xi, eta, zeta;
        std::array<double, DEG + 1> xi_diff, eta_diff, zeta_diff;
        std::array<double, DEG + 1> xi_diff2, eta_diff2, zeta_diff2;
        std::array<double, DEG> xi_lower, eta_lower, zeta_lower;
        std::array<double, (DEG > 1 ? DEG - 1 : 1)> xi_lower2, eta_lower2, zeta_lower2;
    };
    // === Bernstein Basis Fields === End


    /**
     * @brief Evaluates all degree-DEG Bernstein basis polynomials at parameter @p u.
     *
     * Uses the de Casteljau recurrence. Result is written into @p basis.
     *
     * @tparam DEG Polynomial degree.
     * @param u    Parametric coordinate.
     * @param basis Output array of size DEG+1.
     */
    template<int DEG>
    inline void bernsteinBasisAll(double u, std::array<double, DEG + 1>& basis) noexcept {
        if constexpr (DEG == 0) {
            basis[0] = 1.0;
            return;
        }
        basis.fill(0.0);
        basis[0] = 1.0;
        double u_inv = 1.0 - u;

        for (int j = 1; j <= DEG; ++j) {
            double saved = 0.0;
            for (int k = 0; k < j; ++k) {
                double temp = basis[k];
                basis[k] = saved + u_inv * temp;
                saved = u * temp;
            }
            basis[j] = saved;
        }
    }

    /**
     * @brief Computes first derivatives of all degree-DEG Bernstein basis polynomials.
     *
     * Requires degree-(DEG-1) basis values in @p basis_lower (computed via bernsteinBasisAll).
     *
     * @tparam DEG Polynomial degree.
     * @param derivative  Output array of size DEG+1.
     * @param basis_lower Input array of size DEG, holding degree-(DEG-1) basis values.
     */
    template<int DEG>
    inline void bernsteinBasisDiffAll(std::array<double, DEG + 1>& derivative, const std::array<double, DEG>& basis_lower) noexcept {
        if constexpr (DEG == 0) {
            derivative[0] = 0.0;
            return;
        }
        for (int i = 0; i <= DEG; ++i) {
            double val = 0.0;
            if (i > 0) val += basis_lower[i - 1];
            if (i < DEG) val -= basis_lower[i];
            derivative[i] = val * DEG;
        }
    }

    /**
     * @brief Computes second derivatives of all degree-DEG Bernstein basis polynomials.
     *
     * Requires degree-(DEG-2) basis values in @p basis_lower2. Returns zeros if DEG < 2.
     *
     * @tparam DEG Polynomial degree.
     * @param derivative2  Output array of size DEG+1.
     * @param basis_lower2 Input array of size DEG-1, holding degree-(DEG-2) basis values.
     */
    template<int DEG>
    inline void bernsteinBasisDiff2All(std::array<double, DEG + 1>& derivative2, const std::array<double, DEG - 1>& basis_lower2) noexcept {
        if constexpr (DEG < 2) {
            derivative2.fill(0.0);
            return;
        }
        constexpr double factor = static_cast<double>(DEG * (DEG - 1));
        for (int i = 0; i <= DEG; ++i) {
            double val = 0.0;
            if (i >= 2) val += basis_lower2[i - 2];
            if (i >= 1 && i < DEG) val -= 2.0 * basis_lower2[i - 1];
            if (i <= DEG - 2) val += basis_lower2[i];
            derivative2[i] = val * factor;
        }
    }

    /**
     * @brief Caches Bernstein basis values and derivatives at a single parametric point.
     *
     * Call compute() to populate all fields, then access them via the get* accessors.
     * The 2D overload takes (xi, eta); the 3D overload takes (xi, eta, zeta).
     *
     * @tparam ELEM_DIM Parametric dimension (2 or 3).
     * @tparam DEGREE   Polynomial degree.
     */
    template<int ELEM_DIM, int DEGREE>
    class BernsteinBasisCache {
    public:
        using Ntype = LocalPointNFields<ELEM_DIM, DEGREE>;

        // === compute === Start
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 2)>>
        void compute(double xi, double eta) {
            bernsteinBasisAll<DEGREE - 1>(xi, N.xi_lower);
            bernsteinBasisAll<DEGREE - 1>(eta, N.eta_lower);
            bernsteinBasisAll<DEGREE - 2>(xi, N.xi_lower2);
            bernsteinBasisAll<DEGREE - 2>(eta, N.eta_lower2);
            bernsteinBasisAll<DEGREE>(xi, N.xi);
            bernsteinBasisAll<DEGREE>(eta, N.eta);
            bernsteinBasisDiffAll<DEGREE>(N.xi_diff, N.xi_lower);
            bernsteinBasisDiffAll<DEGREE>(N.eta_diff, N.eta_lower);
            bernsteinBasisDiff2All<DEGREE>(N.xi_diff2, N.xi_lower2);
            bernsteinBasisDiff2All<DEGREE>(N.eta_diff2, N.eta_lower2);
        }
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 3)>>
        void compute(double xi, double eta, double zeta) {
            bernsteinBasisAll<DEGREE - 1>(xi, N.xi_lower);
            bernsteinBasisAll<DEGREE - 1>(eta, N.eta_lower);
            bernsteinBasisAll<DEGREE - 1>(zeta, N.zeta_lower);
            bernsteinBasisAll<DEGREE - 2>(xi, N.xi_lower2);
            bernsteinBasisAll<DEGREE - 2>(eta, N.eta_lower2);
            bernsteinBasisAll<DEGREE - 2>(zeta, N.zeta_lower2);
            bernsteinBasisAll<DEGREE>(xi, N.xi);
            bernsteinBasisAll<DEGREE>(eta, N.eta);
            bernsteinBasisAll<DEGREE>(zeta, N.zeta);
            bernsteinBasisDiffAll<DEGREE>(N.xi_diff, N.xi_lower);
            bernsteinBasisDiffAll<DEGREE>(N.eta_diff, N.eta_lower);
            bernsteinBasisDiffAll<DEGREE>(N.zeta_diff, N.zeta_lower);
            bernsteinBasisDiff2All<DEGREE>(N.xi_diff2, N.xi_lower2);
            bernsteinBasisDiff2All<DEGREE>(N.eta_diff2, N.eta_lower2);
            bernsteinBasisDiff2All<DEGREE>(N.zeta_diff2, N.zeta_lower2);
        }
        // === compute === End

        const auto& getNXi() const { return N.xi; }
        const auto& getNEta() const { return N.eta; }
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 3)>>
        const auto& getNZeta() const { return N.zeta; }

        const auto& getNXiDiff() const { return N.xi_diff; }
        const auto& getNEtaDiff() const { return N.eta_diff; }
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 3)>>
        const auto& getNZetaDiff() const { return N.zeta_diff; }

        const auto& getNXiDiff2() const { return N.xi_diff2; }
        const auto& getNEtaDiff2() const { return N.eta_diff2; }
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 3)>>
        const auto& getNZetaDiff2() const { return N.zeta_diff2; }

        Ntype getAllN() const { return N; };

    private:
        Ntype N;
    };

    /**
     * @brief Stores the rational weights for a single element, shared across evaluation calls.
     *
     * Weights are stored as `float` and cast to `double` during evaluation.
     * `NUM_CPTS` = (DEGREE+1)^ELEM_DIM.
     *
     * @tparam ELEM_DIM Parametric dimension (2 or 3).
     * @tparam DEGREE   Polynomial degree.
     */
    template<int ELEM_DIM, int DEGREE>
    class WeightDependentCache {
    public:
        using VectorEi = Eigen::Matrix<int, ELEM_DIM, 1>;
        static constexpr int STRIDE = DEGREE + 1;
        static constexpr int NUM_CPTS = Pow<STRIDE, ELEM_DIM>::value;

        explicit WeightDependentCache(const std::vector<float>& w) : weights(w) {}

        const std::vector<float>& getWeights() const { return weights; }

    private:
        std::vector<float> weights;
    };

    /**
     * @brief Evaluates rational Bezier (NURBS) shape functions and their first and second derivatives.
     *
     * For 2D elements: returns R, dR/dxi, dR/deta, d^2R/dxi^2, d^2R/deta^2, d^2R/dxi*deta per control point.
     * For 3D elements: returns R and 9 partial derivatives (first and mixed second) per control point.
     *
     * ResultType layout per control point [k]:
     * - 2D: [R, R_xi, R_eta, R_xi2, R_eta2, R_xieta]
     * - 3D: [R, R_xi, R_eta, R_zeta, R_xi2, R_eta2, R_zeta2, R_xieta, R_etazeta, R_xizeta]
     *
     * @tparam ELEM_DIM Parametric dimension (2 or 3).
     * @tparam DEGREE   Polynomial degree.
     */
    template<int ELEM_DIM, int DEGREE>
    class BezierFunctionEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            static constexpr int NUM_DERIVS = (ELEM_DIM == 2) ? 5 : 9;

        using WeightCache = Bezier::WeightDependentCache<ELEM_DIM, DEGREE>;
        using BasisCache = Bezier::BernsteinBasisCache<ELEM_DIM, DEGREE>;
        using ResultType = std::array<std::array<double, NUM_DERIVS + 1>, WeightCache::NUM_CPTS>; // [Value, d/dXi, d/dEta, (d/dZeta)] per control point

        explicit BezierFunctionEvaluator(const WeightCache& wc) : weight_cache(wc) {}

        // === evaluation === Start
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 2)>>
        ResultType evaluate(double xi, double eta) const {
            BasisCache basis_cache;
            double denominator = 0;
            Eigen::Matrix<double, NUM_DERIVS, 1> denominator_gradient;

            basis_cache.compute(xi, eta);
            computeDenominator2D(basis_cache.getAllN(), denominator, denominator_gradient);

            const auto& Nxi = basis_cache.getNXi();
            const auto& Neta = basis_cache.getNEta();
            const auto& Nxi_d = basis_cache.getNXiDiff();
            const auto& Neta_d = basis_cache.getNEtaDiff();
            const auto& Nxi_d2 = basis_cache.getNXiDiff2();
            const auto& Neta_d2 = basis_cache.getNEtaDiff2();

            const float* weights_ptr = weight_cache.getWeights().data();

            const double inv_D = 1.0 / denominator;
            const double D_xi = denominator_gradient[0];
            const double D_eta = denominator_gradient[1];
            const double D_xi2 = denominator_gradient[2];
            const double D_eta2 = denominator_gradient[3];
            const double D_xieta = denominator_gradient[4];


            ResultType bezier_result;
            //TODO:parall
            for (int j = 0; j <= DEGREE; ++j) {
                const double n_eta = Neta[j];
                const double n_eta_d = Neta_d[j];
                const double n_eta_d2 = Neta_d2[j];
                for (int i = 0; i <= DEGREE; ++i) {
                    const double n_xi = Nxi[i];
                    const double n_xi_d = Nxi_d[i];
                    const double n_xi_d2 = Nxi_d2[i];

                    const int n = i + WeightCache::STRIDE * j;
                    const double weight_val = static_cast<double>(weights_ptr[n]);

                    const double Num = weight_val * n_xi * n_eta;
                    const double Num_xi = weight_val * n_xi_d * n_eta;
                    const double Num_eta = weight_val * n_xi * n_eta_d;
                    const double Num_xi2 = weight_val * n_xi_d2 * n_eta;
                    const double Num_eta2 = weight_val * n_xi * n_eta_d2;
                    const double Num_xieta = weight_val * n_xi_d * n_eta_d;

                    const double R = Num * inv_D;
                    const double R_xi = (Num_xi - R * D_xi) * inv_D;
                    const double R_eta = (Num_eta - R * D_eta) * inv_D;
                    const double R_xi2 = (Num_xi2 - 2.0 * R_xi * D_xi - R * D_xi2) * inv_D;
                    const double R_eta2 = (Num_eta2 - 2.0 * R_eta * D_eta - R * D_eta2) * inv_D;
                    const double R_xieta = (Num_xieta - R_xi * D_eta - R_eta * D_xi - R * D_xieta) * inv_D;

                    bezier_result[n][0] = R;
                    bezier_result[n][1] = R_xi;
                    bezier_result[n][2] = R_eta;
                    bezier_result[n][3] = R_xi2;
                    bezier_result[n][4] = R_eta2;
                    bezier_result[n][5] = R_xieta;
                }
            }
            return bezier_result;
        }
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 3)>>
        ResultType evaluate(double xi, double eta, double zeta) const {
            BasisCache basis_cache;
            double denominator = 0;
            Eigen::Matrix<double, NUM_DERIVS, 1> denominator_gradient;

            basis_cache.compute(xi, eta, zeta);
            computeDenominator3D(basis_cache.getAllN(), denominator, denominator_gradient);

            const auto& Nxi = basis_cache.getNXi();
            const auto& Neta = basis_cache.getNEta();
            const auto& Nzeta = basis_cache.getNZeta();
            const auto& Nxi_d = basis_cache.getNXiDiff();
            const auto& Neta_d = basis_cache.getNEtaDiff();
            const auto& Nzeta_d = basis_cache.getNZetaDiff();
            const auto& Nxi_d2 = basis_cache.getNXiDiff2();
            const auto& Neta_d2 = basis_cache.getNEtaDiff2();
            const auto& Nzeta_d2 = basis_cache.getNZetaDiff2();

            const float* weights_ptr = weight_cache.getWeights().data();

            const double inv_D = 1.0 / denominator;
            const double D_xi = denominator_gradient[0];
            const double D_eta = denominator_gradient[1];
            const double D_zeta = denominator_gradient[2];
            const double D_xi2 = denominator_gradient[3];
            const double D_eta2 = denominator_gradient[4];
            const double D_zeta2 = denominator_gradient[5];
            const double D_xieta = denominator_gradient[6];
            const double D_etazeta = denominator_gradient[7];
            const double D_xizeta = denominator_gradient[8];


            ResultType bezier_result;
            //TODO:parall
            for (int k = 0; k <= DEGREE; ++k) {
                const double n_zeta = Nzeta[k];
                const double n_zeta_d = Nzeta_d[k];
                const double n_zeta_d2 = Nzeta_d2[k];

                for (int j = 0; j <= DEGREE; ++j) {
                    const double n_eta = Neta[j];
                    const double n_eta_d = Neta_d[j];
                    const double n_eta_d2 = Neta_d2[j];

                    const double term_yz = n_zeta * n_eta;
                    const double term_yz_dy = n_zeta * n_eta_d;
                    const double term_yz_dz = n_zeta_d * n_eta;
                    const double term_yz_dy2 = n_zeta * n_eta_d2;
                    const double term_yz_dz2 = n_zeta_d2 * n_eta;
                    const double term_yz_dydz = n_zeta_d * n_eta_d;

                    for (int i = 0; i <= DEGREE; ++i) {
                        const double n_xi = Nxi[i];
                        const double n_xi_d = Nxi_d[i];
                        const double n_xi_d2 = Nxi_d2[i];

                        const int n = i + j * WeightCache::STRIDE + k * WeightCache::STRIDE * WeightCache::STRIDE;
                        const double weight_val = static_cast<double>(weights_ptr[n]);

                        const double Num = weight_val * term_yz * n_xi;
                        const double Num_xi = weight_val * term_yz * n_xi_d;
                        const double Num_eta = weight_val * term_yz_dy * n_xi;
                        const double Num_zeta = weight_val * term_yz_dz * n_xi;
                        const double Num_xi2 = weight_val * term_yz * n_xi_d2;
                        const double Num_eta2 = weight_val * term_yz_dy2 * n_xi;
                        const double Num_zeta2 = weight_val * term_yz_dz2 * n_xi;
                        const double Num_xieta = weight_val * term_yz_dy * n_xi_d;
                        const double Num_etazeta = weight_val * term_yz_dydz * n_xi;
                        const double Num_xizeta = weight_val * term_yz_dz * n_xi_d;

                        const double R = Num * inv_D;
                        const double R_xi = (Num_xi - R * D_xi) * inv_D;
                        const double R_eta = (Num_eta - R * D_eta) * inv_D;
                        const double R_zeta = (Num_zeta - R * D_zeta) * inv_D;
                        const double R_xi2 = (Num_xi2 - 2.0 * R_xi * D_xi - R * D_xi2) * inv_D;
                        const double R_eta2 = (Num_eta2 - 2.0 * R_eta * D_eta - R * D_eta2) * inv_D;
                        const double R_zeta2 = (Num_zeta2 - 2.0 * R_zeta * D_zeta - R * D_zeta2) * inv_D;
                        const double R_xieta = (Num_xieta - R_xi * D_eta - R_eta * D_xi - R * D_xieta) * inv_D;
                        const double R_etazeta = (Num_etazeta - R_eta * D_zeta - R_zeta * D_eta - R * D_etazeta) * inv_D;
                        const double R_xizeta = (Num_xizeta - R_xi * D_zeta - R_zeta * D_xi - R * D_xizeta) * inv_D;

                        bezier_result[n][0] = R;
                        bezier_result[n][1] = R_xi;
                        bezier_result[n][2] = R_eta;
                        bezier_result[n][3] = R_zeta;
                        bezier_result[n][4] = R_xi2;
                        bezier_result[n][5] = R_eta2;
                        bezier_result[n][6] = R_zeta2;
                        bezier_result[n][7] = R_xieta;
                        bezier_result[n][8] = R_etazeta;
                        bezier_result[n][9] = R_xizeta;
                    }
                }
            }
            return bezier_result;
        }
        // === evaluation === End

        // === evaluateCached === Start
        ResultType evaluate(const typename BasisCache::Ntype& basis_cache) const {
            if constexpr (ELEM_DIM == 2) {
                double denominator = 0;
                Eigen::Matrix<double, NUM_DERIVS, 1> denominator_gradient;
                computeDenominator2D(basis_cache, denominator, denominator_gradient);

                const auto& Nxi = basis_cache.xi;
                const auto& Neta = basis_cache.eta;
                const auto& Nxi_d = basis_cache.xi_diff;
                const auto& Neta_d = basis_cache.eta_diff;
                const auto& Nxi_d2 = basis_cache.xi_diff2;
                const auto& Neta_d2 = basis_cache.eta_diff2;

                const float* weights_ptr = weight_cache.getWeights().data();

                const double inv_D = 1.0 / denominator;
                const double D_xi = denominator_gradient[0];
                const double D_eta = denominator_gradient[1];
                const double D_xi2 = denominator_gradient[2];
                const double D_eta2 = denominator_gradient[3];
                const double D_xieta = denominator_gradient[4];


                ResultType bezier_result;
                //TODO:parall
                for (int j = 0; j <= DEGREE; ++j) {
                    const double n_eta = Neta[j];
                    const double n_eta_d = Neta_d[j];
                    const double n_eta_d2 = Neta_d2[j];

                    for (int i = 0; i <= DEGREE; ++i) {
                        const double n_xi = Nxi[i];
                        const double n_xi_d = Nxi_d[i];
                        const double n_xi_d2 = Nxi_d2[i];

                        const int n = i + WeightCache::STRIDE * j;
                        const double weight_val = static_cast<double>(weights_ptr[n]);

                        const double Num = weight_val * n_xi * n_eta;
                        const double Num_xi = weight_val * n_xi_d * n_eta;
                        const double Num_eta = weight_val * n_xi * n_eta_d;
                        const double Num_xi2 = weight_val * n_xi_d2 * n_eta;
                        const double Num_eta2 = weight_val * n_xi * n_eta_d2;
                        const double Num_xieta = weight_val * n_xi_d * n_eta_d;

                        const double R = Num * inv_D;
                        const double R_xi = (Num_xi - R * D_xi) * inv_D;
                        const double R_eta = (Num_eta - R * D_eta) * inv_D;
                        const double R_xi2 = (Num_xi2 - 2.0 * R_xi * D_xi - R * D_xi2) * inv_D;
                        const double R_eta2 = (Num_eta2 - 2.0 * R_eta * D_eta - R * D_eta2) * inv_D;
                        const double R_xieta = (Num_xieta - R_xi * D_eta - R_eta * D_xi - R * D_xieta) * inv_D;

                        bezier_result[n][0] = R;
                        bezier_result[n][1] = R_xi;
                        bezier_result[n][2] = R_eta;
                        bezier_result[n][3] = R_xi2;
                        bezier_result[n][4] = R_eta2;
                        bezier_result[n][5] = R_xieta;
                    }
                }
                return bezier_result;
            }
            else {
                double denominator = 0;
                Eigen::Matrix<double, NUM_DERIVS, 1> denominator_gradient;

                computeDenominator3D(basis_cache, denominator, denominator_gradient);

                const auto& Nxi = basis_cache.xi;
                const auto& Neta = basis_cache.eta;
                const auto& Nzeta = basis_cache.zeta;
                const auto& Nxi_d = basis_cache.xi_diff;
                const auto& Neta_d = basis_cache.eta_diff;
                const auto& Nzeta_d = basis_cache.zeta_diff;
                const auto& Nxi_d2 = basis_cache.xi_diff2;
                const auto& Neta_d2 = basis_cache.eta_diff2;
                const auto& Nzeta_d2 = basis_cache.zeta_diff2;

                const float* weights_ptr = weight_cache.getWeights().data();

                const double inv_D = 1.0 / denominator;
                const double D_xi = denominator_gradient[0];
                const double D_eta = denominator_gradient[1];
                const double D_zeta = denominator_gradient[2];
                const double D_xi2 = denominator_gradient[3];
                const double D_eta2 = denominator_gradient[4];
                const double D_zeta2 = denominator_gradient[5];
                const double D_xieta = denominator_gradient[6];
                const double D_etazeta = denominator_gradient[7];
                const double D_xizeta = denominator_gradient[8];


                ResultType bezier_result;
                //TODO:parall
                for (int k = 0; k <= DEGREE; ++k) {
                    const double n_zeta = Nzeta[k];
                    const double n_zeta_d = Nzeta_d[k];
                    const double n_zeta_d2 = Nzeta_d2[k];

                    for (int j = 0; j <= DEGREE; ++j) {
                        const double n_eta = Neta[j];
                        const double n_eta_d = Neta_d[j];
                        const double n_eta_d2 = Neta_d2[j];

                        const double term_yz = n_zeta * n_eta;
                        const double term_yz_dz = n_zeta_d * n_eta;
                        const double term_yz_dy = n_zeta * n_eta_d;
                        const double term_yz_dz2 = n_zeta_d2 * n_eta;
                        const double term_yz_dy2 = n_zeta * n_eta_d2;
                        const double term_yz_dydz = n_zeta_d * n_eta_d;

                        for (int i = 0; i <= DEGREE; ++i) {
                            const double n_xi = Nxi[i];
                            const double n_xi_d = Nxi_d[i];
                            const double n_xi_d2 = Nxi_d2[i];

                            const int n = i + j * WeightCache::STRIDE + k * WeightCache::STRIDE * WeightCache::STRIDE;
                            const double weight_val = static_cast<double>(weights_ptr[n]);

                            const double Num = weight_val * term_yz * n_xi;
                            const double Num_xi = weight_val * term_yz * n_xi_d;
                            const double Num_eta = weight_val * term_yz_dy * n_xi;
                            const double Num_zeta = weight_val * term_yz_dz * n_xi;
                            const double Num_xi2 = weight_val * term_yz * n_xi_d2;
                            const double Num_eta2 = weight_val * term_yz_dy2 * n_xi;
                            const double Num_zeta2 = weight_val * term_yz_dz2 * n_xi;
                            const double Num_xieta = weight_val * term_yz_dy * n_xi_d;
                            const double Num_etazeta = weight_val * term_yz_dydz * n_xi;
                            const double Num_xizeta = weight_val * term_yz_dz * n_xi_d;

                            const double R = Num * inv_D;
                            const double R_xi = (Num_xi - R * D_xi) * inv_D;
                            const double R_eta = (Num_eta - R * D_eta) * inv_D;
                            const double R_zeta = (Num_zeta - R * D_zeta) * inv_D;
                            const double R_xi2 = (Num_xi2 - 2.0 * R_xi * D_xi - R * D_xi2) * inv_D;
                            const double R_eta2 = (Num_eta2 - 2.0 * R_eta * D_eta - R * D_eta2) * inv_D;
                            const double R_zeta2 = (Num_zeta2 - 2.0 * R_zeta * D_zeta - R * D_zeta2) * inv_D;
                            const double R_xieta = (Num_xieta - R_xi * D_eta - R_eta * D_xi - R * D_xieta) * inv_D;
                            const double R_etazeta = (Num_etazeta - R_eta * D_zeta - R_zeta * D_eta - R * D_etazeta) * inv_D;
                            const double R_xizeta = (Num_xizeta - R_xi * D_zeta - R_zeta * D_xi - R * D_xizeta) * inv_D;

                            bezier_result[n][0] = R;
                            bezier_result[n][1] = R_xi;
                            bezier_result[n][2] = R_eta;
                            bezier_result[n][3] = R_zeta;
                            bezier_result[n][4] = R_xi2;
                            bezier_result[n][5] = R_eta2;
                            bezier_result[n][6] = R_zeta2;
                            bezier_result[n][7] = R_xieta;
                            bezier_result[n][8] = R_etazeta;
                            bezier_result[n][9] = R_xizeta;
                        }
                    }
                }
                return bezier_result;
            }
        }
        // === evaluateCached === End

    private:
        const WeightCache& weight_cache;

        // === computeDenominator === Start
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 2)>>
        void computeDenominator2D(const typename BasisCache::Ntype& basis_cache, double& denominator, Eigen::Matrix<double, NUM_DERIVS, 1>& denominator_gradient) const {
            denominator = 0.0;
            denominator_gradient.setZero();

            const auto& Nxi = basis_cache.xi;
            const auto& Neta = basis_cache.eta;
            const auto& Nxi_d = basis_cache.xi_diff;
            const auto& Neta_d = basis_cache.eta_diff;
            const auto& Nxi_d2 = basis_cache.xi_diff2;
            const auto& Neta_d2 = basis_cache.eta_diff2;

            const float* weights_ptr = weight_cache.getWeights().data();

            for (int j = 0; j <= DEGREE; ++j) {
                const double n_eta = Neta[j];
                const double n_eta_d = Neta_d[j];
                const double n_eta_d2 = Neta_d2[j];

                for (int i = 0; i <= DEGREE; ++i) {
                    const double n_xi = Nxi[i];
                    const double n_xi_d = Nxi_d[i];
                    const double n_xi_d2 = Nxi_d2[i];

                    const int n = i + j * WeightCache::STRIDE;
                    const double weight_val = static_cast<double>(weights_ptr[n]);

                    denominator += n_xi * n_eta * weight_val;
                    denominator_gradient[0] += n_xi_d * n_eta * weight_val;
                    denominator_gradient[1] += n_xi * n_eta_d * weight_val;
                    denominator_gradient[2] += n_xi_d2 * n_eta * weight_val;
                    denominator_gradient[3] += n_xi * n_eta_d2 * weight_val;
                    denominator_gradient[4] += n_xi_d * n_eta_d * weight_val;
                }
            }
        }
        template <int D = ELEM_DIM, typename = std::enable_if_t<(D == 3)>>
        void computeDenominator3D(const typename BasisCache::Ntype& basis_cache, double& denominator, Eigen::Matrix<double, NUM_DERIVS, 1>& denominator_gradient) const {
            denominator = 0.0;
            denominator_gradient.setZero();

            const auto& Nxi = basis_cache.xi;
            const auto& Neta = basis_cache.eta;
            const auto& Nzeta = basis_cache.zeta;
            const auto& Nxi_d = basis_cache.xi_diff;
            const auto& Neta_d = basis_cache.eta_diff;
            const auto& Nzeta_d = basis_cache.zeta_diff;
            const auto& Nxi_d2 = basis_cache.xi_diff2;
            const auto& Neta_d2 = basis_cache.eta_diff2;
            const auto& Nzeta_d2 = basis_cache.zeta_diff2;

            const float* weights_ptr = weight_cache.getWeights().data();

            for (int k = 0; k <= DEGREE; ++k) {
                const double n_zeta = Nzeta[k];
                const double n_zeta_d = Nzeta_d[k];
                const double n_zeta_d2 = Nzeta_d2[k];

                for (int j = 0; j <= DEGREE; ++j) {
                    const double n_eta = Neta[j];
                    const double n_eta_d = Neta_d[j];
                    const double n_eta_d2 = Neta_d2[j];

                    const double term_yz = n_zeta * n_eta;
                    const double term_yz_dz = n_zeta_d * n_eta;
                    const double term_yz_dy = n_zeta * n_eta_d;
                    const double term_yz_dz2 = n_zeta_d2 * n_eta;
                    const double term_yz_dy2 = n_zeta * n_eta_d2;
                    const double term_yz_dydz = n_zeta_d * n_eta_d;

                    for (int i = 0; i <= DEGREE; ++i) {
                        const double n_xi = Nxi[i];
                        const double n_xi_d = Nxi_d[i];
                        const double n_xi_d2 = Nxi_d2[i];

                        const int n = i + j * WeightCache::STRIDE + k * WeightCache::STRIDE * WeightCache::STRIDE;
                        const double weight_val = static_cast<double>(weights_ptr[n]);

                        denominator += term_yz * n_xi * weight_val;
                        denominator_gradient[0] += term_yz * n_xi_d * weight_val;
                        denominator_gradient[1] += term_yz_dy * n_xi * weight_val;
                        denominator_gradient[2] += term_yz_dz * n_xi * weight_val;
                        denominator_gradient[3] += term_yz * n_xi_d2 * weight_val;
                        denominator_gradient[4] += term_yz_dy2 * n_xi * weight_val;
                        denominator_gradient[5] += term_yz_dz2 * n_xi * weight_val;
                        denominator_gradient[6] += term_yz_dy * n_xi_d * weight_val;
                        denominator_gradient[7] += term_yz_dydz * n_xi * weight_val;
                        denominator_gradient[8] += term_yz_dz * n_xi_d * weight_val;
                    }
                }
            }
        }
        // === computeDenominator === End
    };

} // namespace Bezier