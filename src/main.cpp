#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <filesystem>

#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/StageFactory.hpp>

#include <filters/private/csf/CSF.h>

#include <io/LasReader.hpp>
#include <io/BufferReader.hpp>

#include <open3d/io/PointCloudIO.h>

void extractFeatures(const std::shared_ptr<open3d::geometry::PointCloud>& input_cloud,
                     std::shared_ptr<open3d::geometry::PointCloud>& edges_cloud,
                     std::shared_ptr<open3d::geometry::PointCloud>& planes_cloud) 
{
    std::cout << "\n--- Extracting Geometric Features ---" << std::endl;
    
    // 1. Build a KD-Tree to quickly find neighboring points
    open3d::geometry::KDTreeFlann kdtree(*input_cloud);
    
    int num_neighbors = 10; // The size of our 'S' neighborhood
    double edge_threshold = 0.5;  // High curvature
    double plane_threshold = 0.05; // Low curvature

    size_t num_points = input_cloud->points_.size();
    
    // Pre-allocate some memory (we don't know exact sizes yet, so this is a guess)
    edges_cloud->points_.reserve(num_points / 10);
    planes_cloud->points_.reserve(num_points / 2);

    std::cout << "Calculating smoothness for " << num_points << " points..." << std::endl;

    // Use OpenMP for maximum speed
    #pragma omp parallel
    {
        // Thread-local clouds to prevent nasty race conditions during push_back
        std::vector<Eigen::Vector3d> local_edges;
        std::vector<Eigen::Vector3d> local_planes;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < num_points; ++i) {
            std::vector<int> point_idx;
            std::vector<double> point_sq_dist;
            
            // Search for the 10 closest neighbors
            if (kdtree.SearchKNN(input_cloud->points_[i], num_neighbors, point_idx, point_sq_dist) >= 3) {
                
                Eigen::Vector3d center_point = input_cloud->points_[i];
                Eigen::Vector3d vector_sum(0.0, 0.0, 0.0);
                
                // Sum the differences (X_i - X_j)
                for (size_t j = 1; j < point_idx.size(); ++j) {
                    vector_sum += (center_point - input_cloud->points_[point_idx[j]]);
                }
                
                // Calculate the final curvature 'c'
                double curvature = vector_sum.norm() / (num_neighbors * center_point.norm());
                
                // Sort the point based on its geometry
                if (curvature > edge_threshold) {
                    local_edges.push_back(center_point);
                } else if (curvature < plane_threshold) {
                    local_planes.push_back(center_point);
                }
            }
        }

        // Safely merge the thread-local results back into the main clouds
        #pragma omp critical
        {
            edges_cloud->points_.insert(edges_cloud->points_.end(), local_edges.begin(), local_edges.end());
            planes_cloud->points_.insert(planes_cloud->points_.end(), local_planes.begin(), local_planes.end());
        }
    }

    std::cout << "Extracted " << edges_cloud->points_.size() << " Edges and " 
              << planes_cloud->points_.size() << " Planes!" << std::endl;
}

std::shared_ptr<open3d::geometry::PointCloud> extractBareEarthCSF(
    const std::shared_ptr<open3d::geometry::PointCloud>& cloud)
{
    std::cout << "\n--- Running Cloth Simulation Filter (CSF) ---" << std::endl;

    // 1. Define a PDAL PointTable and layout
    pdal::PointTable table;
    table.layout()->registerDim(pdal::Dimension::Id::X);
    table.layout()->registerDim(pdal::Dimension::Id::Y);
    table.layout()->registerDim(pdal::Dimension::Id::Z);
    table.layout()->registerDim(pdal::Dimension::Id::Classification);

    // ==========================================
    // PHASE 1: PACKING PROGRESS BAR
    // ==========================================
    size_t total_in = cloud->points_.size();
    size_t update_step = std::max<size_t>(1, total_in / 100);
    std::cout << "1/3: Packing Open3D data for PDAL pipeline..." << std::endl;
    
    pdal::PointViewPtr input_view(new pdal::PointView(table));
    for (size_t i = 0; i < total_in; ++i) {
        input_view->setField(pdal::Dimension::Id::X, i, cloud->points_[i].x());
        input_view->setField(pdal::Dimension::Id::Y, i, cloud->points_[i].y());
        input_view->setField(pdal::Dimension::Id::Z, i, cloud->points_[i].z());
        input_view->setField(pdal::Dimension::Id::Classification, i, 0); 
        
        // Terminal update
        if (i % update_step == 0 || i == total_in - 1) {
            int pct = (i * 100) / total_in;
            std::cout << "\r  [" << pct << "%] Packed " << i << " / " << total_in << " points" << std::flush;
        }
    }
    std::cout << std::endl; // Drop to new line

    // 3 & 4 & 5. Setup Buffer, Factory, and Options (unchanged)
    pdal::BufferReader reader;
    reader.addView(input_view);

    pdal::StageFactory factory;
    pdal::Stage* csf_filter = factory.createStage("filters.csf");

    if (!csf_filter) {
        std::cerr << "Error: PDAL was not compiled with filters.csf enabled." << std::endl;
        return nullptr;
    }

    pdal::Options csf_opts;
    csf_opts.add("resolution", 0.5); 
    csf_opts.add("rigidness", 3); 
    csf_opts.add("step", 0.65);
    csf_opts.add("threshold", 0.15); 

    csf_filter->setOptions(csf_opts);
    csf_filter->setInput(reader);

    // ==========================================
    // PHASE 2: EXECUTION STOPWATCH
    // ==========================================
    std::cout << "2/3: Executing CSF Math (This will take a moment)... " << std::flush;
    
    // Start the clock
    auto start_time = std::chrono::high_resolution_clock::now();

    pdal::PointTable out_table;
    csf_filter->prepare(out_table);
    pdal::PointViewSet out_views = csf_filter->execute(out_table);
    pdal::PointViewPtr out_view = *out_views.begin();

    // Stop the clock
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Done! (" << elapsed.count() << " seconds)\n";

    // ==========================================
    // PHASE 3: UNPACKING PROGRESS BAR
    // ==========================================
    std::cout << "3/3: Extracting bare earth points to Open3D..." << std::endl;

    auto cloth_cloud = std::make_shared<open3d::geometry::PointCloud>();
    size_t num_out_points = out_view->size();
    size_t update_step_out = std::max<size_t>(1, num_out_points / 100);
    
    cloth_cloud->points_.reserve(num_out_points); 

    size_t ground_count = 0;
    for (pdal::PointId id = 0; id < num_out_points; ++id) {
        uint8_t classification = out_view->getFieldAs<uint8_t>(pdal::Dimension::Id::Classification, id);
        
        if (classification == 2) {
            cloth_cloud->points_.push_back(Eigen::Vector3d(
                out_view->getFieldAs<double>(pdal::Dimension::Id::X, id),
                out_view->getFieldAs<double>(pdal::Dimension::Id::Y, id),
                out_view->getFieldAs<double>(pdal::Dimension::Id::Z, id)
            ));
            ground_count++;
        }

        // Terminal update
        if (id % update_step_out == 0 || id == num_out_points - 1) {
            int pct = (id * 100) / num_out_points;
            std::cout << "\r  [" << pct << "%] Scanned " << id << " / " << num_out_points << std::flush;
        }
    }
    std::cout << std::endl; // Drop to new line

    std::cout << "CSF Complete: Extracted " << ground_count << " ground points from " 
              << cloud->points_.size() << " total points." << std::endl;

    return cloth_cloud;
}


bool loadLiDARDataWithTime(const std::string& filename,
                           std::shared_ptr<open3d::geometry::PointCloud>& cloud,
                           std::vector<double>& times) 
{
    try {
        pdal::Options options;
        options.add("filename", filename);

        pdal::LasReader reader;
        reader.setOptions(options);

        pdal::PointTable table;
        reader.prepare(table);
        
        pdal::PointViewSet viewSet = reader.execute(table);
        pdal::PointViewPtr view = *viewSet.begin();

        size_t num_points = view->size();
        cloud->points_.resize(num_points);
        times.resize(num_points);

        std::cout << "Successfully read " << num_points << " points from " << filename << std::endl;

        // Compress the massive aerial scan into a 100ms (0.1s) sweep
        double sweep_duration = 0.1; 

        for (pdal::PointId id = 0; id < num_points; ++id) {
            cloud->points_[id] = Eigen::Vector3d(
                view->getFieldAs<double>(pdal::Dimension::Id::X, id),
                view->getFieldAs<double>(pdal::Dimension::Id::Y, id),
                view->getFieldAs<double>(pdal::Dimension::Id::Z, id)
            );
            
            // Generate synthetic, sequential timestamps for our trajectory math
            times[id] = (static_cast<double>(id) / num_points) * sweep_duration;
        }
        return true;
    } 
    catch (const pdal::pdal_error& e) {
        std::cerr << "PDAL Error: " << e.what() << std::endl;
        return false;
    }
}

// ---------------------------------------------------------
// Mock Trajectory Interpolator
// ---------------------------------------------------------
class TrajectoryInterpolator {
public:
    Eigen::Matrix4d getPoseAtTime(double timestamp) const {
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose(0, 3) = 10.0 * timestamp; 
        return pose;
    }
};

std::shared_ptr<open3d::geometry::PointCloud> deskewPointCloudParallel(
    const std::shared_ptr<open3d::geometry::PointCloud>& raw_cloud,
    const std::vector<double>& point_times,
    const TrajectoryInterpolator& trajectory,
    double sweep_start_time) 
{
    auto deskewed_cloud = std::make_shared<open3d::geometry::PointCloud>();
    size_t num_points = raw_cloud->points_.size();
    
    deskewed_cloud->points_.resize(num_points);
    if (raw_cloud->HasColors()) deskewed_cloud->colors_ = raw_cloud->colors_;
    if (raw_cloud->HasNormals()) deskewed_cloud->normals_ = raw_cloud->normals_;

    // 2. Calculate the Base Frame Inverse
    // This is the T_base^{-1} matrix. We transform all points to this specific instant in time.
    Eigen::Matrix4d T_base = trajectory.getPoseAtTime(sweep_start_time);
    Eigen::Matrix4d T_base_inv = T_base.inverse();

    std::atomic<size_t> points_processed(0);
    size_t update_step = std::max<size_t>(1, num_points / 100); 

    std::cout << "Starting parallel deskewing..." << std::endl;

    // 3. The Parallel Execution Block
    // This instructs the compiler to distribute the loop across all CPU threads
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_points; ++i) {
        
        // A. Query the exact pose for this specific point's emission time
        double t_p = point_times[i];
        Eigen::Matrix4d T_point = trajectory.getPoseAtTime(t_p);
        
        // B. Calculate the relative transformation: T_rel = T_base^{-1} * T_point
        Eigen::Matrix4d T_rel = T_base_inv * T_point;
        
        // C. Convert the raw 3D point to homogeneous coordinates (4D vector)
        Eigen::Vector4d p_raw(
            raw_cloud->points_[i](0), 
            raw_cloud->points_[i](1), 
            raw_cloud->points_[i](2), 
            1.0
        );
        
        // D. Apply the transformation to freeze the motion
        Eigen::Vector4d p_deskewed = T_rel * p_raw;
        
        // E. Write the corrected coordinates directly into the pre-allocated array
        deskewed_cloud->points_[i] = p_deskewed.head<3>();


        // Thread-safe Progress Bar
        size_t current_count = ++points_processed; // Atomically add 1
        
        // Only update the terminal every 1% to keep things fast
        if (current_count % update_step == 0 || current_count == num_points) {
            
            int percentage = (current_count * 100) / num_points;
            
            // Force OpenMP threads to wait in line before talking to the terminal
            #pragma omp critical 
            {
                // The '\r' tells the terminal to return to the start of the current line 
                // instead of printing a new line every time!
                std::cout << "\r[" << percentage << "%] Processed " 
                          << current_count << " / " << num_points << " points" << std::flush;
            }
        }
    }
    std::cout << std::endl;

    return deskewed_cloud;
}

int main() {
    std::cout << "--- CTA_LIO: Data Ingestion & Deskewing ---" << std::endl;

    auto raw_cloud = std::make_shared<open3d::geometry::PointCloud>();
    auto cloth_cloud = std::make_shared<open3d::geometry::PointCloud>();

    std::string input_file = "../data/autzen.laz";
    std::string cache_file = "../data/autzen.pcd";

    std::vector<double> point_times;

    if (!loadLiDARDataWithTime(input_file, raw_cloud, point_times)) {
        return -1;
    }
    Eigen::Vector3d center = raw_cloud->GetCenter();
    raw_cloud->Translate(-center);

    TrajectoryInterpolator trajectory;
    double sweep_start_time = point_times[0]; 

    std::cout << "Processing using " << omp_get_max_threads() << " CPU threads..." << std::endl;

    auto deskewed_cloud = deskewPointCloudParallel(raw_cloud, point_times, trajectory, sweep_start_time);

    if (std::filesystem::exists(cache_file)) {
        std::cout << "Cache found! Loading ground points in 1 second..." << std::endl;
        open3d::io::ReadPointCloud(cache_file, *cloth_cloud);
    } else {
        std::cout << "No cache found. Buckle up, running 34-minute CSF..." << std::endl;
        cloth_cloud = extractBareEarthCSF(raw_cloud);
        
        std::cout << "Saving results to cache so we never have to do that again..." << std::endl;
        open3d::io::WritePointCloud(cache_file, *cloth_cloud);
    }

    // E. Visual Verification
    deskewed_cloud->PaintUniformColor({0.5, 0.5, 0.5}); // Paint all points Gray
    if (cloth_cloud) {
        cloth_cloud->PaintUniformColor({0.0, 1.0, 0.0}); // Paint ground points Green
        open3d::visualization::DrawGeometries({deskewed_cloud, cloth_cloud}, "Bare Earth Extraction");
    } else {
        open3d::visualization::DrawGeometries({deskewed_cloud}, "Deskewed Cloud Only");
    }

    return 0;
}