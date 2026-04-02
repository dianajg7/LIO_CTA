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
#include <cmath>
#include <limits>
#include <deque>
#include <mutex>

#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/StageFactory.hpp>

#include <filters/private/csf/CSF.h>

#include <io/LasReader.hpp>
#include <io/BufferReader.hpp>

#include <open3d/io/PointCloudIO.h>
#include <open3d/geometry/KDTreeFlann.h>

#include <cstring> 
#include <ouster/client.h>
#include <ouster/lidar_scan.h>
#include <ouster/types.h>
#include <ouster/xyzlut.h>
#include <ouster/os_pcap.h> 
#include <ouster/pcap.h>  

// --- STEAM & lgmath Headers ---
#include <lgmath/se3/Transformation.hpp>

#include <steam/trajectory/const_vel/interface.hpp>
#include <steam/evaluable/se3/se3_state_var.hpp>
#include <steam/evaluable/vspace/vspace_state_var.hpp>
#include <steam/problem/optimization_problem.hpp>
#include <steam/solver/gauss_newton_solver.hpp>

struct ImuMeasurement {
    double timestamp; // In seconds
    Eigen::Vector3d acceleration; // Linear acceleration (m/s^2)
    Eigen::Vector3d angular_velocity; // Gyroscope (rad/s)
};

// ====================================================================
// NEW: STEAM Continuous-Time Trajectory Interpolator
// ====================================================================
class TrajectoryInterpolator {
private:
    steam::traj::const_vel::Interface traj_; 
    
    bool is_initialized_ = false;
    double first_time_ = 0.0;
    double last_time_ = 0.0;
    lgmath::se3::Transformation last_pose_;
    Eigen::Matrix<double, 6, 1> last_velocity_;

    Eigen::Vector3d gravity_{0.0, 0.0, 9.80665};
    mutable std::mutex buffer_mutex_;

public:
    TrajectoryInterpolator() {
        Eigen::Matrix<double, 6, 1> qc_diag;

        //TODO: tune wrt INS-D specs
        qc_diag.head<3>().setConstant(10.0); // Translational PSD
        qc_diag.tail<3>().setConstant(10.0); // Rotational PSD
        
        traj_ = steam::traj::const_vel::Interface(qc_diag);
    }

    void addImuMeasurement(const ImuMeasurement& imu) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);

        steam::traj::Time knot_time(imu.timestamp);

        // --- FIRST KNOT INITIALIZATION ---
        if (!is_initialized_) {
            first_time_ = imu.timestamp;
            last_time_ = imu.timestamp;
            last_pose_ = lgmath::se3::Transformation(); // Identity
            last_velocity_.setZero();

            auto pose_var = steam::se3::SE3StateVar::MakeShared(last_pose_);
            auto vel_var = steam::vspace::VSpaceStateVar<6>::MakeShared(last_velocity_);
            
            traj_.add(knot_time, pose_var, vel_var);
            
            is_initialized_ = true;
            return;
        }

        double dt = imu.timestamp - last_time_;
        if (dt <= 0) return;

        // --- DEAD-RECKON AN INITIAL GUESS FOR THE OPTIMIZER ---
        Eigen::Vector3d a_global = last_pose_.matrix().block<3,3>(0,0) * imu.acceleration - gravity_;

        Eigen::Vector3d next_v_trans = last_velocity_.head<3>() + (a_global * dt);
        Eigen::Vector3d next_v_rot = imu.angular_velocity; 

        Eigen::Matrix<double, 6, 1> next_velocity;
        next_velocity << next_v_trans, next_v_rot;

        Eigen::Matrix<double, 6, 1> xi = last_velocity_ * dt; 
        lgmath::se3::Transformation T_step(xi);
        lgmath::se3::Transformation next_pose = T_step * last_pose_;

        // --- ADD THE NEW KNOT ---
        auto pose_var = steam::se3::SE3StateVar::MakeShared(next_pose);
        auto vel_var = steam::vspace::VSpaceStateVar<6>::MakeShared(next_velocity);

        traj_.add(knot_time, pose_var, vel_var);

        last_time_ = imu.timestamp;
        last_pose_ = next_pose;
        last_velocity_ = next_velocity;
    }

    Eigen::Matrix4d getPoseAtTime(double timestamp) const {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        if (!is_initialized_) return Eigen::Matrix4d::Identity();

        // Safety Catch: GP interpolation cannot extrapolate.
        // Clamp the query to the bounds of our trajectory knots.
        double clamped_time = std::max(first_time_, std::min(timestamp, last_time_));
        steam::traj::Time query_time(clamped_time);
        
        // STEAM analytical GP evaluator
        auto pose_evaluator = traj_.getPoseInterpolator(query_time);
        lgmath::se3::Transformation T_query = pose_evaluator->value();
        
        return T_query.matrix();
    }
    
    steam::traj::const_vel::Interface& getTrajectory() { return traj_; }
};


// ====================================================================
// PIPELINE FUNCTIONS 
// ====================================================================

bool parseOusterPcap(const std::string& pcap_file, 
                     const std::string& json_meta_file,
                     TrajectoryInterpolator& trajectory,
                     std::shared_ptr<open3d::geometry::PointCloud>& global_cloud,
                     std::vector<double>& point_times,
                     int max_frames = 5) 
{
    std::cout << "\n--- Initializing Ouster PCAP Stream ---" << std::endl;

    auto info = ouster::sdk::core::metadata_from_json(json_meta_file);
    auto pf = ouster::sdk::core::get_format(info); 

    ouster::sdk::core::ScanBatcher batcher(info);
    ouster::sdk::core::LidarScan scan(info);

    std::cout << "Generating Cartesian Lookup Table (LUT)..." << std::endl;
    auto lut = ouster::sdk::core::make_xyz_lut(info, false);

    auto pcap = ouster::sdk::pcap::replay_initialize(pcap_file);
    if (!pcap) {
        std::cerr << "Error: Could not open PCAP file: " << pcap_file << std::endl;
        return false;
    }

    std::vector<uint8_t> packet_buf(65536);
    int frame_count = 0;
    int packet_count = 0;
    double last_imu_time = 0.0;

    std::cout << "Successfully loaded PCAP and Metadata. Beginning stream..." << std::endl;

    while (true) {
        ouster::sdk::pcap::PacketInfo pi;
        if (!ouster::sdk::pcap::next_packet_info(*pcap, pi)) break;
        
        size_t packet_size = ouster::sdk::pcap::read_packet(*pcap, packet_buf.data(), packet_buf.size());
        if (packet_size == 0) break;

        const uint8_t* raw_packet_data = packet_buf.data();
        packet_count++;

        if (packet_size == pf.imu_packet_size) {
            ImuMeasurement imu;
            imu.timestamp = pf.imu_sys_ts(raw_packet_data) / 1e9;
            last_imu_time = imu.timestamp;
            
            constexpr double GRAVITY = 9.80665;
            imu.acceleration = Eigen::Vector3d(
                pf.imu_la_x(raw_packet_data) * GRAVITY,
                pf.imu_la_y(raw_packet_data) * GRAVITY,
                pf.imu_la_z(raw_packet_data) * GRAVITY
            );

            imu.angular_velocity = Eigen::Vector3d(
                pf.imu_av_x(raw_packet_data) * M_PI / 180.0, 
                pf.imu_av_y(raw_packet_data) * M_PI / 180.0,
                pf.imu_av_z(raw_packet_data) * M_PI / 180.0
            );

            trajectory.addImuMeasurement(imu);
        }
        else if (packet_size == pf.lidar_packet_size) {
            
            ouster::sdk::core::LidarPacket lidar_packet(packet_size);
            
            if (lidar_packet.buf.size() < packet_size) {
                lidar_packet.buf.resize(packet_size);
            }
            
            std::memcpy(lidar_packet.buf.data(), raw_packet_data, packet_size);

            if (batcher(lidar_packet, scan)) {
                frame_count++;
            
                auto ouster_points = ouster::sdk::core::cartesian(scan, lut);
                double current_frame_time = last_imu_time;

                for (Eigen::Index i = 0; i < ouster_points.rows(); ++i) {
                    double x = ouster_points(i, 0);
                    double y = ouster_points(i, 1);
                    double z = ouster_points(i, 2);

                    if (std::abs(x) < 1e-6 && std::abs(y) < 1e-6 && std::abs(z) < 1e-6) {
                        continue; 
                    }

                    global_cloud->points_.emplace_back(x, y, z);
                    point_times.push_back(current_frame_time);
                }

                std::cout << "Extracted Frame " << frame_count << " (Total Points: " 
                          << global_cloud->points_.size() << ")" << std::endl;

                if (frame_count >= max_frames) {
                    std::cout << "Reached max frame limit (" << max_frames << "). Stopping stream." << std::endl;
                    break;
                }
            }
        } 
    }
    
    std::cout << "Stream finished. Processed " << packet_count << " total packets." << std::endl;

    return true;
}

bool exportToOBJ(const std::string& filename, 
                 const std::vector<std::vector<Eigen::Vector3d>>& grid_3d) 
{
    std::cout << "\n--- Exporting Curvilinear OBJ 3D Mesh ---" << std::endl;

    if (grid_3d.empty() || grid_3d[0].empty()) return false;

    std::ofstream obj_file(filename);
    if (!obj_file.is_open()) return false;

    size_t num_u = grid_3d.size();
    size_t num_v = grid_3d[0].size();

    std::vector<std::vector<size_t>> vertex_indices(num_u, std::vector<size_t>(num_v, 0));
    size_t current_vertex_index = 1; 

    obj_file << "# Generated by CTA_LIO Curvilinear Pipeline\n";
    obj_file << "o Offroad_Terrain_Spine\n";
    obj_file << std::fixed << std::setprecision(4);

    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            Eigen::Vector3d vertex = grid_3d[u][v];
            if (!std::isnan(vertex.z())) {
                obj_file << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << "\n";
                vertex_indices[u][v] = current_vertex_index++;
            }
        }
    }

    size_t face_count = 0;
    for (size_t u = 0; u < num_u - 1; ++u) {
        for (size_t v = 0; v < num_v - 1; ++v) {
            size_t top_left     = vertex_indices[u][v];
            size_t top_right    = vertex_indices[u][v+1];
            size_t bottom_left  = vertex_indices[u+1][v];
            size_t bottom_right = vertex_indices[u+1][v+1];

            if (top_left && top_right && bottom_left && bottom_right) {
                obj_file << "f " << top_left << " " << bottom_left << " " << top_right << "\n";
                obj_file << "f " << top_right << " " << bottom_left << " " << bottom_right << "\n";
                face_count += 2;
            }
        }
    }

    obj_file.close();
    std::cout << "Successfully exported " << current_vertex_index - 1 << " vertices to " << filename << std::endl;
    return true;
}

bool exportToOpenCRG_Curved(const std::string& filename, 
                            const std::vector<std::vector<Eigen::Vector3d>>& grid_3d,
                            const TrajectoryInterpolator& trajectory,
                            double sweep_start_time,
                            double u_start, double u_inc, 
                            double v_start, double v_inc) 
{
    std::cout << "\n--- Exporting Curvilinear OpenCRG ---" << std::endl;

    if (grid_3d.empty() || grid_3d[0].empty()) return false;

    size_t num_u = grid_3d.size();
    size_t num_v = grid_3d[0].size();

    double u_end = u_start + (num_u - 1) * u_inc;
    double v_end = v_start + (num_v - 1) * v_inc;

    std::ofstream crg_file(filename);
    if (!crg_file.is_open()) return false;

    crg_file << std::fixed << std::setprecision(6);

    crg_file << "* OpenCRG Curvilinear Elevation Grid\n";
    crg_file << "$CT 1.2\n";
    crg_file << "$UB " << u_start << "\n$UE " << u_end << "\n$UI " << u_inc << "\n";
    crg_file << "$VB " << v_start << "\n$VE " << v_end << "\n$VI " << v_inc << "\n";

    crg_file << "* Reference Line X Coordinates\n";
    crg_file << "$X " << num_u << "\n";
    for (size_t u = 0; u < num_u; ++u) {
        double t = (u_start + u * u_inc) / 10.0; 
        crg_file << trajectory.getPoseAtTime(t)(0, 3) << " ";
    }
    crg_file << "\n";

    crg_file << "* Reference Line Y Coordinates\n";
    crg_file << "$Y " << num_u << "\n";
    for (size_t u = 0; u < num_u; ++u) {
        double t = (u_start + u * u_inc) / 10.0;
        crg_file << trajectory.getPoseAtTime(t)(1, 3) << " ";
    }
    crg_file << "\n";

    crg_file << "* Reference Line Heading (PHI)\n";
    crg_file << "$PHI " << num_u << "\n";
    for (size_t u = 0; u < num_u; ++u) {
        double t = sweep_start_time + ((u_start + u * u_inc) / 10.0);
        Eigen::Matrix4d pose = trajectory.getPoseAtTime(t);
        double yaw = std::atan2(pose(1, 0), pose(0, 0)); 
        crg_file << yaw << " ";
    }
    crg_file << "\n";

    crg_file << "* Elevation Data Block\n";
    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            double z_val = grid_3d[u][v].z(); 
            if (std::isnan(z_val)) {
                crg_file << "NaN "; 
            } else {
                crg_file << z_val << " ";
            }
        }
        crg_file << "\n"; 
    }

    crg_file.close();
    std::cout << "Successfully exported curved OpenCRG to " << filename << std::endl;
    return true;
}

std::vector<std::vector<Eigen::Vector3d>> generateCurvilinearGridIDW(
    const std::shared_ptr<open3d::geometry::PointCloud>& ground_cloud,
    const std::vector<double>& grid_u, 
    const std::vector<double>& grid_v,
    const TrajectoryInterpolator& trajectory,
    double sweep_start_time,
    double search_radius = 2.0) 
{
    std::cout << "\n--- Generating Curvilinear Surface Grid (Spine-Aligned) ---" << std::endl;

    auto flat_cloud = std::make_shared<open3d::geometry::PointCloud>();
    flat_cloud->points_ = ground_cloud->points_; 
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < flat_cloud->points_.size(); ++i) {
        flat_cloud->points_[i].z() = 0.0; 
    }

    open3d::geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*flat_cloud);

    size_t num_u = grid_u.size(); 
    size_t num_v = grid_v.size(); 
    int k_neighbors = 8; 
    
    std::vector<std::vector<Eigen::Vector3d>> grid_3d(num_u, std::vector<Eigen::Vector3d>(num_v));

    double velocity_x = 10.0; 

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            
            double t = sweep_start_time + (grid_u[u] / velocity_x);
            Eigen::Matrix4d T_spine = trajectory.getPoseAtTime(t);

            Eigen::Vector4d p_local(0.0, grid_v[v], 0.0, 1.0);
            Eigen::Vector4d p_global = T_spine * p_local;

            Eigen::Vector3d query_point(p_global.x(), p_global.y(), 0.0);
            
            std::vector<int> indices;
            std::vector<double> distances_sq;
            int num_neighbors = kdtree.SearchHybrid(query_point, search_radius, k_neighbors, indices, distances_sq);

            double interpolated_z = std::numeric_limits<double>::quiet_NaN();

            if (num_neighbors > 0) {
                double weighted_z_sum = 0.0;
                double weight_sum = 0.0;

                for (int i = 0; i < num_neighbors; ++i) {
                    double dist = std::sqrt(distances_sq[i]);
                    if (dist < 1e-6) {
                        weighted_z_sum = ground_cloud->points_[indices[i]].z();
                        weight_sum = 1.0;
                        break; 
                    }
                    double weight = 1.0 / dist; 
                    weighted_z_sum += weight * ground_cloud->points_[indices[i]].z();
                    weight_sum += weight;
                }
                interpolated_z = weighted_z_sum / weight_sum;
            }

            grid_3d[u][v] = Eigen::Vector3d(p_global.x(), p_global.y(), interpolated_z);
        }
    }

    std::cout << "Curvilinear Grid Generation Complete." << std::endl;
    return grid_3d;
}

std::shared_ptr<open3d::geometry::PointCloud> extractBareEarthCSF(
    const std::shared_ptr<open3d::geometry::PointCloud>& cloud)
{
    std::cout << "\n--- Running Cloth Simulation Filter (CSF) ---" << std::endl;

    pdal::PointTable table;
    table.layout()->registerDim(pdal::Dimension::Id::X);
    table.layout()->registerDim(pdal::Dimension::Id::Y);
    table.layout()->registerDim(pdal::Dimension::Id::Z);
    table.layout()->registerDim(pdal::Dimension::Id::Classification);

    size_t total_in = cloud->points_.size();
    size_t update_step = std::max<size_t>(1, total_in / 100);
    std::cout << "1/3: Packing Open3D data for PDAL pipeline..." << std::endl;
    
    pdal::PointViewPtr input_view(new pdal::PointView(table));
    for (size_t i = 0; i < total_in; ++i) {
        input_view->setField(pdal::Dimension::Id::X, i, cloud->points_[i].x());
        input_view->setField(pdal::Dimension::Id::Y, i, cloud->points_[i].y());
        input_view->setField(pdal::Dimension::Id::Z, i, cloud->points_[i].z());
        input_view->setField(pdal::Dimension::Id::Classification, i, 0); 
        
        if (i % update_step == 0 || i == total_in - 1) {
            int pct = (i * 100) / total_in;
            std::cout << "\r  [" << pct << "%] Packed " << i << " / " << total_in << " points" << std::flush;
        }
    }
    std::cout << std::endl; 

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

    std::cout << "2/3: Executing CSF Math (This will take a moment)... " << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();

    pdal::PointTable out_table;
    csf_filter->prepare(out_table);
    pdal::PointViewSet out_views = csf_filter->execute(out_table);
    pdal::PointViewPtr out_view = *out_views.begin();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Done! (" << elapsed.count() << " seconds)\n";

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

        if (id % update_step_out == 0 || id == num_out_points - 1) {
            int pct = (id * 100) / num_out_points;
            std::cout << "\r  [" << pct << "%] Scanned " << id << " / " << num_out_points << std::flush;
        }
    }
    std::cout << std::endl; 

    std::cout << "CSF Complete: Extracted " << ground_count << " ground points from " 
              << cloud->points_.size() << " total points." << std::endl;

    return cloth_cloud;
}

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

    Eigen::Matrix4d T_base = trajectory.getPoseAtTime(sweep_start_time);
    Eigen::Matrix4d T_base_inv = T_base.inverse();

    std::atomic<size_t> points_processed(0);
    size_t update_step = std::max<size_t>(1, num_points / 100); 

    std::cout << "Starting parallel deskewing..." << std::endl;

    #pragma omp parallel 
    {
        size_t local_processed = 0; 

        #pragma omp for schedule(static)
        for (size_t i = 0; i < num_points; ++i) {
            
            double t_p = point_times[i];
            Eigen::Matrix4d T_point = trajectory.getPoseAtTime(t_p);
            Eigen::Matrix4d T_rel = T_base_inv * T_point;
            
            Eigen::Vector4d p_raw(
                raw_cloud->points_[i](0), 
                raw_cloud->points_[i](1), 
                raw_cloud->points_[i](2), 
                1.0
            );
            
            Eigen::Vector4d p_deskewed = T_rel * p_raw;
            deskewed_cloud->points_[i] = p_deskewed.head<3>();

            local_processed++;

            if (local_processed == 1000) {
                size_t current_count = (points_processed += local_processed);
                local_processed = 0; 
                
                if (current_count % update_step < 1000) {
                    int percentage = (current_count * 100) / num_points;
                    
                    #pragma omp critical 
                    {
                        std::cout << "\r[" << percentage << "%] Processed " 
                                  << current_count << " / " << num_points << " points" << std::flush;
                    }
                }
            }
        }
        points_processed += local_processed;
    }
    
    std::cout << "\nDeskewing Complete." << std::endl;

    return deskewed_cloud;
}

int main() {
    std::cout << "=== CTA_LIO: Pipeline Execution Start ===" << std::endl;

    auto raw_cloud = std::make_shared<open3d::geometry::PointCloud>();
    auto cloth_cloud = std::make_shared<open3d::geometry::PointCloud>();

    std::string pcap_file = "../data/test/ouster/Urban_Drive.pcap";
    std::string json_file = "../data/test/ouster/Urban_Drive.json";
    
    std::string cache_file = "../data/test/ouster/Urban_Drive.pcd";
    std::string crg_file   = "../data/test/ouster/Urban_Drive.crg";
    std::string obj_file   = "../data/test/ouster/Urban_Drive.obj";

    std::vector<double> point_times;
    TrajectoryInterpolator trajectory;

    if (!std::filesystem::exists(pcap_file) || !std::filesystem::exists(json_file)) {
        std::cerr << "CRITICAL ERROR: Could not find the .pcap or .json files !" << std::endl;
        return -1;
    }

    if (!parseOusterPcap(pcap_file, json_file, trajectory, raw_cloud, point_times, 5)) {
        return -1;
    }

    double sweep_start_time = point_times[0]; 

    std::cout << "Deskewing using " << omp_get_max_threads() << " CPU threads..." << std::endl;
    auto deskewed_cloud = deskewPointCloudParallel(raw_cloud, point_times, trajectory, sweep_start_time);

    std::cout << "\n[Temporary] Cropping cloud for rapid testing..." << std::endl;
    Eigen::Vector3d min_bound(-300.0, -300.0, -1000.0); 
    Eigen::Vector3d max_bound( 300.0,  300.0,  1000.0);
    
    open3d::geometry::AxisAlignedBoundingBox bbox(min_bound, max_bound);
    auto cropped_cloud = deskewed_cloud->Crop(bbox);
    
    std::cout << "Cropped from " << deskewed_cloud->points_.size() 
              << " to " << cropped_cloud->points_.size() << " points." << std::endl;

    std::cout << "Ground Extraction" << std::endl;
    if (std::filesystem::exists(cache_file)) {
        std::cout << "-- Cache found! Loading ground points..." << std::endl;
        open3d::io::ReadPointCloud(cache_file, *cloth_cloud);
    } else {
        std::cout << "-- No cache found. Running Cloth Simulation Filter..." << std::endl;
        cloth_cloud = extractBareEarthCSF(cropped_cloud); 
        std::cout << "-- Saving ground points to cache..." << std::endl;
        open3d::io::WritePointCloud(cache_file, *cloth_cloud);
    }

    std::cout << "Surface Generation" << std::endl;
    if (cloth_cloud && !cloth_cloud->points_.empty()) {
        
        double duration = point_times.back() - point_times.front();
        double mock_velocity = 10.0; 

        double u_start = -2.0; 
        double u_end   = (duration * mock_velocity) + 2.0; 
        
        double v_start = -15.0; 
        double v_end   =  15.0; 
        
        double u_inc = 0.5; 
        double v_inc = 0.5; 

        std::cout << "Curvilinear Grid Size Detected:" << std::endl;
        std::cout << "  U Domain (Forward): [" << u_start << " to " << u_end << "] meters" << std::endl;
        std::cout << "  V Domain (Lateral): [" << v_start << " to " << v_end << "] meters" << std::endl;

        std::vector<double> grid_u, grid_v;
        for (double u = u_start; u <= u_end; u += u_inc) grid_u.push_back(u);
        for (double v = v_start; v <= v_end; v += v_inc) grid_v.push_back(v);

        auto grid_3d = generateCurvilinearGridIDW(cloth_cloud, grid_u, grid_v, trajectory, sweep_start_time, 3.0);

        std::cout << "-- Generating OpenCRG" << std::endl;
        exportToOpenCRG_Curved(crg_file, grid_3d, trajectory, sweep_start_time, u_start, u_inc, v_start, v_inc);
        
        std::cout << "-- Generating OBJ" << std::endl;
        exportToOBJ(obj_file, grid_3d);

    } else {
        std::cerr << "-- Pipeline Error: No ground points available for CRG/OBJ generation." << std::endl;
    }

    std::cout << "Visualizing the OBJ Mesh and Point Cloud..." << std::endl;
    auto terrain_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    
    if (open3d::io::ReadTriangleMesh(obj_file, *terrain_mesh)) {
        
        terrain_mesh->ComputeVertexNormals();
        terrain_mesh->PaintUniformColor({0.54, 0.27, 0.07});
        terrain_mesh->Translate(Eigen::Vector3d(0.0, 0.0, 5));

        cloth_cloud->PaintUniformColor({0.0, 1.0, 0.0});
        cloth_cloud->Translate(Eigen::Vector3d(0.0, 0.0, 1));

        cropped_cloud->PaintUniformColor({0.5, 0.5, 0.5});

        std::cout << "Opening 3D Visualizer..." << std::endl;
        
        open3d::visualization::DrawGeometries(
            {cloth_cloud, cropped_cloud, terrain_mesh}, 
            "OpenCRG Terrain Mesh with LiDAR Overlay"
        );
    } else {
        std::cerr << "Failed to load the OBJ file for visualization." << std::endl;
    }

    return 0;
}