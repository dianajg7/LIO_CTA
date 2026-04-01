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

#include <cstring>         // <-- ADD THIS for std::memcpy
#include <ouster/client.h>
#include <ouster/lidar_scan.h>
#include <ouster/types.h>
#include <ouster/xyzlut.h>
#include <ouster/os_pcap.h> // <-- RESTORE THIS
#include <ouster/pcap.h>    // <-- KEEP THIS


struct ImuMeasurement {
    double timestamp; // In seconds
    Eigen::Vector3d acceleration; // Linear acceleration (m/s^2)
    Eigen::Vector3d angular_velocity; // Gyroscope (rad/s)
};

// class TrajectoryInterpolator {
// private:
//     std::deque<ImuMeasurement> imu_buffer_;
//     mutable std::mutex buffer_mutex_; // <-- FIX 1: Add 'mutable' here

// public:
//     // <-- FIX 2: Add this missing method
//     void addImuMeasurement(const ImuMeasurement& imu) {
//         std::lock_guard<std::mutex> lock(buffer_mutex_);
//         imu_buffer_.push_back(imu);
//     }

//     void clearOldData(double up_to_time) {
//         std::lock_guard<std::mutex> lock(buffer_mutex_);
//         while (!imu_buffer_.empty() && imu_buffer_.front().timestamp < up_to_time) {
//             imu_buffer_.pop_front();
//         }
//     }

//     Eigen::Matrix4d getPoseAtTime(double timestamp) const {
//         std::lock_guard<std::mutex> lock(buffer_mutex_);
//         // ... [Rest of your S-Curve math remains identical] ...
//         Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        
//         double velocity_x = 10.0;
//         double amplitude = 40.0;
//         double frequency = 0.5;

//         double x = velocity_x * timestamp;
//         double y = amplitude * std::sin(frequency * timestamp);

//         double dy_dt = amplitude * frequency * std::cos(frequency * timestamp);
//         double dx_dt = velocity_x;
//         double yaw = std::atan2(dy_dt, dx_dt);

//         pose(0, 0) = std::cos(yaw);
//         pose(0, 1) = -std::sin(yaw);
//         pose(1, 0) = std::sin(yaw);
//         pose(1, 1) = std::cos(yaw);

//         pose(0, 3) = x;
//         pose(1, 3) = y;
//         pose(2, 3) = 0.0; 

//         return pose;
//     }
// };


struct PoseState {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d velocity;
};

class TrajectoryInterpolator {
private:
    std::deque<PoseState> pose_buffer_;
    mutable std::mutex buffer_mutex_;
    
    bool is_initialized_ = false;
    PoseState current_state_;
    Eigen::Vector3d gravity_{0.0, 0.0, 9.80665}; // Standard gravity vector

    // Helper to convert our state to a 4x4 Homogeneous Matrix
    static Eigen::Matrix4d toMatrix(const PoseState& state) {
        Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
        m.block<3,3>(0,0) = state.orientation.toRotationMatrix();
        m.block<3,1>(0,3) = state.position;
        return m;
    }

public:
    void addImuMeasurement(const ImuMeasurement& imu) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        if (!is_initialized_) {
            current_state_.timestamp = imu.timestamp;
            current_state_.position.setZero();
            current_state_.orientation.setIdentity();
            current_state_.velocity.setZero(); 
            
            pose_buffer_.push_back(current_state_);
            is_initialized_ = true;
            return;
        }

        double dt = imu.timestamp - current_state_.timestamp;
        if (dt <= 0) return;

        // 1. Update Orientation (Integrate Angular Velocity)
        Eigen::Vector3d w = imu.angular_velocity;
        Eigen::Quaterniond dq;
        double w_norm = w.norm();
        if (w_norm > 1e-5) {
            dq = Eigen::Quaterniond(Eigen::AngleAxisd(w_norm * dt, w / w_norm));
        } else {
            dq = Eigen::Quaterniond::Identity();
        }
        Eigen::Quaterniond next_q = current_state_.orientation * dq;
        next_q.normalize(); // Prevent floating point drift in the quaternion

        // 2. Update Velocity & Position (Integrate Acceleration)
        // Transform the local IMU acceleration into the global frame
        Eigen::Vector3d a_global = current_state_.orientation * imu.acceleration;
        
        // Remove the Earth's gravity vector to find true linear acceleration
        Eigen::Vector3d a_true = a_global - gravity_;

        // Standard kinematics: p = p_0 + v_0*t + 0.5*a*t^2
        Eigen::Vector3d next_p = current_state_.position + 
                                 (current_state_.velocity * dt) + 
                                 (0.5 * a_true * dt * dt);
        
        // v = v_0 + a*t
        Eigen::Vector3d next_v = current_state_.velocity + (a_true * dt);

        // 3. Save the new state
        current_state_.timestamp = imu.timestamp;
        current_state_.position = next_p;
        current_state_.orientation = next_q;
        current_state_.velocity = next_v;

        pose_buffer_.push_back(current_state_);
    }

    void clearOldData(double up_to_time) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        // Keep at least one element so we don't lose our state entirely
        while (pose_buffer_.size() > 1 && pose_buffer_.front().timestamp < up_to_time) {
            pose_buffer_.pop_front();
        }
    }

    Eigen::Matrix4d getPoseAtTime(double timestamp) const {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        if (pose_buffer_.empty()) return Eigen::Matrix4d::Identity();
        
        // If the requested time is outside our buffer, return the closest known edge
        if (timestamp <= pose_buffer_.front().timestamp) return toMatrix(pose_buffer_.front());
        if (timestamp >= pose_buffer_.back().timestamp) return toMatrix(pose_buffer_.back());

        // Find the two poses that surround our requested timestamp
        auto it_next = std::lower_bound(pose_buffer_.begin(), pose_buffer_.end(), timestamp,
            [](const PoseState& a, double t) { return a.timestamp < t; });
        
        auto it_prev = std::prev(it_next);

        // Calculate how far we are between the two poses (0.0 to 1.0)
        double dt = it_next->timestamp - it_prev->timestamp;
        double alpha = (timestamp - it_prev->timestamp) / dt;

        // Linear interpolation for translation
        Eigen::Vector3d interp_p = it_prev->position + alpha * (it_next->position - it_prev->position);
        
        // Spherical Linear Interpolation (SLERP) for rotation
        Eigen::Quaterniond interp_q = it_prev->orientation.slerp(alpha, it_next->orientation);

        // Construct and return the interpolated 4x4 matrix
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3,3>(0,0) = interp_q.toRotationMatrix();
        pose.block<3,1>(0,3) = interp_p;

        return pose;
    }
};


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

    // --- FIX 1: Generate the massive Lookup Table ONCE before the loop ---
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
        ouster::sdk::pcap::packet_info pi;
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
            
            // --- FIX 2: Force memory allocation before memcpy ---
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


void parseOusterPcap(const std::string& pcap_file, 
                     const std::string& json_meta_file,
                     TrajectoryInterpolator& trajectory) 
{
    std::cout << "\n--- Initializing Ouster PCAP Stream ---" << std::endl;

    auto info = ouster::sdk::core::metadata_from_json(json_meta_file);
    auto pf = ouster::sdk::core::get_format(info); 

    ouster::sdk::core::ScanBatcher batcher(info);
    ouster::sdk::core::LidarScan scan(info);

    auto pcap = ouster::sdk::pcap::replay_initialize(pcap_file);
    if (!pcap) {
        std::cerr << "Error: Could not open PCAP file: " << pcap_file << std::endl;
        return;
    }

    std::cout << "Successfully loaded PCAP and Metadata. Beginning stream..." << std::endl;

    // --- FIX 1: Allocate a buffer to hold the raw packet bytes ---
    std::vector<uint8_t> packet_buf(65536);

    while (true) {
        ouster::sdk::pcap::packet_info pi;
        
        // --- FIX 2: Use the free function to check for the next packet ---
        if (!ouster::sdk::pcap::next_packet_info(*pcap, pi)) {
            break; // End of file
        }
        
        // --- FIX 3: Use the free function to read the actual packet data ---
        size_t packet_size = ouster::sdk::pcap::read_packet(*pcap, packet_buf.data(), packet_buf.size());
        if (packet_size == 0) {
            break;
        }

        const uint8_t* raw_packet_data = packet_buf.data();

        if (packet_size == pf.imu_packet_size) {
            ImuMeasurement imu;
            imu.timestamp = pf.imu_sys_ts(raw_packet_data) / 1e9; 
            
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
            
            // --- FIX 4: Instantiate the derived LidarPacket class directly ---
            ouster::sdk::core::LidarPacket lidar_packet(packet_size);
            
            std::memcpy(lidar_packet.buf.data(), raw_packet_data, packet_size);

            if (batcher(lidar_packet, scan)) {
                std::cout << "Completed full 360 LidarScan Frame. Extracting geometry..." << std::endl;
                
                auto lut = ouster::sdk::core::make_xyz_lut(info, false);
                auto ouster_points = ouster::sdk::core::cartesian(scan, lut);
                
                auto raw_cloud = std::make_shared<open3d::geometry::PointCloud>();
                raw_cloud->points_.reserve(ouster_points.rows());

                for (Eigen::Index i = 0; i < ouster_points.rows(); ++i) {
                    double x = ouster_points(i, 0);
                    double y = ouster_points(i, 1);
                    double z = ouster_points(i, 2);

                    if (std::abs(x) < 1e-6 && std::abs(y) < 1e-6 && std::abs(z) < 1e-6) {
                        continue; 
                    }

                    raw_cloud->points_.emplace_back(x, y, z);
                }

                std::cout << "Successfully mapped " << raw_cloud->points_.size() 
                          << " valid LiDAR hits into Open3D." << std::endl;
            }
        } 
    }
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

    // 1. Write Valid Vertices (Now reading X and Y directly from the curved geometry)
    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            
            Eigen::Vector3d vertex = grid_3d[u][v];
            
            if (!std::isnan(vertex.z())) {
                obj_file << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << "\n";
                vertex_indices[u][v] = current_vertex_index++;
            }
        }
    }

    // 2. Write Triangular Faces (Identical to before)
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

    // 1. Write the Standard Parameters
    crg_file << "* OpenCRG Curvilinear Elevation Grid\n";
    crg_file << "$CT 1.2\n";
    crg_file << "$UB " << u_start << "\n$UE " << u_end << "\n$UI " << u_inc << "\n";
    crg_file << "$VB " << v_start << "\n$VE " << v_end << "\n$VI " << v_inc << "\n";

    // ---------------------------------------------------------
    // 2. Write the Curved Reference Line Arrays
    // ---------------------------------------------------------
    crg_file << "* Reference Line X Coordinates\n";
    crg_file << "$X " << num_u << "\n";
    for (size_t u = 0; u < num_u; ++u) {
        double t = (u_start + u * u_inc) / 10.0; // Using the 10.0m/s mock velocity
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
        // double t = (u_start + u * u_inc) / 10.0;
        double t = sweep_start_time + ((u_start + u * u_inc) / 10.0);
        Eigen::Matrix4d pose = trajectory.getPoseAtTime(t);
        // Extract Yaw from the Rotation Matrix
        double yaw = std::atan2(pose(1, 0), pose(0, 0)); 
        crg_file << yaw << " ";
    }
    crg_file << "\n";

    // ---------------------------------------------------------
    // 3. Write the Elevation Data Block (Z-values only)
    // ---------------------------------------------------------
    crg_file << "* Elevation Data Block\n";
    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            double z_val = grid_3d[u][v].z(); // Extract strictly the Z component
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

bool exportToOpenCRG(const std::string& filename, 
                     const std::vector<std::vector<double>>& z_matrix,
                     double u_start, double u_inc, 
                     double v_start, double v_inc) 
{
    std::cout << "\n--- Exporting to OpenCRG Format ---" << std::endl;

    if (z_matrix.empty() || z_matrix[0].empty()) {
        std::cerr << "Error: Z-matrix is empty." << std::endl;
        return false;
    }

    size_t num_u = z_matrix.size();
    size_t num_v = z_matrix[0].size();

    double u_end = u_start + (num_u - 1) * u_inc;
    double v_end = v_start + (num_v - 1) * v_inc;

    std::ofstream crg_file(filename);
    if (!crg_file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return false;
    }

    crg_file << std::fixed << std::setprecision(6);

    crg_file << "* OpenCRG Elevation Grid generated by CTA_LIO Pipeline\n";
    crg_file << "* Format Version\n";
    crg_file << "$CT 1.2\n";
    
    crg_file << "* Longitudinal (U) Grid Parameters\n";
    crg_file << "$UB " << u_start << "\n";
    crg_file << "$UE " << u_end << "\n";
    crg_file << "$UI " << u_inc << "\n";
    
    crg_file << "* Lateral (V) Grid Parameters\n";
    crg_file << "$VB " << v_start << "\n";
    crg_file << "$VE " << v_end << "\n";
    crg_file << "$VI " << v_inc << "\n";

    crg_file << "* Reference Line Origin\n";
    crg_file << "$X0 0.000000\n";
    crg_file << "$Y0 0.000000\n";
    crg_file << "$PHI0 0.000000\n";

    crg_file << "* Elevation Data Block Starts Here\n";

    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            double z_val = z_matrix[u][v];
            if (std::isnan(z_val)) {
                crg_file << "NaN "; 
            } else {
                crg_file << z_val << " ";
            }
        }
        crg_file << "\n";
    }

    crg_file.close();
    std::cout << "Successfully wrote " << num_u * num_v << " grid nodes to " << filename << std::endl;
    
    return true;
}

std::vector<std::vector<double>> generateSurfaceGridIDW(
    const std::shared_ptr<open3d::geometry::PointCloud>& ground_cloud,
    const std::vector<double>& grid_x, 
    const std::vector<double>& grid_y,
    double search_radius = 2.0)
{
    std::cout << "\n--- Generating Structured Surface Grid (2.5D Hybrid IDW) ---" << std::endl;

    // 1. Create a flattened proxy cloud for the KD-Tree
    auto flat_cloud = std::make_shared<open3d::geometry::PointCloud>();
    flat_cloud->points_ = ground_cloud->points_; 
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < flat_cloud->points_.size(); ++i) {
        flat_cloud->points_[i].z() = 0.0; // Crush all points to Z=0
    }

    // 2. Build the KD-Tree on the FLATTENED cloud
    open3d::geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*flat_cloud);

    size_t num_u = grid_x.size(); 
    size_t num_v = grid_y.size(); 
    
    int k_neighbors = 8; 
    
    std::vector<std::vector<double>> z_matrix(num_u, std::vector<double>(num_v, std::numeric_limits<double>::quiet_NaN()));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            
            Eigen::Vector3d query_point(grid_x[u], grid_y[v], 0.0);
            
            std::vector<int> indices;
            std::vector<double> distances_sq;

            int num_neighbors = kdtree.SearchHybrid(query_point, search_radius, k_neighbors, indices, distances_sq);

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

                    // Inverse Distance Weighting
                    double weight = 1.0 / dist; 
                    weighted_z_sum += weight * ground_cloud->points_[indices[i]].z();
                    weight_sum += weight;
                }

                z_matrix[u][v] = weighted_z_sum / weight_sum;
            }
        }
    }

    std::cout << "Surface Grid Generation Complete (Hybrid Mode)." << std::endl;
    return z_matrix;
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
    
    // Matrix now holds the full 3D coordinate of the vertex, not just Z
    std::vector<std::vector<Eigen::Vector3d>> grid_3d(num_u, std::vector<Eigen::Vector3d>(num_v));

    // The mock velocity used in your TrajectoryInterpolator
    double velocity_x = 10.0; 

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            
            // 1. Calculate the time 't' for this longitudinal position 'U'
            // double t = grid_u[u] / velocity_x;
            double t = sweep_start_time + (grid_u[u] / velocity_x);

            // 2. Get the 6-DoF pose of the spine at this exact moment
            Eigen::Matrix4d T_spine = trajectory.getPoseAtTime(t);

            // 3. Define the local grid node (0 meters forward, V meters lateral)
            // In the vehicle's local frame, X is forward, Y is lateral/left
            Eigen::Vector4d p_local(0.0, grid_v[v], 0.0, 1.0);

            // 4. Transform the local node into global coordinates
            Eigen::Vector4d p_global = T_spine * p_local;

            // Query the KD-Tree using the new curved global coordinates
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

            // 5. Store the final curved 3D vertex
            grid_3d[u][v] = Eigen::Vector3d(p_global.x(), p_global.y(), interpolated_z);
        }
    }

    std::cout << "Curvilinear Grid Generation Complete." << std::endl;
    return grid_3d;
}

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

    // We open the parallel region early so we can create thread-local variables
    #pragma omp parallel 
    {
        // This variable exists independently inside every thread!
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

            // Increment the private counter (Zero contention)
            local_processed++;

            // Batch update the global atomic counter every 1000 points
            if (local_processed == 1000) {
                size_t current_count = (points_processed += local_processed);
                local_processed = 0; // Reset local counter
                
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
        
        // Add any leftover points from this thread to the global counter
        points_processed += local_processed;
    }
    
    std::cout << "\nDeskewing Complete." << std::endl;

    return deskewed_cloud;
}

int main() {
    std::cout << "=== CTA_LIO: Pipeline Execution Start ===" << std::endl;

    // auto raw_cloud = std::make_shared<open3d::geometry::PointCloud>();
    // auto cloth_cloud = std::make_shared<open3d::geometry::PointCloud>();

    // std::string input_file = "../data/autzen.laz";
    // std::string cache_file = "../data/autzen_ground.pcd";
    // std::string crg_file   = "../data/autzen_surface.crg";
    // std::string obj_file   = "../data/autzen_surface.obj";

    // std::vector<double> point_times;

    // if (!loadLiDARDataWithTime(input_file, raw_cloud, point_times)) {
    //     return -1;
    // }

    // // NOTE: For mock data, pre-centering is okay. For real INS data, remove this
    // // or offset the trajectory accordingly.
    // Eigen::Vector3d center = raw_cloud->GetCenter();
    // raw_cloud->Translate(-center);

    // TrajectoryInterpolator trajectory;

    //PCAP TEST START----------------------------
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

    // Eigen::Vector3d center = raw_cloud->GetCenter();
    // raw_cloud->Translate(-center);
    //PCAP TEST END---------------------------

    double sweep_start_time = point_times[0]; 

    std::cout << "Deskewing using " << omp_get_max_threads() << " CPU threads..." << std::endl;
    auto deskewed_cloud = deskewPointCloudParallel(raw_cloud, point_times, trajectory, sweep_start_time);

    // ---------------------------------------------------------
    // TEMPORARY FILTER: Crop to a specific test area
    // ---------------------------------------------------------
    std::cout << "\n[Temporary] Cropping cloud for rapid testing..." << std::endl;
    
    // Define the min and max coordinates for your X, Y square.
    // We set Z to extreme values (-1000 to +1000) so we don't accidentally chop off hilltops or deep ravines.
    Eigen::Vector3d min_bound(-300.0, -300.0, -1000.0); 
    Eigen::Vector3d max_bound( 300.0,  300.0,  1000.0);
    
    open3d::geometry::AxisAlignedBoundingBox bbox(min_bound, max_bound);
    
    // Create a new, drastically smaller point cloud
    auto cropped_cloud = deskewed_cloud->Crop(bbox);
    
    std::cout << "Cropped from " << deskewed_cloud->points_.size() 
              << " to " << cropped_cloud->points_.size() << " points." << std::endl;

    // ---------------------------------------------------------

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
        ////TEST SPINE LINE!!! START
        // Eigen::Vector3d min_bound = cloth_cloud->GetMinBound();
        // Eigen::Vector3d max_bound = cloth_cloud->GetMaxBound();
        // double buffer = 2.0; 
        // double u_start = min_bound.x() - buffer;
        // double u_end   = max_bound.x() + buffer;
        // double v_start = min_bound.y() - buffer;
        // double v_end   = max_bound.y() + buffer;
        // double u_inc = 0.25; 
        // double v_inc = 0.25; 
        // std::cout << "Dynamic Grid Size Detected:" << std::endl;
        // std::cout << "  Length (U): " << (u_end - u_start) << " meters" << std::endl;
        // std::cout << "  Width  (V): " << (v_end - v_start) << " meters" << std::endl;
        // std::vector<double> grid_u, grid_v;
        // for (double u = u_start; u <= u_end; u += u_inc) grid_u.push_back(u);
        // for (double v = v_start; v <= v_end; v += v_inc) grid_v.push_back(v);


        // double u_start = -200.0, u_inc = 0.25, u_end = 200.0;
        // double v_start = -10.0, v_inc = 0.25, v_end = 10.0;
        // std::vector<double> grid_u, grid_v;
        // for (double u = u_start; u <= u_end; u += u_inc) grid_u.push_back(u);
        // for (double v = v_start; v <= v_end; v += v_inc) grid_v.push_back(v);
        // //TEST SPINE LINE!!! END

        // // auto z_matrix = generateSurfaceGridIDW(cloth_cloud, grid_u, grid_v, 3.0);
        // auto grid_3d = generateCurvilinearGridIDW(cloth_cloud, grid_u, grid_v, trajectory, 3.0);

        // if (!std::filesystem::exists(crg_file)) {
        //     std::cout << "-- Generating OpenCRG" << std::endl;
        //     exportToOpenCRG_Curved(crg_file, grid_3d, trajectory, u_start, u_inc, v_start, v_inc);
        //     // exportToOpenCRG(crg_file, z_matrix, u_start, u_inc, v_start, v_inc);
        // }
        // if (!std::filesystem::exists(obj_file)) {
        //     std::cout << "-- Generating OBJ" << std::endl;
        //     exportToOBJ(obj_file, grid_3d);
        //     // exportToOBJ(obj_file, z_matrix, u_start, u_inc, v_start, v_inc);
        // }


        // // --- FIX 6: Use the Dynamic Grid Size! ---
        // Eigen::Vector3d min_bound = cloth_cloud->GetMinBound();
        // Eigen::Vector3d max_bound = cloth_cloud->GetMaxBound();
        
        // // Give the grid a 1-meter padding around the actual ground points
        // double buffer = 1.0; 
        // double u_start = min_bound.x() - buffer;
        // double u_end   = max_bound.x() + buffer;
        // double v_start = min_bound.y() - buffer;
        // double v_end   = max_bound.y() + buffer;
        // double u_inc = 0.5; // Slightly coarser resolution for faster testing
        // double v_inc = 0.5; 

        // std::cout << "Dynamic Grid Size Detected:" << std::endl;
        // std::cout << "  U Domain: [" << u_start << " to " << u_end << "] meters" << std::endl;
        // std::cout << "  V Domain: [" << v_start << " to " << v_end << "] meters" << std::endl;

        // std::vector<double> grid_u, grid_v;
        // for (double u = u_start; u <= u_end; u += u_inc) grid_u.push_back(u);
        // for (double v = v_start; v <= v_end; v += v_inc) grid_v.push_back(v);

        // // Pass the new sweep_start_time parameter!
        // auto grid_3d = generateCurvilinearGridIDW(cloth_cloud, grid_u, grid_v, trajectory, sweep_start_time, 3.0);

        // if (!std::filesystem::exists(crg_file)) {
        //     exportToOpenCRG_Curved(crg_file, grid_3d, trajectory, sweep_start_time, u_start, u_inc, v_start, v_inc);
        // }
        // if (!std::filesystem::exists(obj_file)) {
        //     exportToOBJ(obj_file, grid_3d);
        // }


        // --- FIX 1: Calculate U based on Time, not Global X ---
        // How much time passed between the first and last LiDAR point?
        double duration = point_times.back() - point_times.front();
        
        // In generateCurvilinearGridIDW, your mock velocity is hardcoded to 10.0
        double mock_velocity = 10.0; 

        // U is Longitudinal (Forward) distance. 
        double u_start = -2.0; // Give 2m padding behind the start
        double u_end   = (duration * mock_velocity) + 2.0; // Distance = v*t (plus padding)
        
        // V is Lateral (Left/Right) distance from the car.
        double v_start = -15.0; // 15 meters left
        double v_end   =  15.0; // 15 meters right
        
        double u_inc = 0.5; 
        double v_inc = 0.5; 

        std::cout << "Curvilinear Grid Size Detected:" << std::endl;
        std::cout << "  U Domain (Forward): [" << u_start << " to " << u_end << "] meters" << std::endl;
        std::cout << "  V Domain (Lateral): [" << v_start << " to " << v_end << "] meters" << std::endl;

        std::vector<double> grid_u, grid_v;
        for (double u = u_start; u <= u_end; u += u_inc) grid_u.push_back(u);
        for (double v = v_start; v <= v_end; v += v_inc) grid_v.push_back(v);

        auto grid_3d = generateCurvilinearGridIDW(cloth_cloud, grid_u, grid_v, trajectory, sweep_start_time, 3.0);

        // --- FIX 2: Remove the std::filesystem::exists checks so we overwrite old data! ---
        std::cout << "-- Generating OpenCRG" << std::endl;
        exportToOpenCRG_Curved(crg_file, grid_3d, trajectory, sweep_start_time, u_start, u_inc, v_start, v_inc);
        
        std::cout << "-- Generating OBJ" << std::endl;
        exportToOBJ(obj_file, grid_3d);

    } else {
        std::cerr << "-- Pipeline Error: No ground points available for CRG/OBJ generation." << std::endl;
    }

    std::cout << "Visualizing the OBJ Mesh and Point Cloud..." << std::endl;
    auto terrain_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    
    // Read the newly created OBJ file back into memory
    if (open3d::io::ReadTriangleMesh(obj_file, *terrain_mesh)) {
        
        terrain_mesh->ComputeVertexNormals();
        terrain_mesh->PaintUniformColor({0.54, 0.27, 0.07});
        terrain_mesh->Translate(Eigen::Vector3d(0.0, 0.0, 5));

        cloth_cloud->PaintUniformColor({0.0, 1.0, 0.0});
        cloth_cloud->Translate(Eigen::Vector3d(0.0, 0.0, 1));

        cropped_cloud->PaintUniformColor({0.5, 0.5, 0.5});

        std::cout << "Opening 3D Visualizer..." << std::endl;
        
        // Pass BOTH geometries into the array
        open3d::visualization::DrawGeometries(
            {cloth_cloud, cropped_cloud, terrain_mesh}, 
            "OpenCRG Terrain Mesh with LiDAR Overlay"
        );
    } else {
        std::cerr << "Failed to load the OBJ file for visualization." << std::endl;
    }

    return 0;
}

