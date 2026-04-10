#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <limits>
#include <deque>
#include <mutex>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdint>

#include <Eigen/Dense>

#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/StageFactory.hpp>

#include <filters/private/csf/CSF.h>

#include <io/LasReader.hpp>
#include <io/BufferReader.hpp>

#include <open3d/Open3D.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/geometry/KDTreeFlann.h>

#include <ouster/client.h>
#include <ouster/lidar_scan.h>
#include <ouster/types.h>
#include <ouster/xyzlut.h>
#include <ouster/os_pcap.h> 
#include <ouster/pcap.h>  

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

struct InsNavState {
    double timestamp = -1.0;
    double heading = 0.0, pitch = 0.0, roll = 0.0;
    double lat = 0.0, lon = 0.0, alt = 0.0;
    double v_e = 0.0, v_n = 0.0, v_u = 0.0;
    
    bool has_ori = false, has_pos = false, has_vel = false;

    bool isComplete() const { return has_ori && has_pos && has_vel; }
    void reset() { has_ori = false; has_pos = false; has_vel = false; }
};

// ====================================================================
// STEAM Continuous-Time Trajectory Interpolator
// ====================================================================
class TrajectoryInterpolator {
private:
    steam::traj::const_vel::Interface traj_; 
    
    bool is_initialized_ = false;
    double first_time_ = 0.0;
    double last_time_ = 0.0;

    bool first_nav_ = true;
    double lat0_ = 0.0, lon0_ = 0.0, alt0_ = 0.0;

    mutable std::mutex buffer_mutex_;

public:
    TrajectoryInterpolator() {
        Eigen::Matrix<double, 6, 1> qc_diag;
        qc_diag.head<3>().setConstant(10.0); 
        qc_diag.tail<3>().setConstant(10.0); 
        traj_ = steam::traj::const_vel::Interface(qc_diag);
    }

    void addNavState(const InsNavState& nav) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (!nav.isComplete()) return;

        // Set the Local Tangent Plane origin on the first pose
        if (first_nav_) {
            lat0_ = nav.lat; 
            lon0_ = nav.lon; 
            alt0_ = nav.alt;
            first_nav_ = false;
        }

        // 1. Convert Lat/Lon/Alt to Local X/Y/Z (Meters)
        const double R_EARTH = 6378137.0; // WGS84 Equatorial Radius
        double lat_rad = nav.lat * M_PI / 180.0;
        double lat0_rad = lat0_ * M_PI / 180.0;
        double d_lon_rad = (nav.lon - lon0_) * M_PI / 180.0;
        double d_lat_rad = (nav.lat - lat0_) * M_PI / 180.0;

        double x = R_EARTH * std::cos(lat0_rad) * d_lon_rad; // Easting
        double y = R_EARTH * d_lat_rad;                      // Northing
        double z = nav.alt - alt0_;                          // Altitude

        // 2. Convert Heading/Pitch/Roll to Rotation Matrix
        // ENU Mapping: X=East, Y=North. Heading is CW from North. 
        // We must rotate the vehicle's X-axis (Forward) to align correctly.
        double yaw_rad = (M_PI / 2.0) - nav.heading; 
        Eigen::AngleAxisd yawAngle(yaw_rad, Eigen::Vector3d::UnitZ()); 
        Eigen::AngleAxisd pitchAngle(nav.pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rollAngle(nav.roll, Eigen::Vector3d::UnitX());
        Eigen::Matrix3d R = (yawAngle * pitchAngle * rollAngle).matrix();

        // 3. Build the SE(3) Transformation
        Eigen::Matrix4d T_matrix = Eigen::Matrix4d::Identity();
        T_matrix.block<3,3>(0,0) = R;
        T_matrix.block<3,1>(0,3) = Eigen::Vector3d(x, y, z);
        lgmath::se3::Transformation T_pose(T_matrix);

        // 4. Convert Velocity to Body Frame
        Eigen::Vector3d v_global(nav.v_e, nav.v_n, nav.v_u); 
        Eigen::Matrix<double, 6, 1> velocity;
        velocity.setZero();
        velocity.head<3>() = R.transpose() * v_global;

        // 5. Add Knot to STEAM Trajectory
        steam::traj::Time knot_time(nav.timestamp);
        auto pose_var = steam::se3::SE3StateVar::MakeShared(T_pose);
        auto vel_var = steam::vspace::VSpaceStateVar<6>::MakeShared(velocity);
        
        traj_.add(knot_time, pose_var, vel_var);

        if (!is_initialized_) {
            first_time_ = nav.timestamp;
            is_initialized_ = true;
        }
        last_time_ = nav.timestamp;
    }

    Eigen::Matrix4d getPoseAtTime(double timestamp) const {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (!is_initialized_) return Eigen::Matrix4d::Identity();

        double clamped_time = std::max(first_time_, std::min(timestamp, last_time_));
        
        if (std::abs(timestamp - clamped_time) > 1e-3) {
            static bool warned = false;
            if (!warned) {
                std::cerr << "\n[WARNING] LiDAR time (" << std::fixed << timestamp 
                          << ") is outside INS trajectory bounds (" 
                          << first_time_ << " to " << last_time_ << ")!\n";
                warned = true;
            }
        }

        steam::traj::Time query_time(clamped_time);
        auto pose_evaluator = traj_.getPoseInterpolator(query_time);
        return pose_evaluator->value().matrix();
    }
    
    steam::traj::const_vel::Interface& getTrajectory() { return traj_; }
};

// ====================================================================
// INS Parsing
// ====================================================================

int64_t parse_48bit_signed(const std::vector<uint8_t>& payload) {
    int64_t val = (static_cast<int64_t>(payload[0]) << 40) | (static_cast<int64_t>(payload[1]) << 32) |
                  (static_cast<int64_t>(payload[2]) << 24) | (static_cast<int64_t>(payload[3]) << 16) |
                  (static_cast<int64_t>(payload[4]) << 8)  |  static_cast<int64_t>(payload[5]);
    if (val & 0x0000800000000000LL) val |= 0xFFFF000000000000LL;
    return val;
}

int32_t parse_32bit_signed(const std::vector<uint8_t>& payload) {
    return static_cast<int32_t>((payload[0] << 24) | (payload[1] << 16) | (payload[2] << 8) | payload[3]);
}

struct SyncBuffer {
    double timestamp = -1.0;
    
    bool has_gyro = false;
    Eigen::Vector3d gyro{0, 0, 0};
    
    bool has_accel = false;
    Eigen::Vector3d accel{0, 0, 0};
    
    InsNavState nav;

    void reset(double new_ts) {
        timestamp = new_ts;
        has_gyro = false;
        has_accel = false;
        nav.reset();
        nav.timestamp = new_ts;
    }
};

bool parseCanTrace(const std::string& trc_file, 
                   TrajectoryInterpolator& trajectory,
                   double time_sync_offset = 0.0) 
{
    std::cout << "\n--- Initializing Auto-Adapting CAN Parser ---" << std::endl;
    
    std::ifstream file(trc_file);
    if (!file.is_open()) return false;

    const double KG = 10.0;     // G2000 (2000 deg/sec)
    const double KA = 500.0;    // A40 (40 g)
    const double DEG_TO_RAD = M_PI / 180.0;
    const double G_TO_MPS2  = 9.80665;

    std::string line;
    SyncBuffer buffer;
    
    int stats_full_imu = 0;
    int stats_gyro_only = 0;
    int stats_nav_states = 0;

    auto processBuffer = [&]() {
        if (buffer.timestamp < 0) return; 
        
        // if (buffer.has_gyro) {
        //     ImuMeasurement imu;
        //     imu.timestamp = buffer.timestamp;
        //     imu.angular_velocity = buffer.gyro;
            
        //     if (buffer.has_accel) {
        //         // TIGHTLY-COUPLED strategy
        //         imu.acceleration = buffer.accel;
        //         trajectory.addImuMeasurement(imu);
        //         stats_full_imu++;
        //     } else {
        //         // LOOSELY-COUPLED strategy
        //         imu.acceleration = Eigen::Vector3d(0.0, 0.0, G_TO_MPS2); 
        //         trajectory.addImuMeasurement(imu);
        //         stats_gyro_only++;
        //     }
        // }

        if (buffer.nav.isComplete()) {
            // LOOSELY-COUPLED: Add the INS-D solution to the STEAM trajectory
            trajectory.addNavState(buffer.nav);
            stats_nav_states++;
        }
    };

    while (std::getline(file, line)) {
        if (line.empty() || line.find(';') != std::string::npos) continue; 

        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) tokens.push_back(token);

        if (tokens.size() < 5) continue;

        try {
            double time_ms = std::stod(tokens[1]);
            double current_timestamp = (time_ms / 1000.0) + time_sync_offset;

            if (std::abs(current_timestamp - buffer.timestamp) > 0.005) {
                processBuffer();
                buffer.reset(current_timestamp);
            }

            uint32_t can_id = std::stoul(tokens[3], nullptr, 16);
            int dlc = std::stoi(tokens[4]);

            std::vector<uint8_t> p;
            for (int i = 0; i < dlc && (5 + i) < tokens.size(); ++i) {
                p.push_back(static_cast<uint8_t>(std::stoul(tokens[5 + i], nullptr, 16)));
            }

            uint32_t msg_offset = can_id & 0xF;

            // Populate the buffer based on the message type
            switch(msg_offset) {
                case 0x0: { // Gyroscope
                    int16_t x = static_cast<int16_t>((p[0] << 8) | p[1]);
                    int16_t y = static_cast<int16_t>((p[2] << 8) | p[3]);
                    int16_t z = static_cast<int16_t>((p[4] << 8) | p[5]);
                    buffer.gyro = Eigen::Vector3d(x/KG, y/KG, z/KG) * DEG_TO_RAD;
                    buffer.has_gyro = true;
                    break;
                }
                case 0x1: { // Accelerometer
                    int16_t x = static_cast<int16_t>((p[0] << 8) | p[1]);
                    int16_t y = static_cast<int16_t>((p[2] << 8) | p[3]);
                    int16_t z = static_cast<int16_t>((p[4] << 8) | p[5]);
                    buffer.accel = Eigen::Vector3d(x/KA, y/KA, z/KA) * G_TO_MPS2;
                    buffer.has_accel = true;
                    break;
                }
                case 0x3: { // Orientation
                    buffer.nav.heading = ((p[0] << 8) | p[1]) / 100.0 * DEG_TO_RAD;
                    buffer.nav.pitch   = static_cast<int16_t>((p[2] << 8) | p[3]) / 100.0 * DEG_TO_RAD;
                    buffer.nav.roll    = static_cast<int16_t>((p[4] << 8) | p[5]) / 100.0 * DEG_TO_RAD;
                    buffer.nav.has_ori = true;
                    break;
                }
                case 0x4: buffer.nav.v_e = parse_32bit_signed(p) / 100.0; break;
                case 0x5: buffer.nav.v_n = parse_32bit_signed(p) / 100.0; buffer.nav.has_vel = true; break;
                case 0x6: buffer.nav.v_u = parse_32bit_signed(p) / 100.0; break;
                case 0x7: buffer.nav.lon = parse_48bit_signed(p) / 1e9; break;
                case 0x8: buffer.nav.lat = parse_48bit_signed(p) / 1e9; buffer.nav.has_pos = true; break;
                case 0x9: buffer.nav.alt = parse_32bit_signed(p) / 1000.0; break;
            }

        } catch (...) { continue; }
    }
    
    processBuffer();

    std::cout << "--- Parsing Complete ---" << std::endl;
    std::cout << "Full IMU Pairs (Tightly-Coupled): " << stats_full_imu << std::endl;
    std::cout << "Gyro-Only (Rotational Deskewing): " << stats_gyro_only << std::endl;
    std::cout << "Nav States (Loosely-Coupled):     " << stats_nav_states << std::endl;

    return true;
}

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

        // Ensure ONLY the LiDAR packets are processed
        if (packet_size == pf.lidar_packet_size) {
            
            ouster::sdk::core::LidarPacket lidar_packet(packet_size);
            
            if (lidar_packet.buf.size() < packet_size) {
                lidar_packet.buf.resize(packet_size);
            }
            
            std::memcpy(lidar_packet.buf.data(), raw_packet_data, packet_size);

            if (batcher(lidar_packet, scan)) {
                static double previous_frame_time = (pi.timestamp.count() / 1000000.0) - 0.1;
                double current_frame_time = pi.timestamp.count() / 1000000.0;
                double frame_duration = current_frame_time - previous_frame_time;
                
                frame_count++;
            
                auto ouster_points = ouster::sdk::core::cartesian(scan, lut);
                int W = scan.w; 
                int H = scan.h; 
                
                for (int u = 0; u < W; ++u) {
                    // Time progresses based on the true physical duration of the spin
                    double col_time = previous_frame_time + ((double)u / W) * frame_duration;
                    
                    for (int v = 0; v < H; ++v) {
                        Eigen::Index i = v * W + u; 
                        
                        double x = ouster_points(i, 0);
                        double y = ouster_points(i, 1);
                        double z = ouster_points(i, 2);

                        if (std::abs(x) < 1e-6 && std::abs(y) < 1e-6 && std::abs(z) < 1e-6) continue; 

                        global_cloud->points_.emplace_back(x, y, z);
                        point_times.push_back(col_time);
                    }
                }

                previous_frame_time = current_frame_time;

                std::cout << "Extracted Frame " << frame_count << " (Total Points: " 
                          << global_cloud->points_.size() << ")" << std::endl;

                if (max_frames > 0 && frame_count >= max_frames) {
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

class ArcLengthParameterization {
    std::vector<double> times;
    std::vector<double> distances;
public:
    ArcLengthParameterization(const TrajectoryInterpolator& traj, double t_start, double t_end, double dt = 0.05) {
        times.push_back(t_start);
        distances.push_back(0.0);
        Eigen::Vector3d last_p = traj.getPoseAtTime(t_start).block<3,1>(0,3);
        
        double accumulated_dist = 0.0;
        for (double t = t_start + dt; t <= t_end; t += dt) {
            Eigen::Vector3d p = traj.getPoseAtTime(t).block<3,1>(0,3);
            accumulated_dist += (p - last_p).norm(); // True 3D distance
            times.push_back(t);
            distances.push_back(accumulated_dist);
            last_p = p;
        }
    }
    
    double getTimeAtArcLength(double u) const {
        if (u <= distances.front()) return times.front();
        if (u >= distances.back()) return times.back();
        
        auto it = std::lower_bound(distances.begin(), distances.end(), u);
        int idx = std::distance(distances.begin(), it);
        
        double u0 = distances[idx-1];
        double u1 = distances[idx];
        double t0 = times[idx-1];
        double t1 = times[idx];
        
        // Linear interpolation between the 0.05s steps for absolute precision
        double ratio = (u - u0) / (u1 - u0);
        return t0 + ratio * (t1 - t0);
    }
    
    double getTotalLength() const {
        return distances.empty() ? 0.0 : distances.back();
    }
};

bool exportToOpenCRG_Curved(const std::string& filename, 
                            const std::vector<std::vector<Eigen::Vector3d>>& grid_3d,
                            const TrajectoryInterpolator& trajectory,
                            const ArcLengthParameterization& arc_param,
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
        double t = arc_param.getTimeAtArcLength(u_start + u * u_inc); 
        crg_file << trajectory.getPoseAtTime(t)(0, 3) << " ";
    }
    crg_file << "\n";

    crg_file << "* Reference Line Y Coordinates\n";
    crg_file << "$Y " << num_u << "\n";
    for (size_t u = 0; u < num_u; ++u) {
        double t = arc_param.getTimeAtArcLength(u_start + u * u_inc);
        crg_file << trajectory.getPoseAtTime(t)(1, 3) << " ";
    }
    crg_file << "\n";

    crg_file << "* Reference Line Heading (PHI)\n";
    crg_file << "$PHI " << num_u << "\n";
    for (size_t u = 0; u < num_u; ++u) {
        double t = arc_param.getTimeAtArcLength(u_start + u * u_inc);
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
    const ArcLengthParameterization& arc_param,
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

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t u = 0; u < num_u; ++u) {
        for (size_t v = 0; v < num_v; ++v) {
            
            double t = arc_param.getTimeAtArcLength(grid_u[u]);
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
    const TrajectoryInterpolator& trajectory)
{
    auto deskewed_cloud = std::make_shared<open3d::geometry::PointCloud>();
    size_t num_points = raw_cloud->points_.size();
    
    deskewed_cloud->points_.resize(num_points);
    if (raw_cloud->HasColors()) deskewed_cloud->colors_ = raw_cloud->colors_;
    if (raw_cloud->HasNormals()) deskewed_cloud->normals_ = raw_cloud->normals_;

    std::atomic<size_t> points_processed(0);
    size_t update_step = std::max<size_t>(1, num_points / 100); 

    std::cout << "Starting parallel deskewing (Global Frame)..." << std::endl;

    #pragma omp parallel 
    {
        size_t local_processed = 0; 
        #pragma omp for schedule(static)
        for (size_t i = 0; i < num_points; ++i) {
            
            double t_p = point_times[i];
            Eigen::Matrix4d T_point = trajectory.getPoseAtTime(t_p);
            
            Eigen::Vector4d p_raw(
                raw_cloud->points_[i](0), 
                raw_cloud->points_[i](1), 
                raw_cloud->points_[i](2), 
                1.0
            );
            
            // Transform directly into the Global ENU World!
            Eigen::Vector4d p_deskewed = T_point * p_raw;
            deskewed_cloud->points_[i] = p_deskewed.head<3>();

            // ... (Keep your progress bar logic exactly the same) ...
            local_processed++;
            if (local_processed == 1000) {
                size_t current_count = (points_processed += local_processed);
                local_processed = 0; 
                if (current_count % update_step < 1000) {
                    int percentage = (current_count * 100) / num_points;
                    #pragma omp critical 
                    std::cout << "\r[" << percentage << "%] Processed " << current_count << " / " << num_points << " points" << std::flush;
                }
            }
        }
        points_processed += local_processed;
    }
    std::cout << "\nDeskewing Complete." << std::endl;
    return deskewed_cloud;
}

// ====================================================================
// TIME SYNCHRONIZATION
// ====================================================================
double getFirstCanTimestamp(const std::string& trc_file) {
    std::ifstream file(trc_file);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line.find(';') != std::string::npos) continue;
        std::istringstream iss(line);
        std::string token;
        iss >> token; // msg number
        if (iss >> token) {
            return std::stod(token) / 1000.0;
        }
    }
    return 0.0;
}

double getFirstPcapTimestamp(const std::string& pcap_file) {
    auto pcap = ouster::sdk::pcap::replay_initialize(pcap_file);
    if (!pcap) return 0.0;
    ouster::sdk::pcap::PacketInfo pi;
    if (ouster::sdk::pcap::next_packet_info(*pcap, pi)) {
        return pi.timestamp.count() / 1000000.0;
    }
    return 0.0;
}

// ====================================================================
// MAIN
// ====================================================================
int main() {
    std::cout << "=== CTA_LIO: Pipeline Execution Start ===" << std::endl;
    
    // Start total pipeline timer
    auto t_pipeline_start = std::chrono::high_resolution_clock::now();

    auto raw_cloud = std::make_shared<open3d::geometry::PointCloud>();
    auto cloth_cloud = std::make_shared<open3d::geometry::PointCloud>();

    std::string base_file_name = "../data/plaster_city/section_3/Section3Outof3ForwardAndBack1";

    std::string pcap_file = base_file_name + ".pcap";
    std::string json_file = base_file_name + ".json";
    std::string trc_file  = base_file_name + ".trc";
    
    std::string cache_file = base_file_name + ".pcd";
    std::string crg_file   = base_file_name + ".crg";
    std::string obj_file   = base_file_name + ".obj";

    std::vector<double> point_times;
    TrajectoryInterpolator trajectory;

    if (!std::filesystem::exists(pcap_file) || !std::filesystem::exists(json_file)) {
        std::cerr << "CRITICAL ERROR: Could not find the .pcap or .json files !" << std::endl;
        return -1;
    }

    // --- SEMI-AUTOMATED TIME SYNCHRONIZATION ---
    double first_pcap_time = getFirstPcapTimestamp(pcap_file);
    double first_can_time  = getFirstCanTimestamp(trc_file);
    double base_offset = first_pcap_time - first_can_time;
    double manual_human_delay = 0.0; 
    double time_sync_offset = base_offset + manual_human_delay;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n[Time Sync] First PCAP Time: " << first_pcap_time << " s\n";
    std::cout << "[Time Sync] First CAN Time:  " << first_can_time << " s\n";
    std::cout << "[Time Sync] Applied Offset:  " << time_sync_offset << " s\n";

    // --- STAGE 1: CAN PARSING ---
    auto t_can_start = std::chrono::high_resolution_clock::now();
    if (!parseCanTrace(trc_file, trajectory, time_sync_offset)) {
        return -1;
    }
    auto t_can_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_can = t_can_end - t_can_start;

    // --- STAGE 2: PCAP PARSING ---
    auto t_pcap_start = std::chrono::high_resolution_clock::now();
    if (!parseOusterPcap(pcap_file, json_file, trajectory, raw_cloud, point_times, 5000)) {
        return -1;
    }
    auto t_pcap_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pcap = t_pcap_end - t_pcap_start;

    // --- STAGE 3: DESKEWING ---
    std::cout << "Deskewing using " << omp_get_max_threads() << " CPU threads..." << std::endl;
    auto t_deskew_start = std::chrono::high_resolution_clock::now();
    
    auto deskewed_cloud = deskewPointCloudParallel(raw_cloud, point_times, trajectory);
    
    auto t_deskew_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_deskew = t_deskew_end - t_deskew_start;

    // --- STAGE 4: GROUND EXTRACTION (CSF) ---
    std::cout << "Ground Extraction" << std::endl;
    auto t_csf_start = std::chrono::high_resolution_clock::now();
    if (std::filesystem::exists(cache_file)) {
        std::cout << "-- Cache found! Loading ground points..." << std::endl;
        open3d::io::ReadPointCloud(cache_file, *cloth_cloud);
    } else {
        std::cout << "-- No cache found. Running Cloth Simulation Filter..." << std::endl;
        cloth_cloud = extractBareEarthCSF(deskewed_cloud); 
        std::cout << "-- Saving ground points to cache..." << std::endl;
        open3d::io::WritePointCloud(cache_file, *cloth_cloud);
    }
    auto t_csf_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_csf = t_csf_end - t_csf_start;

    // --- STAGE 5: SURFACE GENERATION ---
    std::cout << "Surface Generation" << std::endl;
    auto t_surface_start = std::chrono::high_resolution_clock::now();
    if (cloth_cloud && !cloth_cloud->points_.empty()) {
        ArcLengthParameterization arc_param(trajectory, point_times.front(), point_times.back());
        double true_driven_distance = arc_param.getTotalLength();

        double u_start = 0.0;
        double u_end   = true_driven_distance; 
        
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

        auto grid_3d = generateCurvilinearGridIDW(cloth_cloud, grid_u, grid_v, trajectory, arc_param, 3.0);

        std::cout << "-- Generating OpenCRG" << std::endl;
        exportToOpenCRG_Curved(crg_file, grid_3d, trajectory, arc_param, u_start, u_inc, v_start, v_inc);
        
        std::cout << "-- Generating OBJ" << std::endl;
        exportToOBJ(obj_file, grid_3d);

    } else {
        std::cerr << "-- Pipeline Error: No ground points available for CRG/OBJ generation." << std::endl;
    }
    auto t_surface_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_surface = t_surface_end - t_surface_start;
    
    auto t_pipeline_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total = t_pipeline_end - t_pipeline_start;

    // ==========================================================
    // EXECUTION TIME SUMMARY
    // ==========================================================
    std::cout << "\n=================================================" << std::endl;
    std::cout << "             EXECUTION TIME SUMMARY              " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. CAN Trace Parsing:       " << duration_can.count() << " seconds" << std::endl;
    std::cout << "2. PCAP Ouster Parsing:     " << duration_pcap.count() << " seconds" << std::endl;
    std::cout << "3. Point Cloud Deskewing:   " << duration_deskew.count() << " seconds" << std::endl;
    std::cout << "4. Ground Extraction (CSF): " << duration_csf.count() << " seconds" << std::endl;
    std::cout << "5. Surface & Mesh Gen:      " << duration_surface.count() << " seconds" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Total Data Processing Time: " << duration_total.count() << " seconds" << std::endl;
    std::cout << "=================================================\n" << std::endl;

    std::cout << "Visualizing the OBJ Mesh and Point Cloud..." << std::endl;
    auto terrain_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    
    if (open3d::io::ReadTriangleMesh(obj_file, *terrain_mesh)) {
        
        terrain_mesh->ComputeVertexNormals();
        terrain_mesh->PaintUniformColor({0.54, 0.27, 0.07});
        terrain_mesh->Translate(Eigen::Vector3d(0.0, 0.0, -5));

        cloth_cloud->PaintUniformColor({0.0, 1.0, 0.0});
        cloth_cloud->Translate(Eigen::Vector3d(0.0, 0.0, -1));

        deskewed_cloud->PaintUniformColor({0.5, 0.5, 0.5});

        std::cout << "Opening 3D Visualizer..." << std::endl;
        
        open3d::visualization::DrawGeometries(
            {cloth_cloud, deskewed_cloud, terrain_mesh},
            "OpenCRG Terrain Mesh with LiDAR Overlay"
        );
    } else {
        std::cerr << "Failed to load the OBJ file for visualization." << std::endl;
    }

    return 0;
}