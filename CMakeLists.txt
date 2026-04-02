cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(CTA_LIO VERSION 1.0 LANGUAGES CXX C)

# --- Force CMake to ignore ROS 2 --- (For LGMATH/STEAM)
set(CMAKE_DISABLE_FIND_PACKAGE_ament_cmake ON CACHE BOOL "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

# --- The GoogleTest Collision Bypass --- (For LGMATH/STEAM)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dummy_gtest)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy_gtest/CMakeLists.txt "project(dummy_gtest)\n")
set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${CMAKE_CURRENT_BINARY_DIR}/dummy_gtest CACHE PATH "" FORCE)

# --- Open3D ---
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE) 
set(BUILD_WEBRTC OFF CACHE BOOL "" FORCE)
set(BUILD_GUI OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/Open3D EXCLUDE_FROM_ALL)

# --- PDAL ---
add_subdirectory(third_party/PDAL EXCLUDE_FROM_ALL)

# --- Ouster SDK ---
set(BUILD_VIZ OFF CACHE BOOL "" FORCE)
set(BUILD_MAPPING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_OSF OFF CACHE BOOL "" FORCE) 
set(BUILD_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ouster_sdk EXCLUDE_FROM_ALL)

# --- LGMATH ---
add_subdirectory(third_party/lgmath EXCLUDE_FROM_ALL)

# --- STEAM ---
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dummy_lgmath_config)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy_lgmath_config/lgmathConfig.cmake "set(lgmath_FOUND TRUE)\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy_lgmath_config/lgmathConfigVersion.cmake 
"set(PACKAGE_VERSION \"3.0.0\")\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\nset(PACKAGE_VERSION_EXACT TRUE)\n")
set(lgmath_DIR ${CMAKE_CURRENT_BINARY_DIR}/dummy_lgmath_config CACHE PATH "" FORCE)
add_subdirectory(third_party/steam EXCLUDE_FROM_ALL)




add_executable(CTA_LIO_Core
    src/main.cpp
)

target_link_libraries(CTA_LIO_Core PRIVATE
    OpenMP::OpenMP_CXX
    Open3D::Open3D
    pdalcpp
    ouster_client
    ouster_sensor
    ouster_pcap
    lgmath
    steam
    Eigen3::Eigen
)

target_include_directories(CTA_LIO_Core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/PDAL
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/PDAL/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/steam/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lgmath/include
)