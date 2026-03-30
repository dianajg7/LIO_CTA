cmake_minimum_required(VERSION 3.20)
project(CTA_LIO VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_subdirectory(third_party/Open3D EXCLUDE_FROM_ALL)
add_subdirectory(third_party/PDAL EXCLUDE_FROM_ALL)

set(BUILD_VIZ OFF CACHE BOOL "" FORCE)
set(BUILD_MAPPING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_OSF OFF CACHE BOOL "" FORCE) 
set(BUILD_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ouster_sdk EXCLUDE_FROM_ALL)

# ... [The rest of your executable linking] ...
add_executable(CTA_LIO_Core
    src/main.cpp
)

target_link_libraries(CTA_LIO_Core PRIVATE
    OpenMP::OpenMP_CXX
    Open3D::Open3D
    pdalcpp
    ouster_client
    ouster_pcap
)

target_include_directories(CTA_LIO_Core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/PDAL
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/PDAL/include
)