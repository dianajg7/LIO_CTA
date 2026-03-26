cmake_minimum_required(VERSION 3.20)
project(CTA_LIO VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_subdirectory(third_party/Open3D EXCLUDE_FROM_ALL)
add_subdirectory(third_party/PDAL EXCLUDE_FROM_ALL)

add_executable(CTA_LIO_Core
    src/main.cpp
)

target_link_libraries(CTA_LIO_Core PRIVATE
    OpenMP::OpenMP_CXX
    Open3D::Open3D
    pdalcpp
)

target_include_directories(CTA_LIO_Core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/PDAL
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/PDAL/include
)