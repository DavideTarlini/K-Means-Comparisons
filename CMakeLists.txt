cmake_minimum_required(VERSION 3.05)
project(KMeans_Comparison)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(KMeans_Comparison
    main.cpp
    src/kmeans_seq.cpp
    src/kmeans_par.cpp
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(KMeans_Comparison PUBLIC OpenMP::OpenMP_CXX)
endif()