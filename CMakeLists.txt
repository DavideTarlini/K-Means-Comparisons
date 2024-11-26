cmake_minimum_required(VERSION 3.5.0)
project(KMeans_Comparison VERSION 0.1.0 LANGUAGES C CXX)

add_executable(KMeans_Comparison main.cpp  src/kmeans_par.cpp)
add_subdirectory(matplotplusplus)
target_link_libraries(KMeans_Comparison PUBLIC matplot)