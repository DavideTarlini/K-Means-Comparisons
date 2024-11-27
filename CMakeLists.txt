cmake_minimum_required(VERSION 3.5.0)
project(KMeans_Comparison VERSION 0.1.0 LANGUAGES C CXX)

add_executable(KMeans_Comparison main.cpp  src/kmeans_seq.cpp src/kmeans_par.cpp src/kmeans_par_simd src/kmeans_seq_single.cpp src/kmeans_par_single.cpp)