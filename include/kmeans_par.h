#pragma once
#include <vector>

inline double euclidean_distance_SIMD(const std::vector<double>& a, const std::vector<double>& b) ;
inline double euclidean_distance(const std::vector<double> &p1, const std::vector<double> &p2);
std::vector<std::vector<double>> kmeans_par(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids);
std::vector<std::vector<double>> kmeans_par_single(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &init_centroids);
std::vector<std::vector<double>> kmeans_par_single_simd(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>>& init_centroids);
