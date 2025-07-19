#pragma once
#include <vector>

inline double euclideanDistanceSquared(const std::vector<double>& a, const std::vector<double>& b);
inline double euclidean_distance_SIMD(const std::vector<double>& a, const std::vector<double>& b);
std::vector<std::vector<double>> kmeans_seq(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &init_centroids);
std::vector<std::vector<double>> kmeans_seq_simd(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>>& init_centroids);
