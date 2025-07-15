#pragma once
#include <vector>

inline double distance_seq(const std::vector<double> &p1, const std::vector<double> &p2);
inline double distance_seq_soa(const std::vector<std::vector<double>>& d1, const std::vector<std::vector<double>>& d2, long Idx1, long Idx2, int numFeatures);
std::vector<std::vector<long>> assign_points_seq(const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &centroids);
std::vector<std::vector<double>> get_new_centroids_seq(const std::vector<std::vector<double>> &points, const std::vector<std::vector<long>> &clusters);
std::vector<std::vector<long>> kmeans_seq(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids);
std::vector<int> kmeans_seq_single(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids);
std::vector<int> kmeans_seq_single_soa(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>> &init_centroids);