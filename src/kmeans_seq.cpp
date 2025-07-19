#include "kmeans_seq.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <limits>

inline double euclideanDistanceSquared(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline double euclidean_distance_SIMD(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0.0;
    int d = a.size();
    #pragma omp simd reduction(+:dist)
    for (int i = 0; i < d; ++i) {
        double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}


std::vector<std::vector<double>> kmeans_seq(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>>& init_centroids) {
    const int MAX_ITERATIONS = 30;
    const double EPSILON = 1;

    int n = points.size();
    int d = points[0].size();

    std::vector<std::vector<double>> centroids = init_centroids;
    std::vector<std::vector<double>> newCentroids(k, std::vector(d, 0.0));
    std::vector<int> clusterSizes(k, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int j = 0; j < k; ++j) {
            std::fill(newCentroids[j].begin(), newCentroids[j].end(), 0.0);
            clusterSizes[j] = 0;
        }

        for (const auto& p : points) {
            double minDist = std::numeric_limits<double>::max();
            int closest = 0;

            for (int j = 0; j < k; ++j) {
                double dist = euclideanDistanceSquared(p, centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    closest = j;
                }
            }

            for (int dim = 0; dim < d; ++dim) {
                newCentroids[closest][dim] += p[dim];
            }
            clusterSizes[closest]++;
        }

        for (int j = 0; j < k; ++j) {
            if (clusterSizes[j] > 0) {
                for (int dim = 0; dim < d; ++dim) {
                    newCentroids[j][dim] /= clusterSizes[j];
                }
            } else {
                newCentroids[j] = centroids[j]; 
            }
        }

        double movement = 0.0;
        for (int j = 0; j < k; ++j) {
            movement += euclideanDistanceSquared(centroids[j], newCentroids[j]);
        }

        centroids = newCentroids;

        if (movement <= EPSILON) {
            break;
        }
    }

    return centroids;
}

std::vector<std::vector<double>> kmeans_seq_simd(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>>& init_centroids) {
    const int MAX_ITERATIONS = 30;
    const double EPSILON = 1;

    int n = points.size();
    int d = points[0].size();

    std::vector<std::vector<double>> centroids = init_centroids;
    std::vector<std::vector<double>> newCentroids(k, std::vector(d, 0.0));
    std::vector<int> clusterSizes(k, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int j = 0; j < k; ++j) {
            std::fill(newCentroids[j].begin(), newCentroids[j].end(), 0.0);
            clusterSizes[j] = 0;
        }

        for (const auto& p : points) {
            double minDist = std::numeric_limits<double>::max();
            int closest = 0;

            for (int j = 0; j < k; ++j) {
                double dist = euclidean_distance_SIMD(p, centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    closest = j;
                }
            }

            #pragma omp simd
            for (int dim = 0; dim < d; ++dim) {
                newCentroids[closest][dim] += p[dim];
            }
            clusterSizes[closest]++;
        }

        for (int j = 0; j < k; ++j) {
            if (clusterSizes[j] > 0) {
                #pragma omp simd
                for (int dim = 0; dim < d; ++dim) {
                    newCentroids[j][dim] /= clusterSizes[j];
                }
            } else {
                newCentroids[j] = centroids[j]; 
            }
        }

        double movement = 0.0;
        for (int j = 0; j < k; ++j) {
            movement += euclidean_distance_SIMD(centroids[j], newCentroids[j]);
        }

        centroids = newCentroids;

        if (movement <= EPSILON) {
            break;
        }
    }

    return centroids;
}