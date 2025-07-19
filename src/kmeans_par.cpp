#include "kmeans_par.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <limits>

inline double euclidean_distance(const std::vector<double> &p1, const std::vector<double> &p2){
    double sum = 0.0;
    for (int i = 0; i < p1.size(); i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

inline double euclidean_distance_SIMD(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0.0;
    int d = a.size();
    #pragma omp simd reduction(+:dist)
    for (int i = 0; i < d; i++) {
        double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}


std::vector<std::vector<double>> kmeans_par(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids){
    std::vector<std::vector<double>> centroids = init_centroids;
    const int MAX_ITERATIONS = 30; 
    const double epsilon = 1;
    long dim = points[0].size();
    int numPoints = points.size();

    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dim, 0.0));
    std::vector<int> clusterSizes(k, 0);
    std::vector<int> currentBestCluster(numPoints, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < k; i++) {
            std::fill(newCentroids[i].begin(), newCentroids[i].end(), 0.0);
            clusterSizes[i] = 0;
        }

        // Assignment step
        #pragma omp parallel for
        for (int i = 0; i < numPoints; i++) {
            double minDist = std::numeric_limits<double>::max();
            int closestCluster = 0;
            const auto p = points[i];

            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(points[i], centroids[j]);

                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }

            currentBestCluster[i] = closestCluster;
        }

        #pragma omp parallel
        {
            std::vector<std::vector<double>> localCentroids(k, std::vector<double>(dim, 0.0));
            std::vector<int> localClusterSizes(k, 0);

            #pragma omp for
            for (int i = 0; i < numPoints; i++) {
                int cluster = currentBestCluster[i];
                for (int j = 0; j < dim; j++) {
                    localCentroids[cluster][j] += points[i][j];
                }
                localClusterSizes[cluster]++;
            }

            #pragma omp critical
            {
                for (int c = 0; c < k; c++) {
                    for (int j = 0; j < dim; j++) {
                        newCentroids[c][j] += localCentroids[c][j];
                    }
                    clusterSizes[c] += localClusterSizes[c];
                }
            }
        }

        // Average centroids
        for (int i = 0; i < k; i++) {
            if (clusterSizes[i] > 0) {
                for (int j = 0; j < dim; j++) {
                    newCentroids[i][j] /= clusterSizes[i];
                }
            } else {
                newCentroids[i] = centroids[i]; 
            }
        }

        // Check convergence
        double dist = 0.0;
        for (int i = 0; i < k; i++) {
            dist += euclidean_distance(centroids[i], newCentroids[i]);
        }

        centroids = newCentroids;

        if (dist <= epsilon)
            break;
    }

    return centroids;
}

std::vector<std::vector<double>> kmeans_par_single(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>>& init_centroids) {
    const int MAX_ITERATIONS = 30;
    const double EPSILON = 1;

    const int n = points.size();
    const int d = points[0].size();

    std::vector<std::vector<double>> centroids = init_centroids;
    std::vector<int> assignments(n, 0);

    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(d, 0.0));
    std::vector<int> clusterSizes(k, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int j = 0; j < k; j++) {
            std::fill(newCentroids[j].begin(), newCentroids[j].end(), 0.0);
            clusterSizes[j] = 0;
        }

        // Assignment Step 
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double minDist = std::numeric_limits<double>::max();
            int closest = 0;
            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(points[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    closest = j;
                }
            }
            assignments[i] = closest;
        }
        

        // Update Step 
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<std::vector<double>>> threadCentroids(num_threads, std::vector<std::vector<double>>(k, std::vector<double>(d, 0.0)));
        std::vector<std::vector<int>> threadSizes(num_threads, std::vector<int>(k, 0));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                for (int dim = 0; dim < d; dim++) {
                    threadCentroids[tid][cluster][dim] += points[i][dim];
                }
                threadSizes[tid][cluster]++;
            }
        }

        for (int t = 0; t < num_threads; t++) {
            for (int j = 0; j < k; ++j) {
                for (int dim = 0; dim < d; dim++) {
                    newCentroids[j][dim] += threadCentroids[t][j][dim];
                }
                clusterSizes[j] += threadSizes[t][j];
            }
        }

        for (int j = 0; j < k; j++) {
            if (clusterSizes[j] > 0) {
                for (int dim = 0; dim < d; dim++) {
                    newCentroids[j][dim] /= clusterSizes[j];
                }
            } else {
                newCentroids[j] = centroids[j]; 
            }
        }

        double movement = 0.0;
        for (int j = 0; j < k; j++) {
            movement += euclidean_distance(centroids[j], newCentroids[j]);
        }

        centroids = newCentroids;

        if (movement <= EPSILON)
            break;
    }

    return centroids;
}

std::vector<std::vector<double>> kmeans_par_single_simd(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>>& init_centroids) {
    const int MAX_ITERATIONS = 30;
    const double EPSILON = 1;

    const int n = points.size();
    const int d = points[0].size();

    std::vector<std::vector<double>> centroids = init_centroids;
    std::vector<int> assignments(n, 0);

    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(d, 0.0));
    std::vector<int> clusterSizes(k, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        for (int j = 0; j < k; j++) {
            std::fill(newCentroids[j].begin(), newCentroids[j].end(), 0.0);
            clusterSizes[j] = 0;
        }

        #pragma omp parallel
        {
            std::vector<std::vector<double>> localCentroids = centroids; 
            // Assignment Step 
            #pragma omp for 
            for (int i = 0; i < n; i++) {
                double minDist = std::numeric_limits<double>::max();
                int closest = 0;
                for (int j = 0; j < k; j++) {
                    double dist = euclidean_distance_SIMD(points[i], centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        closest = j;
                    }
                }
                assignments[i] = closest;
            }
        }
        

        // Update Step
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<std::vector<double>>> threadCentroids(num_threads, std::vector<std::vector<double>>(k, std::vector<double>(d, 0.0)));
        std::vector<std::vector<int>> threadSizes(num_threads, std::vector<int>(k, 0));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                #pragma omp simd
                for (int dim = 0; dim < d; dim++) {
                    threadCentroids[tid][cluster][dim] += points[i][dim];
                }
                threadSizes[tid][cluster]++;
            }
        }

        for (int t = 0; t < num_threads; t++) {
            for (int j = 0; j < k; j++) {
                #pragma omp simd
                for (int dim = 0; dim < d; dim++) {
                    newCentroids[j][dim] += threadCentroids[t][j][dim];
                }
                clusterSizes[j] += threadSizes[t][j];
            }
        }

        for (int j = 0; j < k; j++) {
            if (clusterSizes[j] > 0) {
                #pragma omp simd
                for (int dim = 0; dim < d; dim++) {
                    newCentroids[j][dim] /= clusterSizes[j];
                }
            } else {
                newCentroids[j] = centroids[j]; 
            }
        }

        double movement = 0.0;
        for (int j = 0; j < k; j++) {
            movement += euclidean_distance_SIMD(centroids[j], newCentroids[j]);
        }

        centroids = newCentroids;

        if (movement <= EPSILON)
            break;
    }

    return centroids;
}

