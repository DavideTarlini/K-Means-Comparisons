#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <limits>

inline double distance_par_single_soa(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& centroids, long pointIdx, long clusterIdx, int numFeatures){
    double sum = 0.0;
    for (int i = 0; i < numFeatures; i++) {
        double diff = data[i][pointIdx] - centroids[i][clusterIdx];
        sum += diff * diff;
    }
    return sum;
}

// K-means clustering function
std::vector<int> kmeans_par_single_soa(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>> &init_centroids) {
    const int MAX_ITERATIONS = 100;
    const double epsilon = 100;
    long numPoints = points.size();
    int dim = points[0].size();
    std::vector<int> cluster_labels(numPoints, -1);
    std::vector<std::vector<double>> centroids = init_centroids;

    // Temporary storage for updated centroids
    std::vector<std::vector<double>> newCentroids(dim, std::vector<double>(k, 0.0));
    std::vector<int> clusterSizes(k, 0);

    // Main loop for K-means
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Reset temporary storage
        for (int i = 0; i < k; i++) {
            clusterSizes[i] = 0;
            for (int j = 0; j < dim; j++) {
                newCentroids[j][i] = 0.0;
            }
        }

        // Assign points to the nearest centroid
        #pragma omp parallel
        {
            std::vector<std::vector<double>> localCentroids(dim, std::vector<double>(k, 0.0));
            std::vector<int> localClusterSizes(k, 0);

            #pragma omp for schedule(static)
            for (long i = 0; i < numPoints; i++) {
                double minDist = std::numeric_limits<double>::max();
                int closestCluster = 0;

                for (int j = 0; j < k; j++) {
                    double dist = distance_par_single_soa(points, centroids, i, j, dim);
                    if (dist < minDist) {
                        minDist = dist;
                        closestCluster = j;
                    }
                }

                cluster_labels[i] = closestCluster;

                // Update local centroids
                for (int j = 0; j < dim; j++) {
                    localCentroids[j][closestCluster] += points[j][i];
                }
                localClusterSizes[closestCluster]++;
            }

            // Combine results from threads
            #pragma omp critical
            {
                for (int j = 0; j < k; j++) {
                    for (int w = 0; w < dim; w++) {
                        newCentroids[w][j] += localCentroids[w][j];
                    }
                    clusterSizes[j] += localClusterSizes[j];
                }
            }
        }

        for (int i = 0; i < k; i++) {
            if (clusterSizes[i] > 0) {
                for (int j = 0; j < dim; j++) {
                    newCentroids[i][j] = newCentroids[i][j] / clusterSizes[i];
                }
            }
        }

        double dist = 0;
        for(int j = 0; j<k; j++){
            dist += distance_par_single_soa(centroids, newCentroids, j, j, dim);
        }

        if(dist <= epsilon) 
            break;
        else{
            centroids = newCentroids;
        }
    }

    return cluster_labels;
}
