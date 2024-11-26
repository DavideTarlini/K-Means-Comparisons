#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <limits>

inline double distance_par_single(const std::vector<double> &p1, const std::vector<double> &p2){
    double distance = 0;

    int dim = p1.size();

    for(int i = 0; i<dim; i++){
        double a = p1[i];
        double b = p2[i];
        
        distance += pow((a - b), 2);
    }

    return distance;
}

std::vector<int> kmeans_par_single(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &init_centroids) {
    const int MAX_ITERATIONS = 100;
    const double epsilon = 100;
    long numPoints = points.size();
    int dim = points[0].size();
    std::vector<int> cluster_labels(numPoints, -1);
    std::vector<std::vector<double>> centroids = init_centroids;

    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dim, 0.0));
    std::vector<int> clusterSizes(k, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < k; i++) {
            std::fill(newCentroids[i].begin(), newCentroids[i].end(), 0.0);
            clusterSizes[i] = 0;
        }

        #pragma omp parallel
        {
            std::vector<std::vector<double>> localCentroids(k, std::vector<double>(dim, 0.0));
            std::vector<long> localClusterSizes(k, 0);

            #pragma omp for schedule(static)
            for (int i = 0; i < numPoints; i++) {
                double minDist = std::numeric_limits<double>::max();
                int closestCluster = 0;

                for (int j = 0; j < k; j++) {
                    double dist = distance_par_single(points[i], centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        closestCluster = j;
                    }
                }

                cluster_labels[i] = closestCluster;

                for (int k = 0; k < dim; k++) {
                    localCentroids[closestCluster][k] += points[i][k];
                }
                localClusterSizes[closestCluster]++;
            }

            #pragma omp critical
            {
                for (int j = 0; j < k; j++) {
                    for (int k = 0; k < dim; k++) {
                        newCentroids[j][k] += localCentroids[j][k];
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
        for(int j = 0; j<centroids.size(); j++){
            dist += distance_par_single(centroids[j], newCentroids[j]);
        }

        if(dist <= epsilon) 
            break;
        else{
            centroids = newCentroids;
        }

    }

    return cluster_labels;
}
