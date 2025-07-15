#include "kmeans_pars.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <limits>

inline double distance_par(const std::vector<double> &p1, const std::vector<double> &p2){
    double distance = 0;

    int dim = p1.size();

    for(int i = 0; i<dim; i++){
        double a = p1[i];
        double b = p2[i];
        
        distance += pow((a - b), 2);
    }

    return distance;
}

inline double distance_par_soa(const std::vector<std::vector<double>>& d1, const std::vector<std::vector<double>>& d2, long Idx1, long Idx2, int numFeatures){
    double sum = 0.0;
    for (int i = 0; i < numFeatures; i++) {
        double diff = d1[i][Idx1] - d2[i][Idx2];
        sum += diff * diff;
    }
    return sum;
}


/*
*
*   PARALLEL + DOUBLE
*
*/

std::vector<std::vector<long>> assign_points_par(const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &centroids){
    std::vector<std::vector<long>> clusters_shared(centroids.size(), std::vector<long>());

    #pragma omp parallel
    {
        std::vector<std::vector<long>> clusters_private(centroids.size(), std::vector<long>());

        #pragma omp for schedule(static)
        for(long i = 0; i<points.size(); i++){
            auto p = points[i];
            double minDist = std::numeric_limits<double>::max();
            int closestCluster = 0;

            for(int j=0; j<centroids.size(); j++){
                auto centroid = centroids[j];
                auto dist = distance_par(p, centroid);
                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }

            clusters_private[closestCluster].push_back(i);
        }

        #pragma omp critical
        {
            int i = 0;
            for(auto cls: clusters_private){
                for(auto index: cls){
                    clusters_shared[i].push_back(index);
                }

                i++;
            }
        }
    }

    return clusters_shared;
}

std::vector<std::vector<double>> get_new_centroids_par(const std::vector<std::vector<double>> &points, const std::vector<std::vector<long>> &clusters){
    int cls_num = clusters.size();
    int dim = points[0].size();
    std::vector<std::vector<double>> centroids(cls_num, std::vector<double>());

    #pragma omp parallel for schedule(static)
    for(int i=0; i<cls_num; i++){
        long cls_size = clusters[i].size();
        std::vector<double> cls_centroid(dim, 0);

        for(auto j: clusters[i]){
            auto p = points[j];

            for(int d=0; d<dim; d++){
                cls_centroid[d] += p[d];
            }
        }

        for(int d=0; d<dim; d++){
            cls_centroid[d] = cls_centroid[d]/cls_size;
        }
        
        #pragma omp critical
        centroids[i] = cls_centroid;
    }

    return centroids;
}

std::vector<std::vector<long>> kmeans_par(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids){
    std::vector<std::vector<long>> clusters;
    std::vector<std::vector<double>> centroids = init_centroids;
    const int MAX_ITERATIONS = 100; 
    const double epsilon = 100;

    for(int i = 0; i < MAX_ITERATIONS; i++){
        clusters = assign_points_par(points, centroids);
        auto new_centroids = get_new_centroids_par(points, clusters);

        double dist = 0;
        for(int j = 0; j<centroids.size(); j++){
            dist += distance_par(centroids[j], new_centroids[j]);
        }

        if(dist <= epsilon) 
            break;
        else 
            centroids = new_centroids;
    }

    return clusters;
}

/*
*
*   PARALLEL + SINGLE
*
*/

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
                    double dist = distance_par(points[i], centroids[j]);
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
            dist += distance_par(centroids[j], newCentroids[j]);
        }

        if(dist <= epsilon) 
            break;
        else{
            centroids = newCentroids;
        }

    }

    return cluster_labels;
}

/*
*
*   PARALLEL + SINGLE + SOA
*
*/

std::vector<int> kmeans_par_single_soa(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>> &init_centroids) {
    const int MAX_ITERATIONS = 100;
    const double epsilon = 100;
    long dim = points.size();
    int numPoints = points[0].size();
    std::vector<int> cluster_labels(numPoints, -1);
    std::vector<std::vector<double>> centroids = init_centroids;

    std::vector<std::vector<double>> newCentroids(dim, std::vector<double>(k, 0.0));
    std::vector<int> clusterSizes(k, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < k; i++) {
            clusterSizes[i] = 0;
            for (int j = 0; j < dim; j++) {
                newCentroids[j][i] = 0.0;
            }
        }

        #pragma omp parallel
        {
            std::vector<std::vector<double>> localCentroids(dim, std::vector<double>(k, 0.0));
            std::vector<int> localClusterSizes(k, 0);

            #pragma omp for schedule(static)
            for (long i = 0; i < numPoints; i++) {
                double minDist = std::numeric_limits<double>::max();
                int closestCluster = 0;

                for (int j = 0; j < k; j++) {
                    double dist = distance_par_soa(points, centroids, i, j, dim);
                    if (dist < minDist) {
                        minDist = dist;
                        closestCluster = j;
                    }
                }

                cluster_labels[i] = closestCluster;

                for (int j = 0; j < dim; j++) {
                    localCentroids[j][closestCluster] += points[j][i];
                }
                localClusterSizes[closestCluster]++;
            }

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
                    newCentroids[j][i] = newCentroids[j][i] / clusterSizes[i];
                }
            }
        }

        double dist = 0;
        for(int j = 0; j<k; j++){
            dist += distance_par_soa(centroids, newCentroids, j, j, dim);
        }

        if(dist <= epsilon) 
            break;
        else{
            centroids = newCentroids;
        }
    }

    return cluster_labels;
}