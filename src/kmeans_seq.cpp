#include "kmeans_seq.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <limits>


double distance_seq(const std::vector<double> &p1, const std::vector<double> &p2){
    double distance = 0;

    int dim = p1.size();

    for(int i = 0; i<dim; i++){
        double a = p1[i];
        double b = p2[i];
        
        distance += pow((a - b), 2);
    }

    return distance;
}

inline double distance_seq_soa(const std::vector<std::vector<double>>& d1, const std::vector<std::vector<double>>& d2, long Idx1, long Idx2, int numFeatures){
    double sum = 0.0;
    for (int i = 0; i < numFeatures; i++) {
        double diff = d1[i][Idx1] - d2[i][Idx2];
        sum += diff * diff;
    }
    return sum;
}


/*
*
*   SEQUENTIAL + DOUBLE
*
*/

std::vector<std::vector<long>> assign_points_seq(const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &centroids){
    std::vector<std::vector<long>> clusters(centroids.size(), std::vector<long>());

    long point_index = 0;
    for(const auto p: points){
        double minDist = std::numeric_limits<double>::max();
        int closestCluster = 0;

        for(int j=0; j<centroids.size(); j++){
            auto centroid = centroids[j];
            auto dist = distance_seq(p, centroid);
            if (dist < minDist) {
                minDist = dist;
                closestCluster = j;
            }
        }

        clusters[closestCluster].push_back(point_index);
        point_index++;
    }

    return clusters;
}

std::vector<std::vector<double>> get_new_centroids_seq(const std::vector<std::vector<double>> &points, const std::vector<std::vector<long>> &clusters){
    std::vector<std::vector<double>> centroids;
    int dim = points[0].size();

    for(const auto cls: clusters){
        long cls_size = cls.size();
        std::vector<double> cls_centroid(dim, 0);

        for(auto i: cls){
            auto p = points[i];
            for(int d=0; d<dim; d++){
                cls_centroid[d] += p[d];
            }
        }

        for(int d=0; d<dim; d++){
            cls_centroid[d] = cls_centroid[d]/cls_size;
        }
        
        centroids.push_back(cls_centroid);
    }

    return centroids;
}

std::vector<std::vector<double>> kmeans_seq(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids){
    std::vector<std::vector<long>> clusters;
    std::vector<std::vector<double>> centroids = init_centroids;
    const int MAX_ITERATIONS = 100; 
    const double epsilon = 100;

    for(int i = 0; i < MAX_ITERATIONS; i++){
        clusters = assign_points_seq(points, centroids);
        auto new_centroids = get_new_centroids_seq(points, clusters);

        double dist = 0;
        for(int j = 0; j<centroids.size(); j++){
            dist += distance_seq(centroids[j], new_centroids[j]);
        }

        centroids = new_centroids;

        if(dist <= epsilon) 
            break;
    }

    return centroids;
}


/*
*
*   SEQUENTIAL + SINGLE + AOS
*
*/

std::vector<std::vector<double>> kmeans_seq_single(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids) {
    const int MAX_ITERATIONS = 100;
    const double epsilon = 100;
    long numPoints = points.size();
    int dim = points[0].size();
    std::vector<std::vector<double>> centroids = init_centroids;

    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dim, 0.0));
    std::vector<int> clusterSizes(k, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < k; i++) {
            std::fill(newCentroids[i].begin(), newCentroids[i].end(), 0.0);
            clusterSizes[i] = 0;
        }


        for (int i = 0; i < numPoints; i++) {
            double minDist = std::numeric_limits<double>::max();
            int closestCluster = 0;

            for (int j = 0; j < k; j++) {
                double dist = distance_seq(points[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }

            for (int k = 0; k < dim; k++) {
                newCentroids[closestCluster][k] += points[i][k];
            }
            clusterSizes[closestCluster]++;
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
            dist += distance_seq(centroids[j], newCentroids[j]);
        }

        centroids = newCentroids;

        if(dist <= epsilon) 
            break;

    }

    return centroids;
}


/*
*
*   SEQUENTIAL + SINGLE + AOS + SIMD
*
*/

std::vector<std::vector<double>> kmeans_seq_single_simd(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &init_centroids)
{
    const int MAX_ITERATIONS = 100;
    const double epsilon = 100;
    long numPoints = points.size();
    int dim = points[0].size();
    std::vector<std::vector<double>> centroids = init_centroids;

    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dim, 0.0));
    std::vector<int> clusterSizes(k, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < k; i++) {
            std::fill(newCentroids[i].begin(), newCentroids[i].end(), 0.0);
            clusterSizes[i] = 0;
        }


        for (int i = 0; i < numPoints; i++) {
            double minDist = std::numeric_limits<double>::max();
            int closestCluster = 0;

            for (int j = 0; j < k; j++) {
                double dist = distance_seq(points[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }

            #pragma omp simd
            for (int k = 0; k < dim; k++) {
                newCentroids[closestCluster][k] += points[i][k];
            }
            clusterSizes[closestCluster]++;
        }


        for (int i = 0; i < k; i++) {
            if (clusterSizes[i] > 0) {
                #pragma omp simd
                for (int j = 0; j < dim; j++) {
                    newCentroids[i][j] = newCentroids[i][j] / clusterSizes[i];
                }
            }
        }

        double dist = 0;
        #pragma omp simd
        for(int j = 0; j<centroids.size(); j++){
            dist += distance_seq(centroids[j], newCentroids[j]);
        }

        centroids = newCentroids;

        if(dist <= epsilon) 
            break;

    }

    return centroids;
}


/*
*
*   SEQUENTIAL + SINGLE + SOA
*
*/

std::vector<std::vector<double>> kmeans_seq_single_soa(const int k, const std::vector<std::vector<double>>& points, const std::vector<std::vector<double>> &init_centroids) {
    const int MAX_ITERATIONS = 100;
    const double epsilon = 100;
    long dim = points.size();
    int numPoints = points[0].size();
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

        std::vector<std::vector<double>> localCentroids(dim, std::vector<double>(k, 0.0));
        std::vector<int> localClusterSizes(k, 0);

        for (long i = 0; i < numPoints; i++) {
            double minDist = std::numeric_limits<double>::max();
            int closestCluster = 0;

            for (int j = 0; j < k; j++) {
                double dist = distance_seq_soa(points, centroids, i, j, dim);
                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }

            for (int j = 0; j < dim; j++) {
                newCentroids[j][closestCluster] += points[j][i];
            }
            clusterSizes[closestCluster]++;
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
            dist += distance_seq_soa(centroids, newCentroids, j, j, dim);
        }

        centroids = newCentroids;

        if(dist <= epsilon) 
            break;

    }

    return centroids;
}


/*
*
*   SEQUENTIAL + SINGLE + SOA + SIMD
*
*/

std::vector<std::vector<double>> kmeans_seq_single_soa_simd(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &init_centroids)
{
    const int MAX_ITERATIONS = 100;
    const double epsilon = 100;
    long dim = points.size();
    int numPoints = points[0].size();
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

        std::vector<std::vector<double>> localCentroids(dim, std::vector<double>(k, 0.0));
        std::vector<int> localClusterSizes(k, 0);

        for (long i = 0; i < numPoints; i++) {
            double minDist = std::numeric_limits<double>::max();
            int closestCluster = 0;

            for (int j = 0; j < k; j++) {
                double dist = distance_seq_soa(points, centroids, i, j, dim);
                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }

            #pragma omp simd
            for (int j = 0; j < dim; j++) {
                newCentroids[j][closestCluster] += points[j][i];
            }
            clusterSizes[closestCluster]++;
        }

        for (int i = 0; i < k; i++) {
            if (clusterSizes[i] > 0) {
                #pragma omp simd
                for (int j = 0; j < dim; j++) {
                    newCentroids[j][i] = newCentroids[j][i] / clusterSizes[i];
                }
            }
        }

        double dist = 0;
        #pragma omp simd
        for(int j = 0; j<k; j++){
            dist += distance_seq_soa(centroids, newCentroids, j, j, dim);
        }

        centroids = newCentroids;

        if(dist <= epsilon) 
            break;

    }

    return centroids;
}
