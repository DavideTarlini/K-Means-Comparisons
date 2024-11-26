#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

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