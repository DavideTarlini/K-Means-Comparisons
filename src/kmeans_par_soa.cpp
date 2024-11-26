#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

inline double distance_par_soa(const std::vector<double> &p1, const std::vector<double> &p2){
    double distance = 0;

    int dim = p1.size();

    for(int i = 0; i<dim; i++){
        double a = p1[i];
        double b = p2[i];
        
        distance += pow((a - b), 2);
    }

    return distance;
}

std::vector<std::vector<std::vector<double>>> assign_points_par_soa(const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &centroids){
    std::vector<std::vector<std::vector<double>>> clusters_shared(centroids.size(), std::vector<std::vector<double>>(points[0].size(), std::vector<double>()));

    #pragma omp parallel
    {
        std::vector<std::vector<std::vector<double>>> clusters_private(centroids.size(), std::vector<std::vector<double>>(points[0].size(), std::vector<double>()));

        #pragma omp for schedule(static)
        for(long i = 0; i<points.size(); i++){
            auto p = points[i];
            double minDist = std::numeric_limits<double>::max();
            int closestCluster = 0;

            for(int j=0; j<centroids.size(); j++){
                auto centroid = centroids[j];
                auto dist = distance_par_soa(p, centroid);
                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }

            for(int d = 0; d<points[0].size(); d++){
                clusters_private[closestCluster][d].push_back(p[d]);
            }
        }

        #pragma omp critical
        {
            int i = 0;
            for(auto cls: clusters_private){
                for(int j=0; j<cls[0].size(); j++){
                    for(int d=0; d<cls.size(); d++)
                        clusters_shared[i][d].push_back(cls[d][j]);
                }
                i++;
            }
        }
    }

    return clusters_shared;
}

std::vector<std::vector<double>> get_new_centroids_par_soa(const std::vector<std::vector<double>> &points, const std::vector<std::vector<std::vector<double>>> &clusters){
    int dim = points[0].size();
    std::vector<std::vector<double>> centroids(clusters.size(), std::vector<double>(dim, 0.0));

    #pragma omp parallel for schedule(static)
    for(int k=0; k<clusters.size()*dim; k++){
        int i = k/dim;
        int d = k%dim;
        long cls_size = clusters[i][0].size();
        double cls_centroid_coord = 0;

        #pragma omp simd
        for(int j = 0; j<clusters[i][d].size(); j++){
            cls_centroid_coord += clusters[i][d][j];
        }

        cls_centroid_coord = cls_centroid_coord/cls_size;
        centroids[i][d] = cls_centroid_coord;
    }

    return centroids;
}

std::vector<std::vector<std::vector<double>>> kmeans_par_soa(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids){
    std::vector<std::vector<std::vector<double>>> clusters;
    const int MAX_ITERATIONS = 100; 
    const double epsilon = 100;

    auto centroids = init_centroids;
    for(int i = 0; i < MAX_ITERATIONS; i++){
        clusters = assign_points_par_soa(points, centroids);
        auto new_centroids = get_new_centroids_par_soa(points, clusters);

        double dist = 0;
        for(int j = 0; j<centroids.size(); j++){
            dist += distance_par_soa(centroids[j], new_centroids[j]);
        }

        if(dist <= epsilon)
            break;
        else 
            centroids = new_centroids;
    }

    return clusters;
}