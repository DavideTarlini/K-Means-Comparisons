#include <vector>
#include <algorithm>
#include <random>

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

std::vector<std::vector<long>> assign_points(const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &centroids){
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

std::vector<std::vector<double>> get_new_centroids(const std::vector<std::vector<double>> &points, const std::vector<std::vector<long>> &clusters){
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

std::vector<std::vector<long>> kmeans_seq(const int k, const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> init_centroids){
    std::vector<std::vector<long>> clusters;
    std::vector<std::vector<double>> centroids = init_centroids;
    const int MAX_ITERATIONS = 100; 
    const double epsilon = 100;

    for(int i = 0; i < MAX_ITERATIONS; i++){
        clusters = assign_points(points, centroids);
        auto new_centroids = get_new_centroids(points, clusters);

        double dist = 0;
        for(int j = 0; j<centroids.size(); j++){
            dist += distance_seq(centroids[j], new_centroids[j]);
        }

        if(dist <= epsilon) 
            break;
        else 
            centroids = new_centroids;
    }

    return clusters;
}