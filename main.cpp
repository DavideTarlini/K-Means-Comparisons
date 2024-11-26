//#include "matplotplusplus/source/matplot/matplot.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

#include "src/kmeans_seq.cpp"
#include "src/kmeans_par.cpp"
#include "src/kmeans_par_simd.cpp"
#include "src/kmeans_par_soa.cpp"
#include "src/kmeans_par_single.cpp"
#include "src/kmeans_seq_single.cpp"
#include "src/kmeans_par_single_soa.cpp"

void generatePoints_AOS(int n, int dim, std::vector<std::vector<double>> &points, std::vector<std::vector<double>> &pointsSOA) {
    int lb_gs = 2;
    int ub_gs = 6;
    double lb_mean = -3000;
    double ub_mean = 3000;
    double lb_var = 0.5;
    double ub_var = 6;

    std::uniform_int_distribution<> unif_gaussians(lb_gs,ub_gs);
    std::uniform_real_distribution<> unif_means(lb_mean,ub_mean);
    std::uniform_real_distribution<> unif_var(lb_var,ub_var);
    std::random_device rd; 
	std::mt19937 re(rd());

    int gaussians_size = unif_gaussians(re);
    std::vector<std::normal_distribution<double>> gaussians;
    for(int i=0; i<gaussians_size; i++){
        int mean = unif_means(re);
        int var = unif_var(re);

        gaussians.push_back(std::normal_distribution<double>(mean, var));
    }

    pointsSOA = std::vector<std::vector<double>>(dim, std::vector<double>(n, 0.0));

    std::uniform_int_distribution<> unif(0,gaussians_size);
    for(int i=0; i<n; i++){
        std::vector<double> p;
        for(int j=0; j<dim; j++){
            int k = unif(re);
            std::normal_distribution<double> gs = gaussians[k];
            double v = gs(re);
            p.push_back(v);
            pointsSOA[j][i] = v;
        }

        points.push_back(p);
    }
}

void init_centroids(const std::vector<std::vector<double>> &points, const int k, std::vector<std::vector<double>> &centroids, std::vector<std::vector<double>> &centroidsSOA){
    std::uniform_int_distribution<long> d;
    std::random_device rd; 
	std::mt19937 re(rd());

    std::vector<long> centroids_indeces;
    int i = 0;
    while(i < k){
        long index = d(re)%points.size();
        
        bool seen = false;
        for(const long c_index: centroids_indeces){
            if(index == c_index){
                seen = true;
                break;
            }
        }

        if(!seen){
            centroids_indeces.push_back(index);
            i++;
        }
    }

    centroidsSOA = std::vector<std::vector<double>>(points[0].size(), std::vector<double>(k, 0.0));

    int j = 0;
    for(auto c_i: centroids_indeces){
        centroids.push_back(points[c_i]);

        for(int i=0; i<centroidsSOA.size(); i++){
            centroidsSOA[i][j] = points[c_i][i];
        }

        j++;
    }
}

int main(int, char**){
    using namespace std::chrono_literals;

    const int k = 5;
    const long num_of_points = 100000;
    const int dim = 15;

    std::vector<std::vector<double>> points;
    std::vector<std::vector<double>> pointsSOA;
    std::vector<std::vector<double>> int_centroids;
    std::vector<std::vector<double>> int_centroidsSOA;
    
    generatePoints_AOS(num_of_points, dim, points, pointsSOA);
    init_centroids(points, k, int_centroids, int_centroidsSOA);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto c_seq = kmeans_seq(k, points, int_centroids);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms       seq\n";

    std::this_thread::sleep_for(1s);
    
    t1 = std::chrono::high_resolution_clock::now();
    auto c_par = kmeans_par(k, points, int_centroids);
    t2 = std::chrono::high_resolution_clock::now();

    ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms       par\n";

    std::this_thread::sleep_for(1s);
    
    t1 = std::chrono::high_resolution_clock::now();
    auto c_par_simd = kmeans_par_simd(k, points, int_centroids);
    t2 = std::chrono::high_resolution_clock::now();

    ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms       par_simd\n";

    std::this_thread::sleep_for(1s);
    
    t1 = std::chrono::high_resolution_clock::now();
    auto c_par_soa = kmeans_par_soa(k, points, int_centroids);
    t2 = std::chrono::high_resolution_clock::now();

    ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms       par_soa\n";

    std::this_thread::sleep_for(1s);
    
    t1 = std::chrono::high_resolution_clock::now();
    auto c_par_single = kmeans_par_single(k, points, int_centroids);
    t2 = std::chrono::high_resolution_clock::now();

    ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms       par_single\n";

    std::this_thread::sleep_for(1s);

    t1 = std::chrono::high_resolution_clock::now();
    auto c_seq_single = kmeans_seq_single(k, points, int_centroids);
    t2 = std::chrono::high_resolution_clock::now();

    ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms       seq_single\n";

    std::this_thread::sleep_for(1s);

    t1 = std::chrono::high_resolution_clock::now();
    auto c_par_single_soa = kmeans_par_single_soa(k, pointsSOA, int_centroidsSOA);
    t2 = std::chrono::high_resolution_clock::now();

    ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms       par_single_soa\n";

    /*for(const auto p: points){
        std::cout<< "\n";
        std::cout<< "(";
        int i = 1;
        for(const double c: p){
            std::cout << c;
            if(i < dim)
                std::cout << ", ";
            i++;
        }
        std::cout<< ")";
    }*/

    /*std::cout << "\n\n -------------- Clusters --------------";
    for(auto cls: clusters){
        std::cout << "\n----- cls -----";
        for(const auto i: cls){
            auto p = points[i];
            std::cout<< "\n";
            std::cout<< "(";
            int k = 1;
            for(const double c: p){
                std::cout << c;
                if(k < dim)
                    std::cout << ", ";
                k++;
            }
            std::cout<< ")";
        }
    }

    for(auto cls: c_par_simd){
        std::vector<double> x;
        std::vector<double> y;

        for(auto i: cls){
            x.push_back(points[i][0]);
            y.push_back(points[i][1]);
        }
        
        auto l = matplot::scatter(x,y);
    }

    matplot::show();*/
}
