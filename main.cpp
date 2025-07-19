#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>
#include <omp.h>
#include <algorithm>

#include "kmeans_seq.h"
#include "kmeans_par.h"

auto safe_div(double num, double denom) {
    double res = 0;

    if (std::abs(denom) < 1e-9) {
        std::cerr << "Warning: denominator is zero or very small\n";
        res = -1.0;
    }else{
        res = num/denom;
    }

    return res;
}

void generate_points(int n, int dim, int numClusters, std::vector<std::vector<double>> &points, std::vector<std::vector<double>> &pointsSOA) {
    int num_clusters = numClusters;
    pointsSOA = std::vector<std::vector<double>>(dim, std::vector<double>(n, 0.0));

    std::random_device rd;
    std::mt19937 re(rd());    
    std::uniform_real_distribution<> cluster_center_dist(-400, 400); 
    std::uniform_real_distribution<> cluster_spread_dist(30.0, 50.0);

    std::vector<std::vector<double>> cluster_centers(num_clusters, std::vector<double>(dim));
    std::vector<std::vector<double>> cluster_var(num_clusters, std::vector<double>(dim));
    for (int i = 0; i < num_clusters; i++) {
        for (int d = 0; d < dim; ++d) {
            cluster_centers[i][d] = cluster_center_dist(re);
            cluster_var[i][d] = cluster_spread_dist(re);
        }
    }

    std::uniform_int_distribution<> cluster_picker(0, num_clusters - 1);

    for (int i = 0; i < n; i++) {
        std::vector<double> p(dim);
        int cluster_id = cluster_picker(re);

        for (int d = 0; d < dim; ++d) {
            std::normal_distribution<> point_dist(0.0, 1.0);
            p[d] = cluster_centers[cluster_id][d] + cluster_var[cluster_id][d]*point_dist(re);
            pointsSOA[d][i] = p[d];
        }

        points.push_back(p);
    }
}

void get_initial_centroids(int n, int dim, const int k, const std::vector<std::vector<double>> &points, std::vector<std::vector<double>> &centroids, std::vector<std::vector<double>> &centroidsSOA){
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

    centroidsSOA = std::vector<std::vector<double>>(dim, std::vector<double>(k, 0.0));

    int j = 0;
    for(auto c_i: centroids_indeces){
        centroids.push_back(points[c_i]);

        for(int i=0; i<dim; i++){
            centroidsSOA[i][j] = points[c_i][i];
        }

        j++;
    }
}

void exp_1(){
    using namespace std::chrono_literals;

    int threads[] = {2, 4, 8, 16, 32, 64, 128};
    long num_points[] = {1000, 10000, 100000, 1000000};
    int dimensions[] = {8};
    const int k = 5;

    try{
        std::ofstream sp_file("speedups.csv");

        for(int th=0; th<sizeof(threads)/sizeof(threads[0]); th++){
            omp_set_num_threads(threads[th]);

            for(int y = 0; y<sizeof(dimensions)/sizeof(dimensions[0]); y++){
                const int d = dimensions[y];

                for(int x = 0; x<sizeof(num_points)/sizeof(num_points[0]); x++){
                    const long n = num_points[x];

                    const int runs = 3;
                    std::vector<std::vector<double>> speed_res(4, std::vector<double>(runs, 0.0));

                    for(int w = 0; w<runs; w++){
                        std::cout << "Thread=" << threads[th] << ", D=" << d << ", N=" << n << ", W=" << w << std::endl;
                        std::vector<std::vector<double>> points;
                        std::vector<std::vector<double>> pointsSOA;
                        std::vector<std::vector<double>> init_centroids;
                        std::vector<std::vector<double>> init_centroidsSOA;
                        std::vector<double> points_flat;
                        for (const auto& p : points) {
                            points_flat.insert(points_flat.end(), p.begin(), p.end());
                        }
                        
                        generate_points(n, d, k, points, pointsSOA);
                        get_initial_centroids(n, d, k, points, init_centroids, init_centroidsSOA);

                        /******
                         * 
                         *  SEQUENTIAL
                         * 
                         *  */                      
                        auto t1 = std::chrono::high_resolution_clock::now();
                        auto c_seq_single = kmeans_seq_single(k, points, init_centroids);
                        auto t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_seq_single = t2 - t1;
                        //std::cout << ms_seq_single.count() << "ms       seq_single\n";
                        for (size_t h = 0; h < k; h++)
                        {
                            for (size_t l = 0; l < dimensions[y]; l++)
                            {
                                std::cout << c_seq_single[h][l] << " ";
                            }
                            std::cout << "\n";
                            
                        }
                        std::cout << "\n --seq-- \n\n";

                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_seq_simd = kmeans_seq_single_simd(k, points, init_centroids);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_seq_simd = t2 - t1;
                        //std::cout << ms_seq_single.count() << "ms       seq_single\n";
                        for (size_t h = 0; h < k; h++)
                        {
                            for (size_t l = 0; l < dimensions[y]; l++)
                            {
                                std::cout << c_seq_simd[h][l] << " ";
                            }
                            std::cout << "\n";
                            
                        }
                        std::cout << "\n --seq simd-- \n\n";
           

                        /******
                         * 
                         *  PARALLEL
                         * 
                         *  */ 
                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_par = kmeans_par(k, points, init_centroids);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_par = t2 - t1;
                        //std::cout << ms_par.count() << "ms       par\n";
                        for (size_t h = 0; h < k; h++)
                        {
                            for (size_t l = 0; l < dimensions[y]; l++)
                            {
                                std::cout << c_par[h][l] << " ";
                            }
                            std::cout << "\n"; 
                        }
                        std::cout << "\n --par atomic-- \n\n";

                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_par_single = kmeans_par_single(k, points, init_centroids);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_par_single = t2 - t1;
                        //std::cout << ms_par_single.count() << "ms       par_single\n";
                        for (size_t h = 0; h < k; h++)
                        {
                            for (size_t l = 0; l < dimensions[y]; l++)
                            {
                                std::cout << c_par_single[h][l] << " ";
                            }
                            std::cout << "\n"; 
                        }
                        std::cout << "\n --par no atomic-- \n\n";
                        
                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_par_single_simd = kmeans_par_single_simd(k, points, init_centroids);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_par_single_simd = t2 - t1;
                        //std::cout << ms_par_single_soa.count() << "ms       par_single_soa\n";
                        for (size_t h = 0; h < k; h++)
                        {
                            for (size_t l = 0; l < dimensions[y]; l++)
                            {
                                std::cout << c_par_single_simd[h][l] << " ";
                            }
                            std::cout << "\n"; 
                        }
                        std::cout << "\n --par no atomic simd-- \n\n";   


                        /* Speedups */
                        speed_res[0][w] = safe_div(ms_seq_single.count(), ms_seq_simd.count());
                        speed_res[1][w] = safe_div(ms_seq_single.count(), ms_par.count());
                        speed_res[2][w] = safe_div(ms_seq_single.count(), ms_par_single.count());
                        speed_res[3][w] = safe_div(ms_seq_single.count(), ms_par_single_simd.count());
                        
                    }


                    /******
                         * 
                         *  Speedups
                         * 
                        *  */ 
                    int s = speed_res.size();
                    std::vector<double> sp_means(s);
                    std::vector<double> sp_stds(s);

                    for(int i = 0; i < s; i++){
                        sp_means[i] = std::accumulate(speed_res[i].begin(), speed_res[i].end(), 0.0)/speed_res[i].size();
                    
                        double accum = 0.0;
                        std::for_each(speed_res[i].begin(), speed_res[i].end(), [&](const double j) {
                            accum += (j - sp_means[i]) * (j - sp_means[i]);
                        });
                        sp_stds[i] = sqrt(accum / (speed_res[i].size() - 1));
                    }

                    sp_file << threads[th] << ", "
                        << k << ", "
                        << d << ", "
                        << n;

                    for (int i = 0; i < s; i++) {
                        sp_file << ", " << sp_means[i];
                    }

                    for (int i = 0; i < s; i++) {
                        sp_file << ", " << sp_stds[i];
                    }

                    sp_file << "\n";
                    sp_file.flush(); 
                }
            }
        } 
        
        sp_file.close();
    }catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return;
    }
}

int main(int, char**){
    exp_1();
}
