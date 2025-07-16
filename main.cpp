#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>
#include <omp.h>
#include <vector>
#include <random>
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
    std::uniform_real_distribution<> cluster_center_dist(-3000, 3000); 
    std::uniform_real_distribution<> cluster_spread_dist(30.0, 200.0);

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

void get_initial_centroids(const std::vector<std::vector<double>> &points, const int k, std::vector<std::vector<double>> &centroids, std::vector<std::vector<double>> &centroidsSOA){
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

void exp_1(){
    using namespace std::chrono_literals;

    int threads[] = {2, 4, 8, 12, 16, 24};
    long num_points[] = {1000, 10000, 100000, 1000000};
    int dimensions[] = {8};
    const int k = 5;

    try{
        std::ofstream t_file("time.csv");
        std::ofstream sp_file("speedups.csv");

        for(int th=0; th<sizeof(threads)/sizeof(threads[0]); th++){
            omp_set_num_threads(threads[th]);

            for(int y = 0; y<sizeof(dimensions)/sizeof(dimensions[0]); y++){
                const int d = dimensions[y];

                for(int x = 0; x<sizeof(num_points)/sizeof(num_points[0]); x++){
                    const long n = num_points[x];
                    std::vector<std::vector<double>> time_res(10, std::vector<double>(10, 0.0));
                    std::vector<std::vector<double>> speed_res(9, std::vector<double>(10, 0.0));

                    for(int w = 0; w<10; w++){
                        std::cout << "Thread=" << threads[th] << ", D=" << d << ", N=" << n << ", W=" << w << std::endl;
                        std::vector<std::vector<double>> points;
                        std::vector<std::vector<double>> pointsSOA;
                        std::vector<std::vector<double>> init_centroids;
                        std::vector<std::vector<double>> init_centroidsSOA;
                        
                        generate_points(n, d, k, points, pointsSOA);
                        get_initial_centroids(points, k, init_centroids, init_centroidsSOA);

                        /******
                         * 
                         *  SEQUENTIAL
                         * 
                         *  */ 
                        auto t1 = std::chrono::high_resolution_clock::now();
                        auto c_seq = kmeans_seq(k, points, init_centroids);
                        auto t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_seq = t2 - t1;
                        //std::cout << ms_seq.count() << "ms       seq\n";

                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_seq_single = kmeans_seq_single(k, points, init_centroids);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_seq_single = t2 - t1;
                        //std::cout << ms_seq_single.count() << "ms       seq_single\n";

                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_seq_single_simd = kmeans_seq_single_simd(k, points, init_centroids);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_seq_single_simd = t2 - t1;
                        //std::cout << ms_seq_single_simd.count() << "ms       seq_single_simd\n";  

                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_seq_single_soa = kmeans_seq_single_soa(k, pointsSOA, init_centroidsSOA);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_seq_single_soa = t2 - t1;
                        //std::cout << ms_seq_single_soa.count() << "ms       seq_single_soa\n";                  

                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_seq_single_soa_simd = kmeans_seq_single_soa_simd(k, pointsSOA, init_centroidsSOA);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_seq_single_soa_simd = t2 - t1;
                        //std::cout << ms_seq_single_soa_simd.count() << "ms       seq_single_soa_simd\n";


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
                        
                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_par_single = kmeans_par_single(k, points, init_centroids);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_par_single = t2 - t1;
                        //std::cout << ms_par_single.count() << "ms       par_single\n";
                        
                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_par_single_simd = kmeans_par_single_simd(k, points, init_centroids);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_par_single_simd = t2 - t1;
                        //std::cout << ms_par_single_simd.count() << "ms       par_single_simd\n";                    

                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_par_single_soa = kmeans_par_single_soa(k, pointsSOA, init_centroidsSOA);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_par_single_soa = t2 - t1;
                        //std::cout << ms_par_single_soa.count() << "ms       par_single_soa\n";

                        t1 = std::chrono::high_resolution_clock::now();
                        auto c_par_single_soa_simd = kmeans_par_single_soa_simd(k, pointsSOA, init_centroidsSOA);
                        t2 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> ms_par_single_soa_simd = t2 - t1;
                        //std::cout << ms_par_single_soa_simd.count() << "ms       par_single_soa_simd\n";

                        /* Time */
                        time_res[0][w] = ms_seq.count();
                        time_res[1][w] = ms_seq_single.count();
                        time_res[2][w] = ms_seq_single_simd.count();  
                        time_res[3][w] = ms_seq_single_soa.count();
                        time_res[4][w] = ms_seq_single_soa_simd.count();

                        time_res[5][w] = ms_par.count();
                        time_res[6][w] = ms_par_single.count();
                        time_res[7][w] = ms_par_single_simd.count();
                        time_res[8][w] = ms_par_single_soa.count();
                        time_res[9][w] = ms_par_single_soa_simd.count();


                        /* Speedups */
                        speed_res[0][w] = safe_div(ms_seq.count(), ms_par.count());
                        speed_res[1][w] = safe_div(ms_seq.count(), ms_seq_single.count());

                        speed_res[2][w] = safe_div(ms_seq_single.count(), ms_seq_single_simd.count());
                        speed_res[3][w] = safe_div(ms_seq_single.count(), ms_seq_single_soa.count());
                        speed_res[4][w] = safe_div(ms_seq_single.count(), ms_seq_single_soa_simd.count());

                        speed_res[5][w] = safe_div(ms_seq_single.count(), ms_par_single.count());
                        speed_res[6][w] = safe_div(ms_seq_single.count(), ms_par_single_simd.count());
                        speed_res[7][w] = safe_div(ms_seq_single.count(), ms_par_single_soa.count());
                        speed_res[8][w] = safe_div(ms_seq_single.count(), ms_par_single_soa_simd.count());
                    }

                    /******
                         * 
                         *  Time
                         * 
                         *  */ 
                    int s = time_res.size();
                    std::vector<double> t_means(s);
                    std::vector<double> t_stds(s);
                    
                    for(int i = 0; i < s; i++){
                        t_means[i] = std::accumulate(time_res[i].begin(), time_res[i].end(), 0.0)/time_res[i].size();
                    
                        double accum = 0.0;
                        std::for_each(time_res[i].begin(), time_res[i].end(), [&](const double j) {
                            accum += (j - t_means[i]) * (j - t_means[i]);
                        });
                        t_stds[i] = sqrt(accum / (time_res[i].size() - 1));
                    }

                    t_file << threads[th] << ", "
                        << k << ", "
                        << d << ", "
                        << n;

                    for (int i = 0; i < s; i++) {
                        t_file << ", " << t_means[i];
                    }

                    for (int i = 0; i < s; i++) {
                        t_file << ", " << t_stds[i];
                    }

                    t_file << "\n";
                    t_file.flush();

                    /******
                         * 
                         *  Speedups
                         * 
                        *  */ 
                    s = speed_res.size();
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
        
        t_file.close();
        sp_file.close();
    }catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return;
    }
}

int main(int, char**){
    exp_1();
}
