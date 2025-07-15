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

#include "kmeans_seqs.h"
#include "kmeans_pars.h"


void generatePoints(int n, int dim, int numClusters, std::vector<std::vector<double>> &points, std::vector<std::vector<double>> &pointsSOA) {
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

// Times and Speedups of the 3 versions: 2-step, 1-step, 1-step + SOA
void exp_1(){
    using namespace std::chrono_literals;

    int threads[] = {2, 4, 8, 16};
    long n[] = {1000, 10000, 100000, 1000000};
    int d[] = {2, 5, 8};
    const int k = 5;

    std::ofstream file_1("time.csv");
    file_1 << "th" << ", "
                << "k, "
                << "dim, "
                << "num_points, "
                << "seq_mean, "
                << "par_mean, "
                << "seq_single_mean, "
                << "par_single_mean, "
                << "seq_single_soa_mean, "
                << "par_single_soa_mean, "
                << "seq_std, "
                << "par_std, "
                << "seq_single_std, "
                << "par_single_std, "
                << "seq_single_soa_std, "
                << "par_single_soa_std" 
                << "\n";
                file_1.flush();

    std::ofstream file_2("speedups.csv");
    file_2 << "th" << ", "
                << "k, "
                << "dim, "
                << "num_points, "
                << "par_mean, "
                << "par_single_mean, "
                << "par_single_soa_mean, "
                << "par_single_mix_mean, "
                << "par_std, "
                << "par_single_std, " 
                << "par_single_soa_std, " 
                << "par_single_mix_std"
                << "\n";
                file_2.flush();

    for(int th=0; th<4; th++){
        omp_set_num_threads(threads[th]);

        for(int i = 0; i<3; i++){
            for(int j = 0; j<4; j++){
                std::vector<std::vector<double>> time_res(6, std::vector<double>(10, 0.0));
                std::vector<std::vector<double>> speed_res(4, std::vector<double>(10, 0.0));

                for(int w = 0; w<10; w++){
                    const long num_of_points = n[j];
                    const int dim = d[i];

                    std::vector<std::vector<double>> points;
                    std::vector<std::vector<double>> pointsSOA;
                    std::vector<std::vector<double>> int_centroids;
                    std::vector<std::vector<double>> int_centroidsSOA;
                    
                    generatePoints(num_of_points, dim, k, points, pointsSOA);
                    init_centroids(points, k, int_centroids, int_centroidsSOA);


                    /******
                     * 
                     *  2-step version
                     * 
                     *  */ 
                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto c_seq = kmeans_seq(k, points, int_centroids);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> ms_seq = t2 - t1;
                    std::cout << ms_seq.count() << "ms       seq\n";
                    
                    t1 = std::chrono::high_resolution_clock::now();
                    auto c_par = kmeans_par(k, points, int_centroids);
                    t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> ms_par = t2 - t1;
                    std::cout << ms_par.count() << "ms       par\n";


                    /******
                     * 
                     *  1-step version
                     * 
                     *  */ 
                    t1 = std::chrono::high_resolution_clock::now();
                    auto c_seq_single = kmeans_seq_single(k, points, int_centroids);
                    t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> ms_seq_single = t2 - t1;
                    std::cout << ms_seq_single.count() << "ms       seq_single\n";
                    
                    t1 = std::chrono::high_resolution_clock::now();
                    auto c_par_single = kmeans_par_single(k, points, int_centroids);
                    t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> ms_par_single = t2 - t1;
                    std::cout << ms_par_single.count() << "ms       par_single\n";


                    /******
                     * 
                     *  1-step + SOA version
                     * 
                     *  */ 
                    t1 = std::chrono::high_resolution_clock::now();
                    auto c_seq_single_soa = kmeans_seq_single_soa(k, pointsSOA, int_centroidsSOA);
                    t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> ms_seq_single_soa = t2 - t1;
                    std::cout << ms_seq_single_soa.count() << "ms       seq_single_soa\n";

                    t1 = std::chrono::high_resolution_clock::now();
                    auto c_par_single_soa = kmeans_par_single_soa(k, pointsSOA, int_centroidsSOA);
                    t2 = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double, std::milli> ms_par_single_soa = t2 - t1;
                    std::cout << ms_par_single_soa.count() << "ms       par_single_soa\n";

                    time_res[0][w] = ms_seq.count();
                    time_res[1][w] = ms_par.count();
                    time_res[2][w] = ms_seq_single.count();  
                    time_res[3][w] = ms_par_single.count();
                    time_res[5][w] = ms_seq_single_soa.count();
                    time_res[5][w] = ms_par_single_soa.count();

                    auto speedup_par = ms_seq/ms_par;
                    auto speedup_par_single_mix = ms_seq_single/ms_par_single_soa;
                    auto speedup_par_single = ms_seq_single/ms_par_single;
                    auto speedup_par_single_soa = ms_seq_single_soa/ms_par_single_soa;

                    speed_res[0][w] = speedup_par;
                    speed_res[1][w] = speedup_par_single_mix;
                    speed_res[2][w] = speedup_par_single;  
                    speed_res[3][w] = speedup_par_single_soa;           
                }

                /******
                     * 
                     *  Time
                     * 
                     *  */ 
                auto t_seq_mean = std::accumulate(time_res[0].begin(), time_res[0].end(), 0.0)/time_res[0].size();
                auto t_par_mean = std::accumulate(time_res[1].begin(), time_res[1].end(), 0.0)/time_res[1].size();
                auto t_seq_single_mean = std::accumulate(time_res[2].begin(), time_res[2].end(), 0.0)/time_res[2].size();
                auto t_par_single_mean = std::accumulate(time_res[3].begin(), time_res[3].end(), 0.0)/time_res[3].size();
                auto t_seq_single_soa_mean = std::accumulate(time_res[4].begin(), time_res[4].end(), 0.0)/time_res[4].size();
                auto t_par_single_soa_mean = std::accumulate(time_res[5].begin(), time_res[5].end(), 0.0)/time_res[5].size();

                double accum = 0.0;
                std::for_each (time_res[0].begin(), time_res[0].end(), [&](const double d) {
                    accum += (d - t_seq_mean) * (d - t_seq_mean);
                });

                double t_seq_std = sqrt(accum / (time_res[0].size()-1));

                accum = 0.0;
                std::for_each (time_res[1].begin(), time_res[1].end(), [&](const double d) {
                    accum += (d - t_par_mean) * (d - t_par_mean);
                });

                double t_par_std = sqrt(accum / (time_res[1].size()-1));

                accum = 0.0;
                std::for_each (time_res[2].begin(), time_res[2].end(), [&](const double d) {
                    accum += (d - t_seq_single_mean) * (d - t_seq_single_mean);
                });

                double t_seq_single_std = sqrt(accum / (time_res[2].size()-1));

                accum = 0.0;
                std::for_each (time_res[3].begin(), time_res[3].end(), [&](const double d) {
                    accum += (d - t_par_single_mean) * (d - t_par_single_mean);
                });

                double t_par_single_std = sqrt(accum / (time_res[3].size()-1));

                accum = 0.0;
                std::for_each (time_res[4].begin(), time_res[4].end(), [&](const double d) {
                    accum += (d - t_seq_single_soa_mean) * (d - t_seq_single_soa_mean);
                });

                double t_seq_single_soa_std = sqrt(accum / (time_res[4].size()-1));

                accum = 0.0;
                std::for_each (time_res[5].begin(), time_res[5].end(), [&](const double d) {
                    accum += (d - t_par_single_soa_mean) * (d - t_par_single_soa_mean);
                });

                double t_par_single_soa_std = sqrt(accum / (time_res[5].size()-1));

                file_1 << threads[th] << ", "
                << k << ", "
                << d[i] << ", "
                << n[j] << ", "
                << t_seq_mean << ", "
                << t_par_mean << ", "
                << t_seq_single_mean << ", "
                << t_par_single_mean << ", "
                << t_seq_single_soa_mean << ", "
                << t_par_single_soa_mean << ", "
                << t_seq_std << ", "
                << t_par_std << ", "
                << t_seq_single_std << ", "
                << t_par_single_std << ", "
                << t_seq_single_soa_std << ", "
                << t_par_single_soa_std
                << "\n";
                file_1.flush();

                /******
                     * 
                     *  Speedups
                     * 
                     *  */ 
                auto s_par_mean = std::accumulate(speed_res[0].begin(), speed_res[0].end(), 0.0)/speed_res[0].size();
                auto s_par_single_mix_mean = std::accumulate(speed_res[1].begin(), speed_res[1].end(), 0.0)/speed_res[1].size();
                auto s_par_single_mean = std::accumulate(speed_res[2].begin(), speed_res[2].end(), 0.0)/speed_res[2].size();
                auto s_par_single_soa_mean = std::accumulate(speed_res[3].begin(), speed_res[3].end(), 0.0)/speed_res[3].size();

                accum = 0.0;
                std::for_each (speed_res[0].begin(), speed_res[0].end(), [&](const double d) {
                    accum += (d - s_par_mean) * (d - s_par_mean);
                });

                double s_par_std = sqrt(accum / (speed_res[0].size()-1));

                accum = 0.0;
                std::for_each (speed_res[1].begin(), speed_res[1].end(), [&](const double d) {
                    accum += (d - s_par_single_mix_mean) * (d - s_par_single_mix_mean);
                });

                double s_par_single_mix_std = sqrt(accum / (speed_res[1].size()-1));

                accum = 0.0;
                std::for_each (speed_res[2].begin(), speed_res[2].end(), [&](const double d) {
                    accum += (d - s_par_single_mean) * (d - s_par_single_mean);
                });

                double s_par_single_std = sqrt(accum / (speed_res[2].size()-1));

                accum = 0.0;
                std::for_each (speed_res[3].begin(), speed_res[3].end(), [&](const double d) {
                    accum += (d - s_par_single_soa_mean) * (d - s_par_single_soa_mean);
                });

                double s_par_single_soa_std = sqrt(accum / (speed_res[3].size()-1));

                file_2 << threads[th] << ", "
                << k << ", "
                << d[i] << ", "
                << n[j] << ", "
                << s_par_mean << ", "
                << s_par_single_mean << ", "
                << s_par_single_soa_mean << ", "
                << s_par_single_mix_mean << ", "
                << s_par_std << ", "
                << s_par_single_std << ", "
                << s_par_single_soa_std << ", "
                << s_par_single_mix_std 
                << "\n";
                file_2.flush();
            }
        }
    }

    file_2.close();    
}

int main(int, char**){
    exp_1();
}
