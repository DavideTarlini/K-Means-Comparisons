//#include "matplotplusplus/source/matplot/matplot.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>

#include "src/kmeans_seq.cpp"
#include "src/kmeans_par.cpp"
#include "src/kmeans_par_simd.cpp"
#include "src/kmeans_par_single.cpp"
#include "src/kmeans_seq_single.cpp"
#include "src/kmeans_par_single_soa.cpp"

void generatePoints(int n, int dim, std::vector<std::vector<double>> &points, std::vector<std::vector<double>> &pointsSOA) {
    /*int lb_gs = 3;
    int ub_gs = 7;
    double lb_mean = -4000;
    double ub_mean = 4000;
    double lb_var = 0.5;
    double ub_var = 6;*/

    /*std::uniform_int_distribution<> unif_gaussians(lb_gs,ub_gs);
    std::uniform_real_distribution<> unif_means(lb_mean,ub_mean);
    std::uniform_real_distribution<> unif_var(lb_var,ub_var);*/
    std::random_device rd; 
	std::mt19937 re(rd());

    /*int gaussians_size = unif_gaussians(re);
    std::vector<std::normal_distribution<double>> gaussians;
    for(int i=0; i<gaussians_size; i++){
        int mean = unif_means(re);
        int var = unif_var(re);

        gaussians.push_back(std::normal_distribution<double>(mean, var));
    }*/

    pointsSOA = std::vector<std::vector<double>>(dim, std::vector<double>(n, 0.0));

    //std::uniform_int_distribution<> unif(0,gaussians_size);
    std::uniform_int_distribution<> unif(-3000,3000);
    for(int i=0; i<n; i++){
        std::vector<double> p;
        for(int j=0; j<dim; j++){
            double v = unif(re);
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

    int threads[] = {16, 24, 32};
    long n[] = {5000000, 10000000};
    int d[] = {2};

    std::ofstream file("high_dimensional_speedup.csv");
    file << "th" << ", "
                << "k, "
                << "dim, "
                << "num_points, "
                << "par_mean, "
                //<< "par_simd_mean, "
                << "par_single_mean, "
                << "par_stdev, "
                //<< "par_simd_stdev, "
                << "par_single_stdev" 
                << "\n";
                file.flush();

    for(int th=0; th<3; th++){
        omp_set_num_threads(threads[th]);

        for(int i = 0; i<1; i++){
            for(int j = 0; j<2; j++){

                std::vector<std::vector<double>> speed_res(3, std::vector<double>(5, 0.0));
                for(int w = 0; w<5; w++){
                    const int k = 5;
                    const long num_of_points = n[j];
                    const int dim = d[i];

                    std::vector<std::vector<double>> points;
                    std::vector<std::vector<double>> pointsSOA;
                    std::vector<std::vector<double>> int_centroids;
                    std::vector<std::vector<double>> int_centroidsSOA;
                    
                    generatePoints(num_of_points, dim, points, pointsSOA);
                    init_centroids(points, k, int_centroids, int_centroidsSOA);

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
                    
                    /*t1 = std::chrono::high_resolution_clock::now();
                    auto c_par_simd = kmeans_par_simd(k, points, int_centroids);
                    t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> ms_par_simd = t2 - t1;
                    std::cout << ms_par_simd.count() << "ms       par_simd\n";*/
                    
                    t1 = std::chrono::high_resolution_clock::now();
                    auto c_par_single = kmeans_par_single(k, points, int_centroids);
                    t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> ms_par_single = t2 - t1;
                    std::cout << ms_par_single.count() << "ms       par_single\n";

                    t1 = std::chrono::high_resolution_clock::now();
                    auto c_seq_single = kmeans_seq_single(k, points, int_centroids);
                    t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> ms_seq_single = t2 - t1;
                    std::cout << ms_seq_single.count() << "ms       seq_single\n";

                    /*auto t1 = std::chrono::high_resolution_clock::now();
                    auto c_par_single_soa = kmeans_par_single_soa(k, pointsSOA, int_centroidsSOA);
                    auto t2 = std::chrono::high_resolution_clock::now();

                    auto ms_par_single_soa = t2 - t1;
                    std::cout << ms_par_single_soa.count() << "ms       par_single_soa\n";*/

                    auto speedup_par = ms_seq/ms_par;
                    //auto speedup_par_simd = ms_seq/ms_par_simd;
                    auto speedup_par_single = ms_seq_single/ms_par_single;

                    speed_res[0][w] = speedup_par;
                    //speed_res[1][w] = speedup_par_simd;
                    speed_res[1][w] = speedup_par_single;               
                }

                auto par_mean = std::accumulate(speed_res[0].begin(), speed_res[0].end(), 0.0)/speed_res[0].size();
                //auto par_simd_mean = std::accumulate(speed_res[1].begin(), speed_res[1].end(), 0.0)/speed_res[1].size();
                auto par_single_mean = std::accumulate(speed_res[1].begin(), speed_res[1].end(), 0.0)/speed_res[1].size();

                double accum = 0.0;
                std::for_each (speed_res[0].begin(), speed_res[0].end(), [&](const double d) {
                    accum += (d - par_mean) * (d - par_mean);
                });

                double par_stdev = sqrt(accum / (speed_res[0].size()-1));

                /*accum = 0.0;
                std::for_each (speed_res[1].begin(), speed_res[1].end(), [&](const double d) {
                    accum += (d - par_simd_mean) * (d - par_simd_mean);
                });

                double par_simd_stdev = sqrt(accum / (speed_res[1].size()-1));*/

                accum = 0.0;
                std::for_each (speed_res[1].begin(), speed_res[1].end(), [&](const double d) {
                    accum += (d - par_single_mean) * (d - par_single_mean);
                });

                double par_single_stdev = sqrt(accum / (speed_res[1].size()-1));

                file << threads[th] << ", "
                << 5 << ", "
                << d[i] << ", "
                << n[j] << ", "
                << par_mean << ", "
                //<< par_simd_mean << ", "
                << par_single_mean << ", "
                << par_stdev << ", "
                //<< par_simd_stdev << ", "
                << par_single_stdev 
                << "\n";
                file.flush();
            }
        }
    }

    file.close();    
}
