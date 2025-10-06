#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>   // CUDART_INF (double)

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

__global__ void assign_clusters(
    const double* __restrict__ points,
    const double* __restrict__ centers,
    int* __restrict__ labels,
    int* __restrict__ /*counts*/,
    double* __restrict__ /*sums*/,
    int N,
    int K,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread per point
    if (idx >= N) return;

    double min_dist = CUDART_INF; // double infinity
    int best_center = 0;

    for (int j = 0; j < K; ++j) {
        double dist = 0.0;
        const int pj = idx * D;
        const int cj = j   * D;
        for (int d = 0; d < D; ++d) {
            double diff = points[pj + d] - centers[cj + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_center = j;
        }
    }
    labels[idx] = best_center;
}

__global__ void accumulate(
    const double* __restrict__ points,
    const double* __restrict__ /*centers*/,
    const int* __restrict__ labels,
    int* __restrict__ counts,
    double* __restrict__ sums,
    int N,
    int K,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread per point
    if (idx >= N) return;

    int label = labels[idx];
    if (label < 0 || label >= K) return;

    // counts
    atomicAdd(&counts[label], 1);

    // sums
    const int base_p = idx * D;
    const int base_s = label * D;
    for (int d = 0; d < D; ++d) {
        atomicAdd(&sums[base_s + d], points[base_p + d]);  // native double atomicAdd on sm_60+
    }
}

__global__ void update_centers(
    const double* __restrict__ sums,
    const int* __restrict__ counts,
    double* __restrict__ new_centers,
    const double* __restrict__ old_centers,
    int K,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread per (k,d)
    if (idx >= K * D) return;

    int k = idx / D;
    if (counts[k] > 0) {
        new_centers[idx] = sums[idx] / static_cast<double>(counts[k]);
    } else {
        new_centers[idx] = old_centers[idx];
    }
}

__global__ void compute_shifts(
    const double* __restrict__ old_centers,
    const double* __restrict__ new_centers,
    double* __restrict__ shifts,
    int K,
    int D
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // thread per cluster
    if (k >= K) return;

    double acc = 0.0;
    const int base = k * D;
    for (int d = 0; d < D; ++d) {
        double diff = new_centers[base + d] - old_centers[base + d];
        acc += diff * diff;
    }
    shifts[k] = sqrt(acc);
}

int kmeans_cuda(
    const std::string& input_file,
    int K,
    int dims,
    int max_iter,
    double threshold,
    bool output_centroids,
    unsigned int seed
) {
    kmeans_srand(seed);

    // read data
    std::ifstream infile(input_file);
    if (!infile) {
        std::cerr << "Error: Unable to open input file " << input_file << "\n";
        return 1;
    }

    int _numpoints;
    infile >> _numpoints;

    std::vector<double> points(_numpoints * dims);
    for (int i = 0; i < _numpoints; ++i) {
        int index;
        infile >> index;
        for (int j = 0; j < dims; ++j) {
            infile >> points[i * dims + j];
        }
    }
    infile.close();

    // initial centers
    std::vector<double> centers(K * dims);
    for (int i = 0; i < K; ++i) {
        int index = kmeans_rand() % _numpoints;
        for (int d = 0; d < dims; ++d) {
            centers[i * dims + d] = points[index * dims + d];
        }
    }

    // device buffers
    double *d_points = nullptr, *d_centers = nullptr;
    CHECK_CUDA(cudaMalloc(&d_points,  static_cast<size_t>(_numpoints) * dims * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_centers, static_cast<size_t>(K) * dims * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_points,  points.data(),  static_cast<size_t>(_numpoints) * dims * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centers, centers.data(), static_cast<size_t>(K) * dims * sizeof(double),       cudaMemcpyHostToDevice));

    int *d_labels = nullptr, *d_counts = nullptr;
    double *d_sums = nullptr, *d_new_centers = nullptr, *d_shifts = nullptr;

    CHECK_CUDA(cudaMalloc(&d_labels,      static_cast<size_t>(_numpoints) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_counts,      static_cast<size_t>(K) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sums,        static_cast<size_t>(K) * dims * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_new_centers, static_cast<size_t>(K) * dims * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_shifts,      static_cast<size_t>(K) * sizeof(double)));

    std::vector<int> labels(_numpoints, -1);
    std::vector<double> h_shifts(K, 0.0);

    cudaEvent_t evStart, evStop;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evStop));

    CHECK_CUDA(cudaEventRecord(evStart));

    int iter_to_converge = 0;

    const int threads_per_block = 256;
    const int blocks_numpoints  = (_numpoints + threads_per_block - 1) / threads_per_block;
    const int blocks_k_dims     = (K * dims  + threads_per_block - 1) / threads_per_block;
    const int blocks_k          = (K        + threads_per_block - 1) / threads_per_block;

    while (iter_to_converge < max_iter) {
        // reset counts/sums
        CHECK_CUDA(cudaMemset(d_counts, 0, static_cast<size_t>(K) * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_sums,   0, static_cast<size_t>(K) * dims * sizeof(double)));

        assign_clusters<<<blocks_numpoints, threads_per_block>>>(
            d_points, d_centers, d_labels, d_counts, d_sums, _numpoints, K, dims
        );
        CHECK_CUDA(cudaGetLastError());

        accumulate<<<blocks_numpoints, threads_per_block>>>(
            d_points, d_centers, d_labels, d_counts, d_sums, _numpoints, K, dims
        );
        CHECK_CUDA(cudaGetLastError());

        update_centers<<<blocks_k_dims, threads_per_block>>>(
            d_sums, d_counts, d_new_centers, d_centers, K, dims
        );
        CHECK_CUDA(cudaGetLastError());

        compute_shifts<<<blocks_k, threads_per_block>>>(
            d_centers, d_new_centers, d_shifts, K, dims
        );
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaDeviceSynchronize());

        // convergence check
        CHECK_CUDA(cudaMemcpy(h_shifts.data(), d_shifts, static_cast<size_t>(K) * sizeof(double), cudaMemcpyDeviceToHost));
        double max_shift = 0.0;
        for (int i = 0; i < K; ++i) max_shift = std::max(max_shift, h_shifts[i]);

        CHECK_CUDA(cudaMemcpy(d_centers, d_new_centers, static_cast<size_t>(K) * dims * sizeof(double), cudaMemcpyDeviceToDevice));

        ++iter_to_converge;
        if (max_shift <= threshold) break;
    }

    CHECK_CUDA(cudaEventRecord(evStop));
    CHECK_CUDA(cudaEventSynchronize(evStop));

    float total_time_ms = 0.0f; // cudaEventElapsedTime returns float
    CHECK_CUDA(cudaEventElapsedTime(&total_time_ms, evStart, evStop));
    double time_per_iter_ms = (iter_to_converge > 0) ? static_cast<double>(total_time_ms) / iter_to_converge : 0.0;

    // outputs
    CHECK_CUDA(cudaMemcpy(labels.data(), d_labels, static_cast<size_t>(_numpoints) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(centers.data(), d_centers, static_cast<size_t>(K) * dims * sizeof(double), cudaMemcpyDeviceToHost));

    // CSV: iterations, avg_ms_per_iter
    printf("%d,%.6f\n", iter_to_converge, time_per_iter_ms);

    if (!output_centroids) {
        printf("clusters:");
        for (int i = 0; i < _numpoints; ++i) printf(" %d", labels[i]);
    } else {
        for (int i = 0; i < K; ++i) {
            printf("%d ", i);
            for (int d = 0; d < dims; ++d) {
                printf("%.10f ", centers[i * dims + d]);
            }
            printf("\n");
        }
    }

    // cleanup
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaFree(d_points);
    cudaFree(d_centers);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(d_sums);
    cudaFree(d_new_centers);
    cudaFree(d_shifts);

    return 0;
}