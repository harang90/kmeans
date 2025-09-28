#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
    const float* points,
    const float* centers,
    int* labels,
    int* counts,
    float* sums,
    int N,
    int K,
    int D
) {

    // blocks_per_grid
    // threads_per_block
    // blockDim.x = threads_per_block

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread for each numpoint

    if (idx < N) { // thread for each numpoint

        // find nearest center for points[idx]
        float min_dist = HUGE_VALF;
        int best_center = -1;
        for (int j = 0; j < K; ++j) {
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = points[idx * D + d] - centers[j * D + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_center = j;
            }
        }
        labels[idx] = best_center;

        // accumulate counts for each cluster
        atomicAdd(&counts[best_center], 1);

        // accumulate sums for each cluster
        for (int d = 0; d < D; ++d) {
            atomicAdd(&sums[best_center * D + d], points[idx * D + d]);
        }

    }
}

__global__ void update_centers(
    const float* sums,
    const int* counts,
    float* new_centers,
    float* old_centers,
    int K,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= K * D) return; // thread per dimension * cluster

    int k = idx / D; // cluster index

    if (counts[k] > 0) {
        new_centers[idx] = sums[idx] / counts[k];
    } else {
        new_centers[idx] = old_centers[idx];
    }
}

__global__ void compute_shifts(
    const float* old_centers,
    const float* new_centers,
    float* shifts,
    int K,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= K) return; // thread per cluster num

    float shift = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = new_centers[idx * D + d] - old_centers[idx * D + d];
        shift += diff * diff;
    }
    shifts[idx] = sqrtf(shift);
}

int kmeans_cuda(
    const std::string& input_file,
    int k,
    int dims,
    int max_iter,
    float threshold,
    bool output_centroids,
    unsigned int seed
) {

    kmeans_srand(seed);

    // read data from input_file
    // read first line to get _numpoints

    std::ifstream infile(input_file);
    if (!infile) {
        std::cerr << "Error: Unable to open input file " << input_file << "\n";
        return 1;
    }

    int _numpoints;
    infile >> _numpoints;

    // allocate memory for data points
    // std::vector<std::vector<double>> points(_numpoints, std::vector<double>(dims));
    std::vector<float> points(_numpoints * dims);

    // line has form 'index dim1 dim2 ... dimN'
    for (int i = 0; i < _numpoints; ++i) {
        int index;
        infile >> index;
        for (int j = 0; j < dims; ++j) {
            infile >> points[i * dims + j];
        }
    }
    infile.close();

    // allocate memory for cluster centers
    std::vector<float> centers(k * dims);

    // initialize cluster centers by randomly selecting k points
    for (int i = 0; i < k; ++i) {
        int index = kmeans_rand() % _numpoints;
        for (int d = 0; d < dims; ++d) {
            centers[i * dims + d] = points[index * dims + d];
        }
    }
    
    // copy to device
    float *d_points;
    float *d_centers;

    CHECK_CUDA(cudaMalloc(&d_points, _numpoints * dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_centers, k * dims * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_points, points.data(), _numpoints * dims * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centers, centers.data(), k * dims * sizeof(float), cudaMemcpyHostToDevice))u;

    // allocate device memory for labels, counts, sums, and new_centers
    int *d_labels;
    int *d_counts;
    float *d_sums;
    float *d_new_centers;
    float *d_shifts;

    CHECK_CUDA(cudaMalloc(&d_labels, _numpoints * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_counts, k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sums, k * dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_new_centers, k * dims * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shifts, k * sizeof(float)));

    // prepare array to hold point labels
    std::vector<int> labels(_numpoints, -1);
    std::vector<float> h_shifts(k, 0.0f);

    cudaEvent_t evStart, evStop;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evStop));
    
    // start timing
    CHECK_CUDA(cudaEventRecord(evStart));

    // iteration counter
    int iter_to_converge = 0;

    // iterations
    while (iter_to_converge < max_iter) {

        // assign clusters using CUDA kernel
        int threads_per_block = 256;
        int blocks_numpoints = (_numpoints + threads_per_block - 1) / threads_per_block; 
        int blocks_k_dims = (k * dims + threads_per_block - 1) / threads_per_block;
        int blocks_k = (k + threads_per_block - 1) / threads_per_block;

        // reset counts and sums on device
        CHECK_CUDA(cudaMemset(d_counts, 0, k * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_sums, 0, k * dims * sizeof(float)));

        assign_clusters<<<blocks_numpoints, threads_per_block>>>(
            d_points,
            d_centers,
            d_labels,
            d_counts,
            d_sums,
            _numpoints,
            k,
            dims
        );

        CHECK_CUDA(cudaGetLastError());

        update_centers<<<blocks_k_dims, threads_per_block>>>(
            d_sums,
            d_counts,
            d_new_centers,
            d_centers,
            k,
            dims
        );

        CHECK_CUDA(cudaGetLastError());

        compute_shifts<<<blocks_k, threads_per_block>>>(
            d_centers,
            d_new_centers,
            d_shifts,
            k,
            dims
        );

        CHECK_CUDA(cudaGetLastError());

        // synchronize to ensure all kernels are done
        CHECK_CUDA(cudaDeviceSynchronize());

        // copy shifts back to host
        CHECK_CUDA(cudaMemcpy(h_shifts.data(), d_shifts, k * sizeof(float), cudaMemcpyDeviceToHost));

        float max_shift = 0.0f;
        for (int i = 0; i < k; ++i) {
            max_shift = std::max(max_shift, h_shifts[i]);
        }

        // update centers
        std::swap(d_centers, d_new_centers);

        // update iteration count
        iter_to_converge++;

        // check for convergence
        if (max_shift <= threshold) {
            break;
        }
    }

    // stop timing
    CHECK_CUDA(cudaEventRecord(evStop));
    CHECK_CUDA(cudaEventSynchronize(evStop));

    auto total_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_time_ms, evStart, evStop));

    auto time_per_iter_ms = (iter_to_converge > 0) ? (double)total_time_ms / iter_to_converge : 0.0;

    // copy data back to host
    CHECK_CUDA(cudaMemcpy(labels.data(), d_labels, _numpoints * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(centers.data(), d_centers, k * dims * sizeof(float), cudaMemcpyDeviceToHost));

    printf("%d,%lf\n", iter_to_converge, time_per_iter_ms);

    // output point labels
    if (!output_centroids) {
        printf("clusters:");

        // print cluster id of each point
        for (int i = 0; i < _numpoints; ++i) {
            printf(" %d", labels[i]);
        }
    }

    // output cluster centers if required
    if (output_centroids) {
        for (int i = 0; i < k; ++i) {
            printf("%d ", i);
            for (int d = 0; d < dims; ++d) {
                printf("%f ", centers[i * dims + d]);
            }
            printf("\n");
        }
    }

    // free device memory
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