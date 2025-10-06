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

    int *d_labels = nullptr, *d_counts = nullptr;
    double *d_sums = nullptr, *d_new_centers = nullptr, *d_shifts = nullptr;

    CHECK_CUDA(cudaMalloc(&d_labels,      static_cast<size_t>(_numpoints) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_counts,      static_cast<size_t>(K) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sums,        static_cast<size_t>(K) * dims * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_new_centers, static_cast<size_t>(K) * dims * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_shifts,      static_cast<size_t>(K) * sizeof(double)));

    std::vector<int> labels(_numpoints, -1);
    std::vector<double> h_shifts(K, 0.0);

    // ---------- Timing setup ----------
    cudaEvent_t evTotalStart, evTotalStop, evA, evB;
    CHECK_CUDA(cudaEventCreate(&evTotalStart));
    CHECK_CUDA(cudaEventCreate(&evTotalStop));
    CHECK_CUDA(cudaEventCreate(&evA));
    CHECK_CUDA(cudaEventCreate(&evB));

    float htod_ms = 0.0f;
    float dtoh_ms = 0.0f;
    float total_ms = 0.0f;

    // E2E start BEFORE any transfers to device
    CHECK_CUDA(cudaEventRecord(evTotalStart));

    // ---------- H→D transfers (timed) ----------
    CHECK_CUDA(cudaEventRecord(evA));
    CHECK_CUDA(cudaMemcpy(d_points,  points.data(),  static_cast<size_t>(_numpoints) * dims * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(evB));
    CHECK_CUDA(cudaEventSynchronize(evB));
    float tmp = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&tmp, evA, evB));
    htod_ms += tmp;

    CHECK_CUDA(cudaEventRecord(evA));
    CHECK_CUDA(cudaMemcpy(d_centers, centers.data(), static_cast<size_t>(K) * dims * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(evB));
    CHECK_CUDA(cudaEventSynchronize(evB));
    CHECK_CUDA(cudaEventElapsedTime(&tmp, evA, evB));
    htod_ms += tmp;

    // ---------- Kernel loop (we'll time per-iteration D→H of shifts) ----------
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

        // convergence check: D→H of shifts (timed)
        CHECK_CUDA(cudaEventRecord(evA));
        CHECK_CUDA(cudaMemcpy(h_shifts.data(), d_shifts, static_cast<size_t>(K) * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(evB));
        CHECK_CUDA(cudaEventSynchronize(evB));
        CHECK_CUDA(cudaEventElapsedTime(&tmp, evA, evB));
        dtoh_ms += tmp;

        double max_shift = 0.0;
        for (int i = 0; i < K; ++i) max_shift = std::max(max_shift, h_shifts[i]);

        CHECK_CUDA(cudaMemcpy(d_centers, d_new_centers, static_cast<size_t>(K) * dims * sizeof(double), cudaMemcpyDeviceToDevice));

        ++iter_to_converge;
        if (max_shift <= threshold) break;
    }

    // ---------- Final D→H downloads (timed) ----------
    CHECK_CUDA(cudaEventRecord(evA));
    CHECK_CUDA(cudaMemcpy(labels.data(), d_labels, static_cast<size_t>(_numpoints) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(evB));
    CHECK_CUDA(cudaEventSynchronize(evB));
    CHECK_CUDA(cudaEventElapsedTime(&tmp, evA, evB));
    dtoh_ms += tmp;

    CHECK_CUDA(cudaEventRecord(evA));
    CHECK_CUDA(cudaMemcpy(centers.data(), d_centers, static_cast<size_t>(K) * dims * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(evB));
    CHECK_CUDA(cudaEventSynchronize(evB));
    CHECK_CUDA(cudaEventElapsedTime(&tmp, evA, evB));
    dtoh_ms += tmp;

    // ---------- E2E stop ----------
    CHECK_CUDA(cudaEventRecord(evTotalStop));
    CHECK_CUDA(cudaEventSynchronize(evTotalStop));
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, evTotalStart, evTotalStop));

    // Derived times
    double compute_ms = std::max(0.0, static_cast<double>(total_ms) - static_cast<double>(htod_ms + dtoh_ms));
    double transfer_frac = (total_ms > 0.0f) ? (static_cast<double>(htod_ms + dtoh_ms) / static_cast<double>(total_ms)) : 0.0;

    // Per-iteration kernel-ish time (for your CSV you were printing)
    double time_per_iter_ms = (iter_to_converge > 0) ? (compute_ms / static_cast<double>(iter_to_converge)) : 0.0;

    // CSV: iterations, avg_ms_per_iter (compute only, like before)
    printf("%d,%.6f\n", iter_to_converge, time_per_iter_ms);

    // Extra breakdown (human-readable; keep or remove as you like)
    fprintf(stderr,
            "#E2E(ms)=%.3f  HtoD(ms)=%.3f  DtoH(ms)=%.3f  Compute(ms)=%.3f  Transfer_Frac=%.2f%%\n",
            total_ms, htod_ms, dtoh_ms, compute_ms, 100.0 * transfer_frac);

    if (!output_centroids) {
        // (optional) suppress large prints for big N
        // printf("clusters:"); for (int i = 0; i < _numpoints; ++i) printf(" %d", labels[i]);
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
    cudaEventDestroy(evTotalStart);
    cudaEventDestroy(evTotalStop);
    cudaEventDestroy(evA);
    cudaEventDestroy(evB);
    cudaFree(d_points);
    cudaFree(d_centers);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(d_sums);
    cudaFree(d_new_centers);
    cudaFree(d_shifts);

    return 0;
}