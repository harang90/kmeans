#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

// Random number generator functions
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

// Functor to assign points to nearest cluster
struct assign_cluster_functor {
    const float* points;
    const float* centers;
    int K;
    int dims;
    int numpoints;
    
    assign_cluster_functor(const float* _points, const float* _centers, int _K, int _dims, int _numpoints)
        : points(_points), centers(_centers), K(_K), dims(_dims), numpoints(_numpoints) {}
    
    __host__ __device__
    int operator()(int point_idx) const {
        float min_dist = HUGE_VALF;
        int best_center = -1;
        
        for (int k = 0; k < K; ++k) {
            float dist = 0.0f;
            for (int d = 0; d < dims; ++d) {
                float diff = points[point_idx * dims + d] - centers[k * dims + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_center = k;
            }
        }
        return best_center;
    }
};

// Functor to compute shift per cluster
struct compute_shift_functor {
    const float* old_centers;
    const float* new_centers;
    int dims;
    
    compute_shift_functor(const float* _old, const float* _new, int _dims)
        : old_centers(_old), new_centers(_new), dims(_dims) {}
    
    __host__ __device__
    float operator()(int cluster_idx) const {
        float shift = 0.0f;
        for (int d = 0; d < dims; ++d) {
            float diff = new_centers[cluster_idx * dims + d] - old_centers[cluster_idx * dims + d];
            shift += diff * diff;
        }
        return sqrtf(shift);
    }
};

int kmeans_thrust(
    const std::string& input_file,
    int K,
    int dims,
    int max_iter,
    float threshold,
    bool output_centroids,
    unsigned int seed
) {

    kmeans_srand(seed);

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
    std::vector<float> centers(K * dims);

    // initialize cluster centers by randomly selecting k points
    for (int i = 0; i < K; ++i) {
        int index = kmeans_rand() % _numpoints;
        for (int d = 0; d < dims; ++d) {
            centers[i * dims + d] = points[index * dims + d];
        }
    }

    // prepare array to hold point labels
    std::vector<int> labels(_numpoints, -1);

    thrust::device_vector<float> d_points = points;
    thrust::device_vector<float> d_centers = centers;
    thrust::device_vector<int> d_labels = labels;
    thrust::device_vector<int> d_counts(K);
    thrust::device_vector<float> d_sums(K * dims);
    thrust::device_vector<float> d_new_centers(K * dims);
    thrust::device_vector<float> d_shifts(K);

    // Create temporary vectors for centroid computation
    thrust::device_vector<int> d_point_indices(_numpoints);
    thrust::device_vector<int> d_sorted_labels(_numpoints);
    thrust::device_vector<int> d_sorted_indices(_numpoints);
    thrust::device_vector<float> d_sorted_points(_numpoints * dims);

    // iterations
    int iter_to_converge = 0;

    auto start = std::chrono::steady_clock::now();
    while (iter_to_converge < max_iter) {
        
        // Step 1: Assign points to nearest clusters
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(_numpoints),
            d_labels.begin(),
            assign_cluster_functor(
                thrust::raw_pointer_cast(d_points.data()),
                thrust::raw_pointer_cast(d_centers.data()),
                K, dims, _numpoints
            )
        );

        // Step 2: Sort points by cluster assignment for efficient reduction
        thrust::sequence(d_point_indices.begin(), d_point_indices.end());
        thrust::copy(d_labels.begin(), d_labels.end(), d_sorted_labels.begin());
        thrust::copy(d_point_indices.begin(), d_point_indices.end(), d_sorted_indices.begin());
        
        thrust::sort_by_key(d_sorted_labels.begin(), d_sorted_labels.end(), d_sorted_indices.begin());

        // Rearrange points according to sorted indices
        for (int d = 0; d < dims; ++d) {
            thrust::gather(
                d_sorted_indices.begin(), d_sorted_indices.end(),
                d_points.begin() + d * _numpoints,
                d_sorted_points.begin() + d * _numpoints
            );
        }

        // Step 3: Count points per cluster and compute sums
        thrust::fill(d_counts.begin(), d_counts.end(), 0);
        thrust::fill(d_sums.begin(), d_sums.end(), 0.0f);

        // Use reduce_by_key to count points per cluster
        thrust::device_vector<int> unique_labels(K);
        thrust::device_vector<int> ones(_numpoints, 1);
        thrust::device_vector<int> cluster_counts(K);
        
        auto end_pair = thrust::reduce_by_key(
            d_sorted_labels.begin(), d_sorted_labels.end(),
            ones.begin(),
            unique_labels.begin(),
            cluster_counts.begin()
        );
        
        // Copy counts to the right positions
        for (int i = 0; i < end_pair.first - unique_labels.begin(); ++i) {
            d_counts[unique_labels[i]] = cluster_counts[i];
        }

        // Compute coordinate sums for each cluster and dimension
        for (int d = 0; d < dims; ++d) {
            thrust::device_vector<float> coord_sums(K, 0.0f);
            thrust::device_vector<int> temp_labels(K);
            
            auto sum_end = thrust::reduce_by_key(
                d_sorted_labels.begin(), d_sorted_labels.end(),
                d_sorted_points.begin() + d * _numpoints,
                temp_labels.begin(),
                coord_sums.begin()
            );
            
            // Copy sums to the right positions
            for (int i = 0; i < sum_end.first - temp_labels.begin(); ++i) {
                d_sums[temp_labels[i] * dims + d] = coord_sums[i];
            }
        }

        // Step 4: Compute new cluster centers
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(K * dims),
            d_new_centers.begin(),
            [K, dims, counts_ptr = thrust::raw_pointer_cast(d_counts.data()),
             sums_ptr = thrust::raw_pointer_cast(d_sums.data()),
             old_centers_ptr = thrust::raw_pointer_cast(d_centers.data())] __host__ __device__ (int idx) -> float {
                int cluster = idx / dims;
                if (counts_ptr[cluster] > 0) {
                    return sums_ptr[idx] / counts_ptr[cluster];
                } else {
                    return old_centers_ptr[idx];  // Keep old center if no points assigned
                }
            }
        );

        // Step 5: Compute cluster shifts
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(K),
            d_shifts.begin(),
            compute_shift_functor(
                thrust::raw_pointer_cast(d_centers.data()),
                thrust::raw_pointer_cast(d_new_centers.data()),
                dims
            )
        );

        // Find maximum shift
        float max_shift = thrust::reduce(d_shifts.begin(), d_shifts.end(), 0.0f, thrust::maximum<float>());

        // Update centers
        thrust::copy(d_new_centers.begin(), d_new_centers.end(), d_centers.begin());

        // Increment iteration counter
        iter_to_converge++;

        // Check for convergence
        if (max_shift <= threshold) {
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto time_per_iter_ms = (iter_to_converge > 0) ? (double)total_time_ms / iter_to_converge : 0.0;

    // Copy results back to host
    thrust::copy(d_labels.begin(), d_labels.end(), labels.begin());
    thrust::copy(d_centers.begin(), d_centers.end(), centers.begin());

    printf("%d,%lf\n", iter_to_converge, time_per_iter_ms);

    // output point labels
    if (!output_centroids) {
        printf("clusters:");
        for (int i = 0; i < _numpoints; ++i) {
            printf(" %d", labels[i]);
        }
    }

    // output cluster centers if required
    if (output_centroids) {
        for (int i = 0; i < K; ++i) {
            printf("%d ", i);
            for (int d = 0; d < dims; ++d) {
                printf("%f ", centers[i * dims + d]);
            }
            printf("\n");
        }
    }

    return 0;
}
