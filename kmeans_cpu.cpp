#include <iostream>
#include <fstream>
#include <chrono>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

int kmeans_cpu(
    const std::string& input_file,
    int k,
    int dims,
    int max_iter,
    double threshold,
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
    std::vector<std::vector<double>> points(_numpoints, std::vector<double>(dims));

    // line has form 'index dim1 dim2 ... dimN'
    for (int i = 0; i < _numpoints; ++i) {
        int index;
        infile >> index;
        for (int j = 0; j < dims; ++j) {
            infile >> points[i][j];
        }
    }
    infile.close();

    // allocate memory for cluster centers
    std::vector<std::vector<double>> centers(k, std::vector<double>(dims));

    // initialize cluster centers by randomly selecting k points
    for (int i = 0; i < k; ++i) {
        int index = kmeans_rand() % _numpoints;
        centers[i] = points[index];
    }

    auto start = std::chrono::steady_clock::now();

    int iter_to_converge = 0;

    // iterations

    // prepare array to hold point labels
    std::vector<int> labels(_numpoints, -1);

    while (iter_to_converge <= max_iter) {

        // assign points to nearest cluster center
        for (int i = 0; i < _numpoints; ++i) {

            // find nearest center for points[i]
            double min_dist = std::numeric_limits<double>::max();
            int best_center = -1;

            // for each center
            for (int j = 0; j < k; ++j) {
                double dist = 0.0;
                for (int d = 0; d < dims; ++d) {
                    double diff = points[i][d] - centers[j][d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_center = j;
                }
            }

            labels[i] = best_center;
        }

        // update cluster centers
        // for labels array, count points assigned to each cluster
        std::vector<size_t> counts(static_cast<size_t>(k), 0);
        std::vector<double> sums(static_cast<size_t>(k) * dims, 0.0);
        std::vector<std::vector<double>> new_centers(k, std::vector<double>(dims));

        for (int i = 0; i < _numpoints; ++i) {
            counts[labels[i]]++;
        }

        // get sum of points assigned to each cluster
        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {

                for (int j = 0; j < _numpoints; ++j) {
                    if (labels[j] == i) {
                        for (int d = 0; d < dims; ++d) {
                            sums[i * dims + d] += points[j][d];
                        }
                    }
                }
                for (int d = 0; d < dims; ++d) {
                    new_centers[i][d] = sums[i * dims + d] / counts[i];
                }
            } else {
                int index = kmeans_rand() % _numpoints;
                new_centers[i] = points[index];
            }
        }

        // get max center change
        double max_shift = 0.0;
        for (int i = 0; i < k; ++i) {
            double shift = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = new_centers[i][d] - centers[i][d];
                shift += diff * diff;
            }
            max_shift = std::max(max_shift, std::sqrt(shift));
        }

        // update centers
        centers = new_centers;

        // check for convergence
        if (max_shift <= threshold) {
            break;
        }

        // update iteration count
        iter_to_converge++;
    }

    auto end = std::chrono::steady_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto time_per_iter_ms = (iter_to_converge > 0) ? (double)total_time_ms / iter_to_converge : 0.0;

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
                printf("%lf ", centers[i][d]);
            }
            printf("\n");
        }
    }

    return 0;
}