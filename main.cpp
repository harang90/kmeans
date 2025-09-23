#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>

int kmeans_cpu(
    const std::string& input_file,
    int k,
    int dims,
    int max_iter,
    double threshold,
    bool output_centroids,
    unsigned int seed
);

int kmeans_cuda(
    const std::string& input_file,
    int k,
    int dims,
    int max_iter,
    double threshold,
    bool output_centroids,
    unsigned int seed
);

int main(int argc, char** argv) {

    std::string input_file;
    int k = -1;
    int dims = -1;
    int max_iter = 300;
    double threshold = 1e-6;
    bool output_centroids = false;
    unsigned int seed = 1;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-k") && i + 1 < argc)      k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-d") && i + 1 < argc) dims = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) input_file = argv[++i];
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) max_iter = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) threshold = std::stod(argv[++i]);
        else if (!std::strcmp(argv[i], "-c"))                 output_centroids = true;
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) seed = static_cast<unsigned long>(std::strtoul(argv[++i], nullptr, 10));
        else {
            std::cerr << "Unknown or incomplete option: " << argv[i] << "\n";
            return 1;
        }
    }

    if (k <= 0 || dims <= 0 || input_file.empty()) {
        std::cerr << "Usage: bin/kmeans_cpu -k num_cluster -d dims -i inputfilename "
                     "[-m max_num_iter] [-t threshold] [-c] [-s seed]\n";
        return 1;
    }
    if (max_iter <= 0) {
        std::cerr << "Error: -m must be > 0\n";
        return 1;
    }
    if (threshold < 0.0) {
        std::cerr << "Error: -t must be >= 0\n";
        return 1;
    }

    #ifdef USE_CUDA
    return kmeans_cuda(
        input_file,
        k,
        dims,
        max_iter,
        threshold,
        output_centroids,
        seed
    );
    #else
    return kmeans_cpu(
        input_file,
        k,
        dims,
        max_iter,
        threshold,
        output_centroids,
        seed
    );
    #endif
}