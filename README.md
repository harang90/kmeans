For all your implementations, your program should at least accept the following arguments

-k num_cluster: an integer specifying the number of clusters
-d dims: an integer specifying the dimension of the points
-i inputfilename: a string specifying the input filename
-m max_num_iter: an integer specifying the maximum number of iterations
-t threshold: a double specifying the threshold for convergence test.
-c: a flag to control the output of your program. If -c is specified, your program should output the centroids of all clusters. If -c is not specified, your program should output the labels of all points. See details below.
-s seed: an integer specifying the seed for rand(). This is used by the autograder to simplify the correctness checking process. See details below.
Your program can also accept more arguments to control the behavior of your program for different implementations. These extra arguments can be specified in the submit file. Refer to the instruction in submit file for more details.

-k, -d, and -i should depend on the input files. The max number of iterations -m is used to prevent your program from an infinite loop if something goes wrong. In general, your implementation is expected to converge within 150 iterations. Therefore, an value of 150 to -m should be good enough. Depending on your methods for convergence test, you might want to use different thresholds -t for different implementations. As a reference, the threshold for comparing centers without any non-determinism issues can be as small as 10^-10. However, for comparing centers with non-determinism issues, you might want to use a threshold of 10^-6. The autograder will specify -t 10^-5 for all implementations.