#include "online_regression.h"
#include <ctime>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <iostream>
#include <string>

#define BACKWARD_HAS_BFD 1
#include <backward.hpp>
backward::SignalHandling sh;

__global__ void rand_vec(double *x, curandState *rand_state, size_t count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        x[i] = curand_uniform(rand_state + i);
    }
}

__global__ void init_curand_state(curandState *rand_state, unsigned long seed,
                                  size_t count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        curand_init(seed + i, i, 0, rand_state + i);
    }
}

__global__ void generate_smooth_W(double *W, size_t y_features,
                                  size_t x_features) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < y_features * x_features; i += blockDim.x * gridDim.x) {

        int row = i % y_features;
        int col = i / y_features;

        // Create smooth patterns
        double row_norm = (double)row / y_features;
        double col_norm = (double)col / x_features;

        // Combine several smooth functions
        W[i] = 0.5 * sin(2.0 * M_PI * row_norm) +
               0.3 * cos(4.0 * M_PI * col_norm) +
               0.2 * sin(3.0 * M_PI * (row_norm + col_norm));
    }
}

int main() {
    // Settings
    float ridge = 0.1f;
    size_t num_partial_solves = 5;

    size_t num_samples = 10;
    size_t num_x_features = 3;
    size_t num_y_features = 3;

    printf("num x features; %u, num y features: %u, num samples: %u\n\n",
           num_x_features, num_y_features, num_samples);

    // States
    size_t pad = ((4 - (num_samples * num_y_features % 4)) % 4);
    size_t max_size = std::max(
        std::max(num_x_features * num_samples, num_y_features * num_samples),
        num_x_features * num_y_features);

    double *X, *Y, *W;
    curandState *rand_states;

    cublasHandle_t cublas_handle;

    // Allocations
    ml::online_regression reg(ridge);

    cublasCreate(&cublas_handle);

    cudaMalloc(&rand_states, sizeof(curandState) * max_size);
    cudaMalloc(&W, sizeof(double) * num_y_features * num_x_features);
    cudaMalloc(&X, sizeof(double) * num_samples * num_x_features);
    cudaMalloc(&Y, sizeof(double) * (num_samples * num_y_features + pad));

    // Initialize states
    cudaMemset(Y, 0, sizeof(double) * (num_samples * num_y_features + pad));

    dim3 block_dim(512);
    dim3 grid_dim((max_size + block_dim.x - 1) / block_dim.x);
    init_curand_state<<<grid_dim, block_dim>>>(rand_states, time(nullptr),
                                               max_size);

    grid_dim = dim3(
        ((num_y_features * num_x_features + block_dim.x - 1) / block_dim.x));

    // Choose one of the generation methods
    generate_smooth_W<<<grid_dim, block_dim>>>(W, num_y_features,
                                               num_x_features);

    double alpha = 1, beta = 0;

    grid_dim =
        dim3(((num_x_features * num_samples + block_dim.x - 1) / block_dim.x));

    // Run partial solves
    for (size_t i = 0; i < num_partial_solves; ++i) {
        printf("Partial solve: %u/%u\n", i, num_partial_solves);
        rand_vec<<<grid_dim, block_dim>>>(X, rand_states,
                                          num_x_features * num_samples);
        // Y = W * X
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_y_features,
                    num_samples, num_x_features, &alpha, W, num_y_features, X,
                    num_x_features, &beta, Y, num_y_features);

        reg.partial_fit(X, Y, num_x_features, num_y_features, num_samples);
    }

    printf("Full solve\n");
    reg.fit(nullptr, nullptr, num_x_features, num_y_features, num_samples);

    printf("Predict and score\n");
    rand_vec<<<grid_dim, block_dim>>>(X, rand_states,
                                      num_x_features * num_samples);

    // Y = W * X
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_y_features,
                num_samples, num_x_features, &alpha, W, num_y_features, X,
                num_x_features, &beta, Y, num_y_features);

    double score =
        reg.score(X, Y, num_x_features, num_y_features, num_samples, true);

    printf("score: %.10e\n", score);

    cudaFree(X);
    cudaFree(Y);
    cudaFree(W);
}
