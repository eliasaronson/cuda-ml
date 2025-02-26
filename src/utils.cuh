#pragma once
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream> // Only needed for debugging
#include <string>

#define cusolverErrorCheck(err) __cusolverErrorCheck(err, __FILE__, __LINE__)
inline void __cusolverErrorCheck(cusolverStatus_t err, const char *file,
                                 const int line) {
    if (CUSOLVER_STATUS_SUCCESS != err) {
        fprintf(
            stderr,
            "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",
            __FILE__, __LINE__, err);
        throw std::runtime_error("Cusolver error");
    }
}

#define cublasErrorCheck(err) __cublasErrorCheck(err, __FILE__, __LINE__)
inline void __cublasErrorCheck(cublasStatus_t err, const char *file,
                               const int line) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(
            stderr,
            "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",
            __FILE__, __LINE__, err);
        throw std::runtime_error("Cublas error");
    }
}

#define cudaErrorCheck(ans)                                                    \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "Cuda error: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        throw std::runtime_error("Cuda error");
    }
}
__device__ double4 operator+(const double4 &d1, const double4 &d2) {
    return {d1.x + d2.x, d1.y + d2.y, d1.z + d2.z, d1.w + d2.w};
}

__device__ double4 operator+(const double4 &d1, const double &d2) {
    return {d1.x + d2, d1.y + d2, d1.z + d2, d1.w + d2};
}

__device__ double4 operator-(const double4 &d1, const double4 &d2) {
    return {d1.x - d2.x, d1.y - d2.y, d1.z - d2.z, d1.w - d2.w};
}

__device__ double4 operator-(const double4 &d1, const double &d2) {
    return {d1.x - d2, d1.y - d2, d1.z - d2, d1.w - d2};
}

__device__ double4 operator*(const double4 &d1, const double4 &d2) {
    return {d1.x * d2.x, d1.y * d2.y, d1.z * d2.z, d1.w * d2.w};
}

namespace ml {
template <typename T> __global__ void fill(T *vec, T val, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        vec[i] = val;
    }
}

void printm(std::string label, double *A, size_t m, size_t n) {
    std::cout << label << " " << m << "x" << n << ":\n";
    double h_A[m * n];
    cudaErrorCheck(
        cudaMemcpy(h_A, A, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f", h_A[i + j * m]);
            if (j != n - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

inline size_t num_extra_to_pad(size_t count, size_t pad_to = 4) {
    return ((4 - (count % 4)) % 4);
}

} // namespace ml
