#pragma once
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <vector>
// #include <thrust/device_vector.h>

#define cusolverErrorCheck(err) __cusolverErrorCheck(err, __FILE__, __LINE__)
inline void __cusolverErrorCheck(cusolverStatus_t err, const char *file,
                                 const int line) {
  if (CUSOLVER_STATUS_SUCCESS != err) {
    fprintf(stderr,
            "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",
            __FILE__, __LINE__, err);
    throw std::runtime_error("Cusolver error");
  }
}

#define cublasErrorCheck(err) __cublasErrorCheck(err, __FILE__, __LINE__)
inline void __cublasErrorCheck(cublasStatus_t err, const char *file,
                               const int line) {
  if (CUBLAS_STATUS_SUCCESS != err) {
    fprintf(stderr,
            "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",
            __FILE__, __LINE__, err);
    throw std::runtime_error("Cublas error");
  }
}

#define cudaErrorCheck(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Cuda error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    throw std::runtime_error("Cuda error");
  }
}

// TODO: Move error checking to utility header
// TODO: Remove debug prints
// TODO: Naming case styles
// TODO: Add namespace
// TODO: Add scoring function
// TODO: change to templates or just use float?? How does this affect accuracy
// and performance?
class online_regression {
  cublasHandle_t cublas_handle;

  // Settings
  float ridge;
  int max_iters;
  double tolerance;

  double *W = nullptr;

  // Training data compression accumulated in partial_solve
  double *YXt_partial = nullptr;
  double *XXt_partial = nullptr;

  // Temporary matrices used in partial_solve
  double *YXt = nullptr;
  double *XXt = nullptr;

  // Save sizes to check so that they are not changed
  size_t num_x_features = 0;
  size_t num_y_features = 0;

  int max_threads_per_block = -1;

public:
  online_regression(double ridge = 0.1, int max_iters = 50,
                    double tolerance = 1e-20);
  ~online_regression();

  double *predict(double *X, double *Y, size_t X_m, size_t Y_m, size_t XY_n);
  std::vector<double> predict(std::vector<double> X, size_t X_features,
                              size_t Y_features);

  void partial_fit(double *X, double *Y, size_t X_m, size_t Y_m, size_t XY_n);
  void partial_fit(std::vector<double> X, size_t X_features,
                   std::vector<double> Y, size_t Y_features);

  void fit(double *X, double *Y, size_t X_m, size_t Y_m, size_t XY_n);
  void fit(std::vector<double> X, size_t X_features, std::vector<double> Y,
           size_t Y_features);

  void clear();

  double score(const std::vector<double> &X, size_t X_features,
               const std::vector<double> &Y, size_t Y_features);
  double score(double *X, double *Y, size_t X_m, size_t Y_m, size_t XY_n,
               bool padded = false);
};
