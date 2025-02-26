#pragma once
#include <vector>
// #include <thrust/device_vector.h>

// TODO: Remove debug prints
// TODO: Naming case styles
// TODO: change to templates or just use float?? How does this affect accuracy
// and performance?

typedef struct cublasContext *cublasHandle_t;

namespace ml {
class online_regression {
  cublasHandle_t *cublas_handle;

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
  std::vector<double> predict(const std::vector<double> &X, size_t X_features,
                              size_t Y_features);

  void partial_fit(double *X, double *Y, size_t X_m, size_t Y_m, size_t XY_n);
  void partial_fit(const std::vector<double> &X, size_t X_features,
                   const std::vector<double> &Y, size_t Y_features);

  void fit(double *X, double *Y, size_t X_m, size_t Y_m, size_t XY_n);
  void fit(const std::vector<double> &X, size_t X_features,
           const std::vector<double> &Y, size_t Y_features);

  void clear();

  double score(const std::vector<double> &X, size_t X_features,
               const std::vector<double> &Y, size_t Y_features);
  double score(double *X, double *Y, size_t X_m, size_t Y_m, size_t XY_n,
               bool padded = false);
};
} // namespace ml
