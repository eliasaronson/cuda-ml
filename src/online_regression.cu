#include "online_regression.h"
#include "utils.cuh"
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>

namespace {
// XXᵀ is square so the leading dimension and the minimum dimension is the same
__global__ void add_ridge(double *A, double ridge, int lead_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < lead_dim) {
        A[i * lead_dim + i] += ridge;
    }
}

void transpose(cublasHandle_t &cublas_handle, double *A, double *A_clone,
               double *C, size_t m, size_t n) {
    double alpha = 1, beta = 0;
    cudaErrorCheck(cudaMemcpy(A_clone, A, sizeof(double) * m * n,
                              cudaMemcpyDeviceToDevice));

    cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A_clone,
                n, &beta, A_clone, m, C, m);
}

namespace cg = cooperative_groups;

template <int tile_sz>
__device__ double reduce_sum_tile_shfl(cg::thread_block_tile<tile_sz> g,
                                       double val) {
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane-i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }

    // Thread 0 returns the sum
    return val;
}

/* ----- R2 numerator ----- */
// Handles a double vectors as double4 to do simd instructions. TODO: Check if
// these actually exists for doubles
__device__ double thread_sumed_sqrt_diff(const double4 *y_true,
                                         const double4 *y_pred, int n) {
    double sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4;
         i += blockDim.x * gridDim.x) {
        double4 diff = y_true[i] - y_pred[i];
        diff = diff * diff;
        sum += diff.x + diff.y + diff.z + diff.w;
    }
    return sum;
}

template <int tile_sz>
__global__ void r2_numerator(double *sum, const double *y_true,
                             const double *y_pred, int n) {
    // Allow fewer threads than elements
    double thread_sum =
        thread_sumed_sqrt_diff((double4 *)y_true, (double4 *)y_pred, n);

    auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
    double tile_sum = reduce_sum_tile_shfl<tile_sz>(tile, thread_sum);

    if (tile.thread_rank() == 0) {
        atomicAdd(sum, tile_sum);
    }
}

/* ----- R2 denominator ----- */
// Might be worth to transpose y_true to avoid the modulus operation and for
// better access pattern of y_avg.
__device__ double thread_sumed_sqrt_diff(const double *y_true,
                                         const double *y_avg, int n_features,
                                         int n) {
    double sum = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        int j = i % n_features;
        double diff = y_true[i] - y_avg[j];
        sum += diff * diff;
    }
    return sum;
}

template <int tile_sz>
__global__ void r2_denominator(double *sum, const double *y_true,
                               const double *y_avg, int n_features, int n) {
    // Allow fewer threads than elements
    double thread_sum = thread_sumed_sqrt_diff(y_true, y_avg, n_features, n);

    auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
    double tile_sum = reduce_sum_tile_shfl<tile_sz>(tile, thread_sum);

    if (tile.thread_rank() == 0) {
        atomicAdd(sum, tile_sum);
    }
}
} // namespace

namespace ml {
online_regression::online_regression(double ridge, int max_iters,
                                     double tolerance)
    : ridge(ridge), max_iters(max_iters), tolerance(tolerance) {
    cublas_handle = new cublasHandle_t;
    cublasErrorCheck(cublasCreate(cublas_handle));
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    max_threads_per_block = props.maxThreadsPerBlock;
}

online_regression::~online_regression() {
    clear();
    cublasDestroy_v2(*cublas_handle);
    delete cublas_handle;
}

std::vector<double> online_regression::predict(const std::vector<double> &X,
                                               size_t X_features,
                                               size_t Y_features) {
    size_t n_samples = X.size() / X_features;
    std::vector<double> Y(Y_features * n_samples);

    double *d_X, *d_Y;

    cudaErrorCheck(cudaMalloc(&d_X, sizeof(double) * X.size()));
    cudaErrorCheck(cudaMalloc(&d_Y, sizeof(double) * Y.size()));
    cudaErrorCheck(cudaMemcpy(d_X, X.data(), sizeof(double) * X.size(),
                              cudaMemcpyHostToDevice));

    printf("X_feat; %u, Y_feat: %u, n_samples: %u\n", X_features, Y_features,
           n_samples);
    predict(d_X, d_Y, X_features, Y_features, n_samples);

    cudaErrorCheck(cudaMemcpy(Y.data(), d_Y, sizeof(double) * Y.size(),
                              cudaMemcpyDeviceToHost));

    printm("Y pred", d_Y, Y_features, n_samples);
    cudaErrorCheck(cudaFree(d_X));
    cudaErrorCheck(cudaFree(d_Y));

    return Y;
}

// Y = W * X. Y is used as an output and needs to allocated from the outside
double *online_regression::predict(double *X, double *Y, size_t X_m, size_t Y_m,
                                   size_t XY_n) {
    double alpha = 1, beta = 0;

    printm("X test", X, X_m, XY_n);

    // Y = W * X
    cublasErrorCheck(cublasDgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, Y_m,
                                 XY_n, X_m, &alpha, W, Y_m, X, X_m, &beta, Y,
                                 Y_m));

    return Y;
}

void online_regression::partial_fit(const std::vector<double> &X,
                                    size_t X_features,
                                    const std::vector<double> &Y,
                                    size_t Y_features) {
    if (X.size() / X_features != Y.size() / Y_features) {
        printf("Error! X and Y have different number of samples.\n");
        return;
    }

    double *d_X, *d_Y;
    cudaMalloc(&d_X, sizeof(double) * X.size());
    cudaMalloc(&d_Y, sizeof(double) * Y.size());

    cudaMemcpy(d_X, X.data(),
               X.size() * sizeof(std::decay_t<decltype(X)>::value_type),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y.data(),
               Y.size() * sizeof(std::decay_t<decltype(Y)>::value_type),
               cudaMemcpyHostToDevice);

    partial_fit(d_X, d_Y, X_features, Y_features, X.size() / X_features);

    cudaErrorCheck(cudaFree(d_X));
    cudaErrorCheck(cudaFree(d_Y));
}

/* Our goal is to solve W in Y = WX, where X, Y and W are matrices.
 * Y N y features by N samples, W is N y features by N x features and X is
 * N x features by N samples.
 * If we get continuous online data or we * have to much data to load into
 * memory we can compress this data, by the method from this paper:
 * https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf
 *
 * If we multiply both sides of the equation with the transpose of X, the
 * sample dimensions disappears. We now have the following equation instead:
 * YXᵀ = WXXᵀ
 *
 * Each time we run the partial fit we can you add the new XXᵀ and YXᵀ
 * matrices, to the old one. This does run into precision problems for large
 * data sets however. The paper suggests using hierarchical multistage
 * summation or Kahan summation for better precision, which could be added in
 * the future.
 */
void online_regression::partial_fit(double *X, double *Y, size_t X_m,
                                    size_t Y_m, size_t XY_n) {
    double alpha = 1, beta = 0;

    // printm("X", X, X_m, XY_n);
    // printm("Y", Y, Y_m, XY_n);

    // Allocate temporary and accumulation matrices. Don't reallocate these each
    // time for performance reasons.
    if (num_x_features == 0) {
        cudaMalloc(&YXt, sizeof(double) * Y_m * X_m);
        cudaMalloc(&XXt, sizeof(double) * X_m * X_m);

        cudaMalloc(&YXt_partial, sizeof(double) * Y_m * X_m);
        cudaMemset(YXt_partial, 0, sizeof(double) * Y_m * X_m);

        cudaMalloc(&XXt_partial, sizeof(double) * X_m * X_m);
        cudaMemset(XXt_partial, 0, sizeof(double) * X_m * X_m);

        num_x_features = X_m;
        num_y_features = Y_m;
    }

    if (X_m != num_x_features || Y_m != num_y_features) {
        printf("Warning! The number of features does not match previous "
               "partial_fit dimensions.\nPrevious X features: %u, Y features: "
               "%u\nNew X features: %u, Y features: %u.\nPlease run clear() if "
               "you "
               "want to train a new model.\n",
               num_x_features, num_y_features, X_m, Y_m);
        return;
    }

    // Y * Xᵀ
    cublasErrorCheck(cublasDgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, Y_m,
                                 X_m, XY_n, &alpha, Y, Y_m, X, X_m, &beta, YXt,
                                 Y_m));
    printm("YXt", YXt, Y_m, X_m);

    // X * Xᵀ
    cublasErrorCheck(cublasDgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, X_m,
                                 X_m, XY_n, &alpha, X, X_m, X, X_m, &beta, XXt,
                                 X_m));

    printm("XXt", XXt, X_m, X_m);

    // Accumulate YXᵀ and XXᵀ for further partial fits or to fully solve
    cublasErrorCheck(cublasDgeam(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, Y_m,
                                 X_m, &alpha, YXt_partial, Y_m, &alpha, YXt,
                                 Y_m, YXt_partial, Y_m));

    printm("YXt_partial", YXt_partial, Y_m, X_m);

    cublasErrorCheck(cublasDgeam(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, X_m,
                                 X_m, &alpha, XXt_partial, X_m, &alpha, XXt,
                                 X_m, XXt_partial, X_m));
    printm("XXt_partial", XXt_partial, X_m, X_m);
}

void online_regression::fit(const std::vector<double> &X, size_t X_features,
                            const std::vector<double> &Y, size_t Y_features) {
    partial_fit(X, X_features, Y, Y_features);
    fit(nullptr, nullptr, X_features, Y_features, X.size() / X_features);
}

/* Cusolver's gels solves for X in AX = B.
 * We want to solve for W in YXᵀ = WXXᵀ.
 *
 * To be able to perform the partial fits we need to have X last, but that also
 * means we need to transpose both sides to fit the format of the solver.
 * To do this  we just need to transpose both sides.
 * (YXᵀ)ᵀ = (WXXᵀ)ᵀ
 * (YXᵀ)ᵀ = (XXᵀ)ᵀWᵀ
 * (YXᵀ)ᵀ = XXᵀWᵀ
 *
 * This gives:
 * A = XXᵀ
 * B = (YXᵀ)ᵀ
 * X = W
 * */
void online_regression::fit(double *X, double *Y, size_t X_m, size_t Y_m,
                            size_t XY_n) {
    if (Y != nullptr && Y != nullptr) {
        partial_fit(X, Y, X_m, Y_m, XY_n);
    }

    printm("A", XXt_partial, X_m, X_m);

    if (ridge != 0) {
        printf("ridge: %f\n", ridge);
        int block_size;
        int min_grid_size;
        int grid_size;

        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           add_ridge, 0, 0);

        grid_size = (X_m + block_size - 1) / block_size;
        printm("Pred ridge A", XXt_partial, X_m, X_m);
        add_ridge<<<grid_size, block_size>>>(XXt_partial, ridge, X_m);
        printm("Post ridge A", XXt_partial, X_m, X_m);
    }

    // B is now (YXᵀ)ᵀ. tmp is used for temporary storage in transposes.
    double *B, *tmp;
    cudaMalloc(&B, sizeof(double) * X_m * Y_m);
    cudaMemset(B, 0, sizeof(double) * X_m * Y_m);

    cudaMalloc(&tmp, sizeof(double) * X_m * Y_m);

    transpose(*cublas_handle, YXt_partial, tmp, B, Y_m, X_m);

    printm("B", B, X_m, Y_m);

    /* Allocate data for settings and work */
    int niter = 0;
    size_t work_size = 0;

    double *Wt = nullptr;
    double *work = nullptr;
    int *info = nullptr;

    cusolverDnHandle_t cusolver_handle = nullptr;
    cusolverErrorCheck(cusolverDnCreate(&cusolver_handle));

    cudaErrorCheck(cudaMalloc(&Wt, sizeof(double) * X_m * Y_m));
    cudaErrorCheck(cudaMalloc(&info, sizeof(int)));
    cudaErrorCheck(cudaMalloc(&W, sizeof(double) * Y_m * X_m));

    cusolverDnIRSParams_t gels_irs_params;
    cusolverDnIRSParamsCreate(&gels_irs_params);

    cusolverDnIRSInfos_t gels_irs_infos;
    cusolverDnIRSInfosCreate(&gels_irs_infos);
    cusolverErrorCheck(cusolverDnIRSInfosRequestResidual(gels_irs_infos));

    /* Set all settings */
    // Solver precisions
    cusolverErrorCheck(cusolverDnIRSParamsSetSolverPrecisions(
        gels_irs_params, CUSOLVER_R_64F, CUSOLVER_R_64F));

    // Refinement solver.
    if (Y_m == 1) {
        // Generalized Minimal Residual is more accurate, but can only solve for
        // one right hand side. TODO: Maybe add option to set
        // CUSOLVER_IRS_REFINE_CLASSICAL_GMRES or
        // CUSOLVER_IRS_REFINE_GMRES_GMRES.
        cusolverErrorCheck(cusolverDnIRSParamsSetRefinementSolver(
            gels_irs_params, CUSOLVER_IRS_REFINE_GMRES));
    } else {
        cusolverErrorCheck(cusolverDnIRSParamsSetRefinementSolver(
            gels_irs_params, CUSOLVER_IRS_REFINE_CLASSICAL));
    }
    cusolverErrorCheck(
        cusolverDnIRSParamsSetMaxIters(gels_irs_params, max_iters));
    cusolverErrorCheck(cusolverDnIRSParamsSetTol(gels_irs_params, tolerance));

    // Find the work buffer size from the parameters and allocate it
    cusolverErrorCheck(cusolverDnIRSXgels_bufferSize(
        cusolver_handle, gels_irs_params, X_m, X_m, Y_m, &work_size));
    cudaErrorCheck(cudaMalloc(&work, work_size));

    /* Run the solver */
    cusolverErrorCheck(cusolverDnIRSXgels(
        cusolver_handle, gels_irs_params, gels_irs_infos, X_m, X_m, Y_m,
        XXt_partial, X_m, B, X_m, Wt, X_m, work, work_size, &niter, info));

    printm("Wt", Wt, X_m, Y_m);

    printf("solver iterations: %i\n", niter);

    // No need to transpose if W is a vector
    if (X_m == 1 || Y_m == 1) {
        cudaErrorCheck(cudaMemcpy(W, Wt, sizeof(double) * Y_m * X_m,
                                  cudaMemcpyDeviceToDevice));
    } else {
        transpose(*cublas_handle, Wt, tmp, W, X_m, Y_m);
    }
    printm("W", W, Y_m, X_m);

#define CHECK_AND_FREE(PTR, FREE)                                              \
    if (PTR != nullptr) {                                                      \
        FREE(PTR);                                                             \
    }                                                                          \
    PTR = nullptr;

    // Clean up
    CHECK_AND_FREE(tmp, cudaFree)
    CHECK_AND_FREE(Wt, cudaFree)
    CHECK_AND_FREE(info, cudaFree)
    CHECK_AND_FREE(work, cudaFree)
    CHECK_AND_FREE(B, cudaFree)
    CHECK_AND_FREE(cusolver_handle, cusolverDnDestroy)

    cusolverErrorCheck(cusolverDnIRSParamsDestroy(gels_irs_params));
    cusolverErrorCheck(cusolverDnIRSInfosDestroy(gels_irs_infos));
}

void online_regression::clear() {
    CHECK_AND_FREE(YXt, cudaFree)
    CHECK_AND_FREE(XXt, cudaFree)

    CHECK_AND_FREE(YXt_partial, cudaFree)
    CHECK_AND_FREE(XXt_partial, cudaFree)

    CHECK_AND_FREE(W, cudaFree)

    num_x_features = num_y_features = 0;
}

#undef CHECK_AND_FREE

double online_regression::score(const std::vector<double> &X, size_t X_features,
                                const std::vector<double> &Y,
                                size_t Y_features) {
    size_t n_samples = X.size() / X_features;
    size_t pad = num_extra_to_pad(Y.size());

    double *d_X, *d_Y;

    cudaErrorCheck(cudaMalloc(&d_X, sizeof(double) * X.size()));
    cudaErrorCheck(cudaMalloc(&d_Y, sizeof(double) * (Y.size() + pad)));

    cudaErrorCheck(cudaMemcpy(d_X, X.data(), sizeof(double) * X.size(),
                              cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_Y, Y.data(), sizeof(double) * Y.size(),
                              cudaMemcpyHostToDevice));

    if (pad > 0) {
        cudaErrorCheck(cudaMemset(d_Y + Y.size(), 0, pad * sizeof(double)));
    }

    printf("X_feat; %u, Y_feat: %u, n_samples: %u\n", X_features, Y_features,
           n_samples);
    double res = score(d_X, d_Y, X_features, Y_features, n_samples, true);

    cudaErrorCheck(cudaFree(d_X));
    cudaErrorCheck(cudaFree(d_Y));

    return res;
}

// R2 score. Y should be padded to be divisible by 4 or it will need to be
// reallocated (this is probably not the best solution).
double online_regression::score(double *X, double *Y, size_t X_m, size_t Y_m,
                                size_t XY_n, bool padded) {
    if (XY_n < 2) {
        printf("Warning! R2 requires at least two samples.\n");
    }

    double *Y_padded, *Y_pred, *numerator, *denominator, *Y_average, *Y_div;

    size_t pad = num_extra_to_pad(Y_m * XY_n);
    size_t y_sz = Y_m * XY_n + pad;
    cudaErrorCheck(cudaMalloc(&Y_pred, sizeof(double) * y_sz));
    cudaErrorCheck(cudaMalloc(&numerator, sizeof(double)));
    cudaErrorCheck(cudaMalloc(&denominator, sizeof(double)));
    cudaErrorCheck(cudaMalloc(&Y_div, sizeof(double) * XY_n));
    cudaErrorCheck(cudaMalloc(&Y_average, sizeof(double) * Y_m));

    // Probably not worth it, unless already padded from the outside
    if (pad > 0) {
        cudaErrorCheck(
            cudaMemset(Y_pred + Y_m * XY_n, 0, pad * sizeof(double)));
    }
    if (!padded && pad > 0) {
        cudaErrorCheck(cudaMalloc(&Y_padded, sizeof(double) * y_sz));
        cudaErrorCheck(cudaMemcpy(Y_padded, Y, sizeof(double) * Y_m * XY_n,
                                  cudaMemcpyDeviceToDevice));
        cudaErrorCheck(
            cudaMemset(Y_padded + Y_m * XY_n, 0, pad * sizeof(double)));
    } else {
        Y_padded = Y;
    }

    // Get prediction
    predict(X, Y_pred, X_m, Y_m, XY_n);
    printm("Y", Y, Y_m, XY_n);
    printm("Y_pred", Y_pred, Y_m, XY_n);

    // We should not be register limited, so should not be a problem to max out
    // the blocks when we have large matrices. Might be more efficient to use
    // more SMs with small matrices however.
    dim3 block_dim(max_threads_per_block);
    dim3 grid_dim;
    grid_dim.x = (XY_n + block_dim.x - 1) / block_dim.x;
    fill<double><<<grid_dim, block_dim>>>(Y_div, 1. / XY_n, XY_n);

    grid_dim.x = (y_sz + block_dim.x - 1) / block_dim.x;
    r2_numerator<32>
        <<<grid_dim, block_dim>>>(numerator, Y_padded, Y_pred, y_sz);
    printm("numerator", numerator, 1, 1);

    double alpha = 1., beta = 0.;
    cublasErrorCheck(cublasDgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, Y_m,
                                 1, XY_n, &alpha, Y, Y_m, Y_div, XY_n, &beta,
                                 Y_average, Y_m));
    printm("avg", Y_average, Y_m, 1);

    r2_denominator<32>
        <<<grid_dim, block_dim>>>(denominator, Y, Y_average, Y_m, Y_m * XY_n);
    printm("denominator", denominator, 1, 1);

    double h_denominator, h_numerator;
    cudaMemcpy(&h_numerator, numerator, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_denominator, denominator, sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaErrorCheck(cudaFree(Y_pred));
    cudaErrorCheck(cudaFree(numerator));
    cudaErrorCheck(cudaFree(denominator));
    cudaErrorCheck(cudaFree(Y_div));
    cudaErrorCheck(cudaFree(Y_average));
    if (!padded && pad != 0) {
        cudaFree(Y_padded);
    }

    return 1. - h_numerator / h_denominator;
}
} // namespace ml
