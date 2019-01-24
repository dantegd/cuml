#pragma once
#include "utils.h"
#include <stdio.h>
#include "sigma.h"
#include "cuda_utils.h"
#include "kf_variables.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"

namespace kf {
namespace unscented {

using MLCommon::alignTo;
using namespace kf::sigmagen;
using namespace MLCommon::LinAlg;

// initialize this structure with all relevant pointers
// during first call, pass workspace as a nullptr to evaluate the workspace
// size needed in B, then during the second call, pass the rightfully
// allocated workspace buffer as input
template <typename T>
void set(Variables<T>& var, int _dim_x, int _dim_z,
         const T _alpha_s, const T _beta_s, const T _kappa_s,
         Inverse _inverse_method, T *_x_est, T *_x_up, T *_Phi,
         T *_P_xx, T *_Q, T *_R, T *_H,
         void* workspace, size_t& workspaceSize, cusolverDnHandle_t handle_sol) {
    var.dim_x = _dim_x;
    var.dim_z = _dim_z;
    var.nPoints = _dim_x * 2 + 1;
    var.alpha_s = _alpha_s;
    var.beta_s = _beta_s;
    var.kappa_s = _kappa_s;
    // false below for signaling sigmagen 'not sqrtunkf'
    var.sg = new VanDerMerwe<T> (var.dim_x, var.alpha_s, var.beta_s, var.kappa_s, false);
    size_t ox = var.sg->init();
    CUSOLVER_CHECK(cusolverDnpotrf_bufferSize(handle_sol,
                                              CUBLAS_FILL_MODE_UPPER,
                                              var.dim_x, var.P_zz, var.dim_z,
                                              &var.Lwork));
    workspaceSize = 0;
    const size_t granularity = 256;
    var.K = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * var.dim_x * var.dim_z, granularity);
    var.P_xz = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * var.dim_x * var.dim_z, granularity);
    var.P_zz = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * var.dim_z * var.dim_z, granularity);
    var.X_h = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * var.nPoints * var.dim_z, granularity);
    var.X = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * var.nPoints * var.dim_x, granularity);
    ///@todo: no need to have even number of columns here
    var.X_er = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * (var.nPoints * var.dim_x +\
                      (var.nPoints*var.dim_x)%2), granularity);
    var.X_h_er = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * (var.nPoints * var.dim_z +\
                      (var.nPoints*var.dim_z)%2), granularity);
    var.eig_z = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * var.dim_z, granularity);
    var.workspace_cholesky = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * var.Lwork, granularity);
    var.info = (int *)workspaceSize;
    workspaceSize += alignTo(sizeof(int), granularity);
    var.placeHolder0 = (T *)workspaceSize;
    workspaceSize += alignTo(sizeof(T) * var.dim_z * var.dim_z, granularity);
    var.workspace_sg = (T *)workspaceSize;
    workspaceSize += alignTo(ox, granularity);

    if(workspace){
        ASSERT(!var.initialized, "Variables::set: already initialized!");
        var.inverse_method = _inverse_method;
        var.x_est = _x_est;
        var.x_up = _x_up;
        var.Phi = _Phi;
        var.P_xx = _P_xx;
        var.Q = _Q;
        var.R = _R;
        var.H = _H;
        // initialize all the workspace pointers
        var.K = (T *)((size_t)var.K + (size_t)workspace);
        var.P_xz = (T *)((size_t)var.P_xz + (size_t)workspace);
        var.P_zz = (T *)((size_t)var.P_zz + (size_t)workspace);
        var.X_h = (T *)((size_t)var.X_h + (size_t)workspace);
        var.X = (T *)((size_t)var.X + (size_t)workspace);
        var.X_er = (T *)((size_t)var.X_er + (size_t)workspace);
        var.X_h_er = (T *)((size_t)var.X_h_er + (size_t)workspace);
        var.eig_z = (T *)((size_t)var.eig_z + (size_t)workspace);
        var.workspace_cholesky = (T *)((size_t)var.workspace_cholesky + \
                                   (size_t)workspace);
        var.info = (int *)((size_t)var.info + (size_t)workspace);
        var.placeHolder0 = (T *)((size_t)var.placeHolder0 + (size_t)workspace);
        var.workspace_sg = (T *)((size_t)var.workspace_sg + (size_t)workspace);
        // set workspace for sg
        var.sg->set_workspace(var.workspace_sg);
        // mallocing some small space for internal vals
        var.Wm = (T*)malloc(sizeof(T) * 2);
        var.Wc = (T*)malloc(sizeof(T) * 2);
        var.initialized = true;
    }
}


/**
 * @brief taking initial point at var.x_up, applies Kalman prediction
 *  step to store the predictions in var.x_est, with estimated cov in
 *  var.P_xx.
 */
template <typename T>
void predict_xP(Variables<T>& var, cublasHandle_t handle) {
    // var.X contains cols of sigmas, var.P_xx is junk
    var.sg->give_points(var.P_xx, var.X, var.Wm, var.Wc, var.x_up); // corrupts var.P_xx
    // var.X contains predictions
    T alfa = (T)1.0, beta = (T)0.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N,
                            CUBLAS_OP_N, var.dim_x, var.nPoints, var.dim_x,
                            &alfa, var.Phi, var.dim_x, var.X, var.dim_x,
                            &beta, var.X, var.dim_x));
    // var.x_est conatins mean, var.X_er offsets, var.P_xx cov
    var.sg->find_mean(var.dim_x, var.nPoints, var.Wm, var.X, var.x_est);
    var.sg->find_covariance(var.dim_x, var.nPoints, var.Wc,\
    var.x_est, var.X, var.X_er, var.P_xx);
    alfa = (T)1.0; beta = (T)1.0;
    CUBLAS_CHECK(cublasgeam(handle, CUBLAS_OP_N,
                            CUBLAS_OP_N, var.dim_x, var.dim_x,
                            &alfa, var.P_xx, var.dim_x, &beta,
                            var.Q, var.dim_x, var.P_xx, var.dim_x));
}

template <typename T>
void transform_to_measurement(Variables<T>& var, cublasHandle_t handle) {
    // transformations in var.X_h
    T alfa = (T)1.0, beta = (T)0.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N,
                            CUBLAS_OP_N, var.dim_z, var.nPoints,
                            var.dim_x, &alfa, var.H, var.dim_z, var.X,
                            var.dim_x, &beta, var.X_h, var.dim_z));
}

template <typename T>
void find_P_zz(Variables<T>& var, cublasHandle_t handle) {
    // var.eig_z conatins mean, var.X_h offsets, var.P_zz cov
    var.sg->find_mean(var.dim_z, var.nPoints, var.Wm, var.X_h, var.eig_z);
    var.sg->find_covariance(var.dim_z, var.nPoints, var.Wc, var.eig_z, var.X_h, var.X_h, var.P_zz);
    T alfa = (T)1.0, beta = (T)1.0;
    CUBLAS_CHECK(cublasgeam(handle, CUBLAS_OP_N,
                            CUBLAS_OP_N, var.dim_z, var.dim_z,
                            &alfa, var.P_zz, var.dim_z, &beta,
                            var.R, var.dim_z, var.P_zz, var.dim_z));
}

template <typename T>
void find_P_xz(Variables<T>& var, cublasHandle_t handle) {
    // var.P_xz contains var.P_xz
    T alfa = var.Wc[1], beta = (T)0.0;
    CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N,
                            CUBLAS_OP_T, var.dim_x, var.dim_z,
                            var.nPoints - 1, &alfa, var.X_er + var.dim_x,
                            var.dim_x, var.X_h + var.dim_z, var.dim_z,
                            &beta, var.P_xz, var.dim_x));
    alfa = var.Wc[0];
    CUBLAS_CHECK(cublasger(handle, var.dim_x, var.dim_z,
                           &alfa, var.X_er, 1, var.X_h, 1,
                           var.P_xz, var.dim_x));
}

template <typename T>
void find_kalman_gain(Variables<T>& var, cublasHandle_t handle, cusolverDnHandle_t handle_sol) {
    // decomposing var.P_zz (cholesky, as var.P_zz is symm.)
    CUSOLVER_CHECK(cusolverDnpotrf(handle_sol,
                                   CUBLAS_FILL_MODE_UPPER,
                                   var.dim_z, var.P_zz, var.dim_z,
                                   var.workspace_cholesky, var.Lwork,
                                   var.info));
    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, var.info, sizeof(int),
                          cudaMemcpyDeviceToHost));
    ASSERT(info_h == 0,
           "[UnKF] cholesky decomp, var.info=%d | expected=0", info_h);
    // Implicit: var.P_xz conatins itself and K is uninit
    // Explicit: K will contain itself
    if (var.inverse_method == Explicit) {
        make_ID_matrix(var.placeHolder0, var.dim_z);

        // findind var.P_zz.inv() and placing in var.placeHolder0
        CUSOLVER_CHECK(cusolverDnpotrs(handle_sol,
                                       CUBLAS_FILL_MODE_UPPER,
                                       var.dim_z, var.dim_z, var.P_zz, var.dim_z,
                                       var.placeHolder0, var.dim_z, var.info));
        CUDA_CHECK(cudaMemcpy(&info_h, var.info, sizeof(int),
                              cudaMemcpyDeviceToHost));
        ASSERT(info_h == 0,
               "[UnKF], Explicit inversion var.info=%d | expected=0", info_h);
        T alfa = (T)1.0, beta = (T)0.0;
        CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N,
                                CUBLAS_OP_N, var.dim_x, var.dim_z,
                                var.dim_z, &alfa, var.P_xz, var.dim_x,
                                var.placeHolder0, var.dim_z, &beta,
                                var.K, var.dim_x));
    }
}

template <typename T>
void update_estimate(Variables<T>& var, cublasHandle_t handle, cusolverDnHandle_t handle_sol) {
    // place residuals in z, i.e. z.measured - z.predicted + error
    T alfa = (T)-1.0, beta = (T)1.0;
    CUBLAS_CHECK(cublasaxpy(handle, var.dim_z,
                            &alfa, var.eig_z, 1, var.z, 1));
    if (var.inverse_method == Implicit) {
        CUSOLVER_CHECK(cusolverDnpotrs(handle_sol,
                                       CUBLAS_FILL_MODE_UPPER,
                                       var.dim_z, 1, var.P_zz, var.dim_z,
                                       var.z, var.dim_z, var.info));
        int info_h = 0;
        CUDA_CHECK(cudaMemcpy(&info_h, var.info, sizeof(int),
                              cudaMemcpyDeviceToHost));
        ASSERT(info_h == 0,
               "[UnKF], Implicit, finding B.inv()*residual var.info=%d | expected=0", info_h);
        // z now contains B.inv()*residual
        // K*residual + var.x.est in var.x.up
        CUDA_CHECK(cudaMemcpy(var.x_up, var.x_est, sizeof(T) * var.dim_x,
                              cudaMemcpyDeviceToDevice));
        T alfa = (T)1.0, beta = (T)1.0;
        CUBLAS_CHECK(cublasgemv(handle, CUBLAS_OP_N,
                                 var.dim_x, var.dim_z, &alfa, var.P_xz,
                                 var.dim_x, var.z, 1, &beta, var.x_up, 1));
    }

    else { // Explicit
        // K*residual + var.x.est in var.x.up
        CUDA_CHECK(cudaMemcpy(var.x_up, var.x_est, sizeof(T) * var.dim_x,
                              cudaMemcpyDeviceToDevice));
        alfa = (T)1.0; beta = (T)1.0;
        CUBLAS_CHECK(cublasgemv(handle, CUBLAS_OP_N,
                                 var.dim_x, var.dim_z, &alfa, var.K, var.dim_x,
                                 var.z, 1, &beta, var.x_up, 1));
    }
}

template <typename T>
void update_covariance(Variables<T>& var, cublasHandle_t handle, cusolverDnHandle_t handle_sol) {
    if (var.inverse_method == Explicit) {
        // putting updated cov in var.P_xx = -1.K.var.P_xz' + var.P_xx(old)
        T alfa = (T)-1.0, beta = (T)1.0;
        CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N,
                                CUBLAS_OP_T, var.dim_x, var.dim_x,
                                var.dim_z, &alfa, var.K, var.dim_x,
                                var.P_xz, var.dim_x, &beta,
                                var.P_xx, var.dim_x));
    }

    else { // Implicit
        // transposing var.P_xz and keeping in K
        T alfa = (T)1.0, beta = (T)0.0;
        CUBLAS_CHECK(cublasgeam(handle, CUBLAS_OP_T,
                                CUBLAS_OP_N, var.dim_z, var.dim_x,
                                &alfa, var.P_xz, var.dim_x, &beta,
                                var.P_xz, var.dim_x, var.K, var.dim_z));
        CUSOLVER_CHECK(cusolverDnpotrs(handle_sol,
                                       CUBLAS_FILL_MODE_UPPER,
                                       var.dim_z, var.dim_x, var.P_zz, var.dim_z,
                                       var.K, var.dim_z, var.info));
        // multipling the sol to var.P_xz putting in var.P_xx
        CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N,
                                CUBLAS_OP_N, var.dim_x, var.dim_x,
                                var.dim_z, &alfa, var.P_xz, var.dim_x,
                                var.K, var.dim_z, &beta,
                                var.P_xx, var.dim_x));
    }
}


/**
 * @brief Initialization method for the opaque data structure
 * @tparam T the data type for computation
 * @param var the opaque structure storing all the required state for KF
 * @param _dim_x state vector dimension
 * @param _dim_z measurement vector dimension
 * @param _bias_selected bias while finding means and Covs
 * @param _inverse_method implicit or explicit inversion while calc kalman gain
 * @param _decomposer_selected decomposer to use while finding random points
 * @param _x_est estimated state
 * @param _x_up updated state
 * @param _Phi state transition matrix
 * @param _P_xx updated error covariance (or initital error covariance)
 * @param _Q process noise covariance matrix
 * @param _R measurent noise covariance matrix
 * @param _H state to measurement tranformation matrix
 * @param workspace workspace buffer. Pass nullptr to compute its size
 * @param workspaceSize workspace buffer size in B.
 * @note this must always be called first before calling predict/update
 */
template <typename T>
void init(Variables<T>& var, int _dim_x, int _dim_z, const T _alpha_s,
          const T _beta_s, const T _kappa_s, Inverse _inverse_method, T *_x_est,
          T *_x_up, T *_Phi, T *_P_xx, T *_Q, T *_R, T *_H,
          void* workspace, size_t& workspaceSize, cusolverDnHandle_t handle_sol) {
    set(var, _dim_x, _dim_z, _alpha_s, _beta_s, _kappa_s, _inverse_method,
            _x_est, _x_up, _Phi, _P_xx, _Q, _R, _H, workspace,  workspaceSize, handle_sol);
}


/**
 * @brief Predict the state for the next step, before the measurements are taken
 * @tparam T the data type for computation
 * @param var the opaque structure storing all the required state for KF
 * @note it is assumed that the 'init' function call has already been made with
 * a legal workspace buffer! Also, calling the 'predict' and 'update' functions
 * out-of-order will lead to unknown state!
 */
template <typename T>
void predict(Variables<T>& var, cublasHandle_t handle) {
    ASSERT(var.initialized, "kf::unscented::predict: 'init' not called!");
    predict_xP(var, handle);
}


/**
 * @brief Update the state in-lieu of measurements
 * @tparam T the data type for computation
 * @param var the opaque structure storing all the required state for KF
 * @param _z the measurement vector, if nullptr, fn returns without any state chng
 * @note it is assumed that the 'init' function call has already been made with
 * a legal workspace buffer! Also, calling the 'predict' and 'update' functions
 * out-of-order will lead to unknown state!
 */
template <typename T>
void update(Variables<T>& var, T *_z, cublasHandle_t handle, cusolverDnHandle_t handle_sol) {
    ASSERT(var.initialized, "kf::unscented::update: 'init' not called!");
    var.z = _z;
    transform_to_measurement(var, handle);
    find_P_zz(var, handle);
    // ////////////////////////////////////////
    // T* cpu , *gpu = P_zz;
    // int rows = dim_z, cols = dim_z;
    // cpu = (T *)malloc(sizeof(T)*rows*cols);
    // updateHost(cpu, gpu, rows*cols);
    // printf("P_zz line number %d.\n", __LINE__);
    // for (int i = 0; i < rows; i++){
    //     for (int j = 0; j < cols; j++)
    //         printf("%f | ", cpu[IDX2C(i, j , rows)]);
    //     printf("\n");
    // }
    // /////////////////////////////////////////
    find_P_xz(var, handle);
    find_kalman_gain(var, handle, handle_sol);
    update_estimate(var, handle, handle_sol);
    update_covariance(var, handle, handle_sol);
}


}; // end namespace ensemble
}; // end namespace kf
