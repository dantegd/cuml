#pragma once
#include <cmath>
#include <stdio.h>
#include "utils.h"
#include "cuda_utils.h"
#include "linalg/eltwise.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"
#include "random/mvg.h"
// sigma.h takes in matrices that are column major
// (as in fortran)


namespace kf {
namespace sigmagen {

using MLCommon::Random::fill_uplo;
using MLCommon::Random::Filler;
using MLCommon::LinAlg::scalarMultiply;
using MLCommon::Random::matVecAdd;

template<typename T>
class VanDerMerwe {
private:
    // adjustable stuff
    const int dim, nPoints;
    const double epsilon = 1.e-9;
    T lambda;
    bool sqroot;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    // not so much
    const T alfa, beta, kappa;
    T *P = nullptr, *X = nullptr, *x = nullptr;
    T *workspace_decomp = nullptr;
    int *info, Lwork, info_h;
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    bool deinitilized = false;

    size_t give_buffer_size() {
        size_t granuality = 256, offset = 0;
        workspace_decomp = (T *)offset;
        offset += alignTo(sizeof(T) * Lwork, granuality);
        info = (int *)offset;
        offset += alignTo(sizeof(int), granuality);
        return offset;
    }

public:
    VanDerMerwe() = delete;
    /**
     * @brief initialize VanDerMerwe
     *
     * @tparam T the data type
     * @param dim the dimension of the points to be generated
     * @param alfa parameter alpha
     * @param beta parameter beta
     * @param kappa parameter kappa
     * @param sqroot flag to signal using sigma generator for sqrtunkf
     */
    VanDerMerwe(const int dim, const T alfa,
                const T beta, const T kappa, bool sqroot) :
                dim(dim), alfa(alfa), beta(beta),
                kappa(kappa), nPoints((2 * dim) + 1),
                sqroot(sqroot) { }

    size_t init() {
        CUBLAS_CHECK(cublasCreate(&cublasHandle));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
        CUSOLVER_CHECK(LinAlg::cusolverDnpotrf_bufferSize(cusolverHandle,
                                                          uplo, dim, P, dim,
                                                          &Lwork));
        CUDA_CHECK(cudaDeviceSynchronize());
        return give_buffer_size();
    }

    void set_workspace(T *workarea) {
        workspace_decomp = (T *)((size_t)workspace_decomp + (size_t)workarea);
        info = (int *)((size_t)info + (size_t)workarea);
    }

    /**
     * @brief give the sigma points according to given mean and cov
     *
     * Sigma points are stored as coloumns in X.
     * All the matrices are assumed to be stored in col major form
     *
     * @tparam T the data type
     * @param P the input cov matrix, after calling the function, P is corrupted
     *  If flag sqroot is true, P should be chol_decomposed matrix
     * @param X the output matrix, with cols as SigmaPoints (2*dim + 1)
     * @param Wm Wm[0] = weight of 1st Sigma point, Wm[1] =
     *  weight of rest the points (+/-)
     * @param Wc Wc[0] = weight of 1st Sigma point , Wc[1] =
     *  weight of rest the points (+/-) |
        weight as in contribution to cov matrix
     * @param x mean of the Sigma Points to be generated
     */
    void give_points(T *P, T *X, T *Wm, T *Wc, const T *x = 0) {
        if (sqroot == false) {
            // lower part will contains chol_decomp
            CUSOLVER_CHECK(LinAlg::cusolverDnpotrf(cusolverHandle, uplo, dim, P, dim,
                                                   workspace_decomp, Lwork, info));
            updateHost(&info_h, info, 1);
            ASSERT(info_h == 0, "sigma: error in potrf, info=%d | expected=0", info_h);
            // upper part being filled with 0.0
            dim3 block(32, 32);
            dim3 grid(ceildiv(dim, (int)block.x), ceildiv(dim, (int)block.y));
            fill_uplo<T> <<< grid, block >>>(dim, Filler::UPPER, (T)0.0, P);
            CUDA_CHECK(cudaPeekAtLastError());
        } // else we already have chol_decomp in P
        // find the scalar to be multiplied with chol_decomp
        lambda = (pow(alfa, 2) * (dim + kappa)) - dim;
        T scalar = pow(lambda + dim, 0.5);
        Wm[0] = lambda / (dim + lambda);
        Wc[0] = Wm[0] + (1 - pow(alfa, 2) + beta);
        Wm[1] = 0.5 / (dim + lambda);
        Wc[1] = Wm[1];
        // X set to 0 matrix + adding scaled chol
        CUDA_CHECK(cudaMemset(X, 0, sizeof(T)*dim*nPoints));
        scalarMultiply(X + dim, P, scalar, dim * dim);
        scalarMultiply(X + (dim * (dim + 1)), P, -scalar, dim * dim);
        // if x != 0, bradcasting matrix X
        if(x != nullptr)
            matVecAdd(X, X, x, T(1.0), nPoints, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void deinit() {
        if (deinitilized)
            return;
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));
        CUBLAS_CHECK(cublasDestroy(cublasHandle));
        deinitilized = true;
    }

    /// @todo check why float is giving large errors in finding mean
    /* @brief part of unscented transform i.e. finds mean
     *   of the random points generated, using the weights provided.
     *
     *  Matrix is assumed to contain col vectors
     *
     * @tparam T the data type
     * @param &in dim the dimension of points
     * @param &in nPts, number of cols in m
     * @param &in weights weights for points
     * @param &in ma matrix containing 2*dim+1 random points
     *  generated by give_points method above
     * @param &out v mean is outputted at this gpu pointer
     */
    void find_mean(const int dim, const int nPts, const T *weights, const T *m,
                   T *v) {
        CUDA_CHECK(cudaMemset(v, 0, sizeof(T)*dim));
        vctwiseAccumulate(1, dim, weights[0], m, v);
        CUDA_CHECK(cudaDeviceSynchronize());
        vctwiseAccumulate(nPts - 1, dim, weights[1], m+dim, v);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* @brief finds the covariance of given random points
     *  according to the provided 2D weigths of van der merwe
     *
     *  All the matrices are assumed to be stored in col major form
     *  random points are stored as col vectors in matrix
     *
     * @tparam T the data type
     * @param &in dim the dimension of points
     * @param &in nPts, number of cols in matrix
     * @param &in weights array of size 2 containing weights
     * @param &in mu mean of points
     * @param &in matrix contains 2*dim+1 random points
     *  generated by ver dan merwe method
     * @param &out matrix_diff place to store the offsets w.r.t.
        the mu of random points provided
     * @param &out covariance the covariance of the points
     */
    void find_covariance(const int dim, const int nPts, const T *weights, const T *mu,
                         const T *matrix, T *matrix_diff, T *covariance) {
        // finding the offsets of random points
        matVecAdd(matrix_diff, matrix, mu, T(-1.0), nPts, dim);
        // get the sum of all the covs, except 0th
        T alfa = (T)weights[1], beta = (T)0.0;
        CUBLAS_CHECK(LinAlg::cublasgemm(cublasHandle, CUBLAS_OP_N,
                                        CUBLAS_OP_T, dim, dim, nPts - 1,
                                        &alfa, matrix_diff + dim,
                                        dim, matrix_diff + dim, dim,
                                        &beta, covariance, dim));
        alfa = (T)weights[0];
        CUBLAS_CHECK(LinAlg::cublasger(cublasHandle, dim, dim, &alfa,
                                       matrix_diff, 1, matrix_diff, 1,
                                       covariance, dim));
    }

    ~VanDerMerwe() {
        deinit();
    }
}; // VanDerMerwe

}; // end of namespace sigmagen
}; // end of namespace kf
