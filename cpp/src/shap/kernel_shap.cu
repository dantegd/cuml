/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cuml/explainer/kernel_shap.hpp>

#include <curand.h>
#include <curand_kernel.h>

namespace ML {
namespace Explainer {

template <typename DataT, typename IdxT>
__global__ void exact_rows_kernel_sm(int* X,
                                     IdxT nrows_X,
                                     IdxT M,
                                     DataT* background,
                                     IdxT nrows_background,
                                     DataT* combinations,
                                     DataT* observation){
  extern __shared__ int idx[];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int i, j;

  if(threadIdx.x < nrows_background){
    if(threadIdx.x == 0){
      for(i=0; i<M; i++){
        idx[i] = X[blockIdx.x + i];
      }
    }
    __syncthreads();

#pragma unroll
    for(i=tid; i<nrows_background; i+=blockDim.x){
#pragma unroll
        for(j=0; j<M; j++){
          if (idx[j] == 0){
            combinations[tid * M + j] = background[blockIdx.x * M + j];
          }else{
            combinations[tid * M + j] = observation[j];
          }
        }
      }
  }

}

template <typename DataT, typename IdxT>
__global__ void exact_rows_kernel(int* X,
                                  IdxT nrows_X,
                                  IdxT M,
                                  DataT* background,
                                  IdxT nrows_background,
                                  DataT* combinations,
                                  DataT* observation){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int i, j;

#pragma unroll
    for(i=tid; i<nrows_background; i+=blockDim.x){
#pragma unroll
        for(j=0; j<M; j++){
          if (X[blockIdx.x + j] == 0){
            combinations[tid * M + j] = background[blockIdx.x * M + j];
          }else{
            combinations[tid * M + j] = observation[j];
          }
        }
      }
  }




template <typename DataT, typename IdxT>
__global__ void sampled_rows_kernel(IdxT* nsamples,
                                    int* X,
                                    IdxT nrows_X,
                                    IdxT M,
                                    DataT* background,
                                    IdxT nrows_background,
                                    DataT* combinations,
                                    DataT* observation,
                                    uint64_t seed){
  extern __shared__ int smps[];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int i, j, k_blk;

  // see what k this block will generate
  k_blk = nsamples[blockIdx.x];

  if(threadIdx.x < nrows_background){
    if(threadIdx.x == 0){
      // thread 0 of block generates samples, reducing number of rng calls
      // calling curand only 3 * k times.
      // Sampling algo from: Li, Kim-Hung. "Reservoir-sampling algorithms
      // of time complexity O (n (1+ log (N/n)))." ACM Transactions on Mathematical
      // Software (TOMS) 20.4 (1994): 481-493.
      float w;
      curandState_t state;
      for(i=0; i<k_blk; i++){
        smps[i] = i;
      }
      curand_init((unsigned long long)seed,
                  (unsigned long long)tid,
                  0,
                  &state);

      w = exp(log(curand_uniform(&state)) / k_blk);

      while(i < M){
        i = i + floor(log(curand_uniform(&state)) / log(1 - w)) + 1;
        if(i <= M){
          smps[(int)(curand_uniform(&state) * k_blk)] = i;
          w = w * exp(log(curand_uniform(&state)) / k_blk);
        }
      }

      // write samples to 1-0 matrix
      for(i=0; i<k_blk; i++){
        X[i] = smps[i];
      }
    }


    // all threads write background line to their line

#pragma unroll
    for(i=tid; i<nrows_background; i+=blockDim.x){
#pragma unroll
      for(j=0; j<M; j++){
        combinations[tid * M + j] = background[blockIdx.x * M + j];
      }
    }

    __syncthreads();

    // all threads write observation[samples] into their entry
#pragma unroll
    for(i=tid; i<nrows_background; i+=blockDim.x){
#pragma unroll
      for(j=0; j<k_blk; j++){
        combinations[tid * M + smps[i]] = observation[smps[j]];
      }
    }
  }
}


template <typename DataT, typename IdxT>
void kernel_dataset_impl(const raft::handle_t& handle,
                         int* X,
                         IdxT nrows_X,
                         IdxT M,
                         DataT* background,
                         IdxT nrows_background,
                         DataT* combinations,
                         DataT* observation,
                         int* nsamples,
                         int len_samples,
                         int maxsample,
                         uint64_t seed){
    const auto& handle_impl = handle;
    cudaStream_t stream = handle_impl.get_stream();

    IdxT nblks;
    IdxT Nthreads;
    // todo: check for max of 1024
    Nthreads = int(32 / nrows_background) * 32;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if(M * sizeof(DataT) <= prop.sharedMemPerMultiprocessor){
      // each block calculates the combinations of an entry in X
      nblks = nrows_X - len_samples;
      // at least nrows_background threads per block, multiple of 32
      exact_rows_kernel_sm<<< nblks, Nthreads, M*sizeof(DataT), stream >>>(
        X,
        nrows_X,
        M,
        background,
        nrows_background,
        combinations,
        observation
      );
    } else {
      exact_rows_kernel<<< nblks, Nthreads, 0, stream >>>(
        X,
        nrows_X,
        M,
        background,
        nrows_background,
        combinations,
        observation
      );
    }

    CUDA_CHECK(cudaPeekAtLastError());

    // check if sample is needed
    if(len_samples > 0){
      // each block does a sample
      nblks = len_samples;

      // shared memory shouldn't be a problem since k will be small
      // todo: add check
      sampled_rows_kernel<<< nblks, Nthreads, maxsample*sizeof(int), stream >>>(
        nsamples,
        &X[(nrows_X - len_samples) * M],
        len_samples,
        M,
        background,
        nrows_background,
        combinations,
        observation,
        seed
      );
    }

    CUDA_CHECK(cudaPeekAtLastError());

}

void kernel_dataset(const raft::handle_t& handle,
                    int* X,
                    int nrows_X,
                    int M,
                    float* background,
                    int nrows_background,
                    float* combinations,
                    float* observation,
                    int* nsamples,
                    int len_nsamples,
                    int maxsample,
                    uint64_t seed){

    kernel_dataset_impl(handle,
                        X,
                        nrows_X,
                        M,
                        background,
                        nrows_background,
                        combinations,
                        observation,
                        nsamples,
                        len_nsamples,
                        maxsample,
                        seed);

}


void kernel_dataset(const raft::handle_t& handle,
                    int* X,
                    int nrows_X,
                    int M,
                    double* background,
                    int nrows_background,
                    double* combinations,
                    double* observation,
                    int* nsamples,
                    int len_nsamples,
                    int maxsample,
                    uint64_t seed){

    kernel_dataset_impl(handle,
                        X,
                        nrows_X,
                        M,
                        background,
                        nrows_background,
                        combinations,
                        observation,
                        nsamples,
                        len_nsamples,
                        maxsample,
                        seed);
}


}  // namespace Datasets
}  // namespace ML
