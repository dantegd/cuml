#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np

from cuml.common.handle cimport cumlHandle
from cuml.utils import get_dev_array_ptr, zeros

cdef extern from "datasets/make_blobs.hpp" namespace "ML":
    cdef void make_blobs(const cumlHandle& handle,
                         float* out,
                         int* labels,
                         int n_rows,
                         int n_cols,
                         int n_clusters,
                         const float* centers,
                         const float* cluster_std,
                         const float cluster_std_scalar,
                         bool shuffle,
                         float center_box_min,
                         float center_box_max,
                         uint64_t seed)

    cdef void make_blobs(const cumlHandle& handle,
                         double* out,
                         int* labels,
                         int n_rows,
                         int n_cols,
                         int n_clusters,
                         const double* centers,
                         const double* cluster_std,
                         const double cluster_std_scalar,
                         bool shuffle,
                         double center_box_min,
                         double center_box_max,
                         uint64_t seed)

str_to_dtype = {
    'float': np.float32,
    'double': np.float64
}


# Note: named blobs to avoid cython naming conflict issues, renaming in
# __init__.py to make_blob
def blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0,
          center_box=(-10.0, 10.0), shuffle=True, random_state=None,
          random_state, dtype='float', handle=None):

    if dtype not in ['float', 'double']:
        raise TypeError("dtype must be either 'float' or 'double'")

    handle = cuml.common.handle.Handle() if handle is None else handle

    out = zeros((n_samples, 2), dtype=str_to_dtype[dtype])
    cdef uintptr_t out_ptr = get_cudf_column_ptr(out)

    labels = zeros(n_samples, dtype=np.int32)
    cdef uintptr_t labels_ptr = get_cudf_column_ptr(labels)

    centers = zeros((2, n_features), dtype=str_to_dtype[dtype])
    cdef uintptr_t centers_ptr = get_cudf_column_ptr(centers)
    # cdef uintptr_t centers_ptr = NULL

    cdef uintptr_t cluster_std_ptr = NULL

    center_box_min = center_box[0]
    center_box_max = center_box[1]

    cdef uintptr_t labels_ptr = get_cudf_column_ptr(self.labels_)

    make_blobs(handle_[0],
               <float*> out_ptr,
               <int*> labels_ptr,
               n_samples,
               2,
               n_features,
               <float*> centers_ptr,
               <float*> cluster_std_ptr,
               cluster_std,
               shuffle,
               center_box_min,
               center_box_max,
               0)

    return out, labels
