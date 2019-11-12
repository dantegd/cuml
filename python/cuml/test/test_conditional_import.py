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

import numpy as np
import pytest

from importlib import import_module

from cuml.utils.import_utils import has_dask, has_cupy, has_ucp, \
    has_treelite, has_lightgbm, has_xgboost, has_pytest_benchmark
from cuml.utils.cupy_utils import checked_cupy_unique


functions_libraries = [
    (has_dask, "dask"),
    (has_cupy, "cupy"),
    (has_ucp, "ucp"),
    (has_treelite, "treelite"),
    (has_lightgbm, "lightgbm"),
    (has_xgboost, "xgboost"),
    (has_pytest_benchmark, "pytest_benchmark")
]


@pytest.mark.parametrize('fn_tuple', functions_libraries)
def test_has_library(fn_tuple):
    try:
        import_module(fn_tuple[1])
        assert fn_tuple[0]()
    except ModuleNotFoundError:
        assert not fn_tuple[0]()


def test_cupy_unique():
    if has_cupy():
        import cupy as cp
        a = cp.arange(10)
        assert isinstance(checked_cupy_unique(a), cp.ndarray)
    else:
        a = np.arange(10)
        isinstance(checked_cupy_unique(a), np.ndarray)
