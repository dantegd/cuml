# Copyright (c) 2018, NVIDIA CORPORATION.
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

# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption("--run_stress", action="store_true",
                     default=False, help="run stress tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_stress"):
        # --run_stress given in cli: do not skip stress tests
        return
    skip_stress = pytest.mark.skip(reason="Stress tests run with --run_stress flag." )
    for item in items:
        if "stress" in item.keywords:
            item.add_marker(skip_stress)
