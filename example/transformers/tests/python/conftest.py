#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import pytest

def pytest_addoption(parser):
    parser.addoption("--num_workers", action="store", default=2)
    parser.addoption("--num_dlls", action="store", default=2)
    parser.addoption("--impl", type=str, default="v1", choices=["v0", "v1"])
    

@pytest.fixture
def num_workers(request):
    return request.config.getoption("--num_workers")

@pytest.fixture
def num_dlls(request):
    return request.config.getoption("--num_dlls")

@pytest.fixture
def impl(request):
    return request.config.getoption("--impl")
