#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import os
import pytest

def pytest_addoption(parser):
    parser.addoption("--num_workers", action="store", default=2)
    parser.addoption("--num_dlls", action="store", default=2)
    parser.addoption("--impl", type=str, default="v1", choices=["v0", "v1"])
    parser.addoption("--quant_mode", type=str, default="w8a8", choices=["w8a8", "w8a16"])
    parser.addoption("--w_bit", type=int, default=3, choices=[3, 4])
    
def pytest_configure(config):
    config.addinivalue_line("markers", "quant_combo_skip: skip tests based on quant_mode, impl and device combination")

def pytest_collection_modifyitems(items, config):
    """Mark tests to skip based on cli options """
    impl = config.option.impl
    quant_mode = config.option.quant_mode
    dev = os.getenv("DEVICE")

    if quant_mode == "w8a8":
        # All platforms and impl supports this
        return

    skip_quant_mode_w8a16 = pytest.mark.skip(reason="w8a16, impl, device combination is not supported")
    if (impl == "v0" and quant_mode == "w8a16") or (quant_mode == "w8a16" and dev == "phx"):
        for item in items:
            if "quant_combo_skip" in item.keywords:
                item.add_marker(skip_quant_mode_w8a16)


@pytest.fixture
def num_workers(request):
    return request.config.getoption("--num_workers")

@pytest.fixture
def num_dlls(request):
    return request.config.getoption("--num_dlls")

@pytest.fixture
def impl(request):
    return request.config.getoption("--impl")

@pytest.fixture
def quant_mode(request):
    return request.config.getoption("--quant_mode")

@pytest.fixture
def w_bit(request):
    return request.config.getoption("--w_bit")