#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--quant_mode",
        type=str,
        default="w8a8",
        choices=["none", "w8a8", "w8a16", "w4abf16"],
    )
    parser.addoption("--w_bit", type=int, default=4, choices=[3, 4])


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "quant_combo_skip: skip tests based on quant_mode, impl and device combination",
    )


def pytest_collection_modifyitems(items, config):
    """Mark tests to skip based on cli options"""
    quant_mode = config.option.quant_mode
    dev = os.getenv("DEVICE")

    if quant_mode == "w8a8":
        # All platforms and impl supports this
        return

    skip_quant_mode_w8a16 = pytest.mark.skip(
        reason="w8a16, impl, device combination is not supported."
    )
    if quant_mode == "w8a16" and dev == "phx":
        for item in items:
            if "quant_combo_skip" in item.keywords:
                item.add_marker(skip_quant_mode_w8a16)


@pytest.fixture
def quant_mode(request):
    return request.config.getoption("--quant_mode")


@pytest.fixture
def w_bit(request):
    return request.config.getoption("--w_bit")
