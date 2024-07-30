#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import gc
import logging
import os
import time

import numpy as np
import psutil
import pytest
import qlinear
import ryzenai_torch_cpp
import torch

# m, k, n, k
llama2_7b = [
    {"shape": ((1, 4096), (4096, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 4096), (11008, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 4096), (12288, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 4096), (22528, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 4096), (32768, 4096)), "group_size": 32, "target_err_percent": 1.0},
    {
        "shape": ((1, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (11008, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (12288, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (22528, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (32768, 4096)),
        "group_size": 32,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 50.0,
    },
    {
        "shape": ((2048, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 4096), (11008, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 4096), (12288, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 4096), (22528, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 4096), (32768, 4096)),
        "group_size": 32,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 50.0,
    },
]


@pytest.mark.parametrize("xyshape", llama2_7b)
def test_QLinear_pergrp_gemm(xyshape):
    inp_shape, weight_shape = xyshape["shape"]
    grpsize = xyshape["group_size"]
    target_err_percent = xyshape["target_err_percent"]

    print("")
    print("M: ", inp_shape[0])
    print("K: ", inp_shape[1])
    print("N: ", weight_shape[0])
    print("G: ", grpsize)

    torch.random.manual_seed(123)
    np.random.seed(123)

    x_min, x_max = -42.0, 42.0
    x = np.random.uniform(low=x_min, high=x_max, size=inp_shape)
    x = torch.tensor(x).to(torch.bfloat16)

    y_min, y_max = -1.0, 1.0
    y = np.random.uniform(low=y_min, high=y_max, size=weight_shape).astype(np.float32)

    bias = torch.rand(y.shape[0], dtype=torch.float32)

    gemm_cpu = qlinear.QLinearPerGrp(
        in_features=inp_shape[1],
        out_features=weight_shape[0],
        bias=True,
        device="cpu",
        w_bit=4,
        group_size=grpsize,
    )

    gemm_cpu.weight = torch.from_numpy(y)
    gemm_cpu.bias = bias
    gemm_cpu.quantize_weights()

    gemm_aie = ryzenai_torch_cpp.aie_gemm_torch(
        inp_shape[0], inp_shape[1], weight_shape[0], True
    )
    gemm_aie.initialize_params(
        gemm_cpu.qweight,
        gemm_cpu.qzeros,
        gemm_cpu.scales,
        gemm_cpu.bias,
        gemm_cpu.group_size,
    )

    x_cpu = gemm_cpu(x)
    x_tor = gemm_aie.execute(x)

    # print(x_cpu.shape)
    # print(x_tor.shape)
    assert x_cpu.shape == x_tor.shape

    print(x_cpu)
    print(x_tor)

    result = torch.allclose(x_cpu, x_tor, target_err_percent / 100, 45)

    assert result == True


if __name__ == "__main__":
    test_QLinear_pergrp_gemm(llama2_7b[17])
