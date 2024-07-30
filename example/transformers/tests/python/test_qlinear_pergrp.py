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
import RyzenAI
import torch

torch.random.manual_seed(123)
np.random.seed(123)

opt_1p3b = [
    {"shape": ((1, 2048), (2048, 2048)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 2048), (8192, 2048)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 8192), (2048, 8192)), "group_size": 128, "target_err_percent": 1.0},
    {
        "shape": ((1, 2048), (50272, 2048)),
        "group_size": 128,
        "target_err_percent": 2.21,
    },
    {"shape": ((1, 2048), (2048, 2048)), "group_size": 64, "target_err_percent": 1.0},
    {"shape": ((1, 2048), (8192, 2048)), "group_size": 64, "target_err_percent": 1.51},
    {"shape": ((1, 8192), (2048, 8192)), "group_size": 64, "target_err_percent": 1.0},
    {"shape": ((1, 2048), (50272, 2048)), "group_size": 64, "target_err_percent": 2.5},
    {"shape": ((1, 2048), (2048, 2048)), "group_size": 32, "target_err_percent": 1.0},
    {"shape": ((1, 2048), (8192, 2048)), "group_size": 32, "target_err_percent": 3.907},
    {"shape": ((1, 8192), (2048, 8192)), "group_size": 32, "target_err_percent": 2.28},
    {"shape": ((1, 2048), (50272, 2048)), "group_size": 32, "target_err_percent": 3.58},
    {
        "shape": ((128, 2048), (2048, 2048)),
        "group_size": 128,
        "target_err_percent": 6.7,
    },
    {
        "shape": ((128, 2048), (8192, 2048)),
        "group_size": 128,
        "target_err_percent": 53.16,
    },
    {
        "shape": ((128, 8192), (2048, 8192)),
        "group_size": 128,
        "target_err_percent": 84.38,
    },
    {
        "shape": ((128, 2048), (50272, 2048)),
        "group_size": 128,
        "target_err_percent": 53.13,
    },
    {
        "shape": ((128, 2048), (2048, 2048)),
        "group_size": 64,
        "target_err_percent": 48.44,
    },
    {
        "shape": ((128, 2048), (8192, 2048)),
        "group_size": 64,
        "target_err_percent": 42.969,
    },
    {
        "shape": ((128, 8192), (2048, 8192)),
        "group_size": 64,
        "target_err_percent": 39.84,
    },
    {
        "shape": ((128, 2048), (50272, 2048)),
        "group_size": 64,
        "target_err_percent": 47.66,
    },
    {
        "shape": ((128, 2048), (2048, 2048)),
        "group_size": 32,
        "target_err_percent": 28.91,
    },
    {
        "shape": ((128, 2048), (8192, 2048)),
        "group_size": 32,
        "target_err_percent": 42.19,
    },
    {
        "shape": ((128, 8192), (2048, 8192)),
        "group_size": 32,
        "target_err_percent": 11.17,
    },
    {
        "shape": ((128, 2048), (50272, 2048)),
        "group_size": 32,
        "target_err_percent": 48.44,
    },
]

llama2_7b = [
    {"shape": ((1, 4096), (4096, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 4096), (11008, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {
        "shape": ((1, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {"shape": ((1, 4096), (32000, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (32000, 4096)), "group_size": 32, "target_err_percent": 3.9},
    {
        "shape": ((128, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 7.6,
    },
    {
        "shape": ((128, 4096), (11008, 4096)),
        "group_size": 128,
        "target_err_percent": 7.6,
    },
    {
        "shape": ((128, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 4.4,
    },
    {
        "shape": ((128, 4096), (32000, 4096)),
        "group_size": 128,
        "target_err_percent": 38.9,
    },
    {
        "shape": ((128, 4096), (32000, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
]

qwen1p5_7b = [
    {
        "shape": ((1, 4096), (151936, 4096)),
        "group_size": 128,
        "target_err_percent": 7.9,
    },
    {
        "shape": ((128, 4096), (151936, 4096)),
        "group_size": 128,
        "target_err_percent": 55.6,
    },
    {
        "shape": ((1, 4096), (151936, 4096)),
        "group_size": 32,
        "target_err_percent": 28.2,
    },
    {
        "shape": ((128, 4096), (151936, 4096)),
        "group_size": 32,
        "target_err_percent": 50.0,
    },
]

starcoder = [
    {"shape": ((1, 6144), (6400, 6144)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 6400), (6144, 6400)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 6144), (24576, 6144)), "group_size": 128, "target_err_percent": 2.4},
    {
        "shape": ((1, 24576), (6144, 24576)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {"shape": ((1, 6144), (49152, 6144)), "group_size": 128, "target_err_percent": 2.4},
    {"shape": ((1, 6144), (49152, 6144)), "group_size": 32, "target_err_percent": 3.0},
    {
        "shape": ((128, 6144), (6400, 6144)),
        "group_size": 128,
        "target_err_percent": 9.4,
    },
    {
        "shape": ((128, 6400), (6144, 6400)),
        "group_size": 128,
        "target_err_percent": 4.7,
    },
    {
        "shape": ((128, 6144), (24576, 6144)),
        "group_size": 128,
        "target_err_percent": 41.5,
    },
    {
        "shape": ((128, 24576), (6144, 24576)),
        "group_size": 128,
        "target_err_percent": 71.9,
    },
    {
        "shape": ((128, 6144), (49152, 6144)),
        "group_size": 128,
        "target_err_percent": 41.5,
    },
    {
        "shape": ((128, 6144), (49152, 6144)),
        "group_size": 32,
        "target_err_percent": 52.35,
    },
]

codellama2_7b = [
    {"shape": ((1, 4096), (32016, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {
        "shape": ((128, 4096), (32016, 4096)),
        "group_size": 128,
        "target_err_percent": 38.3,
    },
    {"shape": ((1, 4096), (32016, 4096)), "group_size": 32, "target_err_percent": 3.9},
    {
        "shape": ((128, 4096), (32016, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
]

chatglm = [
    {"shape": ((1, 4096), (16384, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {
        "shape": ((1, 16384), (4096, 16384)),
        "group_size": 128,
        "target_err_percent": 1.1,
    },
    {"shape": ((1, 4096), (12288, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {
        "shape": ((1, 4096), (130528, 4096)),
        "group_size": 128,
        "target_err_percent": 7.9,
    },
    {"shape": ((1, 4096), (16384, 4096)), "group_size": 32, "target_err_percent": 3.9},
    {"shape": ((1, 16384), (4096, 16384)), "group_size": 32, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (12288, 4096)), "group_size": 32, "target_err_percent": 1.1},
    {
        "shape": ((1, 4096), (130528, 4096)),
        "group_size": 32,
        "target_err_percent": 28.2,
    },
    {
        "shape": ((128, 4096), (16384, 4096)),
        "group_size": 128,
        "target_err_percent": 7.6,
    },
    {
        "shape": ((128, 16384), (4096, 16384)),
        "group_size": 128,
        "target_err_percent": 4.6,
    },
    {
        "shape": ((128, 4096), (12288, 4096)),
        "group_size": 128,
        "target_err_percent": 7.6,
    },
    {
        "shape": ((128, 4096), (130528, 4096)),
        "group_size": 128,
        "target_err_percent": 55.6,
    },
    {
        "shape": ((128, 4096), (16384, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
    {
        "shape": ((128, 16384), (4096, 16384)),
        "group_size": 32,
        "target_err_percent": 14.9,
    },
    {
        "shape": ((128, 4096), (12288, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
    {
        "shape": ((128, 4096), (130528, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
]

chatglm3 = [
    {"shape": ((1, 4096), (4608, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (4096, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (27392, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {
        "shape": ((1, 13696), (4096, 13696)),
        "group_size": 128,
        "target_err_percent": 2.4,
    },
    {"shape": ((1, 4096), (65024, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (4608, 4096)), "group_size": 32, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (4096, 4096)), "group_size": 32, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (27392, 4096)), "group_size": 32, "target_err_percent": 3.9},
    {"shape": ((1, 13696), (4096, 13696)), "group_size": 32, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (65024, 4096)), "group_size": 32, "target_err_percent": 28.2},
    {
        "shape": ((128, 4096), (4608, 4096)),
        "group_size": 128,
        "target_err_percent": 7.6,
    },
    {
        "shape": ((128, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 7.6,
    },
    {
        "shape": ((128, 4096), (27392, 4096)),
        "group_size": 128,
        "target_err_percent": 38.3,
    },
    {
        "shape": ((128, 13696), (4096, 13696)),
        "group_size": 128,
        "target_err_percent": 13.8,
    },
    {
        "shape": ((128, 4096), (65024, 4096)),
        "group_size": 128,
        "target_err_percent": 38.3,
    },
    {
        "shape": ((128, 4096), (4608, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
    {
        "shape": ((128, 4096), (4096, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
    {
        "shape": ((128, 4096), (27392, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
    {
        "shape": ((128, 13696), (4096, 13696)),
        "group_size": 32,
        "target_err_percent": 62.5,
    },
    {
        "shape": ((128, 4096), (65024, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
]

gemma7b = [
    {"shape": ((1, 3072), (4096, 3072)), "group_size": 128, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (3072, 4096)), "group_size": 128, "target_err_percent": 1.1},
    {"shape": ((1, 3072), (24576, 3072)), "group_size": 128, "target_err_percent": 2.0},
    {
        "shape": ((1, 24576), (3072, 24576)),
        "group_size": 128,
        "target_err_percent": 1.1,
    },
    {
        "shape": ((1, 3072), (256000, 3072)),
        "group_size": 128,
        "target_err_percent": 12.5,
    },
    {"shape": ((1, 3072), (4096, 3072)), "group_size": 32, "target_err_percent": 1.1},
    {"shape": ((1, 4096), (3072, 4096)), "group_size": 32, "target_err_percent": 1.1},
    {"shape": ((1, 3072), (24576, 3072)), "group_size": 32, "target_err_percent": 1.0},
    {"shape": ((1, 24576), (3072, 24576)), "group_size": 32, "target_err_percent": 1.1},
    {"shape": ((1, 3072), (256000, 3072)), "group_size": 32, "target_err_percent": 6.3},
    {
        "shape": ((128, 3072), (4096, 3072)),
        "group_size": 128,
        "target_err_percent": 4.0,
    },
    {
        "shape": ((128, 4096), (3072, 4096)),
        "group_size": 128,
        "target_err_percent": 5.8,
    },
    {
        "shape": ((128, 3072), (24576, 3072)),
        "group_size": 128,
        "target_err_percent": 45.4,
    },
    {
        "shape": ((128, 24576), (3072, 24576)),
        "group_size": 128,
        "target_err_percent": 18.8,
    },
    {
        "shape": ((128, 3072), (256000, 3072)),
        "group_size": 128,
        "target_err_percent": 45.4,
    },
    {"shape": ((128, 3072), (4096, 3072)), "group_size": 32, "target_err_percent": 6.3},
    {
        "shape": ((128, 4096), (3072, 4096)),
        "group_size": 32,
        "target_err_percent": 49.3,
    },
    {
        "shape": ((128, 3072), (24576, 3072)),
        "group_size": 32,
        "target_err_percent": 25.8,
    },
    {
        "shape": ((128, 24576), (3072, 24576)),
        "group_size": 32,
        "target_err_percent": 20.4,
    },
    {
        "shape": ((128, 3072), (256000, 3072)),
        "group_size": 32,
        "target_err_percent": 56.3,
    },
]


test_shapes = (
    opt_1p3b
    + llama2_7b
    + qwen1p5_7b
    + starcoder
    + codellama2_7b
    + chatglm
    + chatglm3
    + gemma7b
)


@pytest.mark.parametrize("xyshape", test_shapes)
def test_QLinear_pergrp(xyshape, w_bit):
    inp_shape, weight_shape = xyshape["shape"]
    grpsize = xyshape["group_size"]

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
        w_bit=w_bit,
        group_size=grpsize,
    )
    gemm_aie = qlinear.QLinearPerGrp(
        in_features=inp_shape[1],
        out_features=weight_shape[0],
        bias=True,
        device="aie",
        w_bit=w_bit,
        group_size=grpsize,
    )

    gemm_cpu.weight = torch.from_numpy(y)
    gemm_cpu.bias = bias
    gemm_cpu.quantize_weights()

    gemm_aie.weight = torch.from_numpy(y)
    gemm_aie.bias = bias
    gemm_aie.quantize_weights()

    res0 = gemm_cpu(x).to(torch.float64)
    res2 = gemm_aie(x).to(torch.float64)

    print(res0)
    print(res2)

    target_err_percent = xyshape["target_err_percent"]

    print((res0 - res2).abs().max())
    nz = torch.nonzero(res0, as_tuple=True)
    errpercent = ((res0[nz] - res2[nz]).abs() / res0[nz]).max() * 100.0
    if errpercent <= target_err_percent:
        print(f"***** PASS: res0 vs res2: {errpercent}")
        result3 = True
    else:
        print(f"***** FAIL: res0 vs res2: {errpercent}")
        result3 = False

    assert result3 == True


llama2_7b_mladf = [
    {
        "shape": ((1, 11008), (4096, 11008)),
        "group_size": 128,
        "target_max_error": 64.0,
        "target_mean_error": 12.1,
    },
    {
        "shape": ((1, 4096), (4096, 4096)),
        "group_size": 128,
        "target_max_error": 32.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((1, 4096), (11008, 4096)),
        "group_size": 128,
        "target_max_error": 32.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((1, 4096), (12288, 4096)),
        "group_size": 128,
        "target_max_error": 32.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((1, 4096), (22528, 4096)),
        "group_size": 128,
        "target_max_error": 32.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((1, 4096), (32768, 4096)),
        "group_size": 32,
        "target_max_error": 48.0,
        "target_mean_error": 6.9,
    },
    {
        "shape": ((128, 11008), (4096, 11008)),
        "group_size": 128,
        "target_max_error": 96.0,
        "target_mean_error": 11.9,
    },
    {
        "shape": ((128, 4096), (4096, 4096)),
        "group_size": 128,
        "target_max_error": 48.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((128, 4096), (11008, 4096)),
        "group_size": 128,
        "target_max_error": 48.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((128, 4096), (12288, 4096)),
        "group_size": 128,
        "target_max_error": 48.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((128, 4096), (22528, 4096)),
        "group_size": 128,
        "target_max_error": 64.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((128, 4096), (32768, 4096)),
        "group_size": 32,
        "target_max_error": 48.0,
        "target_mean_error": 6.9,
    },
    {
        "shape": ((2048, 11008), (4096, 11008)),
        "group_size": 128,
        "target_max_error": 96.0,
        "target_mean_error": 11.9,
    },
    {
        "shape": ((2048, 4096), (4096, 4096)),
        "group_size": 128,
        "target_max_error": 64.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((2048, 4096), (11008, 4096)),
        "group_size": 128,
        "target_max_error": 64.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((2048, 4096), (12288, 4096)),
        "group_size": 128,
        "target_max_error": 64.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((2048, 4096), (22528, 4096)),
        "group_size": 128,
        "target_max_error": 64.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((2048, 4096), (32768, 4096)),
        "group_size": 32,
        "target_max_error": 64.0,
        "target_mean_error": 6.9,
    },
]


other_shapes_mladf = [
    {
        "shape": ((1, 4096), (4096, 4096)),
        "group_size": 128,
        "target_max_error": 32.0,
        "target_mean_error": 6.8,
    },
    {
        "shape": ((199, 11008), (4096, 11008)),
        "group_size": 128,
        "target_max_error": 8272.0,
        "target_mean_error": 520.2,
    },
    {
        "shape": ((37, 4096), (4096, 4096)),
        "group_size": 128,
        "target_max_error": 96.0,
        "target_mean_error": 12.2,
    },
]


@pytest.mark.parametrize("xyshape", llama2_7b_mladf + other_shapes_mladf)
def test_QLinear_pergrp_mladf(xyshape, w_bit):
    if not os.getenv("MLADF"):
        pytest.skip("Test only relevant for MLADF transaction binaries.")
    inp_shape, weight_shape = xyshape["shape"]
    grpsize = xyshape["group_size"]

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
        w_bit=w_bit,
        group_size=grpsize,
    )
    gemm_aie = qlinear.QLinearPerGrp(
        in_features=inp_shape[1],
        out_features=weight_shape[0],
        bias=True,
        device="aie",
        w_bit=w_bit,
        group_size=grpsize,
    )

    gemm_cpu.weight = torch.from_numpy(y)
    gemm_cpu.bias = bias
    gemm_cpu.quantize_weights()

    gemm_aie.weight = torch.from_numpy(y)
    gemm_aie.bias = bias
    gemm_aie.quantize_weights()

    res0 = gemm_cpu(x).to(torch.float64)
    res2 = gemm_aie(x).to(torch.float64)

    print(res0)
    print(res2)

    target_max_error = xyshape["target_max_error"]
    target_mean_error = xyshape["target_mean_error"]

    err_max = (res0 - res2).abs().max()
    err_mean = (res0 - res2).abs().mean()
    print(f"max error: {err_max}, target {target_max_error}")
    print(f"mean error: {err_mean}, target {target_mean_error}")

    if (err_max <= target_max_error) and (err_mean <= target_mean_error):
        print(f"***** PASS: res0 vs res2")
        result3 = True
    else:
        print(
            f"***** FAIL: res0 vs res2: target max error {target_max_error}, target mean error {target_mean_error}"
        )
        result3 = False

    assert result3 == True


"""
@pytest.mark.parametrize("xyshape", llama2_shapes)
@pytest.mark.parametrize("grpsize", grp_sizes)
@pytest.mark.skip(reason="test only for perf measurement")
def test_QLinear_awq_perf(xyshape, grpsize, w_bit):
    dev = os.getenv("DEVICE")
    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(2)
    log_dir = "./logs_pytest"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_awq.log"
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    inp_shape, weight_shape = xyshape

    x_min, x_max = -42.0, 42.0
    x = np.random.uniform(low=x_min, high=x_max, size=inp_shape)
    x = torch.tensor(x).to(torch.bfloat16)

    num_grps = int(weight_shape[1]/grpsize)
    qw = torch.randint(0, 8, size=weight_shape).to(torch.int8)
    qz = torch.randint(0, 8, size=(weight_shape[0], num_grps)).to(torch.int8)
    sc = torch.rand((weight_shape[0], num_grps)).to(torch.bfloat16)
    bias = None
    gemm_aie = qlinear.QLinearPerGrp(in_features=inp_shape[1], out_features=weight_shape[0], bias=True,
                            device='aie', w_bit = w_bit, group_size = grpsize, profiler = True)

    gemm_aie.bias   = bias
    gemm_aie.group_size = grpsize
    gemm_aie.qweight = qw
    gemm_aie.qzeros = qz
    gemm_aie.scales = sc
    gemm_aie.quantize_weights()

    num_runs = 1000
    time_stamps = []
    for i in range(num_runs):
        start = time.perf_counter()
        res2 = gemm_aie(x)
        stop = time.perf_counter()
        time_stamps.append((stop - start) * 1000000)

    time_stamps = np.array(time_stamps)
    print(f"inp_shape, weight_shape, grpsize, time_mean, time_min, time_max, time_median : {inp_shape} {weight_shape} {grpsize} {time_stamps.mean():.0f}us {time_stamps.min():.0f}us {time_stamps.max():.0f}us {np.median(time_stamps):.0f}us")

    gc.collect()

    assert True == True


if __name__ == "__main__":
    test_QLinear_awq_perf(((1, 4096), (11008, 4096)), 128, 3)
    test_QLinear_awq_perf(((1, 4096), (32000, 4096)), 32, 3)

"""
