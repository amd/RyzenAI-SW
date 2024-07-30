#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import logging
import os
import time

import numpy as np
import pytest
import qlinear
import torch

torch.random.manual_seed(123)
import ryzenai_torch_cpp

opt_1p3b_shapes = [
    ((8, 2048), (2048, 2048)),
    ((8, 2048), (8192, 2048)),
    ((8, 8192), (2048, 8192)),
    ((8, 2048), (50272, 2048)),
    ((1, 2048), (2048, 2048)),
    ((1, 2048), (8192, 2048)),
    ((1, 8192), (2048, 8192)),
    ((1, 2048), (50272, 2048)),
]


@pytest.mark.parametrize("xyshape", opt_1p3b_shapes)
@pytest.mark.skip("qlinear_experimental is not supported")
def test_QLinearExperimentalCPU_quantmode1(xyshape):
    (inp_shape, weight_shape) = xyshape
    lo, hi = -42.0, 42.0
    x = (hi - lo) * torch.rand(inp_shape) + lo
    lo, hi = -0.5, 0.5
    y = (hi - lo) * torch.rand(weight_shape) + lo

    y_scale = torch.max(torch.abs(y)) / 128.0

    yq = torch.round(y / y_scale)
    yq = torch.clip(yq, -128, 128).to(torch.int32)
    requantize_in_scale = 1.0
    requantize_out_scale = 128.0

    # reference compute
    x_scale = torch.max(torch.abs(x)) / 128
    xq = torch.round(x / x_scale).to(torch.int32)
    xq = torch.clip(xq, -128, 128)
    zq = torch.matmul(xq, yq.transpose(0, 1))
    zq = zq * (requantize_in_scale / requantize_out_scale)
    zq = torch.clip(zq, -32768, 32767)
    z = zq.to(torch.float32) * x_scale * y_scale * requantize_out_scale

    gemm = qlinear_experimental.QLinearExperimentalCPU(
        in_features=inp_shape[1],
        out_features=weight_shape[0],
        bias=False,
        x_scale=x_scale,
        y_scale=y_scale,
        quant_mode=1,
        requantize_in_scale=requantize_in_scale,
        requantize_out_scale=requantize_out_scale,
    )

    # set the quantization packed params
    zero_point = 0.0
    weight_q = torch.quantize_per_tensor(
        y, y_scale, torch.tensor(zero_point), dtype=torch.qint8
    )
    # print(f"weight_q: {weight_q}")
    bias = torch.zeros(y.shape[0], dtype=torch.float)
    lpp = torch.ao.nn.quantized.modules.linear.LinearPackedParams()
    lpp.set_weight_bias(weight_q, bias)
    # print(f"lpp._weight_bias(): {lpp._weight_bias()}")
    gemm.weight_bias = lpp._weight_bias()
    gemm.quantize_weights()

    cpu_out = gemm(x)

    result = torch.allclose(cpu_out, z, atol=3)
    if result:
        print(f"***** PASS: z vs cpu_out: {result}")
    else:
        print(f"z[0] : {z[0]}")
        print(f"cpu_out[0] : {cpu_out[0]}")
        for i in range(z.shape[0]):
            err = cpu_out[i] - z[i]
            print(
                f"i:{i} : err.min():{err.min()} err.max():{err.max()} z[i].max():{z[i].max()}"
            )
        print(f"***** FAIL: z vs cpu_out: {result}")
    # input("Enter a key")
    assert result == True


@pytest.mark.parametrize("xyshape", opt_1p3b_shapes)
def test_ryzenai_torch_cpp_cpu_linear_mmul(xyshape):
    (inp_shape, weight_shape) = xyshape
    x = torch.rand(inp_shape)
    w = torch.rand(weight_shape)
    ryzenai_cpu_linear = ryzenai_torch_cpp.cpu_linear()

    out1 = ryzenai_cpu_linear.mmul(x, w.transpose(0, 1))
    out2 = torch.matmul(x, w.transpose(0, 1))

    result = torch.allclose(out1, out2)
    if result:
        print(f"***** PASS: z vs aie_out: {result}")
    else:
        print(f"out1[0] : {out1[0]}")
        print(f"out2[0] : {out2[0]}")
        err = out1 - out2
        print(f"err : {err.min()} {err.max()}")
        print(f"***** FAIL: z vs aie_out: {result}")
    assert result == True


@pytest.mark.parametrize("xyshape", opt_1p3b_shapes)
def test_ryzenai_torch_cpp_cpu_linear_qmmul(xyshape):
    (inp_shape, weight_shape) = xyshape
    lo, hi = -42.0, 42.0
    x = (hi - lo) * torch.rand(inp_shape) + lo
    lo, hi = -0.5, 0.5
    y = (hi - lo) * torch.rand(weight_shape) + lo

    y_scale = torch.max(torch.abs(y)) / 128.0

    yq = torch.round(y / y_scale)
    yq = torch.clip(yq, -128, 128).to(torch.int32)
    requantize_in_scale = torch.tensor(1.0)
    requantize_out_scale = torch.tensor(128.0)

    # reference compute
    x_scale = torch.max(torch.abs(x)) / 128
    xq = torch.round(x / x_scale).to(torch.int32)
    xq = torch.clip(xq, -128, 128)
    zq = torch.matmul(xq, yq.transpose(0, 1))
    zq = zq * (requantize_in_scale / requantize_out_scale)
    zq = torch.clip(zq, -32768, 32767)
    z = zq.to(torch.float32) * x_scale * y_scale * requantize_out_scale

    ryzenai_cpu_qlinear = ryzenai_torch_cpp.cpu_qlinear(
        yq, y_scale, requantize_in_scale, requantize_out_scale
    )

    # print(yq, yq.dtype)
    # print(x, x.dtype)
    # print(y_scale, y_scale.dtype)
    out1 = ryzenai_cpu_qlinear.qmmul(x)

    result = torch.allclose(out1, z)
    if result:
        print(f"***** PASS: z vs aie_out: {result}")
    else:
        print(f"out1[0] : {out1[0]}")
        print(f"out2[0] : {z[0]}")
        err = out1 - z
        print(f"err : {err.min()} {err.max()}")
        print(f"***** FAIL: z vs aie_out: {result}")

    assert result == True
