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
import torch

torch.random.manual_seed(123)
import RyzenAI

# weight shapes are transposes

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

llama2_shapes = [
    ((8, 4096), (4096, 4096)),
    ((1, 4096), (4096, 4096)),
    ((8, 4096), (11008, 4096)),
    ((1, 4096), (11008, 4096)),
    ((8, 11008), (4096, 11008)),
    ((1, 11008), (4096, 11008)),
    ((8, 4096), (32000, 4096)),
    ((1, 4096), (32000, 4096)),
]


@pytest.mark.parametrize("xyshape", opt_1p3b_shapes + llama2_shapes)
@pytest.mark.quant_combo_skip
def test_QLinear_quantmode1(xyshape, quant_mode):
    """Compare qlinear.QLinear's AIE output in quant_mode=1 with np.matmul() output"""
    inp_shape, weight_shape = xyshape

    # print(f"***** inp_shape = {inp_shape}  weight_shape = {weight_shape}")

    x_min, x_max = -42.0, 42.0
    x_scale = (x_max - x_min) / 256
    y_min, y_max = -0.5, 0.5
    y_scale = (y_max - y_min) / 256

    xq = np.random.randint(-128, 127, inp_shape).astype(np.int8)
    xq = np.clip(np.round(xq), -128, 127).astype(np.int8)
    x = xq.astype(np.float32) * x_scale

    yq = np.random.randint(-128, 127, weight_shape).astype(np.int8)
    yq = np.clip(np.round(yq), -128, 127).astype(np.int8)
    y = yq.astype(np.float32) * y_scale

    xq = (x / x_scale).astype(np.int8)

    gemm = qlinear.QLinear(
        in_features=inp_shape[1],
        out_features=weight_shape[0],
        bias=True,
        device="aie",
        x_scale=x_scale,
        y_scale=y_scale,
        quant_mode=quant_mode,
        profiler=False,
    )

    # set the quantization packed params
    zero_point = 0.0
    weight_q = torch.quantize_per_tensor(
        torch.tensor(y),
        torch.tensor(y_scale),
        torch.tensor(zero_point),
        dtype=torch.qint8,
    )
    # print(f"weight_q: {weight_q}")
    bias = torch.zeros(y.shape[0], dtype=torch.float)
    lpp = torch.ao.nn.quantized.modules.linear.LinearPackedParams()
    lpp.set_weight_bias(weight_q, bias)
    # print(f"lpp._weight_bias(): {lpp._weight_bias()}")
    gemm.weight_bias = lpp._weight_bias()
    gemm.quantize_weights()

    aie_out = gemm(torch.tensor(x))

    yq = torch.int_repr(weight_q).numpy()
    zq = np.matmul(xq.astype(np.int32), yq.astype(np.int32).transpose())
    zq = zq.astype(np.int32)
    z = zq.astype(np.float32) * x_scale * y_scale
    z = torch.tensor(z).to(torch.float32)

    result = torch.allclose(aie_out, z, atol=1)
    if result:
        print(f"***** PASS: z vs aie_out: {result}")
    else:
        print(f"z[0] : {z[0]}")
        print(f"aie_out[0] : {aie_out[0]}")
        for i in range(z.shape[0]):
            err = aie_out[i] - z[i]
            print(
                f"i:{i} : err.min():{err.min()} err.max():{err.max()} z[i].max():{z[i].max()}"
            )
        print(f"***** FAIL: z vs aie_out: {result}")
    # input("Enter a key")
    del gemm
    gc.collect()
    assert result == True


@pytest.mark.parametrize("xyshape", opt_1p3b_shapes)
@pytest.mark.quant_combo_skip
def test_QLinear_manual_load(xyshape, quant_mode):
    """Compare qlinear.QLinear's AIE output in quant_mode=1 with np.matmul() output"""
    inp_shape, weight_shape = xyshape
    x_min, x_max = -42.0, 42.0
    x_scale = (x_max - x_min) / 256
    y_min, y_max = -0.5, 0.5
    y_scale = (y_max - y_min) / 256

    xq = np.random.randint(-128, 127, inp_shape).astype(np.int8)
    xq = np.clip(np.round(xq), -128, 127).astype(np.int8)
    x = xq.astype(np.float32) * x_scale

    yq = np.random.randint(-128, 127, weight_shape).astype(np.int8)
    yq = np.clip(np.round(yq), -128, 127).astype(np.int8)
    y = yq.astype(np.float32) * y_scale

    xq = (x / x_scale).astype(np.int8)

    gemm = qlinear.QLinear(
        in_features=inp_shape[1],
        out_features=weight_shape[0],
        bias=False,
        device="aie",
        x_scale=x_scale,
        y_scale=y_scale,
        quant_mode=quant_mode,
        profiler=False,
    )

    gemm.weight = None
    gemm.bias = None
    gemm.weight_q = np.ascontiguousarray(yq.transpose().astype(np.int8))
    gemm.y_scale = y_scale
    gemm.weight_bias = None
    gemm.wshape = gemm.weight_q.shape
    gemm.aiegemm.initialize_weights(gemm.weight_q)
    gemm.weight_q = None

    if quant_mode == "w8a8":
        gemm.accum_type = np.int32
        gemm.act_type = np.int8
        gemm.x_scaled_abs_max_fp = 128.0
        gemm.x_scaled_max = 127
        gemm.x_scaled_min = -128
    else:
        gemm.accum_type = np.int64
        gemm.act_type = np.int16
        gemm.x_scaled_abs_max_fp = 32768.0
        gemm.x_scaled_max = 32767
        gemm.x_scaled_min = -32768

    gemm.c_fp = np.empty((1, gemm.out_features), dtype=np.float32)
    gemm.c_token = np.zeros((1, gemm.out_features), dtype=gemm.accum_type)
    gemm.x_abs = np.empty((1, gemm.in_features), dtype=np.float32)
    gemm.x_round = np.empty((1, gemm.in_features), dtype=np.float32)
    gemm.x_scaled = np.empty((1, gemm.in_features), dtype=np.float32)
    gemm.x_clip = np.empty((1, gemm.in_features), dtype=np.float32)
    gemm.x_max = np.array(1.0, dtype=np.float32)

    aie_out = gemm(torch.tensor(x))

    zq = np.matmul(xq.astype(np.int32), yq.astype(np.int32).transpose())
    zq = zq.astype(np.int32)
    z = zq.astype(np.float32) * x_scale * y_scale
    z = torch.tensor(z).to(torch.float32)

    result = torch.allclose(aie_out, z, atol=1)
    if result:
        print(f"***** PASS: z vs aie_out: {result}")
    else:
        print(f"z[0] : {z[0]}")
        print(f"aie_out[0] : {aie_out[0]}")
        for i in range(z.shape[0]):
            err = aie_out[i] - z[i]
            print(
                f"i:{i} : err.min():{err.min()} err.max():{err.max()} z[i].max():{z[i].max()}"
            )
        print(f"***** FAIL: z vs aie_out: {result}")
    # input("Enter a key")
    assert result == True


@pytest.mark.skip(reason="test only for perf measurement")
@pytest.mark.parametrize("xyshape", opt_1p3b_shapes + llama2_shapes)
def test_QLinear_quantmode1_perf(xyshape, quant_mode):
    """Compare qlinear.QLinear's AIE output in quant_mode=1 with np.matmul() output"""
    dev = os.getenv("DEVICE")
    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(2)
    log_dir = "./logs_pytest"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_w8a16.log"
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.CRITICAL,
    )
    inp_shape, weight_shape = xyshape

    # print(f"***** num_dlls = {num_dlls}  num_workers = {num_workers}")
    # print(f"***** kernel_x_shape = {kernel_x_shape}  kernel_y_shape = {kernel_y_shape}")
    # print(f"***** inp_shape = {inp_shape}  weight_shape = {weight_shape}")

    x_min, x_max = -42.0, 42.0
    x_scale = (x_max - x_min) / 256
    y_min, y_max = -0.5, 0.5
    y_scale = (y_max - y_min) / 256

    xq = np.random.randint(-128, 127, inp_shape).astype(np.int8)
    xq = np.clip(np.round(xq), -128, 127).astype(np.int8)
    x = xq.astype(np.float32) * x_scale

    yq = np.random.randint(-128, 127, weight_shape).astype(np.int8)
    yq = np.clip(np.round(yq), -128, 127).astype(np.int8)
    y = yq.astype(np.float32) * y_scale

    xq = (x / x_scale).astype(np.int8)

    gemm = qlinear.QLinear(
        in_features=inp_shape[1],
        out_features=weight_shape[0],
        bias=True,
        device="aie",
        x_scale=x_scale,
        y_scale=y_scale,
        quant_mode=quant_mode,
        profiler=True,
    )

    # set the quantization packed params
    zero_point = 0.0
    weight_q = torch.quantize_per_tensor(
        torch.tensor(y),
        torch.tensor(y_scale),
        torch.tensor(zero_point),
        dtype=torch.qint8,
    )
    # print(f"weight_q: {weight_q}")
    bias = torch.zeros(y.shape[0], dtype=torch.float)
    lpp = torch.ao.nn.quantized.modules.linear.LinearPackedParams()
    lpp.set_weight_bias(weight_q, bias)
    # print(f"lpp._weight_bias(): {lpp._weight_bias()}")
    gemm.weight_bias = lpp._weight_bias()
    gemm.quantize_weights()

    num_runs = 1000
    time_stamps = []
    for i in range(num_runs):
        start = time.perf_counter()
        aie_out = gemm(torch.tensor(x))
        stop = time.perf_counter()
        time_stamps.append((stop - start) * 1000000)

    time_stamps = np.array(time_stamps)
    print(
        f"inp_shape, weight_shape, time_mean, time_min, time_max, time_median : {inp_shape} {weight_shape} {time_stamps.mean():.0f}us {time_stamps.min():.0f}us {time_stamps.max():.0f}us {np.median(time_stamps):.0f}us"
    )

    gc.collect()

    assert True == True


if __name__ == "__main__":
    test_QLinear_quantmode1_perf(((1, 4096), (32000, 4096)), (2048, 2048), "w8a16")
