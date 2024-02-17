#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import qlinear
import numpy as np
import pytest
import psutil
import logging
import time
import gc 
import os

import torch
torch.random.manual_seed(123)
import RyzenAI

# weight shapes are transposes 

opt_1p3b_shapes = [((8, 2048), (2048, 2048)),
                   ((8, 2048), (8192, 2048)),
                   ((8, 8192), (2048, 8192)),
                   ((8, 2048), (50272, 2048)),
                   ((1, 2048), (2048, 2048)),
                   ((1, 2048), (8192, 2048)),
                   ((1, 8192), (2048, 8192)),
                   ((1, 2048), (50272, 2048))
                    ]

llama2_shapes = [  ((8, 4096), (4096, 4096)),
                   ((1, 4096), (4096, 4096)),                  
                   ((8, 4096), (11008, 4096)),
                   ((1, 4096), (11008, 4096)),
                   ((8, 11008), (4096, 11008)),
                   ((1, 11008), (4096, 11008)),
                   ((8, 4096), (32000, 4096)),
                   ((1, 4096), (32000, 4096)),
                   ]


kernel_shapes =   [ ((8, 2048), (2048, 2048)),
                    ((4, 2048), (2048, 2048)),
                    ((8, 4096), (4096, 4096)),
                   #((8, 2048), (2048, 6144)), 
                    
                   ]

@pytest.mark.parametrize("xyshape", opt_1p3b_shapes+llama2_shapes)
@pytest.mark.parametrize("kernelshape", kernel_shapes)
@pytest.mark.quant_combo_skip
def test_QLinear_quantmode1(xyshape, kernelshape, num_dlls, num_workers, impl, quant_mode):
    """ Compare qlinear.QLinear's AIE output in quant_mode=1 with np.matmul() output
    """
    num_dlls = int(num_dlls)
    num_workers = int(num_workers)
    kernel_x_shape, kernel_y_shape = kernelshape
    inp_shape, weight_shape = xyshape
    
    #print(f"***** num_dlls = {num_dlls}  num_workers = {num_workers}")
    #print(f"***** kernel_x_shape = {kernel_x_shape}  kernel_y_shape = {kernel_y_shape}")
    #print(f"***** inp_shape = {inp_shape}  weight_shape = {weight_shape}")


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

    gemm = qlinear.QLinear(in_features=inp_shape[1], out_features=weight_shape[0], bias=True, device='aie',
                           x_scale=x_scale, y_scale=y_scale,
                           quant_mode=quant_mode,
                           profiler=False, num_workers=num_workers, num_dlls=num_dlls,
                           kernel_x_shape = kernel_x_shape, kernel_y_shape = kernel_y_shape, impl=impl)

    # set the quantization packed params
    zero_point = 0.0
    weight_q = torch.quantize_per_tensor(
        torch.tensor(y),
        torch.tensor(y_scale),
        torch.tensor(zero_point),
        dtype=torch.qint8)
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
                f"i:{i} : err.min():{err.min()} err.max():{err.max()} z[i].max():{z[i].max()}")
        print(f"***** FAIL: z vs aie_out: {result}")
    # input("Enter a key")
    del gemm
    gc.collect()
    assert result == True