#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import qlinear
import numpy as np
import pytest
import gc 
import logging
import os
import psutil
import time

import torch
torch.random.manual_seed(123)
np.random.seed(123)
import RyzenAI

opt_1p3b_shapes = [((8, 2048), (2048, 2048)),
                   ((8, 2048), (8192, 2048)),
                   ((8, 8192), (2048, 8192)),
                   ((8, 2048), (50272, 2048)),
                   ((1, 2048), (2048, 2048)),
                   ((1, 2048), (8192, 2048)),
                   ((1, 8192), (2048, 8192)),
                   ((1, 2048), (50272, 2048))
                    ]

llama2_shapes = [
                   ((8, 4096), (4096, 4096)),
                   ((1, 4096), (4096, 4096)),                  
                   ((8, 4096), (11008, 4096)),
                   ((1, 4096), (11008, 4096)),
                   ((8, 11008), (4096, 11008)),
                   ((1, 11008), (4096, 11008)),
                   ((8, 4096), (32000, 4096)),
                   ((1, 4096), (32000, 4096))
                   ]

grp_sizes    =  [128, 64, 32]

#@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize("xyshape", llama2_shapes+opt_1p3b_shapes)
@pytest.mark.parametrize("grpsize", grp_sizes)
def test_QLinear_pergrp(xyshape, grpsize, w_bit):
    inp_shape, weight_shape = xyshape

    x_min, x_max = -42.0, 42.0
    x = np.random.uniform(low=x_min, high=x_max, size=inp_shape)
    x = torch.tensor(x).to(torch.bfloat16) 
    
    y_min, y_max = -8, 7
    y = np.random.uniform(low=y_min, high=y_max, size=weight_shape).astype(np.float32)

    bias = torch.rand(y.shape[0], dtype=torch.float32)
    
    gemm_cpu = qlinear.QLinearPerGrp(in_features=inp_shape[1], out_features=weight_shape[0], bias=True, 
                            device='cpu', w_bit = w_bit, group_size = grpsize)
    gemm_aie = qlinear.QLinearPerGrp(in_features=inp_shape[1], out_features=weight_shape[0], bias=True, 
                            device='aie', w_bit = w_bit, group_size = grpsize)
    #gemm_aie_emu = qlinear.QLinearPerGrp(in_features=inp_shape[1], out_features=weight_shape[0], bias=True, 
    #                        device='aie_emu', w_bit = w_bit, group_size = grpsize)
    
    gemm_cpu.weight = torch.from_numpy(y)
    gemm_cpu.bias = bias
    gemm_cpu.quantize_weights()
    
    gemm_aie.weight = torch.from_numpy(y)
    gemm_aie.bias = bias
    gemm_aie.quantize_weights()
    
    #gemm_aie_emu.weight = torch.from_numpy(y)
    #gemm_aie_emu.bias = bias
    #gemm_aie_emu.quantize_weights()
    
    #print(f"inp_shape, weight_shape, grpsize : {inp_shape} {weight_shape} {grpsize}")
    #print(f"gemm_aie_emu.qweight: {gemm_aie_emu.qweight.shape} {gemm_aie_emu.qweight.min()} {gemm_aie_emu.qweight.max()}")
    #print(f"gemm_aie_emu.qzeros: {gemm_aie_emu.qzeros.shape} {gemm_aie_emu.qzeros.min()} {gemm_aie_emu.qzeros.max()}")
    #print(f"gemm_aie_emu.scales: {gemm_aie_emu.scales.shape}")
    
    #print(f"gemm_cpu.weight: {gemm_cpu.weight.shape}")
    
    res0 = gemm_cpu(x).to(torch.float64)
    #res1 = gemm_aie_emu(x).to(torch.float64)
    res2 = gemm_aie(x).to(torch.float64)

    del gemm_cpu, gemm_aie#, gemm_aie_emu
    gc.collect()
    
    #errpercent = ((res0-res1).abs()/res0).max()
    #if (errpercent < 1.0):
    #    print(f"***** PASS: res0 vs res1: {errpercent}") 
    #    result1 = True
    #else:
    #    print(f"***** FAIL: res0 vs res1: {errpercent}")
    #    result1 = False

    #errpercent = ((res0-res2).abs()/res0).max()
    #if (errpercent < 1.0):
    #    print(f"***** PASS: res0 vs res2: {errpercent}") 
    #    result2 = True
    #else:
    #    print(f"***** FAIL: res0 vs res2: {errpercent}")
    #    result2 = False

    print(res0)
    print(res2)
    print((res0-res2).abs().max())
    errpercent = ((res0-res2).abs()/res0).max()
    if (errpercent < 1.0):
        print(f"***** PASS: res0 vs res2: {errpercent}") 
        result3 = True
    else:
        print(f"***** FAIL: res0 vs res2: {errpercent}")
        result3 = False

    #assert result1 == True
    #assert result2 == True
    assert result3 == True

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