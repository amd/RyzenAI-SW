#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import logging
from typing import Tuple
from torch.nn.parameter import Parameter
from torch import Tensor
import os
import math
import gc
import numpy as np
import time
import platform
from collections import defaultdict

import torch
torch.random.manual_seed(123)

from _ffi import register_object , register_func, register_extension, get_global_func
import RyzenAI

class QLinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int = 1, out_features: int = 1, bias: bool = True, device=None,
                 x_scale: float = 1, y_scale: float = 1, quant_mode = "w8a8",
                 num_dlls: int = 2, profiler: bool = False, dtype="float32",
                 kernel_x_shape: Tuple = (8, 2048), kernel_y_shape: Tuple = (2048, 2048),
                 num_workers: int = 2, pack_weights=True, impl="v1") -> None:
        factory_kwargs = {'device': 'cpu', 'dtype': None}
        super().__init__()
        self.device = device
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        if self.quant_mode == "w8a8": 
            self.weight = None
            self.accum_type = np.int32
            self.act_type = np.int8
            self.x_scaled_abs_max_fp = 128.0
            self.x_scaled_max = 127
            self.x_scaled_min = -128
        elif self.quant_mode == "w8a16":
            self.weight = None
            self.accum_type = np.int64
            self.act_type = np.int16
            self.x_scaled_abs_max_fp = 32768.0
            self.x_scaled_max = 32767
            self.x_scaled_min = -32768
        else:
            self.weight = torch.empty((in_features, out_features), **factory_kwargs)
        self.weight_q = None
        self.dev = os.getenv("DEVICE")
        if self.dev is None:
            print("DEVICE environment variable is not set")
            raise SystemExit
        # Perform checks and error out
        if self.quant_mode == "w8a16" and impl == "v0":
            print(f"{self.quant_mode} is not supported on impl {impl}")              
            raise SystemExit
        if self.quant_mode == "w8a16" and self.dev != "stx":
            print(f"{self.quant_mode} is not supported on {self.dev} platform")
            raise SystemExit
        if bias is True:                                                               
            self.bias = torch.empty(out_features, **factory_kwargs)                    
        else:                                                                          
            self.register_parameter('bias', None)                                      
            self.bias = None
        self.kernel_x_shape = kernel_x_shape
        self.kernel_y_shape = kernel_y_shape
        self.x_scale = np.array(x_scale, dtype=np.float32)
        self.y_scale = np.array(y_scale, dtype=np.float32)
        self.num_dlls = num_dlls
        self.num_workers = num_workers
        self.pack_weights = pack_weights
        self.impl = impl
        if self.device == 'aie':
            if not os.path.exists("./logs"):
                os.makedirs("./logs")
            self.init_aiegemm()

        self.forward_dict  = defaultdict(lambda: self.forward_prefill, {1: self.forward_token})

    def init_aiegemm_v0(self) -> None:
        dll_path_base =  os.getenv("PYTORCH_AIE_PATH") + "/dll/" + self.dev + "/qlinear/"
        self.dll_path = dll_path_base + f"libGemmQnnAie_{self.kernel_x_shape[0]}x{self.kernel_x_shape[1]}_{self.kernel_y_shape[0]}x{self.kernel_y_shape[1]}.dll"
        if self.num_dlls == 2:
            self.dll_token_path = dll_path_base + f"libGemmQnnAie_{1}x{self.kernel_x_shape[1]}_{self.kernel_y_shape[0]}x{self.kernel_y_shape[1]}.dll"
        if self.num_dlls == 2:
            self.aiegemm = RyzenAI.qlinear(
                [self.dll_token_path, self.dll_path],
                [(1, self.kernel_x_shape[1]), self.kernel_x_shape],
                self.kernel_y_shape,
                self.num_workers, self.num_dlls, 4, self.pack_weights,
                f"./logs/log_aiegemm_cpp.csv")
        elif self.num_dlls ==1:
            self.aiegemm = RyzenAI.qlinear(
                [self.dll_path],
                [self.kernel_x_shape],
                self.kernel_y_shape,
                self.num_workers, self.num_dlls, 4, self.pack_weights,
                f"./logs/log_aiegemm_cpp.csv")
        else:
            print("Unsupported dll config")
            raise SystemExit

    def init_aiegemm_v1(self) -> None:
        if self.quant_mode == "w8a16":
            self.aiegemm = RyzenAI.qlinear_2_a16w8acc64("int16", "int8", "int64")
        else:
            self.aiegemm = RyzenAI.qlinear_2_a8w8acc32("int8", "int8", "int32")
        
    
    def init_aiegemm(self) -> None:
        if self.impl == "v1":
            self.init_aiegemm_v1()
        elif self.impl == "v0":
            self.init_aiegemm_v0()

    def __repr__(self):
        if self.bias is None:
            return f"ryzenAI.QLinear(in_features:{self.in_features}, out_features:{self.out_features}, bias:None, device:{self.device}, quant_mode:{self.quant_mode} kernel_x_shape:{self.kernel_x_shape}, kernel_y_shape:{self.kernel_y_shape} )"
        else:
            return f"ryzenAI.QLinear(in_features:{self.in_features}, out_features:{self.out_features}, bias:{self.bias.shape}, device:{self.device}, quant_mode:{self.quant_mode} kernel_x_shape:{self.kernel_x_shape}, kernel_y_shape:{self.kernel_y_shape} )"

    def quantize_weights(self):
        if (self.quant_mode == "w8a8") or (self.quant_mode == "w8a16"):
            self.weight_q = torch.int_repr(
                self.weight_bias[0]).numpy().astype(np.int8)
            self.weight_q = np.ascontiguousarray(self.weight_q.transpose())
            self.y_scale = np.array(self.weight_bias[0].q_scale(), dtype=np.float32)
            if self.weight_bias[1] is not None:
                if self.weight_bias[1].data.dtype == torch.bfloat16:
                    self.bias = self.weight_bias[1].data.to(torch.float32).numpy().astype(np.float32)
                else:
                    self.bias = self.weight_bias[1].data.numpy().astype(np.float32)
            else:
                self.bias = None
            self.aiegemm.initialize_weights(self.weight_q)
            self.wshape = self.weight_q.shape
            del self.weight_q, self.weight_bias

            self.c_fp = np.empty((1,self.out_features), dtype=np.float32)
            self.c_token = np.zeros((1,self.out_features), dtype=self.accum_type)
            self.x_abs = np.empty((1,self.in_features), dtype=np.float32)
            self.x_round = np.empty((1,self.in_features), dtype=np.float32)
            self.x_scaled = np.empty((1,self.in_features), dtype=np.float32)
            self.x_clip  = np.empty((1,self.in_features), dtype=np.float32)
            self.x_max = np.array(1.0, dtype=np.float32)

    def forward_prefill(self, x):
        x_scale = np.max(np.fabs(x)) / self.x_scaled_abs_max_fp
        x = np.clip(np.round(x/x_scale), self.x_scaled_min , self.x_scaled_max).astype(self.act_type, copy=False)

        c = np.zeros((x.shape[0], self.out_features), dtype=self.accum_type)
        self.aiegemm.execute(x, c)
        y = c.astype(np.float32, copy=False) * (x_scale * self.y_scale)
        if self.bias is not None:
            y = y + self.bias 
        return y

    def forward_token(self, x):
        self.c_token.fill(0)
        
        np.abs(x, out=self.x_abs)
        np.divide(np.max(self.x_abs), self.x_scaled_abs_max_fp, out=self.x_scale)
        np.divide(x, self.x_scale, out=self.x_scaled)
        np.clip(np.round(self.x_scaled, out=self.x_round), self.x_scaled_min , self.x_scaled_max, out=self.x_clip)
        
        self.aiegemm.execute(self.x_clip.astype(self.act_type, copy=False), self.c_token)
        
        np.multiply(self.c_token.astype(np.float32, copy=False), np.multiply(self.x_scale, self.y_scale), out=self.c_fp)
        if self.bias is not None:
            np.add(self.c_fp, self.bias, out=self.c_fp)
        return self.c_fp
        
    
    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.squeeze(0)
            has_batch = True
        else: has_batch = False
        y = self.forward_dict[x.shape[0]](x.numpy())  
        if has_batch is False:
            return torch.from_numpy(y)
        else:
            return torch.from_numpy(np.expand_dims(y, axis=0))
            
class QLinearPerGrp(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int = 1, out_features: int = 1, bias: bool = True, device=None, 
                 w_bit:int=3, group_size:int=128, profiler:bool=False) -> None:
        factory_kwargs = {'device': 'cpu', 'dtype': None}
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.weight = None
        self.qweight = None
        self.qzeros = None
        self.scales = None
        if bias is True:
            self.bias = torch.empty(out_features, **factory_kwargs)
            self.biasshape = self.bias.shape
        else:
            self.bias = None
            self.biasshape = "None"
        
        
    def __repr__(self):
        return f"ryzenAI.QLinearPerGrp(in_features:{self.in_features}, out_features:{self.out_features}, bias:{self.biasshape}, device:{self.device}, w_bit:{self.w_bit} group_size:{self.group_size}  )"

    @torch.no_grad()
    def quantize_weights(self):
        if self.bias is not None:
            self.bias = self.bias.to(torch.bfloat16).to(torch.float32)
            self.biasshape = self.bias.shape
        else: 
            self.biasshape = "None"
        if (self.qweight is None) and (self.weight is not None):
            self.w_shape_orig = self.weight.shape
            w = self.weight.reshape(-1, self.group_size)

            # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            
            # Calculate the scale factor and zero point. 
            max_int = 2 ** self.w_bit - 1
            self.scales = ((max_val - min_val).clamp(min=1e-5) / max_int).to(torch.bfloat16)
            assert self.scales.shape == max_val.shape
            self.qzeros = (-torch.round(min_val / self.scales)).clamp_(0, max_int).to(torch.int8)
            assert self.scales.shape == min_val.shape

            assert torch.isnan(self.scales).sum() == 0
            assert torch.isnan(w).sum() == 0

            # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
            self.qweight = torch.clamp(torch.round(w / self.scales) + self.qzeros, 0, max_int).to(torch.int8)
            
            assert self.qweight.dim() == 2 and self.qweight.size(0) == self.scales.size(0) and self.qweight.size(1) == self.group_size

            self.qweight = self.qweight.reshape(self.w_shape_orig)
            self.qzeros = self.qzeros.reshape((self.w_shape_orig[0], int(self.w_shape_orig[1]/self.group_size))).to(torch.int8)
            self.scales = self.scales.reshape((self.w_shape_orig[0], int(self.w_shape_orig[1]/self.group_size)))
            del self.weight, max_val, min_val, w

        self.weight = (self.qweight - torch.repeat_interleave(self.qzeros, self.group_size, dim=1))
        self.weight = self.weight * torch.repeat_interleave(self.scales, self.group_size, dim=1) 
        self.wshape = self.weight.shape

        self.weight = self.weight.transpose(0,1).to(torch.float)
        self.qweight = self.qweight.transpose(0,1)
        self.qzeros = self.qzeros.transpose(0,1)
        self.scales = self.scales.transpose(0,1)
        if self.bias is None:
            self.bias = torch.zeros((self.wshape[0]), dtype=torch.float32)
        if self.device == "aie":
            if self.w_bit == 3:
                self.aiegemm = RyzenAI.qlinear_2_a16fw4acc32f("bfloat16", "int4", "float32")
            else:
                self.aiegemm = RyzenAI.qlinear_2_a16fw4acc32f("bfloat16", "uint4", "float32")
            if not os.path.exists("./logs"):
                os.makedirs("./logs")
            self.aiegemm.initialize_weights(self.qweight.numpy(), self.qzeros.numpy(), self.scales.to(torch.float).numpy(), self.bias.to(torch.float).numpy(), self.group_size)
            self.c_token = np.zeros((1, self.out_features), dtype=np.float32)
            del self.qweight, self.qzeros, self.scales, self.weight, self.bias
        elif self.device == "aie_emu":
            del self.weight
        elif self.device == "cpu":
            del self.qweight, self.qzeros, self.scales 

        if self.device == "aie":
            self.forward_func = self.forward_aie
            self.forward_dict_aie = defaultdict(lambda: self.forward_aie_prefill, {1: self.forward_aie_token})
        elif self.device == "aie_emu":
            self.forward_func = self.forward_aie_emu
        else:
            self.forward_func = self.forward_cpu

        gc.collect()
        
    def forward_cpu(self, x: Tensor) -> Tensor:
        x = x.to(torch.bfloat16).to(torch.float)
        x = torch.matmul(x, self.weight )
        if self.bias is not None:
            x = x + self.bias
        return x

    def forward_aie_emu(self, x: Tensor) -> Tensor:
        """ Code emulating AIE kernel call with exact loopnests 
        test_qlinear_pergrp verifies the functionality of forward_op() 
        with forward_awq() call. 
        forward_awq() call is used to measure perplexity on CPU with AWQ"""
        c = torch.empty((x.shape[0], self.qweight.shape[1]), dtype=torch.float)
        x = x.to(torch.bfloat16)
        num_grps = int(x.shape[1]/self.group_size)
                    
        for rx in range(x.shape[0]):
            for cw in range(self.qweight.shape[1]):
                if self.bias is not None:
                    acc = self.bias[cw].clone().detach().to(torch.float) #torch.tensor(self.bias[cw], dtype=torch.float)
                else:
                    acc = torch.tensor(0.0, dtype=torch.float)
                for g in range(num_grps):
                    x_grp = x[rx, g*self.group_size:((g+1)*self.group_size)]
                    qw_grp = self.qweight[g*self.group_size:((g+1)*self.group_size), cw]
                    qz_grp = self.qzeros[g, cw]
                    sc_grp = self.scales[g, cw]
                    w_grp = (qw_grp - qz_grp).to(torch.bfloat16) * sc_grp
                    acc += torch.dot(x_grp.to(torch.float), w_grp.to(torch.float))
                c[rx][cw] = acc
        return c 

    def forward_aie_token(self, x: Tensor) -> Tensor:
        xv_i16 = x.view(torch.int16)
        self.aiegemm.execute(xv_i16, self.c_token)
        return torch.from_numpy(self.c_token)
    
    def forward_aie_prefill(self, x: Tensor) -> Tensor:
        c = torch.empty((x.shape[0], self.wshape[0]), dtype=torch.float).numpy()
        self.aiegemm.execute(x.view(torch.int16), c)
        return torch.from_numpy(c)

    def forward_aie(self, x: Tensor) -> Tensor:
        return self.forward_dict_aie[x.shape[0]](x)

    def forward(self, x: Tensor) -> Tensor:
        #print(f"shapes: {x.shape} {self.weight_q.shape} {self.wshape}")
        if len(x.shape) == 3:
            x = x.squeeze(0)
            has_batch = True
        else: has_batch = False
        y = self.forward_func(x)
        if has_batch is False:
            return y.to(torch.bfloat16)
        else:
            return y.unsqueeze(0).to(torch.bfloat16)