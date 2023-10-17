import logging
from typing import Tuple
from torch.nn.parameter import Parameter
from torch import Tensor
import os
import math
import numpy as np
import time
import platform


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
                 x_scale: float = 1, y_scale: float = 1, quant_mode: int = 1,
                 num_dlls: int = 2, dtype="float32", profiler = False,
                 kernel_x_shape: Tuple = (8, 2048), kernel_y_shape: Tuple = (2048, 2048), num_workers: int = 2, pack_weights=True) -> None:
        factory_kwargs = {'device': 'cpu', 'dtype': None}
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        if (self.quant_mode == 1):
            self.weight = None
        else:
            self.weight = Parameter(
                torch.empty(
                    (out_features, in_features), **factory_kwargs))
        self.weight_q = None
        if bias is True:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.bias = None
        self.kernel_x_shape = kernel_x_shape
        self.kernel_y_shape = kernel_y_shape
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.num_dlls = num_dlls
        self.num_workers = num_workers
        self.pack_weights = pack_weights
        dev = os.getenv("DEVICE")
        if dev is None:
            print("DEVICE environment variable is not set")
            raise SystemExit
        dll_path_base =  os.getenv("PYTORCH_AIE_PATH") + "/dll/" + dev + "/qlinear/"
        self.dll_path = dll_path_base + f"libGemmQnnAie_{self.kernel_x_shape[0]}x{self.kernel_x_shape[1]}_{self.kernel_y_shape[0]}x{self.kernel_y_shape[1]}.dll"
        if self.num_dlls == 2:
            self.dll_token_path = dll_path_base + f"libGemmQnnAie_{1}x{self.kernel_x_shape[1]}_{self.kernel_y_shape[0]}x{self.kernel_y_shape[1]}.dll"
        if self.device == 'aie':
            if not os.path.exists("./logs"):
                os.makedirs("./logs")
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
        
        if self.quant_mode == None:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        if self.bias is None:
            return f"ryzenAI.QLinear(in_features:{self.in_features}, out_features:{self.out_features}, bias:None, device:{self.device}, quant_mode:{self.quant_mode} kernel_x_shape:{self.kernel_x_shape}, kernel_y_shape:{self.kernel_y_shape} )"
        else:
            return f"ryzenAI.QLinear(in_features:{self.in_features}, out_features:{self.out_features}, bias:{self.bias.shape}, device:{self.device}, quant_mode:{self.quant_mode} kernel_x_shape:{self.kernel_x_shape}, kernel_y_shape:{self.kernel_y_shape} )"

    def quantize_weights(self):
        if (self.quant_mode == 1):
            self.weight_q = torch.int_repr(
                self.weight_bias[0]).numpy().astype(
                np.int8)
            self.weight_q = np.ascontiguousarray(self.weight_q.transpose())
            self.y_scale = self.weight_bias[0].q_scale()
            if self.weight_bias[1] is not None:
                self.bias = torch.nn.Parameter(self.weight_bias[1])
            else:
                self.bias = None
            self.weight_bias = None
            self.wshape = self.weight_q.shape
            self.aiegemm.initialize_weights(self.weight_q)
            del self.weight_q
        else:  # self.device=='cpu'
            self.wshape = self.weight.data.transpose(0,1).shape

    def execute(self, a):
        c = np.zeros((a.shape[0], self.out_features)).astype(np.int32)
        self.aiegemm.execute(a, c)
        return c

    def forward_quant_mode_1(self, x: Tensor) -> Tensor:
        # quantize inputs
        self.x_scale = x.abs().max() / 128.0
        x = x.div(self.x_scale).round_().clip_(-128, 127)

        # quantized matmul in aie
        if self.dtype == "bfloat16":
            x = x.to(torch.float)
        x = torch.tensor(self.execute(x.numpy().astype(np.int8)), dtype=torch.float)
        
        # dequantize
        x.mul_(self.x_scale * self.y_scale)        
        return x 
    
    def forward(self, x: Tensor) -> Tensor:
        has_batch = True
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            has_batch = False
        y = torch.empty(x.shape[0], x.shape[1], self.wshape[1], dtype=torch.float)
        for i in range(x.shape[0]):
            y[i, :, :] = self.forward_quant_mode_1(x[i, :, :])
        if self.bias is not None:
            y = y + self.bias
        if has_batch is False:
            y = y[0, :, :]
        if self.dtype == "bfloat16":
            return y.to(torch.bfloat16)
        else: return y

#################################################################################  
#License
#Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
