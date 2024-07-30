#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import gc
import logging
import math
import os
import platform
import time
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

torch.random.manual_seed(123)

import onnxruntime
import RyzenAI


class QLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = True,
        device=None,
        x_scale: float = 1,
        y_scale: float = 1,
        quant_mode="w8a8",
        profiler: bool = False,
        dtype="float32",
    ) -> None:
        factory_kwargs = {"device": "cpu", "dtype": None}
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
        if self.quant_mode == "w8a16" and self.dev != "stx":
            print(f"{self.quant_mode} is not supported on {self.dev} platform")
            raise SystemExit
        if bias is True:
            self.bias = torch.empty(out_features, **factory_kwargs)
        else:
            self.register_parameter("bias", None)
            self.bias = None
        self.x_scale = np.array(x_scale, dtype=np.float32)
        self.y_scale = np.array(y_scale, dtype=np.float32)
        self.profiler = profiler
        if self.profiler:
            self.aie_time_execute_start = 0
            self.aie_time_execute_end = 0
            self.aie_compile_time_start = 0
            self.aie_compile_time_end = 0
            self.quant_time_start = 0
            self.quant_time_end = 0
            self.dequant_time_start = 0
            self.dequant_time_end = 0
            self.pre_time_start = 0
            self.pre_time_end = 0
            self.post_time_start = 0
            self.post_time_end = 0
            self.exec_pybind_time_start = 0
            self.exec_pybind_time_end = 0
            self.exec_c_time_start = 0
            self.exec_c_time_end = 0
            self.bias_add_time_start = 0
            self.bias_add_time_end = 0
        if self.device == "aie":
            if self.profiler:
                self.aie_compile_time_start = time.perf_counter()
            if not os.path.exists("./logs"):
                os.makedirs("./logs")
            self.init_aiegemm()
            if self.profiler:
                self.aie_compile_time_end = time.perf_counter()

        self.forward_dict = defaultdict(
            lambda: self.forward_prefill, {1: self.forward_token}
        )

    def init_aiegemm(self) -> None:
        if self.quant_mode == "w8a16":
            self.aiegemm = RyzenAI.qlinear_2_a16w8acc64("int16", "int8", "int64")
        else:
            self.aiegemm = RyzenAI.qlinear_2_a8w8acc32("int8", "int8", "int32")

    def __repr__(self):
        return f"ryzenAI.QLinear(in_features:{self.in_features}, out_features:{self.out_features}, bias:{True if self.bias is not None else False}, device:{self.device}, quant_mode:{self.quant_mode} )"

    def quantize_weights(self):
        if (self.quant_mode == "w8a8") or (self.quant_mode == "w8a16"):
            self.weight_q = torch.int_repr(self.weight_bias[0]).numpy().astype(np.int8)
            self.weight_q = np.ascontiguousarray(self.weight_q.transpose())
            self.y_scale = np.array(self.weight_bias[0].q_scale(), dtype=np.float32)
            if self.weight_bias[1] is not None:
                if self.weight_bias[1].data.dtype == torch.bfloat16:
                    self.bias = (
                        self.weight_bias[1]
                        .data.to(torch.float32)
                        .numpy()
                        .astype(np.float32)
                    )
                else:
                    self.bias = self.weight_bias[1].data.numpy().astype(np.float32)
            else:
                self.bias = None
            self.aiegemm.initialize_weights(self.weight_q)
            self.wshape = self.weight_q.shape
            del self.weight_q, self.weight_bias

            self.c_fp = np.empty((1, self.out_features), dtype=np.float32)
            self.c_token = np.zeros((1, self.out_features), dtype=self.accum_type)
            self.x_abs = np.empty((1, self.in_features), dtype=np.float32)
            self.x_round = np.empty((1, self.in_features), dtype=np.float32)
            self.x_scaled = np.empty((1, self.in_features), dtype=np.float32)
            self.x_clip = np.empty((1, self.in_features), dtype=np.float32)
            self.x_max = np.array(1.0, dtype=np.float32)

    def forward_prefill(self, x):
        x_scale = np.max(np.fabs(x)) / self.x_scaled_abs_max_fp
        x = np.clip(np.round(x / x_scale), self.x_scaled_min, self.x_scaled_max).astype(
            self.act_type, copy=False
        )

        c = np.zeros((x.shape[0], self.out_features), dtype=self.accum_type)
        self.aiegemm.execute(x, c)
        y = c.astype(np.float32, copy=False) * (x_scale * self.y_scale)
        if self.bias is not None:
            y = y + self.bias
        return y

    def forward_token(self, x):
        if self.profiler:
            self.exec_c_time_start = time.perf_counter()
        self.c_token.fill(0)
        if self.profiler:
            self.exec_c_time_end = time.perf_counter()

        if self.profiler:
            self.quant_time_start = time.perf_counter()
        np.abs(x, out=self.x_abs)
        np.divide(np.max(self.x_abs), self.x_scaled_abs_max_fp, out=self.x_scale)
        np.divide(x, self.x_scale, out=self.x_scaled)
        np.clip(
            np.round(self.x_scaled, out=self.x_round),
            self.x_scaled_min,
            self.x_scaled_max,
            out=self.x_clip,
        )
        if self.profiler:
            self.quant_time_end = time.perf_counter()

        if self.profiler:
            self.exec_pybind_time_start = time.perf_counter()
        self.aiegemm.execute(
            self.x_clip.astype(self.act_type, copy=False), self.c_token
        )
        if self.profiler:
            self.exec_pybind_time_end = time.perf_counter()

        if self.profiler:
            self.dequant_time_start = time.perf_counter()
        np.multiply(
            self.c_token.astype(np.float32, copy=False),
            np.multiply(self.x_scale, self.y_scale),
            out=self.c_fp,
        )
        if self.profiler:
            self.dequant_time_end = time.perf_counter()
        if self.profiler:
            self.bias_add_time_start = time.perf_counter()
        if self.bias is not None:
            np.add(self.c_fp, self.bias, out=self.c_fp)
        if self.profiler:
            self.bias_add_time_end = time.perf_counter()
        return self.c_fp

    def forward(self, x: Tensor) -> Tensor:
        if self.profiler:
            self.aie_time_execute_start = time.perf_counter()
        if len(x.shape) == 3:
            x = x.squeeze(0)
            has_batch = True
        else:
            has_batch = False
        y = self.forward_dict[x.shape[0]](x.numpy())
        if self.profiler:
            self.aie_time_execute_end = time.perf_counter()
            logging.critical(
                f"[PROFILE][AIE] {x.shape[0]} {x.shape[1]} {self.wshape[0]} {self.wshape[1]} {self.aie_compile_time_start} {self.aie_compile_time_end} {self.pre_time_start} {self.pre_time_end} {self.quant_time_start} {self.quant_time_end} {self.aie_time_execute_start} {self.aie_time_execute_end} {self.dequant_time_start} {self.dequant_time_end} {self.post_time_start} {self.post_time_end} {self.exec_pybind_time_start} {self.exec_pybind_time_end} {self.exec_c_time_start} {self.exec_c_time_end} {self.bias_add_time_start} {self.bias_add_time_end}"
            )
        if has_batch is False:
            return torch.from_numpy(y)
        else:
            return torch.from_numpy(np.expand_dims(y, axis=0))


class QLinearPerGrpWithProfile(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = True,
        device=None,
        w_bit: int = 3,
        group_size: int = 128,
        profiler: bool = False,
    ) -> None:
        factory_kwargs = {"device": "cpu", "dtype": None}
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
        self.profiler = profiler
        if self.profiler:
            self.aie_time_execute_start = 0
            self.aie_time_execute_end = 0
            self.aie_compile_time_start = 0
            self.aie_compile_time_end = 0
            self.quant_time_start = 0
            self.quant_time_end = 0
            self.dequant_time_start = 0
            self.dequant_time_end = 0
            self.pre_time_start = 0
            self.pre_time_end = 0
            self.post_time_start = 0
            self.post_time_end = 0
            self.exec_pybind_time_start = 0
            self.exec_pybind_time_end = 0
            self.exec_c_time_start = 0
            self.exec_c_time_end = 0
            self.bias_add_time_start = 0
            self.bias_add_time_end = 0
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
            self.bias.data = self.bias.to(torch.bfloat16).to(torch.float32)
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
            max_int = 2**self.w_bit - 1
            self.scales = ((max_val - min_val).clamp(min=1e-5) / max_int).to(
                torch.bfloat16
            )
            assert self.scales.shape == max_val.shape
            self.qzeros = (
                (-torch.round(min_val / self.scales)).clamp_(0, max_int).to(torch.int8)
            )
            assert self.scales.shape == min_val.shape

            assert torch.isnan(self.scales).sum() == 0
            assert torch.isnan(w).sum() == 0

            # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
            self.qweight = torch.clamp(
                torch.round(w / self.scales) + self.qzeros, 0, max_int
            ).to(torch.int8)

            assert (
                self.qweight.dim() == 2
                and self.qweight.size(0) == self.scales.size(0)
                and self.qweight.size(1) == self.group_size
            )

            self.qweight = self.qweight.reshape(self.w_shape_orig)
            self.qzeros = self.qzeros.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            ).to(torch.int8)
            self.scales = self.scales.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            )
            del self.weight, max_val, min_val, w

        self.weight = self.qweight - torch.repeat_interleave(
            self.qzeros, self.group_size, dim=1
        )
        self.weight = self.weight * torch.repeat_interleave(
            self.scales, self.group_size, dim=1
        )
        self.wshape = self.weight.shape

        self.weight = self.weight.transpose(0, 1).to(torch.float)
        self.qweight = self.qweight.transpose(0, 1)
        self.qzeros = self.qzeros.transpose(0, 1)
        self.scales = self.scales.transpose(0, 1)
        if self.bias is None:
            self.bias = torch.zeros((self.wshape[0]), dtype=torch.float32)
        if self.device == "aie":
            if self.profiler:
                self.aie_compile_time_start = time.perf_counter()
            if self.w_bit == 3:
                if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc16f(
                        "bfloat16", "int4", "bfloat16"
                    )
                else:
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc32fo16f(
                        "bfloat16", "int4", "float32"
                    )
            else:
                if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc16f(
                        "bfloat16", "uint4", "bfloat16"
                    )
                else:
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc32fo16f(
                        "bfloat16", "uint4", "float32"
                    )
            if self.profiler:
                self.aie_compile_time_end = time.perf_counter()
            if not os.path.exists("./logs"):
                os.makedirs("./logs")
            self.aiegemm.initialize_weights(
                self.qweight.numpy(),
                self.qzeros.numpy(),
                self.scales.to(torch.float).numpy(),
                self.bias.to(torch.float).numpy(),
                self.group_size,
            )
            # self.c_token = torch.zeros(1, self.out_features, dtype=torch.float32)
            self.c_token = torch.zeros(1, self.out_features, dtype=torch.bfloat16)
            del self.qweight, self.qzeros, self.scales, self.weight, self.bias
        elif self.device == "aie_emu":
            del self.weight
        elif self.device == "cpu":
            del self.qweight, self.qzeros, self.scales

        if self.device == "aie":
            self.forward_func = self.forward_aie
            self.forward_dict_aie = defaultdict(
                lambda: self.forward_aie_prefill, {1: self.forward_aie_token}
            )
        elif self.device == "aie_emu":
            self.forward_func = self.forward_aie_emu
        else:
            self.forward_func = self.forward_cpu

        gc.collect()

    def forward_cpu(self, x: Tensor) -> Tensor:
        x = x.to(torch.bfloat16).to(torch.float)
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def forward_aie_emu(self, x: Tensor) -> Tensor:
        """Code emulating AIE kernel call with exact loopnests
        test_qlinear_pergrp verifies the functionality of forward_op()
        with forward_awq() call.
        forward_awq() call is used to measure perplexity on CPU with AWQ"""
        c = torch.empty((x.shape[0], self.qweight.shape[1]), dtype=torch.float)
        x = x.to(torch.bfloat16)
        num_grps = int(x.shape[1] / self.group_size)

        for rx in range(x.shape[0]):
            for cw in range(self.qweight.shape[1]):
                if self.bias is not None:
                    acc = (
                        self.bias[cw].clone().detach().to(torch.float)
                    )  # torch.tensor(self.bias[cw], dtype=torch.float)
                else:
                    acc = torch.tensor(0.0, dtype=torch.float)
                for g in range(num_grps):
                    x_grp = x[rx, g * self.group_size : ((g + 1) * self.group_size)]
                    qw_grp = self.qweight[
                        g * self.group_size : ((g + 1) * self.group_size), cw
                    ]
                    qz_grp = self.qzeros[g, cw]
                    sc_grp = self.scales[g, cw]
                    w_grp = (qw_grp - qz_grp).to(torch.bfloat16) * sc_grp
                    acc += torch.dot(x_grp.to(torch.float), w_grp.to(torch.float))
                c[rx][cw] = acc
        return c

    def forward_aie_token(self, x: Tensor) -> Tensor:
        if self.profiler:
            self.exec_pybind_time_start = time.perf_counter()
        self.aiegemm.execute(x.view(torch.int16), self.c_token.view(torch.int16))
        if self.profiler:
            self.exec_pybind_time_end = time.perf_counter()
        return self.c_token

    def forward_aie_prefill(self, x: Tensor) -> Tensor:
        if self.profiler:
            self.exec_c_time_start = time.perf_counter()
        # c = torch.empty((x.shape[0], self.wshape[0]), dtype=torch.float)
        c = torch.empty((x.shape[0], self.wshape[0]), dtype=torch.bfloat16)
        if self.profiler:
            self.exec_c_time_end = time.perf_counter()

        if self.profiler:
            self.exec_pybind_time_start = time.perf_counter()
        self.aiegemm.execute(x.view(torch.int16), c.view(torch.int16))
        if self.profiler:
            self.exec_pybind_time_end = time.perf_counter()
        return c

    def forward_aie(self, x: Tensor) -> Tensor:
        return self.forward_dict_aie[x.shape[0]](x)

    def forward(self, x: Tensor) -> Tensor:
        if self.profiler:
            self.aie_time_execute_start = time.perf_counter()
        if len(x.shape) == 3:
            has_batch = True
        else:
            x = x.unsqueeze(0)
            has_batch = False
        # y = torch.empty((x.shape[0], x.shape[1], self.wshape[0]))
        y = torch.empty((x.shape[0], x.shape[1], self.wshape[0]), dtype=torch.bfloat16)
        for i in range(x.shape[0]):
            y[i] = self.forward_func(x[i])
        # print(f"[QLinearPerGrp] {x.shape} {self.wshape} {y.shape}")
        if self.profiler:
            self.aie_time_execute_end = time.perf_counter()
            logging.critical(
                f"[PROFILE][AIE] {x.shape[0]} {x.shape[1]} {self.wshape[0]} {self.wshape[1]} {self.aie_compile_time_start} {self.aie_compile_time_end} {self.pre_time_start} {self.pre_time_end} {self.quant_time_start} {self.quant_time_end} {self.aie_time_execute_start} {self.aie_time_execute_end} {self.dequant_time_start} {self.dequant_time_end} {self.post_time_start} {self.post_time_end} {self.exec_pybind_time_start} {self.exec_pybind_time_end} {self.exec_c_time_start} {self.exec_c_time_end} {self.bias_add_time_start} {self.bias_add_time_end}"
            )
        if has_batch is False:
            # return y.squeeze(0).to(torch.bfloat16)
            return y.squeeze(0)
        else:
            # return y.to(torch.bfloat16)
            return y


class QLinearPerGrp(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = True,
        device=None,
        w_bit: int = 3,
        group_size: int = 128,
        profiler: bool = False,
    ) -> None:
        factory_kwargs = {"device": "cpu", "dtype": None}
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
        self.profiler = profiler
        if bias is True:
            self.bias = torch.empty(out_features, **factory_kwargs)
            self.biasshape = self.bias.shape
        else:
            self.bias = None
            self.biasshape = "None"

    def __repr__(self):
        return f"ryzenAI.QLinearPerGrp(in_features:{self.in_features}, out_features:{self.out_features}, bias:{self.biasshape}, device:{self.device}, w_bit:{self.w_bit} group_size:{self.group_size}  )"

    @torch.no_grad()
    def prune_weights(self):
        self.forward_func = self.forward_cpu
        self.wshape = self.weight.shape
        w = self.weight.data.transpose(0, 1).to(torch.bfloat16)
        del self.weight
        self.weight = w.clone()
        del w
        if self.bias is not None:
            self.bias = self.bias.to(torch.bfloat16)

    @torch.no_grad()
    def quantize_weights(self):
        if self.bias is not None:
            self.bias.data = self.bias.to(torch.bfloat16).to(torch.float32)
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
            max_int = 2**self.w_bit - 1
            self.scales = ((max_val - min_val).clamp(min=1e-5) / max_int).to(
                torch.bfloat16
            )
            assert self.scales.shape == max_val.shape
            self.qzeros = (
                (-torch.round(min_val / self.scales)).clamp_(0, max_int).to(torch.int8)
            )
            assert self.scales.shape == min_val.shape

            assert torch.isnan(self.scales).sum() == 0
            assert torch.isnan(w).sum() == 0

            # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
            self.qweight = torch.clamp(
                torch.round(w / self.scales) + self.qzeros, 0, max_int
            ).to(torch.int8)

            assert (
                self.qweight.dim() == 2
                and self.qweight.size(0) == self.scales.size(0)
                and self.qweight.size(1) == self.group_size
            )

            self.qweight = self.qweight.reshape(self.w_shape_orig)
            self.qzeros = self.qzeros.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            ).to(torch.int8)
            self.scales = self.scales.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            )
            del self.weight, max_val, min_val, w

        self.weight = self.qweight - torch.repeat_interleave(
            self.qzeros, self.group_size, dim=1
        )
        self.weight = self.weight * torch.repeat_interleave(
            self.scales, self.group_size, dim=1
        )
        self.wshape = self.weight.shape

        self.weight = self.weight.transpose(0, 1).to(torch.float)
        self.qweight = self.qweight.transpose(0, 1)
        self.qzeros = self.qzeros.transpose(0, 1)
        self.scales = self.scales.transpose(0, 1)
        if self.bias is None:
            self.bias = torch.zeros((self.wshape[0]), dtype=torch.float32)
        if (self.device == "aie") or (self.device == "npugpu"):
            if self.w_bit == 3:
                if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc16f(
                        "bfloat16", "int4", "bfloat16"
                    )
                else:
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc32fo16f(
                        "bfloat16", "int4", "float32"
                    )
            else:
                if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc16f(
                        "bfloat16", "uint4", "bfloat16"
                    )
                else:
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc32fo16f(
                        "bfloat16", "uint4", "float32"
                    )
            if not os.path.exists("./logs"):
                os.makedirs("./logs")
            self.aiegemm.initialize_weights(
                self.qweight.numpy(),
                self.qzeros.numpy(),
                self.scales.to(torch.float).numpy(),
                self.bias.to(torch.float).numpy(),
                self.group_size,
            )
            self.c_token = torch.zeros(1, self.out_features, dtype=torch.bfloat16)
            del (
                self.qweight,
                self.qzeros,
                self.scales,
            )
            del self.weight, self.bias
            if self.device == "npugpu":
                self.session = None

        elif self.device == "cpu":
            # del self.qweight, self.qzeros, self.scales
            self.weight = self.weight.to(torch.bfloat16)
            if self.bias is not None:
                self.bias = self.bias.to(torch.bfloat16)

        if self.device == "aie":
            self.forward_func = self.forward_aie
            self.forward_dict_aie = defaultdict(
                lambda: self.forward_aie_prefill, {1: self.forward_aie_token}
            )
        elif self.device == "npugpu":
            self.forward_func = self.forward_aie
            self.forward_dict_aie = defaultdict(
                lambda: self.forward_aie_prefill, {1: self.forward_gpu_token}
            )
        else:
            self.forward_func = self.forward_cpu

        gc.collect()

    def forward_cpu(self, x: Tensor) -> Tensor:
        x = x.to(torch.bfloat16)
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def forward_gpu_token(self, x: Tensor) -> Tensor:
        y = x.to(torch.float16).numpy()
        y = self.session.run(None, {"input": y})
        y = torch.from_numpy(y[0]).to(torch.bfloat16)
        return y

    def forward_aie_token(self, x: Tensor) -> Tensor:
        self.aiegemm.execute(x.view(torch.int16), self.c_token.view(torch.int16))
        # if self.device == "npugpu": print("You aren't using GPU in token phase")
        return self.c_token

    def forward_aie_prefill(self, x: Tensor) -> Tensor:
        c = torch.empty((x.shape[0], self.wshape[0]), dtype=torch.bfloat16)
        self.aiegemm.execute(x.view(torch.int16), c.view(torch.int16))
        return c

    def forward_aie(self, x: Tensor) -> Tensor:
        return self.forward_dict_aie[x.shape[0]](x)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            has_batch = True
        else:
            x = x.unsqueeze(0)
            has_batch = False
        y = torch.empty((x.shape[0], x.shape[1], self.wshape[0]), dtype=torch.bfloat16)
        for i in range(x.shape[0]):
            y[i] = self.forward_func(x[i])
        if has_batch is False:
            return y.squeeze(0)
        else:
            return y
