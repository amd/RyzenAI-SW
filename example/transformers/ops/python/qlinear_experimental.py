#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import gc
import math
import os
import platform
import time

import numpy as np
import torch
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.target import versal_aie
from tvm.relay import transform
from tvm.relay.op.contrib.versal_aie import partition_for_aie

torch.random.manual_seed(123)
import logging
from typing import Tuple

import RyzenAI
import ryzenai_torch_cpp
from torch import Tensor
from torch.nn.parameter import Parameter


class QLinearExperimentalPyTiling(torch.nn.Module):
    """
    AIE _ CPU implementation of following steps
    1.      FP32 -> inputs and weights
    2. CPU: int8 Quantization of Weights at compile time
    3. CPU: int8 quantization of inputs at dynamic runtime
    4. AIE: int8 * int8 = int32 multiplication of GEMM
    5. AIE: int32 -> int16 requantize
    6. CPU: int16 -> FP32 dequantize
    7. CPU: GEMM + FP32 Bias Add
    """

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
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        quant_mode=0,
        use_tvm_dll: bool = True,
        profiler: bool = False,
        dtype=None,
        kernel_x_shape: Tuple = (8, 2048),
        kernel_y_shape: Tuple = (2048, 2048),
        pack_weights=True,
    ) -> None:
        factory_kwargs = {"device": "cpu", "dtype": None}
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = None
        if self.quant_mode == 1:
            self.weight = None
        else:
            self.weight = Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
        self.weight_q = None
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.kernel_x_shape = kernel_x_shape
        self.kernel_y_shape = kernel_y_shape
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.profiler = profiler
        self.pack_weights = pack_weights
        lib_extension = "dll" if platform.system() == "Windows" else "so"
        self.use_tvm_dll = use_tvm_dll
        dev = os.getenv("DEVICE")
        if dev is None:
            print("DEVICE environment variable is not set")
            raise SystemExit
        self.tvm_dll = (
            os.getenv("PYTORCH_AIE_PATH")
            + "/dll/"
            + dev
            + "/qlinear/"
            + f"libGemmQnnAie_{self.kernel_x_shape[0]}x{self.kernel_x_shape[1]}_{self.kernel_y_shape[0]}x{self.kernel_y_shape[1]}.{lib_extension}"
        )
        self.weight_packing = None
        if dev == "phx" and self.pack_weights == True:
            self.weight_packing = ["QN", "NO", "KO", "TK", "K0", "TN", "TDN", "N0"]
        if dev == "stx" and self.pack_weights == True:
            self.weight_packing = ["QN", "NO", "KO", "TK", "K0", "TN", "N0"]
        if self.quant_mode != 1:
            self.reset_parameters()
        self.weight_bias = None
        if self.profiler:
            self.aie_time_execute = 0
            self.quant_time = 0
            self.dequant_time = 0
            self.pre_time = 0
            self.post_rime = 0
        if self.device == "aie":
            self.quant_mode = quant_mode
            self.tvm_func = self._gemm_qnn_layer()
            self.graph_executor_aie = self._compile_aie()
        self.forward_dict = {
            0: self.forward_quant_mode_0,
            1: self.forward_quant_mode_1,
            None: self.forward_quant_mode_None,
        }

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # self.quantize_weights()
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"ryzenAI.QLinearExperimentalPyTiling(in_features:{self.in_features}, out_features:{self.out_features}, device:{self.device}, quant_mode:{self.quant_mode} kernel_x_shape:{self.kernel_x_shape}, kernel_y_shape:{self.kernel_y_shape} )"

    def quantize_weights(self):
        if self.quant_mode == 1:
            self.weight_q = (
                torch.int_repr(self.weight_bias[0]).numpy().astype(np.int32)
            )  # .transpose()
            self.y_scale = self.weight_bias[0].q_scale()
            if self.weight_bias[1] is not None:
                self.bias = torch.nn.Parameter(self.weight_bias[1])
            else:
                self.bias = None
            self.weight_q = np.ascontiguousarray(self.weight_q.transpose())
            self.weight_bias = None
        elif self.quant_mode == 0:
            self.weight_q = (1 / self.y_scale) * self.weight.data
            self.weight_q = torch.clip(torch.round(self.weight_q), -128, 127)
            self.weight_q = self.weight_q.detach().numpy()  # .transpose()
            self.weight_q = np.ascontiguousarray(self.weight_q.transpose())
        else:  # self.device=='cpu'
            pass

    def run(self, a, b):
        m = self.graph_executor_aie
        m.set_input("x", tvm.nd.array(a.astype("int8")))
        m.set_input("y", tvm.nd.array(b.astype("int8")))
        m.run()
        tvm_output = m.get_output(0)
        return tvm_output.numpy()

    def _gemm_qnn_layer(
        self,
        x_name="x",
        y_name="y",
        i_dtype="int8",
        o_dtype="int16",
    ):
        x_zp = 0
        y_zp = 0
        x = relay.var(x_name, shape=self.kernel_x_shape, dtype=i_dtype)
        y = relay.var(y_name, shape=self.kernel_y_shape, dtype=i_dtype)
        z = relay.qnn.op.aie.matmul(
            x,
            y,
            units=self.kernel_y_shape[1],
            out_dtype="int32",
            weight_packing=self.weight_packing,
        )

        func = relay.Function([x, y], z)
        return func

    def _get_xclbin_and_config_paths(self):
        dev = os.getenv("DEVICE")
        if dev is None:
            print("DEVICE environment variable is not set")
            raise SystemExit
        if dev == "phx":
            target_design = os.getenv("TARGET_DESIGN")
            if target_design == "ASR4x2":
                aie_target = "aieml-gemm-asr"
                xclbin = (
                    os.getenv("PYTORCH_AIE_PATH") + "/xclbin/phx/aieml_gemm_asr.xclbin"
                )
                aiectrl_json = (
                    os.getenv("PYTORCH_AIE_PATH") + "/xclbin/phx/aieml_gemm_asr.json"
                )
            else:
                aie_target = "aieml-gemm-vm-phx-4x4"
                xclbin = (
                    os.getenv("PYTORCH_AIE_PATH")
                    + "/xclbin/phx/aieml_gemm_vm_phx_4x4.xclbin"
                )
                aiectrl_json = (
                    os.getenv("PYTORCH_AIE_PATH")
                    + "/xclbin/phx/aieml_gemm_vm_phx_4x4.json"
                )
            compile_flags = "-mattr=+bdrp,+opt,+double-buffer,+tct"
        elif dev == "stx":
            aie_target = "aie2p-gemm-vm-strix-4x4"
            xclbin = (
                os.getenv("PYTORCH_AIE_PATH")
                + "/xclbin/stx/aie2p_gemm_vm_strix_4x4.xclbin"
            )
            aiectrl_json = (
                os.getenv("PYTORCH_AIE_PATH")
                + "/xclbin/stx/aie2p_gemm_vm_strix_4x4.json"
            )
            compile_flags = "-mattr=+bdrp,+opt,+double-buffer,+tct"

        return aie_target, xclbin, aiectrl_json, compile_flags

    def _compile_aie(self):
        if self.use_tvm_dll == False:
            mod = tvm.IRModule()
            mod["main"] = self.tvm_func

            aie_target, xclbin, aiectrl_json, compile_flags = (
                self._get_xclbin_and_config_paths()
            )

            tvm_target = "llvm"
            tvm_target = {"llvm": "llvm", "versal_aie": "versal_aie"}
            with versal_aie.Environment.from_target(aie_target):
                with tvm.transform.PassContext(
                    opt_level=3,
                    config={
                        "relay.ext.versal_aie.options.target": f"{aie_target} -device=aiemaize {compile_flags}",
                        "relay.ext.versal_aie.options.xclbin": xclbin,
                        "relay.ext.versal_aie.options.aiectrl": aiectrl_json,
                        "relay.ext.versal_aie.options.mode": "xrt",
                    },
                ):
                    mod = partition_for_aie(mod, should_partition=True)
                    lib = relay.build(mod, tvm_target, params={})
                    lib.export_library(self.tvm_dll)
                    ctx = tvm.cpu(0)
                    m = graph_executor.GraphModule(lib["default"](ctx, tvm.aie()))
        else:
            ctx = tvm.cpu(0)
            lib = tvm.runtime.load_module(self.tvm_dll)
            m = graph_executor.GraphModule(lib["default"](ctx, tvm.aie()))
        return m

    def ceil_for_me(self, x, y):
        """Return nearest value of x that is a multiple of y"""
        return y * math.ceil(x * 1.0 / y)

    def execute(self, a, b):
        ra_orig = a.shape[0]
        cb_orig = b.shape[1]

        # pad to the nearest multiple of the GEMM Size we are offloading to aie
        a_new_shape = (
            self.ceil_for_me(a.shape[0], self.kernel_x_shape[0]),
            self.ceil_for_me(a.shape[1], self.kernel_x_shape[1]),
        )
        b_new_shape = (
            self.ceil_for_me(b.shape[0], self.kernel_y_shape[0]),
            self.ceil_for_me(b.shape[1], self.kernel_y_shape[1]),
        )
        a_pad_shape = (
            (0, a_new_shape[0] - a.shape[0]),
            (0, a_new_shape[1] - a.shape[1]),
        )
        b_pad_shape = (
            (0, b_new_shape[0] - b.shape[0]),
            (0, b_new_shape[1] - b.shape[1]),
        )
        a_new = np.pad(a, a_pad_shape, mode="constant", constant_values=0)
        b_new = np.pad(b, b_pad_shape, mode="constant", constant_values=0)
        c_new = np.zeros((a_new.shape[0], b_new.shape[1])).astype(np.int32)

        for ra in range(0, a_new.shape[0], self.kernel_x_shape[0]):
            for cb in range(0, b_new.shape[1], self.kernel_y_shape[1]):
                for ca in range(0, a_new.shape[1], self.kernel_x_shape[1]):
                    rb = ca
                    a_tile = a_new[
                        ra : ra + self.kernel_x_shape[0],
                        ca : ca + self.kernel_x_shape[1],
                    ]  # .astype(np.int32)
                    b_tile = b_new[
                        rb : rb + self.kernel_y_shape[0],
                        cb : cb + self.kernel_y_shape[1],
                    ]  # .astype(np.int32)
                    c_new[
                        ra : ra + self.kernel_x_shape[0],
                        cb : cb + self.kernel_y_shape[1],
                    ] += self.run(a_tile, b_tile)

        return c_new[:ra_orig, :cb_orig]

    def forward_quant_mode_None(self, x: Tensor) -> Tensor:
        if self.profiler:
            start = time.time()
        x = torch.matmul(x, self.weight.transpose(0, 1))
        if self.profiler:
            end = time.time()
            self.cpu_time = end - start
        return x

    def forward_quant_mode_0(self, x: Tensor) -> Tensor:
        # fixed scales at compile time
        x = torch.round((1 / self.x_scale) * x).to(torch.int32)  # + self.zp
        x = torch.clip(x, -128, 127)
        x = torch.tensor(self.execute(x, self.weight_q)).to(torch.float)
        x = x * self.x_scale * self.y_scale
        return x

    def forward_quant_mode_1(self, x: Tensor) -> Tensor:
        # quantize inputs
        if self.profiler:
            start = time.time_ns()
        self.x_scale = torch.max(torch.abs(x)).item() / 128
        x = torch.round((1 / self.x_scale) * x)  # + self.zp
        x = torch.clip(x, -128, 127).numpy().astype(np.int8)
        if self.profiler:
            end = time.time_ns()
            self.quant_time = end - start

        # quantized matmul in aie
        if self.profiler:
            start = time.time_ns()
        x = torch.tensor(self.execute(x, self.weight_q)).to(torch.float)
        if self.profiler:
            end = time.time_ns()
            self.aie_time_execute = end - start

        # dequantize
        if self.profiler:
            start = time.time_ns()
        x = x * self.x_scale * self.y_scale
        if self.profiler:
            end = time.time_ns()
            self.dequant_time = end - start
        return x

    def forward(self, x: Tensor) -> Tensor:
        xshape = x.shape
        if self.profiler:
            start = time.time()
        has_batch = False
        if len(x.shape) == 3:
            has_batch = True
            x = x.squeeze(0)
            xshape = x.shape
        if self.profiler:
            end = time.time()
            self.pre_time = end - start

        x = self.forward_dict[self.quant_mode](x)

        if self.profiler:
            start = time.time()
        if self.bias is not None:
            x = x + self.bias
        if has_batch:
            x = torch.unsqueeze(x, 0)
        if self.profiler:
            end = time.time()
            self.post_time = end - start
        if self.profiler:
            logging.critical(
                f"[PROFILE][AIE] {xshape[0]} {xshape[1]} {self.weight_q.shape[0]} {self.weight_q.shape[1]} {self.pre_time} {self.quant_time} {self.aie_time_execute} {self.dequant_time} {self.post_time} "
            )
        return x


class QLinearExperimentalCPU(torch.nn.Module):
    """
    All CPU implementation of following steps
    1. FP32 -> inputs and weights
    2. int8 Quantization of Weights at compile time
    3. int8 quantization of inputs at dynamic runtime
    4. int8 * int8 = int32 multiplication of GEMM
    5. int32 -> int16 rescale
    6. int16 -> FP32 dequantize
    7. Fp32 Bias add
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = False,
        x_scale: float = 1 / 64,
        y_scale: float = 1 / 64,
        quant_mode=None,
        requantize_in_scale: int = 1,
        requantize_out_scale: int = 16,
        profiler: bool = False,
        collect_stats: bool = False,
    ) -> None:
        factory_kwargs = {"device": "cpu", "dtype": None}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        self.collect_stats = collect_stats
        if (self.quant_mode == 1) or (self.quant_mode == 0):
            self.weight = None
        else:
            self.weight = Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
        self.weight_q = None
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.x_scale = x_scale
        self.y_scale = y_scale
        if (self.quant_mode == 1) or (self.quant_mode == 0):
            self.weight_bias = None
        self.requantize_in_scale = requantize_in_scale
        self.requantize_out_scale = requantize_out_scale

        self.ryzenai_cpu_linear = ryzenai_torch_cpp.cpu_linear()

        if (self.quant_mode != 1) and (self.quant_mode != 0):
            self.reset_parameters()
        self.forward_dict = {
            0: self.forward_quant_mode_0,
            1: self.forward_quant_mode_1,
            2: self.forward_quant_mode_2,
            3: self.forward_quant_mode_3,
            4: self.forward_quant_mode_4,
            5: self.forward_quant_mode_5,
            6: self.forward_quant_mode_6,
            7: self.forward_quant_mode_7,
            None: self.forward_quant_mode_None,
        }
        self.profiler = profiler
        if self.profiler:
            self.pre_time = 0
            self.post_time = 0
            self.quant_time = 0
            self.dequant_time = 0
            self.execute_time = 0
        if self.collect_stats:
            self.mmul_shapes = {}
            self.min_int32 = 0
            self.max_int32 = 0
            self.data = []

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"ryzenAI.QLinearExperimentalCPU(self.in_features:{self.in_features}, self.out_features:{self.out_features}, quant_mode:{self.quant_mode} )"

    def quantize_weights(self):
        if (
            (self.quant_mode == 0)
            or (self.quant_mode == 1)
            or (self.quant_mode == 2)
            or (self.quant_mode == 3)
            or (self.quant_mode == 4)
            or (self.quant_mode == 5)
            or (self.quant_mode == 6)
            or (self.quant_mode == 7)
        ):
            # 2 <class 'tuple'> <class 'torch.Tensor'> <class 'NoneType'>
            #                  'torch.quantized.QInt8Tensor
            # print(len(self.weight_bias), type(self.weight_bias), type(self.weight_bias[0]), type(self.weight_bias[1]) )
            self.weight_q = (
                torch.int_repr(self.weight_bias[0]).transpose(0, 1).to(torch.int32)
            )
            self.y_scale = self.weight_bias[0].q_scale()
            if self.weight_bias[1] is not None:
                self.bias = torch.nn.Parameter(self.weight_bias[1])
            else:
                self.bias = None
            self.weight_bias = None
            self.ryzenai_cpu_qlinear = ryzenai_torch_cpp.cpu_qlinear(
                self.weight_q.transpose(0, 1),
                torch.tensor(self.y_scale),
                torch.tensor(self.requantize_in_scale),
                torch.tensor(self.requantize_out_scale),
            )

        else:
            pass
        self.req_scale = self.requantize_in_scale / self.requantize_out_scale

    def forward_quant_mode_None(self, x: Tensor) -> Tensor:
        x = torch.matmul(x, self.weight.transpose(0, 1))
        return x

    def forward_quant_mode_0(self, x: Tensor) -> Tensor:
        # fixed scales at compile time
        x = torch.round((1 / self.x_scale) * x).to(torch.int32)  # + self.zp
        x = torch.clip(x, -128, 127)

        # with new torch cpp extension
        # x = torch.matmul(x, self.weight_q)
        x = self.ryzenai_cpu_linear.mmul(x, self.weight_q)
        x = self.req_scale * x
        x = torch.clip(x, -32768, 32767)

        # self.output_min.append(out.min())
        # self.output_max.append(out.max())
        # requantize to fit in int16
        # then dequantize
        x = (
            x.to(torch.float32)
            * self.x_scale
            * self.y_scale
            * self.requantize_out_scale
        )
        # requantize + dequantize in step
        # out = out.to(torch.float32)  * self.x_scale * self.y_scale
        return x

    def forward_quant_mode_1_cpp(self, x: Tensor) -> Tensor:
        # Python versio is used so user can extract int32 intermediate values for computing requantize_scale
        x = self.ryzenai_cpu_qlinear.qmmul(x)
        return x

    def forward_quant_mode_1(self, x: Tensor) -> Tensor:
        self.x_scale = torch.max(torch.abs(x)) / 128
        x = torch.round((1 / self.x_scale) * x).to(torch.int32)  # + self.zp
        x = torch.clip(x, -128, 127)

        # quant matmul
        # with new torch cpp extension
        x = self.ryzenai_cpu_linear.mmul(x, self.weight_q)
        if self.collect_stats:
            self.min_int32 = min(self.min_int32, x.min().item())
            self.max_int32 = max(self.max_int32, x.max().item())
            key = str(tuple(x.shape)) + str(tuple(self.weight_q.shape))
            if self.mmul_shapes.get(key) is None:
                self.mmul_shapes[key] = 0
            self.mmul_shapes[key] += 1

        x = torch.round(self.req_scale * x)
        x = torch.clip(x, -32768, 32767)

        # dequant
        x = (
            x.to(torch.float32)
            * self.x_scale
            * self.y_scale
            * self.requantize_out_scale
        )
        return x

    def forward_quant_mode_2(self, x: Tensor) -> Tensor:
        """fp32 = fp32 * int8"""
        # print(f"Linear2: x: {x.min().item()} {x.max().item()}")
        x = x.to(torch.float32)
        w = self.weight_q.to(torch.float32)
        # print(f"Linear2: xw: {x.min().item()} {x.max().item()} {w.min().item()} {w.max().item()}")
        x = torch.matmul(x.to(torch.float32), w.to(torch.float32))
        del w
        x = x * self.y_scale
        return x

    def forward_quant_mode_3(self, x: Tensor) -> Tensor:
        """fp32 = fp16 * int8"""
        # y = torch.matmul(x, self.weight_q.to(torch.float32) )
        # print(f"Linear2: x,y: {x.min().item()} {x.max().item()} {y.min().item()} {y.max().item()}")
        # del y
        x = x.to(torch.half)
        w = self.weight_q.to(torch.half)
        x = torch.matmul(x.to(torch.float32), w.to(torch.float32))
        del w
        x = x * self.y_scale
        return x

    def forward_quant_mode_4(self, x: Tensor) -> Tensor:
        """fp32 = bfloat16 * int8"""
        # y = torch.matmul(x, self.weight_q.to(torch.float32) )
        # print(f"Linear2: x,y: {x.min().item()} {x.max().item()} {y.min().item()} {y.max().item()}")
        # del y
        x = x.to(torch.bfloat16)
        w = self.weight_q.to(torch.bfloat16)
        x = torch.matmul(x.to(torch.float32), w.to(torch.float32))
        del w
        x = x * self.y_scale
        return x

    def forward_quant_mode_5(self, x: Tensor) -> Tensor:
        """bfloat16 = bfloat16 * int8"""
        x = x.to(torch.bfloat16)
        w = self.weight_q.to(torch.bfloat16)
        x = torch.matmul(x, w)
        del w
        x = x.to(torch.float32) * self.y_scale
        return x

    def forward_quant_mode_6(self, x: Tensor) -> Tensor:
        self.x_scale = torch.max(torch.abs(x)) / 128
        x = torch.round((1 / self.x_scale) * x).to(torch.int32)  # + self.zp
        x = torch.clip(x, -128, 127)
        x = self.ryzenai_cpu_linear.mmul(x, self.weight_q)
        x = x.to(torch.float32) * self.x_scale * self.y_scale
        return x

    def forward_quant_mode_7(self, x: Tensor) -> Tensor:
        """int32 = int16 @ int8"""
        self.x_scale = torch.max(torch.abs(x)) / 2**15
        x = torch.round((1 / self.x_scale) * x).to(torch.int32)  # + self.zp
        x = torch.clip(x, -32768, 32767)

        # quant matmul
        # with new torch cpp extension
        x = self.ryzenai_cpu_linear.mmul(x, self.weight_q)

        # dequant
        x = x.to(torch.float32) * self.x_scale * self.y_scale
        return x

    def forward(self, x: Tensor) -> Tensor:
        has_batch = True
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            has_batch = False
        # For quant_mode None or 2, use self.weight.shape[1]
        if (self.quant_mode == None) or (self.quant_mode == 2):
            wshape = self.weight.transpose(0, 1).shape
        else:
            wshape = self.weight_q.shape
        y = torch.empty(x.shape[0], x.shape[1], wshape[1])

        for i in range(x.shape[0]):
            # print(f"shapes in linear {x[i].shape} {wshape} {y[i].shape}")
            y[i, :, :] = self.forward_dict[self.quant_mode](x[i, :, :])
        if self.bias is not None:
            y = y + self.bias
        if self.profiler:
            logging.critical(
                f"[PROFILE][AIE] {x.shape[0]} {x.shape[1]} {wshape[0]} {wshape[1]} {self.aie_compile_time_start} {self.aie_compile_time_end} {self.pre_time_start} {self.pre_time_end} {self.quant_time_start} {self.quant_time_end} {self.aie_time_execute_start} {self.aie_time_execute_end} {self.dequant_time_start} {self.dequant_time_end} {self.post_time_start} {self.post_time_end} {self.exec_pybind_time_start} {self.exec_pybind_time_end} {self.exec_c_time_start} {self.exec_c_time_end}"
            )
        if has_batch:
            return y
        else:
            return y[0, :, :]
