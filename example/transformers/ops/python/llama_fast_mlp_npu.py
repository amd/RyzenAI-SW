#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy
from collections import defaultdict

import numpy as np
import ryzenai_torch_cpp
import torch
import torch.nn.functional as F
from qmodule import WQLinear


class LlamaFastMLP_prior(torch.nn.Module):
    def __init__(self, precision):
        super().__init__()
        self.precision = precision
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
        self.act_fn = None  # torch.nn.SiLU(inplace=True)

    def merge_gate_up_w4abf16(self):
        self.gate_end_idx = self.gate_proj.out_features
        self.gate_up_proj = WQLinear(
            self.gate_proj.w_bit,
            self.gate_proj.group_size,
            self.gate_proj.in_features,
            self.gate_proj.out_features * 2,
            bias=False,
            dev="cpu",
        )
        self.gate_up_proj.qweight = torch.empty(
            self.gate_proj.qweight.shape[0] * 2,
            self.gate_proj.qweight.shape[1],
            dtype=torch.int8,
            device="cpu",
        )
        self.gate_up_proj.qzeros = torch.empty(
            self.gate_proj.qzeros.shape[0] * 2,
            self.gate_proj.qzeros.shape[1],
            dtype=torch.int8,
            device="cpu",
        )
        self.gate_up_proj.scales = torch.empty(
            self.gate_proj.scales.shape[0] * 2,
            self.gate_proj.scales.shape[1],
            dtype=torch.bfloat16,
            device="cpu",
        )

        self.gate_up_proj.qweight[: self.gate_proj.out_features, :].copy_(
            self.gate_proj.qweight
        )
        self.gate_up_proj.qweight[
            self.gate_proj.out_features : self.gate_proj.out_features * 2, :
        ].copy_(self.up_proj.qweight)

        self.gate_up_proj.qzeros[: self.gate_proj.out_features, :].copy_(
            self.gate_proj.qzeros
        )
        self.gate_up_proj.qzeros[
            self.gate_proj.out_features : self.gate_proj.out_features * 2, :
        ].copy_(self.up_proj.qzeros)

        self.gate_up_proj.scales[: self.gate_proj.out_features, :].copy_(
            self.gate_proj.scales
        )
        self.gate_up_proj.scales[
            self.gate_proj.out_features : self.gate_proj.out_features * 2, :
        ].copy_(self.up_proj.scales)

        if self.gate_proj.bias is not None:
            self.gate_bias = copy.deepcopy(self.gate_proj.bias)
            self.gate_bias = self.gate_bias.to(torch.bfloat16)
        else:
            self.gate_bias = None
        if self.gate_proj.bias is not None:
            self.up_bias = copy.deepcopy(self.up_proj.bias)
            self.up_bias = self.up_bias.to(torch.bfloat16)
        else:
            self.up_bias = None

    def init_fastmlp(self):
        self.gate_shape = (self.gate_proj.in_features, self.gate_proj.out_features)
        self.down_shape = (self.down_proj.in_features, self.down_proj.out_features)
        if self.precision == "w4abf16":
            self.merge_gate_up_w4abf16()
        else:
            print(f"Only supported for w4abf16 currently")
            raise NotImplementedError
        del self.gate_proj, self.up_proj
        self.c0_token = torch.zeros((1, self.gate_shape[1] * 2), dtype=torch.bfloat16)
        self.c1_token = torch.zeros((1, self.down_shape[1]), dtype=torch.bfloat16)
        self.forward_funcs = defaultdict(
            lambda: self.forward_prefill, {1: self.forward_token}
        )

    def matmul_gu_bias(self, x):
        """WIP - for other models with non zero bias"""
        c0 = torch.empty((x.shape[1], self.gate_shape[1] * 2), dtype=torch.bfloat16)
        self.gate_up_proj.aiegemm.execute(x[0].view(torch.int16), c0.view(torch.int16))
        if self.gate_bias is not None:
            c0[:, : self.gate_end_idx] += self.gate_bias
        if self.up_bias is not None:
            c0[:, self.gate_end_idx :] += self.up_bias
        return c0

    def forward_prefill(self, x):
        c0 = torch.empty((x.shape[0], self.gate_shape[1] * 2), dtype=torch.bfloat16)
        self.gate_up_proj.aiegemm.execute(x.view(torch.int16), c0.view(torch.int16))
        x = self.act_fn(c0[:, : self.gate_end_idx]) * c0[:, self.gate_end_idx :]
        c1 = torch.empty((x.shape[0], self.down_shape[1]), dtype=torch.bfloat16)
        self.down_proj.aiegemm.execute(x.view(torch.int16), c1.view(torch.int16))
        return c1

    def forward_token(self, x):
        self.gate_up_proj.aiegemm.execute(
            x.view(torch.int16), self.c0_token.view(torch.int16)
        )
        x = (
            self.act_fn(self.c0_token[:, : self.gate_end_idx])
            * self.c0_token[:, self.gate_end_idx :]
        )
        self.down_proj.aiegemm.execute(
            x.view(torch.int16), self.c1_token.view(torch.int16)
        )
        return self.c1_token

    def forward_unmerged(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def forward(self, x):
        return self.forward_funcs[x.shape[1]](x[0])


mlp_npu = ryzenai_torch_cpp.aie_mlp_npu_torch()


class LlamaFastMLP(torch.nn.Module):
    def __init__(self, precision):
        super().__init__()
        self.precision = precision
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
        self.act_fn = None  # torch.nn.SiLU(inplace=True)

        self.forward_funcs = defaultdict(
            lambda: self.forward_cpu,
            {1: self.forward_npu, 2048: self.forward_npu},
        )

    def init_fastmlp(self):
        pass

    def forward_npu(self, x):  # cpu
        y0 = self.gate_proj(x)[0]
        # y0 = silu_mladf.execute(y0)
        y1 = self.up_proj(x)
        x = mlp_npu.execute(y0, y1)
        # x = elemmul_mladf.execute(y0, y1)
        x = self.down_proj(x)
        return x

    def forward_cpu(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def forward(self, x):
        return self.forward_funcs[x.shape[1]](x)
