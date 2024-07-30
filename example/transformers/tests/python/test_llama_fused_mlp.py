#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#


import copy

import pytest
import torch
import torch.nn.functional as F

torch.manual_seed(0)


class Llama2MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.gate_proj = torch.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = torch.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = torch.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


if __name__ == "__main__":
    m = Llama2MLP()
    x = torch.randn(1, 4096, requires_grad=True)
    torch.onnx.export(
        m,
        x,
        "llama2_mlp.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
