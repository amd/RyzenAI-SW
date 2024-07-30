#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

from typing import Optional

import torch
from torch import Tensor

torch.random.manual_seed(123)


class Softmax1(torch.nn.Module):
    __constants__ = ["dim"]
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None, device=None, dtype=None) -> None:
        super().__init__()
        self.device = device
        self.dim = dim
        self.err_percent = 6

    def forward(self, input: Tensor) -> Tensor:
        if self.device == "cpu":
            out = torch.nn.functional.softmax(input, dim=self.dim)

        elif self.device == "aie":
            # uniform_noise = np.random.uniform(0, self.err_percent/100.0, size=input.shape) + 1
            uniform_noise = 1 + self.err_percent / 100
            input = torch.tensor(uniform_noise) * input
            input -= torch.max(input)
            out = torch.exp(input)  # + uniform_noise
            outsum = out.sum(dim=-1, keepdim=True)
            out /= outsum
        return out.to(torch.float32)

    def __repr__(self):
        return f"aie2p.Softmax(self.dim:{self.dim}, self.device:{self.device})"
