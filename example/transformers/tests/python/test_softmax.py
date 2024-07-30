#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy
import logging

import pytest
import softmax
import torch

torch.random.manual_seed(123)


def test_softmax():
    x_min, x_max = -3.0, 3.0
    inp_shape = (3, 3, 3)
    x = (x_max - x_min) * torch.rand(inp_shape) - x_min
    xc = copy.deepcopy(x)
    # print(f"\n{x=}")

    s = softmax.Softmax1(dim=-1, device="cpu")
    o = s(x)
    print(f"{o=}\n")

    s.device = "aie"
    oaie = s(x)
    print(f"{oaie=}")
    ideal = torch.nn.Softmax(dim=-1)(xc).to(torch.float)
    print(f"{ideal=}")
    print(f"{torch.allclose(ideal, o)=}")
    print(f"{torch.allclose(ideal, oaie)=}")
    s.err_percent = 0
    oaie = s(x)
    result = torch.allclose(ideal, oaie)
    print(f"{torch.allclose(ideal, oaie)=}")
    assert result == True
