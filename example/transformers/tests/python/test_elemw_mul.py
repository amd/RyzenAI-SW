#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)

prompt_lengths = [1, 128, 256, 512, 1024, 2048]


@pytest.mark.parametrize("m", prompt_lengths)
def test_ryzenai_aie_elem_mult(m):
    inpshape = (m, 11008)
    a, b = -42, 42
    x = (b - a) * torch.rand(inpshape).to(torch.bfloat16) + a
    y = (b - a) * torch.rand(inpshape).to(torch.bfloat16) + a

    if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
        import ryzenai_torch_cpp

        op = ryzenai_torch_cpp.aie_elemw_mul_torch(inpshape[0] * 11008)
        out = op.execute(x, y)
    else:
        print("No AIE Impl")
        out = x * y

    ref = x * y
    result = torch.allclose(out, ref)

    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True
