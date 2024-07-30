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
def test_ryzenai_torch_cpp_silu(m):
    a, b = -42, 42
    x = (b - a) * torch.rand(m, 11008).to(torch.bfloat16) + a

    if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
        import ryzenai_torch_cpp

        op = ryzenai_torch_cpp.aie_silu_torch(m * 11008)
        out = op.execute(x)

    else:
        print("NO AIE IMPL")
        out = torch.nn.functional.silu(x)

    ref = torch.nn.functional.silu(x)

    result = torch.allclose(out, ref, atol=0.8e-1, rtol=1.0e-1)
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True
