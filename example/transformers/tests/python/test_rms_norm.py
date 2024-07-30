#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)
import ryzenai_torch_cpp

from transformers.models.llama.modeling_llama import LlamaRMSNorm


@pytest.mark.parametrize("m", [128, 256, 512, 1024, 2048])
def test_ryzenai_torch_cpp_rmsnorm(m):
    golden = LlamaRMSNorm(hidden_size=4096)
    x = torch.rand((m, 4096)).to(torch.bfloat16)
    ref = golden(x)

    # placeholder
    npu = golden(x)
    op = ryzenai_torch_cpp.aie_rmsnorm_torch()
    y = op.execute(x)

    result = torch.allclose(ref, npu)
    result = result and torch.allclose(x, y)
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True
