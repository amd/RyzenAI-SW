#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)
import ryzenai_torch_cpp


def test_ryzenai_torch_cpp_scalar():
    x = np.random.randint(-128, 128, [1, 64]).astype(np.int32)
    x = torch.tensor(x)
    scalar_mult = ryzenai_torch_cpp.aie_scalar_mult(torch.numel(x))
    y = scalar_mult.execute(x)
    result = torch.allclose(y, x * 3)
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True
