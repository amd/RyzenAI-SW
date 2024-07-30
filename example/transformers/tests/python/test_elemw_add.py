#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)
import ryzenai_torch_cpp

dims = [4096]


@pytest.mark.parametrize("m", dims)
def test_ryzenai_aie_elemw_add(m):
    inpshape = (m, 4096)
    a, b = -42, 42
    x = (b - a) * torch.rand(inpshape).to(torch.bfloat16) + a

    if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
        import ryzenai_torch_cpp

        print("In AIE Impl")
        op = ryzenai_torch_cpp.aie_elemw_add_torch()
        out = op.execute(x, x)
    else:
        print("No AIE Impl")
        out = x + x

    ref = x + x
    result = torch.allclose(out, ref)

    err = (out - ref).abs()
    print(f"err: {err.max()}")
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True


def profile_add():
    mladfadd = ryzenai_torch_cpp.aie_mladfadd_torch()
    magic_shape = (4096, 4096)
    x = torch.rand(magic_shape).to(torch.bfloat16)
    y = torch.rand(magic_shape).to(torch.bfloat16)

    import time

    tarr = []
    for i in range(10):
        st = time.perf_counter()
        res = mladfadd.execute(x, y)
        tarr.append(time.perf_counter() - st)

    tarr = np.array(tarr)
    print(f"mladfadd tarr:   {tarr}")
    print(f"mladfadd avg:    {tarr.mean()}")
    print(f"mladfadd median: {np.median(tarr)}")


if __name__ == "__main__":
    profile_add()
