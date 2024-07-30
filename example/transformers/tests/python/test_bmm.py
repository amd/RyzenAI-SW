#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)


def test_ryzenai_torch_cpp_bmmA():
    TRANSPOSE = False
    sizeA = [32, 2048, 2048]
    sizeB = [32, 2048, 128]
    xFloat = (np.random.rand(*sizeA) - 0.5) * 1.5
    yFloat = (np.random.rand(*sizeB) - 0.5) * 1.5

    x = torch.from_numpy(xFloat).to(torch.bfloat16)
    y = torch.from_numpy(yFloat).to(torch.bfloat16)

    ref = torch.matmul(x, y)

    if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
        import ryzenai_torch_cpp

        op = ryzenai_torch_cpp.aie_bmm_torch(TRANSPOSE)
        out = op.execute(x, y)
    else:
        print("Not implemented in NPU")
        out = torch.matmul(x, y)

    result = torch.allclose(ref, out, atol=0.5, rtol=0.5)

    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True


def test_ryzenai_torch_cpp_bmmC():
    TRANSPOSE = True
    sizeA = [32, 2048, 128]
    sizeB = [32, 2048, 128]

    xFloat = (np.random.rand(*sizeA) - 0.5) * 1.5
    yFloat = (np.random.rand(*sizeB) - 0.5) * 1.5

    yTrans = torch.transpose(torch.tensor(yFloat), 1, 2)
    zGold = torch.matmul(
        torch.tensor(xFloat).to(torch.bfloat16), yTrans.to(torch.bfloat16)
    )

    if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
        import ryzenai_torch_cpp

        bm = ryzenai_torch_cpp.aie_bmm_torch(TRANSPOSE)
        z = bm.execute(
            torch.tensor(xFloat).to(torch.bfloat16),
            torch.tensor(yFloat).to(torch.bfloat16),
        )
    else:  # not implemented on phx
        print("Not implemented in NPU")
        z = torch.matmul(
            torch.tensor(xFloat).to(torch.bfloat16), yTrans.to(torch.bfloat16)
        )

    result = torch.allclose(
        z.to(torch.bfloat16), zGold.to(torch.bfloat16), atol=0.08, rtol=0.2
    )

    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True


def profile_bmm1():
    bmm1 = ryzenai_torch_cpp.aie_bmm_torch(True)
    x = torch.rand((32, 2048, 128)).to(torch.bfloat16)
    y = torch.rand((32, 2048, 128)).to(torch.bfloat16)

    import time

    tarr = []
    for i in range(10):
        st = time.perf_counter()
        # res = bmm1.execute(x, y)
        res = torch.matmul(x, y.transpose(1, 2))
        tarr.append(time.perf_counter() - st)

    tarr = np.array(tarr)
    print(f"bmm1 tarr:   {tarr}")
    print(f"bmm1 avg:    {tarr.mean()}")
    print(f"bmm1 median: {np.median(tarr)}")


def profile_bmm2():
    bmm2 = ryzenai_torch_cpp.aie_bmm_torch(False)
    x = torch.rand((32, 2048, 2048)).to(torch.bfloat16)
    y = torch.rand((32, 128, 2048)).to(torch.bfloat16)

    import time

    tarr = []
    for i in range(10):
        st = time.perf_counter()
        # res = bmm2.execute(x, y)
        res = torch.matmul(x, y.transpose(1, 2))
        res = res.unsqueeze(0)
        tarr.append(time.perf_counter() - st)

    tarr = np.array(tarr)
    print(f"bmm2 tarr:   {tarr}")
    print(f"bmm2 avg:    {tarr.mean()}")
    print(f"bmm2 median: {np.median(tarr)}")


if __name__ == "__main__":
    profile_bmm1()
    profile_bmm2()
