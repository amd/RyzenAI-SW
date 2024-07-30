#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import math

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)
import ryzenai_torch_cpp

torch.random.manual_seed(123)


def mask_softmax(input, mask):
    shape = (32, 2048, 2048)
    r = (np.random.rand(*shape) - 0.5) * 42.0
    M = input.shape[1]
    B = input.shape[0]
    K = input.shape[2]
    for batch in range(input.shape[0]):
        for m in range(input.shape[1]):
            add_att = input[batch][m] + mask[0][m]
            r[batch][m] = np.exp(add_att)
            e = np.sum(r[batch][m])
            r[batch][m] = r[batch][m] / e
            # for k in range(K):
            #    r[batch][m][k] = math.exp(input[batch][m][k]+mask[0][m][k])
            #    runSum += r[batch][m][k]
            # for k in range(K):
            #    r[batch][m][k] = r[batch][m][k]/runSum

    return r


def test_softmax():
    x_min, x_max = -3.0, 3.0
    inp_shape = (32, 2048, 2048)
    msk_shape = (1, 2048, 2048)

    x = (np.random.rand(*inp_shape) - 0.5) * 10.0
    m = (np.random.rand(*msk_shape) - 0.5) * 10.0
    m = np.full((1, 2048, 2048), -np.inf, dtype=float)
    for i in range(2048):
        for j in range(2048):
            if j <= i:
                m[0][i][j] = 0
    s = ryzenai_torch_cpp.aie_softmax_torch()
    print("x: ", x.shape)
    print("m: ", m.shape)
    o = s.execute(
        torch.tensor(x).to(torch.bfloat16), torch.tensor(m).to(torch.bfloat16)
    )
    g = mask_softmax(
        torch.tensor(x).to(torch.bfloat16).to(torch.float).numpy(),
        torch.tensor(m).to(torch.bfloat16).to(torch.float).numpy(),
    )

    result = torch.allclose(
        o.to(torch.bfloat16), torch.tensor(g).to(torch.bfloat16), atol=2e-2, rtol=2.0e-2
    )
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True


def profile_softmax():
    ss = ryzenai_torch_cpp.aie_softmax_torch()
    x = torch.rand((32, 2048, 2048)).to(torch.bfloat16)
    a = torch.rand((1, 2048, 2048)).to(torch.bfloat16)
    import time

    tarr = []
    for i in range(10):
        st = time.perf_counter()
        # res = ss.execute(x, a)
        x = x + a
        x = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32).to(
            torch.bfloat16
        )
        tarr.append(time.perf_counter() - st)

    tarr = np.array(tarr)
    print(f"ss tarr:   {tarr}")
    print(f"ss avg:    {tarr.mean()}")
    print(f"ss median: {np.median(tarr)}")


if __name__ == "__main__":
    profile_softmax()
