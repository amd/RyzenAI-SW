#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import numpy as np
import pytest
import qlinear
import ryzenai_torch_cpp
import torch
from test_qlinear import opt_1p3b_shapes

torch.random.manual_seed(123)

test_shapes = [
    ([256, 2048], [2048, 2048]),
    ([128, 2048], [2048, 2048]),
    ([512, 2048], [2048, 2048]),
    ([512, 2048], [2048, 4096]),
    ([512, 2048 * 5], [2048 * 5, 4096]),
    ([110, 600], [600, 500]),
    ([769, 1531], [1531, 899]),
    ([128, 3000], [3000, 2048]),
    ([128, 3000], [3000, 4096]),
    ([513, 3000], [3000, 4096]),
]

data_range = [[-10, 10], [-10000, 10000], [-1000000, 1000000]]


@pytest.mark.parametrize("shapes", test_shapes)
def test_ryzenai_torch_linear_bfloat16(shapes):
    """
    Compare bfloat16 matmul op on AIE with torch matmul
    """
    xshape = shapes[0]
    yshape = shapes[1]

    d_min = -1000000000
    d_max = 1000000000
    x = (d_max - d_min) * torch.rand(xshape, dtype=torch.bfloat16) + d_max
    y = (d_max - d_min) * torch.rand(yshape, dtype=torch.bfloat16) + d_max

    aie_linear = ryzenai_torch_cpp.aie_linear_bf16((256, 2048), (2048, 2048))
    aie_linear.initialize_weights(y)
    aie_out = aie_linear.execute(x)
    z = torch.matmul(x, y)

    result = torch.allclose(aie_out, z, rtol=5.0e-2)

    assert result == True


@pytest.mark.parametrize("input_data_range", data_range)
def test_ryzenai_bfloat16_accuracy(input_data_range):
    """
    validate bfloat16 accuracy for kernel shape 256x2048x2048
    This does not use CPU tiling
    """

    xshape = [256, 2048]
    yshape = [2048, 2048]

    d_min = input_data_range[0]
    d_max = input_data_range[1]

    x = (d_min - d_max) * torch.rand(xshape, dtype=torch.bfloat16) + d_max
    y = (d_min - d_max) * torch.rand(yshape, dtype=torch.bfloat16) + d_max

    aie_linear = ryzenai_torch_cpp.aie_linear_bf16((256, 2048), (2048, 2048))
    aie_linear.initialize_weights(y)
    aie_out = aie_linear.execute(x)
    z = torch.matmul(x, y)

    err = abs((z - aie_out) / z)

    # print(f"x: {x[0:10]}")
    # print(f"y: {y[0:10]}")
    # print(f"aie_out: {aie_out[0:10]}")

    print(
        f"Data range: {d_min, d_max}, Err min: {err.min() * 100}, Err median: {err.median() * 100}, Err max: {err.max() * 100}, Err Mean: {err.mean() * 100}"
    )

    result = torch.allclose(aie_out, z, rtol=5.0e-2)

    assert result == True


@pytest.mark.parametrize("input_data_range", data_range)
def test_ryzenai_bfloat16_tiled_accuracy1(input_data_range):
    """
    validate bfloat16 accuracy for kernel shape 256x2048x2048
    Two tiles in K dimension
    """

    xshape = [256, 2048 * 2]
    yshape = [2048 * 2, 2048]

    d_min = input_data_range[0]
    d_max = input_data_range[1]

    x = (d_min - d_max) * torch.rand(xshape, dtype=torch.bfloat16) + d_max
    y = (d_min - d_max) * torch.rand(yshape, dtype=torch.bfloat16) + d_max

    aie_linear = ryzenai_torch_cpp.aie_linear_bf16((256, 2048), (2048, 2048))
    aie_linear.initialize_weights(y)
    aie_out = aie_linear.execute(x)
    z = torch.matmul(x, y)

    err = abs((z - aie_out) / z)

    # print(f"x: {x[0:10]}")
    # print(f"y: {y[0:10]}")
    # print(f"aie_out: {aie_out[0:10]}")

    print(
        f"Data range: {d_min, d_max}, Err min: {err.min() * 100}, Err median: {err.median() * 100}, Err max: {err.max() * 100}, Err Mean: {err.mean() * 100}"
    )

    # Due to tiling on CPU, the error is signficant. Check for mean error till this is handled in a proper way.
    assert err.mean() < 0.05


@pytest.mark.parametrize("opt_shapes", opt_1p3b_shapes)
def test_ryzenai_bfloat16_opt_shapes(opt_shapes):
    """
    validate bfloat16 kernel for Opt shapes
    """
    (xshape, yshape) = opt_shapes
    # TODO: Torch matmul with transpose has an issue which results in incorrect results.
    # For now, handle it in shapes instead of an explicit transpose.
    yshape = (yshape[1], yshape[0])

    d_min = -10
    d_max = 10

    x = (d_min - d_max) * torch.rand(xshape, dtype=torch.bfloat16) + d_max
    y = (d_min - d_max) * torch.rand(yshape, dtype=torch.bfloat16) + d_max

    aie_linear = ryzenai_torch_cpp.aie_linear_bf16((256, 2048), (2048, 2048))
    aie_linear.initialize_weights(y)
    aie_out = aie_linear.execute(x)
    z = torch.matmul(x, y)

    err = abs((z - aie_out) / z)

    # print(f"x: {x[0:10]}")
    # print(f"y: {y[0:10]}")
    # print(f"aie_out: {aie_out[0:10]}")

    print(
        f"X shape: {x.shape[0]}x{x.shape[1]}, Y shape: {y.shape[0]}x{y.shape[1]}, Err min: {err.min() * 100}, Err median: {err.median() * 100}, Err max: {err.max() * 100}, Err Mean: {err.mean() * 100}"
    )

    # Due to tiling on CPU, the error is signficant. Check for mean error till this is handled in a proper way.
    assert err.mean() < 0.05
