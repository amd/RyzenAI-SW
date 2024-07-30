# this model implements BF16xInt4 Matmul
# We use torch.bfloat16 to generate scales, activations, bias
# matmul is computed in float to model AIE bf16 MAC behavior
# Kernel implementation can be found here
# https://gitenterprise.xilinx.com/yraparti/vitis-aie-kernels_orig/blob/bf16_gemm_phx_uint4/models/attention_model/bf16_gemm/kernels.cc

import torch


def gemm_int4_bfloat16(
    input_act: torch.Tensor,
    qweights: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    Mgemm = input_act.shape[0]
    Kgemm = qweights.shape[0]
    Ngemm = qweights.shape[1]

    # Assert that input data types match expected
    assert input_act.dtype == torch.bfloat16
    assert qweights.dtype == torch.int8
    assert qzeros.dtype == torch.int8
    assert scales.dtype == torch.bfloat16
    assert bias.dtype == torch.bfloat16
    # Assert that input shapes are valid
    assert input_act.shape == (Mgemm, Kgemm)
    assert qweights.shape == (Kgemm, Ngemm)
    assert qzeros.shape == (Kgemm // group_size, Ngemm)
    assert scales.shape == (Kgemm // group_size, Ngemm)
    assert bias.shape == (Ngemm,)

    # NOTE: To properly emulate AIE bfloat16 x bfloat16 multiply, we must
    #       do the following steps.
    #
    #       1. Round the input to the nearest even bfloat16 value
    #       2. Upshift the rounded result to float32
    #       3. Compute the multiply in float32
    #
    #       These steps are required because AIE intrinsics take two input
    #       bfloat16 vectors, and produce a float32 result.

    # Dequantize weights
    diff = qweights - torch.repeat_interleave(qzeros, group_size, dim=0)
    diff = diff.to(torch.float32)
    scales = scales.to(torch.float32)
    weights = diff * torch.repeat_interleave(scales, group_size, dim=0)
    weights = weights.to(torch.bfloat16)
    # Compute the GeMM
    input_act = input_act.to(torch.float32)
    weights = weights.to(torch.float32)
    bias = bias.to(torch.float32)
    output_act = (input_act @ weights) + bias

    # Assert that output activate type matches expected
    assert output_act.dtype == torch.float32

    return output_act


def main():
    Mgemm = 1
    Kgemm = 4096
    Ngemm = 4096
    group_size = 32
    min_act = -42.0
    max_act = 42.0
    min_quant = 0
    max_quant = 7
    min_scale = 0.0
    max_scale = 1.0
    min_bias = 0.0
    max_bias = 0.0

    # Generate random inputs
    torch.random.manual_seed(42)
    input_act = (min_act - max_act) * torch.rand(
        (Mgemm, Kgemm), dtype=torch.bfloat16
    ) + max_act
    qweights = torch.randint(min_quant, max_quant + 1, (Kgemm, Ngemm), dtype=torch.int8)
    qzeros = torch.randint(
        min_quant, max_quant + 1, (Kgemm // group_size, Ngemm), dtype=torch.int8
    )
    scales = (min_scale - max_scale) * torch.rand(
        (Kgemm // group_size, Ngemm), dtype=torch.bfloat16
    ) + max_scale
    bias = (min_bias - max_bias) * torch.rand(Ngemm, dtype=torch.bfloat16) + max_bias

    # Compute the output activation
    output_act = gemm_int4_bfloat16(
        input_act, qweights, qzeros, scales, bias, group_size
    )

    # Save results to a file
    with open("input_act.bin", "wb") as f:
        f.write(input_act.view(torch.int16).numpy().tobytes())
    with open("qweights.bin", "wb") as f:
        f.write(qweights.numpy().tobytes())
    with open("qzeros.bin", "wb") as f:
        f.write(qzeros.numpy().tobytes())
    with open("scales.bin", "wb") as f:
        f.write(scales.view(torch.int16).numpy().tobytes())
    with open("bias.bin", "wb") as f:
        f.write(bias.view(torch.int16).numpy().tobytes())
    with open("output_act.bin", "wb") as f:
        f.write(output_act.numpy().tobytes())


if __name__ == "__main__":
    main()
