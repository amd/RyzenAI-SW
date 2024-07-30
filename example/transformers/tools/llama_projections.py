#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import os


def ten_hut(
    x=(1, 4096),
    q_proj=(4096, 4096),
    k_proj=(4096, 4096),
    v_proj=(4096, 4096),
    seq_len=10,
    o_proj=(4096, 4096),
    grp_size=128,
):

    tops = {
        "int4_elemw_add": {"proj": 0},
        "bf16_elemw_mul": {"proj": 0, "softmax": 0, "bmm1": 0, "bmm2": 0, "rope": 0},
        "bf16_elemw_add": {"softmax": 0, "rope": 0},
        "bf16_matmul": {"proj": 0, "bmm1": 0, "bmm2": 0},
    }

    # 1 - QKV projections - no bias add for Llama2
    # Q_proj
    tops["int4_elemw_add"]["proj"] += q_proj[0] * q_proj[1]  # (w-z)
    tops["bf16_elemw_mul"]["proj"] += q_proj[0] * q_proj[1]  # (w-z).s
    tops["bf16_matmul"]["proj"] += x[0] * q_proj[0] * q_proj[1] * 2

    # K_proj
    tops["int4_elemw_add"]["proj"] += k_proj[0] * k_proj[1]  # (w-z)
    tops["bf16_elemw_mul"]["proj"] += k_proj[0] * k_proj[1]  # (w-z).s
    tops["bf16_matmul"]["proj"] += x[0] * k_proj[0] * k_proj[1] * 2

    # V_proj
    tops["int4_elemw_add"]["proj"] += v_proj[0] * v_proj[1]  # (w-z)
    tops["bf16_elemw_mul"]["proj"] += v_proj[0] * q_proj[1]  # (w-z).s
    tops["bf16_matmul"]["proj"] += x[0] * v_proj[0] * v_proj[1] * 2

    # 2 - RoPE
    # q = (seq_len_+1)
    # k = (seq_len_+1)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    tops["bf16_elemw_mul"]["rope"] += seq_len + 1
    tops["bf16_elemw_mul"]["rope"] += seq_len + 1
    tops["bf16_elemw_add"]["rope"] += seq_len + 1

    tops["bf16_elemw_mul"]["rope"] += seq_len + 1
    tops["bf16_elemw_mul"]["rope"] += seq_len + 1
    tops["bf16_elemw_add"]["rope"] += seq_len + 1

    # 3 - bmm1
    B, H, L, d = 1, 32, seq_len, 128
    tops["bf16_matmul"]["bmm1"] += H * d * (seq_len + 1) * 2
    #  *scale
    tops["bf16_elemw_mul"]["bmm1"] += B * H * 1 * (seq_len + 1)

    # 4 Softmax
    # x = (B, H, 1, seqlen+1)
    # x = x - x_max
    # e_x = exp(x) = ax+b
    # sum_ex = sum(e_x)
    # softmax = e_x/sum_ex
    tops["bf16_elemw_add"]["softmax"] += B * H * 1 * (seq_len + 1)  # x = x-x_max
    tops["bf16_elemw_add"]["softmax"] += B * H * 1 * (seq_len + 1)  # exp(x) + b
    tops["bf16_elemw_mul"]["softmax"] += B * H * 1 * (seq_len + 1)  # exp(x) ax+b
    tops["bf16_elemw_add"]["softmax"] += B * H * 1 * (seq_len + 1)  # sum_ex
    tops["bf16_elemw_mul"]["softmax"] += B * H * 1 * (seq_len + 1)  # e_x/sum_ex

    # 5 bmm2
    tops["bf16_matmul"]["bmm2"] += H * d * (seq_len + 1) * 2

    # 6 O_proj
    tops["int4_elemw_add"]["proj"] += o_proj[0] * o_proj[1]  # (w-z)
    tops["bf16_elemw_mul"]["proj"] += o_proj[0] * o_proj[1]  # (w-z).s
    tops["bf16_matmul"]["proj"] += x[0] * o_proj[0] * o_proj[1] * 2

    for k1 in tops.keys():
        for k2 in tops[k1].keys():
            tops[k1][k2] *= 32  # layers
            tops[k1][k2] /= 1e12  # ops -> tops
    return tops


def mlp(
    seq_len=1,
    hidden_size=4096,
    intermediate_size=12288,
):

    tops = {
        "int4_elemw_add": {"gate_proj": 0, "up_proj": 0, "down_proj": 0},
        "bf16_elemw_mul": {"gate_proj": 0, "up_proj": 0, "down_proj": 0, "silu": 0},
        "bf16_elemw_add": {"silu": 0},
        "bf16_matmul": {"gate_proj": 0, "up_proj": 0, "down_proj": 0},
    }

    # gate_proj
    tops["int4_elemw_add"]["gate_proj"] += hidden_size * intermediate_size  # (w-z)
    tops["bf16_elemw_mul"]["gate_proj"] += hidden_size * intermediate_size  # (w-z).s
    tops["bf16_matmul"]["gate_proj"] += seq_len * hidden_size * intermediate_size * 2

    # up_proj
    tops["int4_elemw_add"]["up_proj"] += hidden_size * intermediate_size  # (w-z)
    tops["bf16_elemw_mul"]["up_proj"] += hidden_size * intermediate_size  # (w-z).s
    tops["bf16_matmul"]["up_proj"] += seq_len * hidden_size * intermediate_size * 2

    # silu
    # x = (seqlen, intermediate_size)
    # e_x = exp(x) = ax+b
    # sigmoid = e_x / (1+e_x)
    # silu = x * sigmoid
    tops["bf16_elemw_add"]["silu"] += seq_len * intermediate_size  # exp(x) +b
    tops["bf16_elemw_mul"]["silu"] += seq_len * intermediate_size  # exp(x) ax+b
    tops["bf16_elemw_add"]["silu"] += seq_len * intermediate_size  # +1
    tops["bf16_elemw_mul"]["silu"] += seq_len * intermediate_size  # e_x / (1+e_x)
    tops["bf16_elemw_mul"]["silu"] += seq_len * intermediate_size  # x * sigmoid

    # down_proj
    tops["int4_elemw_add"]["down_proj"] += intermediate_size * hidden_size  # (w-z)
    tops["bf16_elemw_mul"]["down_proj"] += intermediate_size * hidden_size  # (w-z).s
    tops["bf16_matmul"]["down_proj"] += seq_len * intermediate_size * hidden_size * 2

    for k1 in tops.keys():
        for k2 in tops[k1].keys():
            tops[k1][k2] *= 32  # layers
            tops[k1][k2] /= 1e12  # ops -> tops
    return tops


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        help="prefill or decode",
        type=str,
        default="decode",
        choices=["prefill", "decode"],
    )
    parser.add_argument("--seq_len", help="starting seq len", type=int, default=2000)
    parser.add_argument(
        "--new_tokens", help="number of new tokens", type=int, default=2
    )
    args = parser.parse_args()
    print(f"{args}")

    if args.phase == "prefill":
        # Attntion
        tops = ten_hut(x=(args.seq_len, 4096), seq_len=args.seq_len + 1)
        for k1 in tops.keys():
            tops_count = 0
            for k2 in tops[k1].keys():
                tops_count += tops[k1][k2]
                print(f"Prefill {k1} {k2} : {tops[k1][k2]}")
            print(f"Prefill {k1} : {tops_count} ============")

        # MLP
        tops = mlp(seq_len=args.seq_len)
        for k1 in tops.keys():
            tops_count = 0
            for k2 in tops[k1].keys():
                tops_count += tops[k1][k2]
                print(f"Prefill {k1} {k2} : {tops[k1][k2]}")
            print(f"Prefill {k1} : {tops_count} ============")

    else:  # token
        # MLP
        for i in range(args.new_tokens):
            print(f"====== New token {i+1} ======")

            # Attention
            tops = ten_hut(x=(1, 4096), seq_len=args.seq_len + 1)
            for k1 in tops.keys():
                tops_count = 0
                for k2 in tops[k1].keys():
                    tops_count += tops[k1][k2]
                    print(f"Token {k1} {k2} : {tops[k1][k2]}")
                print(f"Token {k1} : {tops_count} =============")

            # MLP
            tops = mlp(seq_len=1)
            for k1 in tops.keys():
                tops_count = 0
                for k2 in tops[k1].keys():
                    tops_count += tops[k1][k2]
                    print(f"Token {k1} {k2} : {tops[k1][k2]}")
                print(f"Token {k1} : {tops_count} =============")
