##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy as np
import onnxruntime as ort


def srs_uint8_even_fast(inp, shift):
    sign_inp = np.sign(inp)
    inp = abs(inp)
    inp_floor = inp >> shift
    inp_frac = inp - (inp_floor << shift)
    frac_lead_bit = inp_frac >> (shift - 1)

    frac_lead_bit_nonzero_bool_mtrx = frac_lead_bit != 0

    frac_lead_bit_nonzero_bool_mtrx = frac_lead_bit_nonzero_bool_mtrx.astype(np.int64)

    frac_lead_bit_zero_bool_mtrx = 1 - frac_lead_bit_nonzero_bool_mtrx

    inp_floor_even_bool_mtrx = inp_floor % 2
    inp_floor_odd_bool_mtrx = 1 - inp_floor_even_bool_mtrx

    inp_frac_eq_half_bool_mtrx = inp_frac == (1 << shift - 1)
    inp_frac_eq_half_bool_mtrx = inp_frac_eq_half_bool_mtrx.astype(np.int64)
    inp_frac_neq_half_bool_mtrx = 1 - inp_frac_eq_half_bool_mtrx

    inp_floor_plus_1 = inp_floor + 1

    round_res = frac_lead_bit_zero_bool_mtrx * inp_floor

    round_res += (
        frac_lead_bit_nonzero_bool_mtrx * inp_frac_neq_half_bool_mtrx * inp_floor_plus_1
    )
    round_res += (
        frac_lead_bit_nonzero_bool_mtrx
        * inp_frac_eq_half_bool_mtrx
        * inp_floor_odd_bool_mtrx
        * inp_floor_plus_1
    )
    round_res += (
        frac_lead_bit_nonzero_bool_mtrx
        * inp_frac_eq_half_bool_mtrx
        * inp_floor_even_bool_mtrx
        * inp_floor
    )
    # code snippet that is commented out is implemented in np.array format for speed
    """
  if frac_lead_bit != 0:
    round_res = inp_floor+1
    if inp_frac == (1<<shift-1):
      if inp_floor%2==0:
        round_res = inp_floor
      else:
        round_res = inp_floor+1
  else:
    round_res = inp_floor
  """
    round_res = sign_inp * round_res
    round_res = np.clip(round_res, 0, 255)

    return round_res.astype(np.uint8)


def srs_int32_even_fast(inp, shift):
    if shift == 0:
        round_res = np.clip(inp, -2147483648, 2147483647)
        return round_res.astype(np.int32)
    else:
        sign_inp = np.sign(inp)
        inp = abs(inp)
        inp_floor = inp >> shift
        inp_frac = inp - (inp_floor << shift)
        frac_lead_bit = inp_frac >> (shift - 1)

        frac_lead_bit_nonzero_bool_mtrx = frac_lead_bit != 0

        frac_lead_bit_nonzero_bool_mtrx = frac_lead_bit_nonzero_bool_mtrx.astype(
            np.int64
        )

        frac_lead_bit_zero_bool_mtrx = 1 - frac_lead_bit_nonzero_bool_mtrx

        inp_floor_even_bool_mtrx = inp_floor % 2
        inp_floor_odd_bool_mtrx = 1 - inp_floor_even_bool_mtrx

        inp_frac_eq_half_bool_mtrx = inp_frac == (1 << shift - 1)
        inp_frac_eq_half_bool_mtrx = inp_frac_eq_half_bool_mtrx.astype(np.int64)
        inp_frac_neq_half_bool_mtrx = 1 - inp_frac_eq_half_bool_mtrx

        inp_floor_plus_1 = inp_floor + 1

        round_res = frac_lead_bit_zero_bool_mtrx * inp_floor

        round_res += (
            frac_lead_bit_nonzero_bool_mtrx
            * inp_frac_neq_half_bool_mtrx
            * inp_floor_plus_1
        )
        round_res += (
            frac_lead_bit_nonzero_bool_mtrx
            * inp_frac_eq_half_bool_mtrx
            * inp_floor_odd_bool_mtrx
            * inp_floor_plus_1
        )
        round_res += (
            frac_lead_bit_nonzero_bool_mtrx
            * inp_frac_eq_half_bool_mtrx
            * inp_floor_even_bool_mtrx
            * inp_floor
        )
        # code snippet that is commented out is implemented in np.array format for speed
        """
    if frac_lead_bit != 0:
      round_res = inp_floor+1
      if inp_frac == (1<<shift-1):
        if inp_floor%2==0:
          round_res = inp_floor
        else:
          round_res = inp_floor+1
    else:
      round_res = inp_floor
    """
        round_res = sign_inp * round_res
        round_res = np.clip(round_res, -2147483648, 2147483647)

        return round_res.astype(np.int32)


# SRS with round even
def right_broadcasting(arr, target):
    return arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))


def find_closest_shifted_int32(float_val):
    INT32_MAX = 16777216  # 2147483647
    prev_rel_err = 1e9
    curr_float_val = float_val
    best_float_val = float(0)
    shift_val = np.int16
    shift_val = 0
    best_int = np.int32
    closest_curr_int = np.int32
    best_shift_val = np.int16(0)

    while curr_float_val <= INT32_MAX:
        closest_curr_int = round(curr_float_val)
        cur_rel_err = abs(float_val - closest_curr_int / (2**shift_val)) / float_val

        if cur_rel_err < prev_rel_err:
            prev_rel_err = cur_rel_err
            best_float_val = float(closest_curr_int >> shift_val)
            best_shift_val = shift_val
            best_int = closest_curr_int

        curr_float_val *= 2
        shift_val += 1
    return [best_int, best_shift_val]


def compute_qdq_coeff_matmul_bias(
    activations,
    a_dq_xscale,
    a_dq_xzero_pt,
    weights,
    w_dq_xscale,
    w_dq_xzero_pt,
    bias,
    b_dq_xscale,
    b_dq_xzero_pt,
    a_q_yscale,
    a_q_yzero_pt,
):
    # assert (
    #     isinstance(w_dq_xscale, float)
    #     and isinstance(a_dq_xscale, float)
    #     and isinstance(a_q_yscale, float)
    #     and isinstance(b_dq_xscale, float)
    # ), "input scale, wts scale and output scale must be float values"

    # assert (
    #     isinstance(a_dq_xzero_pt, np.ndarray)
    #     and isinstance(w_dq_xzero_pt, np.ndarray)
    #     and isinstance(b_dq_xzero_pt, np.ndarray)
    #     and isinstance(a_q_yzero_pt, np.ndarray)
    # ), "zero points of input, weights, bias and output should be numpy array"

    a_dq_xzero_pt = a_dq_xzero_pt.astype(np.int64)
    w_dq_xzero_pt = w_dq_xzero_pt.astype(np.int64)
    a_q_yzero_pt = a_q_yzero_pt.astype(np.int64)

    assert len(weights.shape) == 2, "weights shape should be 2 dims"

    weights_in_ch = np.int64(weights.shape[-2])

    if (activations.dtype == "uint8") and (weights.dtype == "uint16"):
        matmul_shift = np.int64(max(25 + np.ceil(np.log2(weights_in_ch)) - 33, 0))
    elif (activations.dtype == "uint16") and (weights.dtype == "uint16"):
        matmul_shift = np.int64(max(33 + np.ceil(np.log2(weights_in_ch)) - 33, 0))
    elif (activations.dtype == "uint16") and (weights.dtype == "uint8"):
        matmul_shift = np.int64(max(25 + np.ceil(np.log2(weights_in_ch)) - 33, 0))
    else:
        # print('uint8 x uint8')
        matmul_shift = 0
    # matmul_shift=0
    # activations = activations.astype(np.int64)
    weights = weights.astype(np.int64)
    bias_min_zp = bias.astype(np.int64) - b_dq_xzero_pt

    c2_coeff = float((a_dq_xscale * w_dq_xscale) / a_q_yscale)
    c4_coeff = float(b_dq_xscale / a_q_yscale)
    [c2_coeff_prime, shft_c2] = find_closest_shifted_int32(c2_coeff)
    [c4_coeff_prime, shft_c4] = find_closest_shifted_int32(c4_coeff)
    if shft_c2 != shft_c4:
        diff_shft_c2_c4 = shft_c2 - shft_c4
        # print(diff_shft_c2_c4)
        abs_diff_shft_c2_c4 = np.abs(np.int64(diff_shft_c2_c4))
        if diff_shft_c2_c4 > 0:
            c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4
        elif diff_shft_c2_c4 < 0:
            c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4
        else:
            c4_coeff_prime = c4_coeff_prime

    c2_coeff_prime = np.int64(c2_coeff_prime)

    # c1_coeff can be computed at compile time
    c1_coeff = (
        (-a_dq_xzero_pt) * c2_coeff_prime * np.sum(weights, axis=(-2), dtype=np.int64)
        + (a_q_yzero_pt << shft_c2)
        + (bias_min_zp * c4_coeff_prime)
    )
    c1_coeff = np.int64(c1_coeff)
    num_weights_unrolled = weights_in_ch
    c3_coeff_offset = np.int32(-a_dq_xzero_pt * num_weights_unrolled)
    c3_coeff_scale = np.int64(-c2_coeff_prime * w_dq_xzero_pt)
    ####################################################
    c3_coeff_scale_shift = 0
    # right shift c3 coeff_scale to ensure fits into int32
    if np.abs(c3_coeff_scale) > 2147483647:  # Max int32 number
        c3_coeff_scale_shift = np.int64(np.ceil(np.log2(np.abs(c3_coeff_scale))) - 31)

    else:
        c3_coeff_scale_shift = 0

    c3_coeff_scale = (c3_coeff_scale >> c3_coeff_scale_shift).astype(np.int32)

    return (
        c1_coeff.astype(np.int32),  # TODO:: Add saturation
        c3_coeff_scale,
        c2_coeff_prime,
        c3_coeff_scale_shift,
        shft_c2,
    )


def qdq_matmul_uint8_uint8_cstm(
    activations,
    weights,
    a_dq_xscale,
    a_dq_xzero_pt,
    w_dq_xscale,
    w_dq_xzero_pt,
    a_q_yscale,
    a_q_yzero_pt,
    **kwargs,
):

    ################

    print("matmul_uint8_uint8")
    ##############
    a_dq_xzero_pt = a_dq_xzero_pt.astype(np.int64)
    w_dq_xzero_pt = w_dq_xzero_pt.astype(np.int64)
    a_q_yzero_pt = a_q_yzero_pt.astype(np.int64)
    # print('matmul_uint8_uint8')
    weights_in_ch = np.int64(weights.shape[-2])

    if (activations.dtype == "uint8") and (weights.dtype == "uint16"):
        matmul_shift = np.int64(max(25 + np.ceil(np.log2(weights_in_ch)) - 33, 0))
    elif (activations.dtype == "uint16") and (weights.dtype == "uint16"):
        matmul_shift = np.int64(max(33 + np.ceil(np.log2(weights_in_ch)) - 33, 0))
    elif (activations.dtype == "uint16") and (weights.dtype == "uint8"):
        matmul_shift = np.int64(max(25 + np.ceil(np.log2(weights_in_ch)) - 33, 0))
    else:
        # print('uint8 x uint8')
        matmul_shift = 0
    # print('Matmul shift = ' + str(matmul_shift))
    # matmul_shift=0
    activations = activations.astype(np.int64)
    weights = weights.astype(np.int64)

    c2_coeff = float((a_dq_xscale * w_dq_xscale) / a_q_yscale)

    [c2_coeff_prime, shft_c2] = find_closest_shifted_int32(c2_coeff)
    print(f"c2_coeff_prime {c2_coeff_prime}  shft_c2 {shft_c2}")

    c2_coeff_prime = np.int64(c2_coeff_prime)

    # Coeff2 = Sa*Sw/So
    # Coeff1 = (-Coeff2 * Za )(sum(Wk)) + Zo << shift_in
    # Coeff3= -Sa*Sw*Zw/So

    # c1_coeff can be computed at compile time
    c1_coeff = (-a_dq_xzero_pt) * c2_coeff_prime * np.sum(
        weights, axis=(-2), dtype=np.int64
    ) + (a_q_yzero_pt << shft_c2)

    c1_coeff = np.int64(c1_coeff)

    # breakpoint()
    num_weights_unrolled = weights_in_ch
    c3_coeff_offset = np.int32(-a_dq_xzero_pt * num_weights_unrolled)
    c3_coeff_scale = np.int64(-c2_coeff_prime * w_dq_xzero_pt)

    ####################################################
    c3_coeff_scale_shift = 0
    # right shift c3 coeff_scale to ensure fits into int32
    if np.abs(c3_coeff_scale) > 2147483647:  # Max int32 number
        c3_coeff_scale_shift = np.int64(np.ceil(np.log2(np.abs(c3_coeff_scale))) - 31)

    else:
        c3_coeff_scale_shift = 0

    c3_coeff_scale = (c3_coeff_scale >> c3_coeff_scale_shift).astype(np.int32)

    # int32_matmul = np.matmul(activations, weights).astype(np.int64)
    # print(np.mean(np.abs(int32_matmul.astype(np.float32))))
    # print(np.max(np.abs(int32_matmul.astype(np.float32))))
    # print(np.min(np.abs(int32_matmul.astype(np.float32))))
    ####################################################
    int32_matmul = srs_int32_even_fast(np.matmul(activations, weights), matmul_shift)
    # int32 matmul typecast as int64 to ensure the output of int32xint32 is int64
    temp_out = (c2_coeff_prime << matmul_shift).astype(np.int32) * (
        (int32_matmul.astype(np.int64))
    )
    # 2nd operand is typecast as int64 to ensure product is int64
    # Compensate for right shift of c3 coeff_scale in second operand, experiment shows for PSJ 2nd operand really only utilizes around 10 bits of 32 bits for PSJ
    new_term = c3_coeff_scale * (
        (
            (
                (np.sum(activations, axis=(-1), dtype=np.int32))
                + c3_coeff_offset.astype(np.int64)
            )
            << c3_coeff_scale_shift
        ).astype(np.int64)
    ).astype(np.int64)

    # if len(c1_coeff.shape) == (len(temp_out.shape) - 1):
    #     c1_coeff = right_broadcasting(c1_coeff, temp_out)
    #     c1_coeff = np.swapaxes(c1_coeff, -2, -1)

    temp_out += c1_coeff
    temp_out += right_broadcasting(new_term, temp_out)
    temp_out = srs_uint8_even_fast(temp_out, shft_c2)
    output = np.reshape(temp_out, temp_out.shape)

    print("Matmul shift   = " + str(matmul_shift))
    print("c3 coeff shift = " + str(c3_coeff_scale_shift))
    print("c2 coeff = " + str(c2_coeff_prime))
    print("c2 coeff shift = " + str(shft_c2))
    breakpoint()
    # print(new_term)

    return (
        output,
        c1_coeff,
        c3_coeff_scale,
        c2_coeff_prime,
        c3_coeff_scale_shift,
        shft_c2,
    )


# if __name__ == "__main__":

#     sa = 0.10114588588476181
#     sw = 0.0008763571968302131
#     so = 0.02840903401374817
#     bs = 0.00029612990329042077
#     b_zp = 128
#     za = 129
#     zw = 128
#     zo = 122

#     wts = np.load("tensor.npy")
#     activations = np.load("intermediate_output_0.npz")[
#         "/tulrv6/embeddings/LayerNorm/Add_1_output_0_convert_QuantizeLinear_Output"
#     ]

#     # out_cpu, c0, c1, c2, shift_qb, shift_out = qdq_matmul_uint8_uint8_cstm(
#     #     activations[0], wts, sa, np.asarray(za), sw, np.asarray(zw), so, np.asarray(zo)
#     # )
#     bias = np.random.uniform(low=0, high=255, size=(wts.shape[1]))
#     c0, c1, c2, shift_qb, shift_out = compute_qdq_coeff_matmul_bias(
#         activations[0],

#         wts,
#         bias,
#
#         sw,
#         np.asarray(zw),
#         bs,
#         np.asarray(b_zp),
#         so,
#         np.asarray(zo),
#     )

#     # ort_sess = ort.InferenceSession(
#     #     "_tulrv6_encoder_layer.0_attention_self_query_MatMul.onnx"
#     # )
#     # outputs = ort_sess.run(
#     #     None,
#     #     {
#     #         "/tulrv6/embeddings/LayerNorm/Add_1_output_0_convert_QuantizeLinear_Output": activations
#     #     },
#     # )

#     # diff = np.abs(out_cpu.flatten() - outputs[0].flatten())
#     # print(f"Abs max {diff.max()}")
#     # print(f"out_cpu sum {out_cpu.sum()}")
#     # print(f"outputs sum {outputs[0].sum()}")
#     # outfile = "golden_out.bin"
#     # infile = "input.bin"
#     # wtsfile = "weights.bin"
#     # print(wts.sum())

#     # with open(infile, "wb") as inf:
#     #     inf.write(activations.data)
#     # with open(outfile, "wb") as of:
#     #     of.write(out_cpu.data)
#     # with open(wtsfile, "wb") as wf:
#     #     wf.write(wts.data)
