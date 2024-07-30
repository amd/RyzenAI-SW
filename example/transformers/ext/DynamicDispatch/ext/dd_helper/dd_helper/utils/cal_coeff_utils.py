##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy as np
import sys

# from ml_dtypes import bfloat16


def f2bf(data, bits=16):
    data = data.astype(np.float32)
    z = data.view(np.uint32)
    lsb = (z >> 16) & (1)
    bias = 0x7FFF + lsb
    z_new = z + bias
    x16 = z_new >> bits
    return x16.astype(np.uint16)


def dequantize_bf16(input, scale, zero_point):
    input = input.astype(np.float32)
    zero_point = np.float32(zero_point)
    scale = np.float32(scale)
    output_fp32 = scale * (input - zero_point)
    return f2bf(output_fp32)


# QKT matmul
def qdq_act_matmul_uint8_uint8_cstm(
    a_dq_xscale,
    a_dq_xzero_pt,
    weights_in_ch,
    w_dq_xscale,
    w_dq_xzero_pt,
    a_q_yscale,
    a_q_yzero_pt,
):
    a_dq_xzero_pt = a_dq_xzero_pt.astype(np.int64)
    w_dq_xzero_pt = w_dq_xzero_pt.astype(np.int64)

    a_q_yzero_pt = a_q_yzero_pt.astype(np.int64)

    c2_coeff = float((a_dq_xscale * w_dq_xscale) / a_q_yscale)

    [c2_coeff_prime, shft_c2] = find_closest_shifted_int32(c2_coeff, 8388607)

    c2_coeff_prime = np.int64(c2_coeff_prime)

    # c1_coeff can be computed at compile time
    weight_coeff_scale = np.int64(-c2_coeff_prime * a_dq_xzero_pt)
    weight_coeff_scale_shift = 0
    # right shift c3 coeff_scale to ensure fits into int32
    if np.abs(weight_coeff_scale) > 2147483647:  # Max int32 number
        weight_coeff_scale_shift = np.int64(
            np.ceil(np.log2(np.abs(weight_coeff_scale))) - 31
        )

    else:
        weight_coeff_scale_shift = 0
    weight_coeff_scale = (weight_coeff_scale >> weight_coeff_scale_shift).astype(
        np.int32
    )

    c1_coeff = a_q_yzero_pt << shft_c2

    c1_coeff = np.int64(c1_coeff)
    num_weights_unrolled = weights_in_ch

    c3_coeff_offset = np.int32(-a_dq_xzero_pt * num_weights_unrolled)
    c3_coeff_scale = np.int64(-c2_coeff_prime * w_dq_xzero_pt)
    c1_coeff += c3_coeff_scale * c3_coeff_offset.astype(np.int64)

    ####################################################
    c3_coeff_scale_shift = 0
    # right shift c3 coeff_scale to ensure fits into int32
    if np.abs(c3_coeff_scale) > 2147483647:  # Max int32 number
        c3_coeff_scale_shift = np.int64(np.ceil(np.log2(np.abs(c3_coeff_scale))) - 31)

    else:
        c3_coeff_scale_shift = 0

    c3_coeff_scale = (c3_coeff_scale >> c3_coeff_scale_shift).astype(np.int32)

    # print("a_dq_xscale " + str(a_dq_xscale))
    # print("a_dq_xzero_pt " + str(a_dq_xzero_pt))
    # print("w_dq_xscale " + str(w_dq_xscale))
    # print("w_dq_xzero_pt " + str(w_dq_xzero_pt))
    # print("a_q_yscale " + str(a_q_yscale))
    # print("a_q_yzero_pt " + str(a_q_yzero_pt))

    c3_coeff_scale_shift = np.int32(c3_coeff_scale_shift)
    shft_c2 = np.int32(shft_c2)
    matmul_shift = 0

    # c2_coeff_prime is C2_scalar_32. c3_coeff_scale is C1_scalar_32. weight_coeff_scale is C3_scalar_32. c1_coeff is C0_scalar_64
    return (
        c1_coeff,  # C0
        c3_coeff_scale,  # C1
        c2_coeff_prime,  # C2
        weight_coeff_scale,  # C3
        c3_coeff_scale_shift,  # shift_qb
        shft_c2,  # shift_out
        matmul_shift,
    )


def mha_qdq_params_fill(
    qkt_qdq, smv_qdq, sm_qdq_before, sm_qdq_after, is_qkt_smv_int16
):
    qdq_params = np.zeros(96).astype(np.int32)

    qry_subv_rows = 32
    qry_subv_cols = 96
    key_subv_rows = 64
    key_subv_rows_int16 = 16
    key_subv_cols = 96
    val_subv_rows = 64
    val_subv_cols = 64
    out_subv_rows = 32
    out_subv_cols = 64

    # QKT
    # qdq_params.view(np.int64)[0] = qkt_qdq[0]
    qdq_params[(16 * 0) + 2] = qkt_qdq[1]
    qdq_params[(16 * 0) + 3] = qkt_qdq[2]
    qdq_params[(16 * 0) + 4] = qkt_qdq[3]
    qdq_params[(16 * 0) + 5] = qry_subv_rows
    qdq_params[(16 * 0) + 6] = key_subv_rows
    qdq_params[(16 * 0) + 7] = qkt_qdq[4]
    qdq_params[(16 * 0) + 8] = qkt_qdq[5]
    qdq_params[(16 * 0) + 9] = qkt_qdq[6]
    qdq_params[(16 * 0) + 10] = is_qkt_smv_int16

    # SM *V
    # qdq_params.view(np.int64)[16] = smv_qdq[0]
    qdq_params[(16 * 1) + 2] = smv_qdq[1]
    qdq_params[(16 * 1) + 3] = smv_qdq[2]
    qdq_params[(16 * 1) + 4] = smv_qdq[3]
    qdq_params[(16 * 1) + 5] = qry_subv_rows
    qdq_params[(16 * 1) + 6] = val_subv_cols
    qdq_params[(16 * 1) + 7] = smv_qdq[4]
    qdq_params[(16 * 1) + 8] = smv_qdq[5]
    qdq_params[(16 * 1) + 9] = smv_qdq[6]
    qdq_params[(16 * 1) + 10] = is_qkt_smv_int16

    # DQ before SM
    qdq_params[(16 * 2) + 0] = sm_qdq_before[1]
    qdq_params[(16 * 2) + 1] = sm_qdq_before[0]

    # Q after SM
    qdq_params[(16 * 3) + 0] = sm_qdq_after[1]
    qdq_params[(16 * 3) + 1] = sm_qdq_after[0]
    qdq_params[(16 * 3) + 2] = is_qkt_smv_int16

    return qdq_params


def gprb_vec32_fill(
    coeff_grpb,
    act_scale,
    act_zero_point,
    wgt_scale,
    wgt_zero_point,
    model_a,
    model_a_scale,
    model_a_zp,
    model_b,
    model_b_scale,
    model_b_zp,
    model_c,
    model_c_scale,
    model_c_zp,
    is_grpb_int16,
):
    gprb_vec32 = np.zeros(32).astype(np.int32)

    qdq_c0_idx = 0
    qdq_c1_idx = 2
    qdq_c2_idx = 3
    qdq_c3_idx = 4
    qdq_Mv_idx = 5
    qdq_Nv_idx = 6
    qdq_SQb_idx = 7
    qdq_Sout_idx = 8
    qdq_Stdm_idx = 9

    gprb_act_scale_idx = 10
    gprb_act_zero_idx = 11
    gprb_wgt_scale_idx = 12
    gprb_wgt_zero_idx = 13
    gprb_model_a_idx = 14
    gprb_model_b_idx = 26
    gprb_model_c_idx = 27
    gprb_isint16_idx = 28

    num_heads = 12

    gprb_vec32[qdq_c1_idx] = coeff_grpb[0]
    gprb_vec32[qdq_c2_idx] = coeff_grpb[1]
    gprb_vec32[qdq_c3_idx] = 0
    gprb_vec32[qdq_Mv_idx] = 32
    gprb_vec32[qdq_Nv_idx] = 8
    gprb_vec32[qdq_SQb_idx] = coeff_grpb[2]
    gprb_vec32[qdq_Sout_idx] = coeff_grpb[3]
    gprb_vec32[qdq_Stdm_idx] = coeff_grpb[4]

    gprb_vec32[gprb_act_scale_idx] = int(f2bf(act_scale))
    gprb_vec32[gprb_act_zero_idx] = int(act_zero_point)
    gprb_vec32[gprb_wgt_scale_idx] = int(f2bf(wgt_scale))
    gprb_vec32[gprb_wgt_zero_idx] = int(wgt_zero_point)
    gprb_vec32[gprb_isint16_idx] = is_grpb_int16

    model_a_bf = dequantize_bf16(model_a, model_a_scale, model_a_zp).flatten()
    for h in range(num_heads):
        gprb_vec32[gprb_model_a_idx + h] = model_a_bf[h]

    gprb_vec32[gprb_model_b_idx] = int(
        (dequantize_bf16(model_b, model_b_scale, model_b_zp))
    )
    gprb_vec32[gprb_model_c_idx] = int(
        (dequantize_bf16(model_c, model_c_scale, model_c_zp))
    )
    return gprb_vec32


def grpb_qgprb_vec64_fill(bias, qk_qdq_c0, smv_qdq_c0):
    gprb_vec64 = np.zeros(11).astype(np.int64)
    # gprb_vec32 = np.zeros(32).astype(np.int32)

    for i in range(8):
        gprb_vec64[i] = bias[i]
    # gprb_c0_scalar_idx = 8
    qk_qdq_c0_scalar_idx = 9
    smv_qdq_c0_scalar_idx = 10

    # gprb_vec64[gprb_c0_scalar_idx] = gprb_dataqdq_c0
    gprb_vec64[qk_qdq_c0_scalar_idx] = qk_qdq_c0
    gprb_vec64[smv_qdq_c0_scalar_idx] = smv_qdq_c0

    return gprb_vec64


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


def find_closest_shifted_int16(float_val):
    INT16_MAX = 32767
    prev_rel_err = 1e9
    curr_float_val = float_val
    best_float_val = float(0)
    shift_val = np.int16
    shift_val = 0
    best_int = np.int16
    closest_curr_int = np.int16
    best_shift_val = np.int16

    while curr_float_val <= INT16_MAX:
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


def find_closest_shifted_int32(float_val, max_value):
    INT32_MAX = max_value  # 2147483647
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

    matmul_shift = 0
    weights = weights.astype(np.int64)

    bias_min_zp = bias.astype(np.int64) - b_dq_xzero_pt

    c2_coeff = float((a_dq_xscale * w_dq_xscale) / a_q_yscale)
    c4_coeff = float(b_dq_xscale / a_q_yscale)
    [c2_coeff_prime, shft_c2] = find_closest_shifted_int32(c2_coeff, 8388607)
    [c4_coeff_prime, shft_c4] = find_closest_shifted_int32(c4_coeff, 8388607)
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
    c1_coeff += c3_coeff_scale * c3_coeff_offset.astype(np.int64)

    return (
        c1_coeff,
        c3_coeff_scale,
        c2_coeff_prime,
        c3_coeff_scale_shift,
        shft_c2,
        matmul_shift,
    )


def qdq_matmul_uint8_uint8_cstm(args_list, bias=None):
    a_dq_xscale = args_list[0]
    a_dq_xzero_pt = args_list[1]
    weights = args_list[2]
    w_dq_xscale = args_list[3]
    w_dq_xzero_pt = args_list[4]
    a_q_yscale = args_list[5]
    a_q_yzero_pt = args_list[6]

    a_dq_xzero_pt = a_dq_xzero_pt.astype(np.int64)
    w_dq_xzero_pt = w_dq_xzero_pt.astype(np.int64)
    a_q_yzero_pt = a_q_yzero_pt.astype(np.int64)
    weights_in_ch = np.int64(weights.shape[-2])

    matmul_shift = 0

    weights = weights.astype(np.int64)

    c2_coeff = float((a_dq_xscale * w_dq_xscale) / a_q_yscale)

    [c2_coeff_prime, shft_c2] = find_closest_shifted_int32(c2_coeff, 8388607)

    c2_coeff_prime = np.int64(c2_coeff_prime)

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
    c1_coeff += c3_coeff_scale * c3_coeff_offset.astype(np.int64)
    # print("Matmul shift   = " + str(matmul_shift))
    # print("c3 coeff shift = " + str(c3_coeff_scale_shift))
    # print("c2 coeff = " + str(c2_coeff_prime))
    # print("c2 coeff shift = " + str(shft_c2))
    # print(new_term)

    return (
        c1_coeff,
        c3_coeff_scale,
        c2_coeff_prime,
        c3_coeff_scale_shift,
        shft_c2,
        matmul_shift,
    )


def qdq_act_matmul_uint16_uint16_cstm(
    a_dq_xscale,
    a_dq_xzero_pt,
    in_ch_dim,
    w_dq_xscale,
    w_dq_xzero_pt,
    a_q_yscale,
    a_q_yzero_pt,
):
    a_dq_xzero_pt = a_dq_xzero_pt.astype(np.int64)
    w_dq_xzero_pt = w_dq_xzero_pt.astype(np.int64)
    a_q_yzero_pt = a_q_yzero_pt.astype(np.int64)

    matmul_shift = np.int64(min(max(np.ceil(np.log2(in_ch_dim)) + 1, 0), 15))

    c2_coeff = float((a_dq_xscale * w_dq_xscale) / a_q_yscale)
    [c2_coeff_prime, shft] = find_closest_shifted_int16(c2_coeff)
    c2_coeff_prime = np.int64(c2_coeff_prime)

    c3_coeff_scale = np.int64(-c2_coeff_prime * w_dq_xzero_pt)

    # right shift c3 coeff_scale to ensure fits into int32
    if np.abs(c3_coeff_scale) > 2147483647:  # Max int32 number
        c3_coeff_scale_shift = np.int64(np.ceil(np.log2(np.abs(c3_coeff_scale))) - 31)
        sys.exit(
            "Current AIE uint16A_uint16A qdq implementation does not support ifm sum shift"
        )

    else:
        c3_coeff_scale_shift = 0

    c3_coeff_scale = (c3_coeff_scale >> c3_coeff_scale_shift).astype(np.int32)

    # Parameter naming below according to (C3*gemm_result+c2*IFM1+c1*IFM2+C0) >> shft

    C3 = (c2_coeff_prime << matmul_shift).astype(np.int32)
    C2 = c3_coeff_scale.astype(np.int32)
    C1 = ((-a_dq_xzero_pt) * c2_coeff_prime).astype(np.int32)
    if np.abs(C1) > 2147483647:  # Max int32 number
        sys.exit(
            "Current AIE uint16A_uint16A qdq implementation does not support ifm sum shift"
        )

    C0 = (
        (a_q_yzero_pt << shft)
        + np.int64(
            np.int64(a_dq_xzero_pt)
            * np.int64(w_dq_xzero_pt)
            * np.int64(in_ch_dim)
            * c2_coeff_prime.astype(np.int64)
        )
    ).astype(np.int64)

    right_shft_matmul = matmul_shift

    shft_final = shft

    return np.int64(C0), C2, C3, C1, c3_coeff_scale_shift, shft_final, right_shft_matmul


def dq_uint16A_uint8W_bias_matmul_q_param_gen(
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
    a_dq_xzero_pt = a_dq_xzero_pt.astype(np.int64)
    w_dq_xzero_pt = w_dq_xzero_pt.astype(np.int64)
    a_q_yzero_pt = a_q_yzero_pt.astype(np.int64)
    in_ch_dim = weights.shape[-2]
    weights = weights.astype(np.int64)
    bias_min_zp = bias.astype(np.int64) - b_dq_xzero_pt

    matmul_shift = np.int64(min(max(np.ceil(np.log2(in_ch_dim)) - 7, 0), 7))
    c2_coeff = float((a_dq_xscale * w_dq_xscale) / a_q_yscale)
    c4_coeff = float(b_dq_xscale / a_q_yscale)

    [c2_coeff_prime, shft_c2] = find_closest_shifted_int32(c2_coeff, 8388607)
    [c4_coeff_prime, shft_c4] = find_closest_shifted_int32(c4_coeff, 8388607)

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

    c1_coeff = (
        (-a_dq_xzero_pt) * c2_coeff_prime * np.sum(weights, axis=(-2), dtype=np.int64)
        + (a_q_yzero_pt << shft_c2)
        + (bias_min_zp * c4_coeff_prime)
    )

    c1_coeff = np.int64(c1_coeff)

    c3_coeff_scale = np.int64(-c2_coeff_prime * w_dq_xzero_pt)
    c3_coeff_offset = np.int32(-a_dq_xzero_pt * in_ch_dim)

    # right shift c3 coeff_scale to ensure fits into int32
    if np.abs(c3_coeff_scale) > 2147483647:  # Max int32 number
        c3_coeff_scale_shift = np.int64(np.ceil(np.log2(np.abs(c3_coeff_scale))) - 31)
        print(c3_coeff_scale)
        sys.exit(
            "Current AIE uint16A_uint8W qdq implementation does not support ifm sum shift"
        )

    else:
        c3_coeff_scale_shift = 0

    c3_coeff_scale = (c3_coeff_scale >> c3_coeff_scale_shift).astype(np.int32)

    # Parameter naming below according to (C2*(gemm_result>>matmul_shft)+c1*(IFM1_sum)+C0) >> final_shft

    C2 = (c2_coeff_prime << matmul_shift).astype(np.int32)

    C1 = (c3_coeff_scale).astype(np.int32)

    C0 = (
        c3_coeff_scale.astype(np.int64)
        * (c3_coeff_offset.astype(np.int64) << c3_coeff_scale_shift).astype(np.int64)
    ).astype(np.int64) + c1_coeff

    shft_final = shft_c2

    return C0, C1, C2, c3_coeff_scale_shift, shft_final, matmul_shift


def qdq_matmul_uint16_uint8_cstm(args_list):
    a_dq_xscale = args_list[0]
    a_dq_xzero_pt = args_list[1]
    weights = args_list[2]
    w_dq_xscale = args_list[3]
    w_dq_xzero_pt = args_list[4]
    a_q_yscale = args_list[5]
    a_q_yzero_pt = args_list[6]

    a_dq_xzero_pt = a_dq_xzero_pt.astype(np.int64)
    w_dq_xzero_pt = w_dq_xzero_pt.astype(np.int64)
    a_q_yzero_pt = a_q_yzero_pt.astype(np.int64)
    # print("matmul_uint16_uint8")
    weights_in_ch = np.int64(weights.shape[-2])

    matmul_shift = np.int64(min(max(25 + np.ceil(np.log2(weights_in_ch)) - 32, 0), 7))
    weights = weights.astype(np.int64)

    c2_coeff = float((a_dq_xscale * w_dq_xscale) / a_q_yscale)

    [c2_coeff_prime, shft_c2] = find_closest_shifted_int32(c2_coeff, 8388607)

    c2_coeff_prime = np.int64(c2_coeff_prime)

    # c1_coeff can be computed at compile time
    c1_coeff = (-a_dq_xzero_pt) * c2_coeff_prime * np.sum(
        weights, axis=(-2), dtype=np.int64
    ) + (a_q_yzero_pt << shft_c2)

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

    C2 = (c2_coeff_prime << matmul_shift).astype(np.int32)
    C1 = c3_coeff_scale.astype(np.int32)
    C0 = (
        c3_coeff_scale.astype(np.int64)
        * (c3_coeff_offset.astype(np.int64) << c3_coeff_scale_shift).astype(np.int64)
    ).astype(np.int64) + c1_coeff

    return (C0, C1, C2, c3_coeff_scale_shift, shft_c2, matmul_shift)


class EltwiseAdd:
    def __init__(self, a_scale, a_zp, b_scale, b_zp):
        self.a_scale = a_scale
        self.a_zp = a_zp
        self.b_scale = b_scale
        self.b_zp = b_zp
        # assert isinstance(self.a_scale, float), "a_scale must be float value"
        # assert isinstance(self.a_zp, int), "a_zp must be int value"

        # assert isinstance(self.b_scale, float), "b_scale must be float value"
        # assert isinstance(self.b_zp, int), "b_zp must be int value"

    def cal_coeff(self):
        co_eff1 = f2bf(np.asarray(self.a_scale))
        co_eff2 = self.a_zp

        co_eff3 = f2bf(np.asarray(self.b_scale))
        co_eff4 = self.b_zp

        return co_eff1, co_eff2, co_eff3, co_eff4


class LRN:
    def __init__(self, out_scale, out_zp):
        self.out_scale = out_scale
        self.out_zp = out_zp

        # assert isinstance(self.out_scale, float), "scale must be float value"
        # assert isinstance(self.out_zp, int), "scale must be int value"
        # breakpoint()

    def cal_coeff(self):
        co_eff2 = self.out_zp
        co_eff1 = f2bf(np.asarray(self.out_scale))
        return co_eff1, co_eff2
