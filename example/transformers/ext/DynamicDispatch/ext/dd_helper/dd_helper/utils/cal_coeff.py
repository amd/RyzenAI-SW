##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy as np


class MatMul_coeff:
    def __init__(self, args_list, bias=None):
        # def __init__(
        #     self, in_scale, in_zp,weights, wts_scale, wts_zp, out_scale, out_zp, bias=None
        # ):

        # GEMM output = srs(GEMM * Coeff2 + Ifmsum*Coeff1 + Coeff0, shift_out)

        # Coeff2 = Sa*Sw/So
        # Coeff1= -Sa*Sw*Zw/So
        # Coeff0= (-Sa*Sw*Za/So (K*Zw-sum(Wk))+Zo)<<shift_in

        # self.wts = weights
        # assert len(self.wts.shape) == 2, "weight shape should be two dimension"

        # self.wts_scale = wts_scale
        # self.wts_zp = wts_zp
        # self.in_scale = in_scale
        # self.in_zp = in_zp
        # self.out_scale = out_scale
        # self.out_zp = out_zp
        # self.wts_sum = np.sum(self.wts, axis=0)
        # self.bias = bias
        # print(args_list)

        self.in_scale = args_list[0]
        self.in_zp = args_list[1]
        self.wts = args_list[2]

        assert len(self.wts.shape) == 2, "weight shape should be two dimension"

        self.wts_scale = args_list[3]
        self.wts_zp = args_list[4]

        self.out_scale = args_list[5]
        self.out_zp = args_list[6]
        self.wts_sum = np.sum(self.wts, axis=0)
        self.bias = bias
        # breakpoint()
        # assert (
        #     isinstance(self.wts_scale, float)
        #     and isinstance(self.in_scale, float)
        #     and isinstance(self.out_scale, float)
        # ), "input scale, wts scale and output scale must be float values"

        # assert (
        #     isinstance(self.in_zp, int)
        #     and isinstance(self.out_zp, int)
        #     and isinstance(self.wts_zp, int)
        # ), "input zp, wts zp and output zp must be integer values"

        # assert len(self.bias.shape) == 1, "bias shape should single dimension"

    def cal_coeff2(self):
        return (self.in_scale * self.wts_scale) / self.out_scale

    def cal_coeff1(self):
        x = -self.cal_coeff2() * self.wts_zp
        return x

    def cal_coeff0(self):
        x = -self.cal_coeff2() * self.in_scale
        kzw = self.wts.shape[0] * self.wts_zp
        sum_wk = self.wts_sum
        if self.bias is None:
            return ((x * (kzw - sum_wk)) + self.out_zp).astype(np.float32)
        else:
            return ((x * (kzw - sum_wk + self.bias)) + self.out_zp).astype(np.float32)

    def convert_float_to_qint(self, in_f):
        ret = 0
        ret = np.asarray(in_f).astype(np.float32).getfield(np.int32)
        ret &= 0x7FFFFFFF
        return ret

    def get_shift_from_int32_rep(self, rep):
        shift = 127 - (((rep >> 23) & 255) + 1) + (8 * 4 - 2)
        return shift

    def get_pos_mantissa(self, temp, floatBits):
        result1 = temp | (1 << 29) | ((floatBits & 0x007FFFFF) << 6)
        return result1

    def get_neg_mantissa(self, result, floatBits):
        val = (
            np.bitwise_not(
                np.bitwise_or(1 << 29, (np.bitwise_and(floatBits, 0x007FFFFF) << 6))
            )
            + 1
        )
        result = np.bitwise_or(result, val)
        return result

    def get_int32_representation_from_float(self, val):
        val = np.asarray(val).astype(np.float32)
        floatBits = val.getfield(np.uint32)
        exponent = (floatBits >> 23) & 0xFF
        mantissa = floatBits & 0x7FFFFF
        # If the mantissa and exponent are equal to 0, return 0 directly.
        # if (0 == exponent) and (0 == mantissa):
        #     return 0
        result = np.array(0).astype(np.int32)
        temp = np.bitwise_and(floatBits, 0x80000000)
        result = np.bitwise_or(temp, result)
        result = result >> 1
        # result = np.where(
        #     result < 0,
        #     #self.get_pos_mantissa(result, floatBits),
        #     0,
        #     self.get_pos_mantissa(result, floatBits)
        #     #self.get_neg_mantissa(result, floatBits),
        # )
        # print(f"result {result}")

        if val < 0:
            result |= ~((1 << 29) | ((floatBits & 0x007FFFFF) << 6)) + 1
        else:
            result |= (1 << 29) | ((floatBits & 0x007FFFFF) << 6)

        result_int = np.asarray(0).astype(np.int32)
        result_int = result
        return result_int

    def compute_coeff(self):
        c0 = self.cal_coeff0()
        c1 = self.cal_coeff1()
        c2 = self.cal_coeff2()

        c0_qint = self.convert_float_to_qint(c0)
        c0_shift = self.get_shift_from_int32_rep(c0_qint)
        # print(f"C0 {c0.shape}")
        c0_int = []
        for i in range(c0.shape[0]):
            c0_int.append(self.get_int32_representation_from_float(c0[i]))

        c1_qint = self.convert_float_to_qint(c1)
        # print(f"c1 {c1}")
        c1_shift = self.get_shift_from_int32_rep(c1_qint)
        c1_int = self.get_int32_representation_from_float(c1)

        c2_qint = self.convert_float_to_qint(c2)
        c2_shift = self.get_shift_from_int32_rep(c2_qint)
        c2_int = self.get_int32_representation_from_float(c2)

        min_exp = min(c2_shift, c1_shift)

        c2_int = c2_int >> (c2_shift - min_exp)
        c1_int = c1_int >> (c1_shift - min_exp)

        # shift_out = min_exp
        # shift_qb = c0_shift - min_exp

        # print(f"c0_qint {c0_qint[0]} c1_shift {c0_shift[0]} c1_int {c0_int[0]}")
        # print(f"c1_qint {c1_qint} c1_shift {c1_shift} c1_int {c1_int}")
        # print(f"c2_qint {c2_qint} c2_shift {c2_shift} c2_int {c2_int}")

        shift_qb = min_exp - c0_shift
        shift_out = min_exp
        # print(f"shift_qb {shift_qb[0]} shift_out {shift_out} ")
        return (
            np.asarray(c0_int).astype(np.int32),
            c1_int,
            c2_int,
            np.min(shift_qb),
            shift_out,
        )


class Dequantize:
    def __init__(self, out_scale, out_zp):
        self.out_scale = out_scale
        self.out_zp = out_zp
        assert isinstance(self.out_scale, float), "scale must be float value"
        assert isinstance(self.out_zp, int), "scale must be int value"

    def f2bf(self, data, bits=16):
        return (
            data.astype(np.float32).getfield(np.int32) & ~(2 ** (32 - bits) - 1)
        ).getfield(np.float32)

    def cal_coeff(self):
        co_eff2 = self.out_zp
        co_eff1 = self.f2bf(np.asarray(self.out_scale))
        return co_eff1, co_eff2


class quantize:
    def __init__(self, out_scale, out_zp):
        self.out_scale = out_scale
        self.out_zp = out_zp
        assert isinstance(self.out_scale, float), "scale must be float value"
        assert isinstance(self.out_zp, int), "scale must be int value"

    def f2bf(self, data, bits=16):
        return (
            data.astype(np.float32).getfield(np.int32) & ~(2 ** (32 - bits) - 1)
        ).getfield(np.float32)

    def shift_out(self):
        return 8  ## TODO

    def cal_coeff(self):
        co_eff2 = self.out_zp
        co_eff1 = self.f2bf(np.asarray(self.out_scale))
        return co_eff1, co_eff2


############################

# sa = 0.101
# sw = 0.0008
# so = 0.0284
# za = 129
# zw = 128
# zo = 122

# wts = np.load("tensor.npy")
# bias = np.random.uniform(-1, 1, wts.shape[1])
# const_dict = ()
# const_dict["wts"]
# const_dict["wts_scale"]
# const_dict["wts_zp"]
# const_dict["in_scale"]
# const_dict["in_zp"]
# const_dict[""]
# const_dict["weights"]
# const_dict["weights"]

# matul_layer = MatMul_coeff(wts, sw, zw, sa, za, so, zo, bias=bias)
# # print(matul_layer.get_shift_from_int32_rep(993686620))
# c0, c1, c2, shift_qb, shift_out = matul_layer.compute_coeff()

# print(f"c0 {c0}, c1 {c1}, c2 {c2}, shift_in {shift_qb}, shift_out {shift_out}")


# """
# so = 0.0284
# za = 129
# dequant_layer = Dequantize(so, za)
# co_eff1, co_eff2 = dequant_layer.cal_coeff()
# print(f"co_eff1 {co_eff1} co_eff2 {co_eff2}")
# """
