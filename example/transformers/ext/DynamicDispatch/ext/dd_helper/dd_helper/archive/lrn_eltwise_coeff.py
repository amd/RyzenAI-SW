##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy as np


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

    def f2bf(self, data, bits=16):
        xf = (
            data.astype(np.float32).getfield(np.int32) & ~(2 ** (32 - bits) - 1)
        ).getfield(np.float32)
        x32 = xf.view(np.uint32)
        x16 = (x32 >> 16).astype(np.uint16)
        return x16

    def cal_coeff(self):
        co_eff1 = self.f2bf(np.asarray(self.a_scale))
        co_eff2 = self.a_zp

        co_eff3 = self.f2bf(np.asarray(self.b_scale))
        co_eff4 = self.b_zp

        return co_eff1, co_eff2, co_eff3, co_eff4


class LRN:
    def __init__(self, out_scale, out_zp):
        self.out_scale = out_scale
        self.out_zp = out_zp

        # assert isinstance(self.out_scale, float), "scale must be float value"
        # assert isinstance(self.out_zp, int), "scale must be int value"
        # breakpoint()

    def f2bf(self, data, bits=16):
        xf = (
            data.astype(np.float32).getfield(np.int32) & ~(2 ** (32 - bits) - 1)
        ).getfield(np.float32)
        x32 = xf.view(np.uint32)
        x16 = (x32 >> 16).astype(np.uint16)
        return x16

    def cal_coeff(self):
        co_eff2 = self.out_zp
        co_eff1 = self.f2bf(np.asarray(self.out_scale))
        return co_eff1, co_eff2


if __name__ == "__main__":

    # LayerNormalization_fused_ReduceMean_0
    so = 0.000393
    za = 33087
    lrn = LRN(so, za)
    co_eff1, co_eff2 = lrn.cal_coeff()
    print(f"co_eff1 {co_eff1} co_eff2 {co_eff2}")

    # /tulrv6/encoder/layer.0/output/Add
    a_scale = 0.0353
    a_zp = 138
    b_scale = 0.11115
    b_zp = 131

    eltwise = EltwiseAdd(a_scale, a_zp, b_scale, b_zp)
    co_eff1, co_eff2, co_eff3, co_eff4 = eltwise.cal_coeff()
    print(f"co_eff1 {co_eff1} co_eff2 {co_eff2} co_eff3 {co_eff3} co_eff4 {co_eff4}")
