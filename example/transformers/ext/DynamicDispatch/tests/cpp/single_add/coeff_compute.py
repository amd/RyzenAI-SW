import numpy as np


class EltwiseAdd:
    def __init__(self, a_scale, a_zp, b_scale, b_zp):
        self.a_scale = a_scale
        self.a_zp = a_zp
        self.b_scale = b_scale
        self.b_zp = b_zp
        assert isinstance(self.a_scale, float), "a_scale must be float value"
        assert isinstance(self.a_zp, int), "a_zp must be int value"

        assert isinstance(self.b_scale, float), "b_scale must be float value"
        assert isinstance(self.b_zp, int), "b_zp must be int value"

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

        return (co_eff1, co_eff2, co_eff3, co_eff4)
