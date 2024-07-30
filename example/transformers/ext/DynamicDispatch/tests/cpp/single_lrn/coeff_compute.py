import numpy as np


class LRN:
    def __init__(self, out_scale, out_zp):
        self.out_scale = out_scale
        self.out_zp = out_zp
        assert isinstance(self.out_scale, float), "scale must be float value"
        assert isinstance(self.out_zp, int), "scale must be int value"

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
