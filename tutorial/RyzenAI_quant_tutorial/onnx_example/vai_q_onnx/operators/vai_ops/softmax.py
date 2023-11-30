#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import onnx

from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase


class QDQSoftmax(QDQOperatorBase):

    def quantize(self):
        super().quantize()
        if self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.UINT8:
            if self.quantizer.is_activation_symmetric:
                out_scale = 1 / 128.0
                out_zero_point = 128
            else:
                out_scale = 1 / 256.0
                out_zero_point = 0
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.INT8:
            if self.quantizer.is_activation_symmetric:
                out_scale = 1 / 128.0
                out_zero_point = 0
            else:
                out_scale = 1 / 256.0
                out_zero_point = -128
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.UINT16:
            if self.quantizer.is_activation_symmetric:
                out_scale = 1 / 32768.0
                out_zero_point = 32768
            else:
                out_scale = 1 / 65536.0
                out_zero_point = 0
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.INT16:
            if self.quantizer.is_activation_symmetric:
                out_scale = 1 / 32768.0
                out_zero_point = 0
            else:
                out_scale = 1 / 65536.0
                out_zero_point = -32768
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.UINT32:
            if self.quantizer.is_activation_symmetric:
                out_scale = 1. / 2**31
                out_zero_point = 2**31
            else:
                out_scale = 1. / 2**32
                out_zero_point = 0
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.INT32:
            if self.quantizer.is_activation_symmetric:
                out_scale = 1. / 2**31
                out_zero_point = 0
            else:
                out_scale = 1. / 2**32
                out_zero_point = -2**31
        else:
            return
        self.quantizer.set_quant_scale_zp(self.node.output[0],
                                          (out_scale, out_zero_point))
