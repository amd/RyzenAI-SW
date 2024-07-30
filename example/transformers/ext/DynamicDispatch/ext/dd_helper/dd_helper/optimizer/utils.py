##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy as np
import copy
import os
import csv
from collections import OrderedDict
from colorama import init, Fore

init(autoreset=True)

from .. import onnx_tool
from ..onnx_tool.fusion import *
from ..utils.cal_coeff_utils import *

# Node Names for pattern extraction
matmul_add_nodes = [
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
    "onnx::MatMul_2195_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/MatMul_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/Add",
    "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_QuantizeLinear",
]
matmul_add_gelu_nodes = [
    "/tulrv6/encoder/layer.0/attention/output/LayerNorm/Add_1_output_0_DequantizeLinear",
    "onnx::MatMul_2209_DequantizeLinear",
    "/tulrv6/encoder/layer.0/intermediate/dense/MatMul",
    "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/intermediate/dense/MatMul_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/intermediate/dense/Add",
    "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_DequantizeLinear",
    "Gelu_363",
    "/tulrv6/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1_output_0_QuantizeLinear",
]
mha_pattern_in_PSF = [
    "/tulrv6/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
    "/tulrv6/Constant_12_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
    "/tulrv6/Mul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
    "onnx::MatMul_2204_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
    "/tulrv6/Constant_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Sub",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
    "/tulrv6/embeddings/LayerNorm/Constant_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_2",
    "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
    "/tulrv6/GatherElements_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear",
]
PSF_a8w8_MHAGRPB = [
    "/tulrv6/encoder/layer.0/attention/self/value/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
    "/tulrv6/Constant_12_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_convert_DequantizeLinear",
    "/tulrv6/Mul_output_0_convert_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear__1",
    "onnx::MatMul_2204_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/MatMul_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_1_output_0_DequantizeLinear",
    "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
    "/tulrv6/Constant_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Sub",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
    "/tulrv6/embeddings/LayerNorm/Constant_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_2",
    "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_2_output_0_DequantizeLinear",
    "/tulrv6/GatherElements_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_4_output_0_convert_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_convert_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4_output_0_QuantizeLinear",
]
PSH_MHAGRPB = [
    "279_DequantizeLinear",
    "Reshape_198",
    "334_QuantizeLinear",
    "334_DequantizeLinear",
    "Transpose_199",
    "335_QuantizeLinear",
    "335_DequantizeLinear",
    "276_DequantizeLinear",
    "Reshape_186",
    "316_QuantizeLinear",
    "316_DequantizeLinear",
    "Transpose_200",
    "336_QuantizeLinear",
    "336_DequantizeLinear",
    "274_DequantizeLinear",
    "Reshape_173",
    "297_QuantizeLinear",
    "297_DequantizeLinear",
    "Transpose_174",
    "298_QuantizeLinear",
    "298_DequantizeLinear",
    "MatMul_201",
    "337_QuantizeLinear",
    "337_DequantizeLinear",
    "1062_DequantizeLinear",
    "Div_203",
    "339_QuantizeLinear",
    "339_DequantizeLinear",
    "110_DequantizeLinear",
    "Add_204",
    "340_QuantizeLinear",
    "340_DequantizeLinear",
    "298_DequantizeLinear__1",
    "1077_DequantizeLinear",
    "MatMul_214",
    "351_QuantizeLinear",
    "351_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "Add_215",
    "352_QuantizeLinear",
    "352_DequantizeLinear",
    "Reshape_223",
    "366_QuantizeLinear",
    "366_DequantizeLinear",
    "ReduceSum_225",
    "368_QuantizeLinear",
    "368_DequantizeLinear",
    "Sigmoid_226",
    "369_QuantizeLinear",
    "369_DequantizeLinear",
    "Slice_237",
    "380_QuantizeLinear",
    "380_DequantizeLinear",
    "369_DequantizeLinear__1",
    "Slice_240",
    "383_QuantizeLinear",
    "383_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "Mul_241",
    "384_QuantizeLinear",
    "384_DequantizeLinear",
    "107_DequantizeLinear__1",
    "Sub_243",
    "386_QuantizeLinear",
    "386_DequantizeLinear",
    "Mul_244",
    "387_QuantizeLinear",
    "387_DequantizeLinear",
    "130_DequantizeLinear",
    "Add_246",
    "389_QuantizeLinear",
    "389_DequantizeLinear",
    "271_DequantizeLinear",
    "Mul_247",
    "390_QuantizeLinear",
    "390_DequantizeLinear",
    "Add_248",
    "391_QuantizeLinear",
    "391_DequantizeLinear",
    "Softmax_249",
    "392_QuantizeLinear",
    "392_DequantizeLinear",
    "MatMul_250",
    "393_QuantizeLinear",
    "393_DequantizeLinear",
    "Transpose_251",
    "394_QuantizeLinear",
    "394_DequantizeLinear",
    "Reshape_263",
    "409_QuantizeLinear",
]
PSH_QMatMulADD = [
    "138_DequantizeLinear",
    "1068_DequantizeLinear",
    "MatMul_157",
    "273_QuantizeLinear",
    "273_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.query.bias_DequantizeLinear",
    "Add_158",
    "274_QuantizeLinear",
]
PSH_QMatMulADDGELU = [
    "424_DequantizeLinear",
    "1082_DequantizeLinear",
    "MatMul_278",
    "426_QuantizeLinear",
    "426_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
    "Add_279",
    "427_QuantizeLinear",
    "427_DequantizeLinear",
    "Gelu_229",
    "435_QuantizeLinear",
]
PSH_QMatMul = [
    "138_DequantizeLinear__1",
    "1069_DequantizeLinear",
    "MatMul_159",
    "276_QuantizeLinear",
]
PSH_SkipAdd = [
    "412_DequantizeLinear",
    "138_DequantizeLinear__3",
    "Add_265",
    "412_QuantizeLinear",
]
QDQ = [
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_DequantizeLinear",
]
PSH_a8w8_MHAGRPB = [
    "274_DequantizeLinear",
    "Reshape_173",
    "297_QuantizeLinear",
    "297_DequantizeLinear",
    "Transpose_174",
    "298_QuantizeLinear",
    "298_DequantizeLinear",
    "276_DequantizeLinear",
    "Reshape_186",
    "316_QuantizeLinear",
    "316_DequantizeLinear",
    "Transpose_200",
    "336_QuantizeLinear",
    "336_DequantizeLinear",
    "MatMul_201",
    "337_QuantizeLinear",
    "337_DequantizeLinear",
    "1062_DequantizeLinear",
    "Div_203",
    "339_QuantizeLinear",
    "339_DequantizeLinear",
    "339_convert_QuantizeLinear",
    "339_convert_DequantizeLinear",
    "110_convert_DequantizeLinear",
    "Add_204",
    "340_QuantizeLinear",
    "340_DequantizeLinear",
    "298_DequantizeLinear__1",
    "1077_DequantizeLinear",
    "MatMul_214",
    "351_QuantizeLinear",
    "351_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "Add_215",
    "352_QuantizeLinear",
    "352_DequantizeLinear",
    "Reshape_223",
    "366_QuantizeLinear",
    "366_DequantizeLinear",
    "ReduceSum_225",
    "368_QuantizeLinear",
    "368_DequantizeLinear",
    "Sigmoid_226",
    "369_QuantizeLinear",
    "369_DequantizeLinear",
    "Slice_237",
    "380_QuantizeLinear",
    "380_DequantizeLinear",
    "369_DequantizeLinear__1",
    "Slice_240",
    "383_QuantizeLinear",
    "383_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "Mul_241",
    "384_QuantizeLinear",
    "384_DequantizeLinear",
    "107_DequantizeLinear__1",
    "Sub_243",
    "386_QuantizeLinear",
    "386_DequantizeLinear",
    "Mul_244",
    "387_QuantizeLinear",
    "387_DequantizeLinear",
    "130_DequantizeLinear",
    "Add_246",
    "389_QuantizeLinear",
    "389_DequantizeLinear",
    "271_DequantizeLinear",
    "Mul_247",
    "390_QuantizeLinear",
    "390_DequantizeLinear",
    "390_convert_QuantizeLinear",
    "390_convert_DequantizeLinear",
    "Add_248",
    "391_QuantizeLinear",
    "391_DequantizeLinear",
    "Softmax_249",
    "392_QuantizeLinear",
    "392_DequantizeLinear",
    "392_convert_QuantizeLinear",
    "392_convert_DequantizeLinear",
    "279_DequantizeLinear",
    "Reshape_198",
    "334_QuantizeLinear",
    "334_DequantizeLinear",
    "Transpose_199",
    "335_QuantizeLinear",
    "335_DequantizeLinear",
    "MatMul_250",
    "393_QuantizeLinear",
    "393_DequantizeLinear",
    "Transpose_251",
    "394_QuantizeLinear",
    "394_DequantizeLinear",
    "Reshape_263",
    "409_QuantizeLinear",
]
PSH_a8w8_MatMulAdd_Gelu = [
    "424_DequantizeLinear",
    "1082_DequantizeLinear",
    "MatMul_278",
    "426_QuantizeLinear",
    "426_DequantizeLinear",
    "426_convert_QuantizeLinear",
    "426_convert_DequantizeLinear",
    "roberta_encoder_src.encoder.layer.0.intermediate.dense.bias_DequantizeLinear",
    "Add_279",
    "427_QuantizeLinear",
    "427_DequantizeLinear",
    "Gelu_fused_Erf_0",
    "435_QuantizeLinear",
]
### pattern for unit test not present in PSF
Qmatmul = ["DeQuantizeLinear_1", "matmul_1", "QuantizeLinear_2"]

QDQMatmul = [
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear__1",
    "onnx::MatMul_2196_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_QuantizeLinear",
]

QKTMATMUL = [
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
]
LAYERNORM = [
    "/tulrv6/embeddings/Add_2_output_0_DequantizeLinear",
    "tulrv6.embeddings.LayerNorm.weight_DequantizeLinear",
    "tulrv6.embeddings.LayerNorm.bias_DequantizeLinear",
    "LayerNormalization_242",
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear",
]
PSF_a8w8_LayerNorm = [
    "/tulrv6/embeddings/Add_2_output_0_DequantizeLinear",
    "tulrv6.embeddings.LayerNorm.weight_DequantizeLinear",
    "tulrv6.embeddings.LayerNorm.bias_DequantizeLinear",
    "LayerNormalization_fused_ReduceMean_0",
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_QuantizeLinear",
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear",
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_convert_QuantizeLinear",
]
Add = [
    "/tulrv6/embeddings/LayerNorm/Add_1_output_0_DequantizeLinear__3",
    "/tulrv6/encoder/layer.0/attention/output/dense/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/output/Add",
    "/tulrv6/encoder/layer.0/attention/output/Add_output_0_QuantizeLinear",
]

SIGMOID = [
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_QuantizeLinear",
]
SOFTMAX = [
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_QuantizeLinear",
]

MUL = [
    "tulrv6.encoder.layer.1.attention.self.eco_a_DequantizeLinear",
    "/tulrv6/encoder/layer.1/attention/self/Slice_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.1/attention/self/Mul_2",
    "/tulrv6/encoder/layer.1/attention/self/Mul_2_output_0_QuantizeLinear",
]
DIV = [
    "/tulrv6/Constant_12_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_QuantizeLinear",
]
SUB = [
    "/tulrv6/Constant_output_0_DequantizeLinear__1",
    "/tulrv6/encoder/layer.0/attention/self/Mul_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sub",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_QuantizeLinear",
]

GELU = [
    "/tulrv6/encoder/layer.0/intermediate/dense/Add_output_0_DequantizeLinear",
    "Gelu_363",
    "/tulrv6/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1_output_0_QuantizeLinear",
]
RESHAPE = [
    "/tulrv6/encoder/layer.0/attention/self/key/MatMul_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_QuantizeLinear",
]
TRANSPOSE = [
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_QuantizeLinear",
]
## Add and divide operators are having split(dq layer inputs to multiple mul/add nodes)
SLICE = [
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Slice",
    "/tulrv6/encoder/layer.0/attention/self/Slice_output_0_QuantizeLinear",
]
REDUCE_SUM = [
    "/tulrv6/encoder/layer.8/attention/self/Reshape_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.8/attention/self/ReduceSum",
    "/tulrv6/encoder/layer.8/attention/self/ReduceSum_output_0_QuantizeLinear",
]


# Pattern for MHA after fusing all the small ops and removing qdq after shape ops
MHA = [
    "/tulrv6/encoder/layer.0/attention/self/Reshape_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_2_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape",
    "/tulrv6/encoder/layer.0/attention/self/Transpose",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear",
    "/tulrv6/Constant_12_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Div_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Add_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Softmax_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_2",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_1_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1",
    "/tulrv6/encoder/layer.0/attention/self/MatMul_1_output_0_QuantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_3",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_4",
]

# unused but can be used to fuse GRPB seperately
GRPB = [
    "/tulrv6/encoder/layer.0/attention/self/query/Add_output_0_DequantizeLinear_1",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_output_0_DequantizeLinear_1",
    "/tulrv6/encoder/layer.0/attention/self/Transpose_output_0_DequantizeLinear_1",
    "tulrv6.encoder.layer.0.attention.self.gate_ur_linear.bias_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/gate_ur_linear/Add_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Reshape_3_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/ReduceSum_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear_1",
    "tulrv6.encoder.layer.0.attention.self.eco_a_DequantizeLinear",
    "/tulrv6/Constant_output_0_DequantizeLinear_1",
    "/tulrv6/encoder/layer.0/attention/self/Sigmoid_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Sub_output_0_DequantizeLinear",
    "/tulrv6/encoder/layer.0/attention/self/Mul_3_output_0_DequantizeLinear",
    "/tulrv6/GatherElements_output_0_DequantizeLinear",
]


dict1 = {
    "QMatMul": QDQMatmul,
    "QLayerNorm": LAYERNORM,
    "Qsigmoid": SIGMOID,
    "QMul": MUL,
    "QSoftmax": SOFTMAX,
    "QGelu": GELU,
    "QDiv": DIV,
    "QAdd": Add,
    "QSub": SUB,
    # "QReshape":RESHAPE,
    # "QTranspose":TRANSPOSE,
    "QReduceSum": REDUCE_SUM,
    # "QSlice":SLICE,
    "MHA": MHA,
    # "GRPB":GRPB,
}

dict2 = {"matmul_unit": QDQMatmul}

dict_PSF_a16 = {
    "QMHAGRPB": mha_pattern_in_PSF,
    "QLayerNorm": LAYERNORM,
    "QMatMulAddGelu": matmul_add_gelu_nodes,
    "QMatMulAdd": matmul_add_nodes,
    "QMatMul": QDQMatmul,
    "QSkipAdd": Add,
}

dict_PSF_a8 = {
    "QMHAGRPB": PSF_a8w8_MHAGRPB,
    "QLayerNorm": PSF_a8w8_LayerNorm,
}
dict_PSH_a16 = {
    "QMHAGRPB": PSH_MHAGRPB,
    "QMatMulAddGelu": PSH_QMatMulADDGELU,
    "QMatMulAdd": PSH_QMatMulADD,
    "QSkipAdd": PSH_SkipAdd,
}

dict_PSH_a8 = {
    "QMHAGRPB": PSH_a8w8_MHAGRPB,
    "QMatMulAddGelu": PSH_a8w8_MatMulAdd_Gelu,
}

layername_dict = {
    "QMHAGRPB": "MHAGRPB",
    "QLayerNorm": "LayerNorm",
    "QMatMulAddGelu": "MatMulAddGelu",
    "QMatMulAdd": "MatMulAdd",
    "QMatMul": "MatMul",
    "QSkipAdd": "Add",
}


def calc_q_params(g, nodes):
    correct_matmul = True
    QKT_input_qparams = []
    QKT_output_qparams = []
    VSQKT_input_qparams = []
    VSQKT_output_qparams = []
    softmax_input_qparams = []
    softmax_output_qparams = []
    params = []
    ini = []
    sub_scale = []
    add_scale = []
    sigmoid_params = []
    div_params = []
    grpb_matmul_add_out_params = []
    for node in nodes:
        if g.nodemap[node].op_type == "MatMul":
            correct_matmul = True
            parents = g.nodemap[node].prevnodes
            for i in parents:
                if i.op_type == "DequantizeLinear" and len(i.prevnodes) < 1:
                    correct_matmul = False
            QKT_matmul = False
            if correct_matmul == True:
                # print(node)
                if (
                    g.nodemap[node].prevnodes[0].prevnodes[0].prevnodes[0].op_type
                    == g.nodemap[node].prevnodes[1].prevnodes[0].prevnodes[0].op_type
                ):
                    for i in parents:
                        QKT_input_qparams.append(i.input[1])
                        QKT_input_qparams.append(i.input[2])
                    i = g.nodemap[node].nextnodes[0]
                    QKT_output_qparams.append(i.input[1])
                    QKT_output_qparams.append(i.input[2])

                elif (
                    g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].op_type
                    == "Transpose"
                ):
                    # print(node)
                    for i in parents:
                        VSQKT_input_qparams.append(i.input[1])
                        VSQKT_input_qparams.append(i.input[2])
                    i = g.nodemap[node].nextnodes[0]
                    VSQKT_output_qparams.append(i.input[1])
                    VSQKT_output_qparams.append(i.input[2])

        if g.nodemap[node].op_type == "Softmax":
            i = g.nodemap[node].prevnodes[0]
            softmax_input_qparams.append(i.input[1])
            softmax_input_qparams.append(i.input[2])
            if (
                g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].op_type
                == "MatMul"
            ):
                i = g.nodemap[node].nextnodes[0].nextnodes[0]
            elif (
                g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].op_type
                == "QuantizeLinear"
            ):
                i = g.nodemap[node].nextnodes[0].nextnodes[0].nextnodes[0].nextnodes[0]
            softmax_output_qparams.append(i.input[1])
            softmax_output_qparams.append(i.input[2])
            # breakpoint()

        if g.nodemap[node].op_type == "Sub":
            parents = g.nodemap[node].prevnodes
            for parent in parents:
                if parent.op_type == "DequantizeLinear" and len(parent.prevnodes) < 1:
                    sub_scale.extend(parent.input)

        if g.nodemap[node].op_type == "Div":
            parents = g.nodemap[node].prevnodes
            for parent in parents:
                if parent.op_type == "DequantizeLinear" and len(parent.prevnodes) < 1:
                    div_params.extend(parent.input)

        if g.nodemap[node].op_type == "Add":
            correct_add = False
            grpb_matmul_add = False
            parents = g.nodemap[node].prevnodes
            for parent in parents:
                if (
                    len(parent.prevnodes)
                    and parent.prevnodes[0].prevnodes[0].op_type == "Mul"
                ):
                    correct_add = True
            if correct_add == True:
                for parent in parents:
                    if (
                        parent.op_type == "DequantizeLinear"
                        and len(parent.prevnodes) < 1
                    ):
                        add_scale.extend(parent.input)
                        # print(add_scale)
            if correct_add == False:
                for parent in parents:
                    if (
                        parent.op_type == "DequantizeLinear"
                        and len(parent.prevnodes) < 1
                    ):
                        if g.tensormap[parent.input[0]].shape == (8,):
                            grpb_matmul_add = True
                if grpb_matmul_add == True:
                    nextnode = g.nodemap[node].nextnodes[0]
                    grpb_matmul_add_out_params.extend(nextnode.input[1:])

        if g.nodemap[node].op_type == "Sigmoid":
            parents = g.nodemap[node].prevnodes
            for parent in parents:
                if parent.op_type == "DequantizeLinear":
                    sigmoid_params.extend(parent.input[1:])
            nextnode = g.nodemap[node].nextnodes[0]
            sigmoid_params.extend(nextnode.input[1:])

    # params=params.extend()
    # print(add_scale)
    # breakpoint()
    return (
        QKT_input_qparams,
        QKT_output_qparams,
        VSQKT_input_qparams,
        VSQKT_output_qparams,
        softmax_input_qparams,
        softmax_output_qparams,
        sub_scale,
        add_scale,
        sigmoid_params,
        div_params,
        grpb_matmul_add_out_params,
    )


# Check if Matmul contains wts as one of the input
# if both the inputs are activations, we are not fusing them right now
def check_if_wts_matmul(g, nodes, tensormap):
    fuse = False
    mm_shape = ""
    for node in nodes:
        if g.nodemap[node].op_type == "MatMul" and g.nodemap[node].name != "MatMul_791":
            parents = g.nodemap[node].prevnodes
            mm_shape = tensormap[g.nodemap[node].output[0]].shape
            for parent in parents:
                if parent.op_type == "DequantizeLinear" and len(parent.prevnodes) < 1:
                    fuse = True

    return fuse, mm_shape


def check_if_wts_add(g, nodes):
    fuse = True

    for node in nodes:
        if g.nodemap[node].op_type == "Add" and g.nodemap[node].name != "Add_775":
            inputs = g.nodemap[node].input
            parents = g.nodemap[node].prevnodes

            for parent in parents:
                if parent.op_type == "DequantizeLinear":
                    if len(parent.prevnodes) < 1 and parent.input[0] not in g.input:
                        fuse = False
                    elif len(parent.prevnodes) < 1 and parent.input[0] in g.input:
                        fuse = True
            if inputs[0] == g.nodemap[nodes[0]].output[0]:
                continue
            else:
                nodes[0], nodes[1] = nodes[1], nodes[0]
    return fuse, nodes


def check_if_wts_add_profile(g, nodes):
    fuse = True

    for node in nodes:
        if g.nodemap[node].op_type == "Add" and g.nodemap[node].name != "Add_775":
            inputs = g.nodemap[node].input
            parents = g.nodemap[node].prevnodes

            for parent in parents:
                if parent.op_type == "DequantizeLinear":
                    if len(parent.prevnodes) < 1 and parent.input[0] not in g.input:
                        fuse = False
                    elif len(parent.prevnodes) < 1 and parent.input[0] in g.input:
                        fuse = True

    return fuse


def remove_abovedq_belowq_for_op(g, op_type):
    """
    Function to remove Dq layers above the op and Q layer below a given op_type like Transpose/Reshape/Slice etc
    - As QDQ on shape ops is not necessary
    """

    rm_node_list = []
    for node in g.nodemap:
        if g.nodemap[node].op_type == op_type:
            if g.nodemap[node].prevnodes[0].op_type == "DequantizeLinear":
                above_dq = g.nodemap[node].prevnodes[0].name

                rm_node_list.append(above_dq)
            if g.nodemap[node].nextnodes[0].op_type == "QuantizeLinear":
                belowq = g.nodemap[node].nextnodes[0].name

                rm_node_list.append(belowq)

    for rm_node in rm_node_list:
        g.skip_node(rm_node)
        g.graph_reorder_nodes()
    return g


def duplicate_layer(m, g, node_optype, save=False):
    """
    Duplicate a layer with multiple outputs, to facilitate fusion at a node with multiple outputs
    """

    node_keys = list(g.nodemap.keys())

    for node_name in node_keys:
        if (g.nodemap[node_name].op_type == node_optype) and (
            len(g.nodemap[node_name].nextnodes) > 1
        ):
            node = g.nodemap[node_name]

            orig_out_tensor = g.tensormap[node.output[0]]
            for i in range(1, len(node.nextnodes)):
                """
                1.Create new node
                2.new node's next node will be one of the next nodes
                3. Add new node in nodemap
                """

                new_node = copy.copy(node)
                new_node.name = node_name + "__" + str(i)
                new_node.nextnodes = [node.nextnodes[i]]
                if new_node.name not in g.nodemap.keys():
                    g.nodemap[new_node.name] = new_node
                """
                    4. Create output tensor
                    5. add it to tensormap
                    6. add output tensor name in new_node's output
                    """
                new_output_tensor = copy.copy(orig_out_tensor)
                new_output_tensor.name = orig_out_tensor.name + "_" + str(i)
                new_node.output = [new_output_tensor.name]
                """
                    7. Add tensor in tensormap
                    8. update preoducedby[new_tensor] with new_node
                    9. update consumedby[new_tensor] with one of the next nodes (ith node's name)
                    10. update the corresponding input of the consumer node with the created output tensor
                    """
                if new_output_tensor.name not in g.tensormap.keys():
                    g.tensormap[new_output_tensor.name] = new_output_tensor

                if new_output_tensor.name not in g.producedby:
                    g.producedby[new_output_tensor.name] = [new_node.name]
                # print(node.nextnodes[i].name)

                if new_output_tensor.name not in g.consumedby:
                    g.consumedby[new_output_tensor.name] = [node.nextnodes[i].name]

                    con_node = g.nodemap[node.nextnodes[i].name]
                    for i in range(len(con_node.input)):
                        if (
                            con_node.input[i] == node.output[0]
                        ):  # check old node's output
                            con_node.input[i] = new_node.output[
                                0
                            ]  # update new node's output
                            # new node's output consumed by update
                """
                    11. Update the consumed by of input tensor of the new_node ( currently it has the old node only )
                    """
                input_tensor_to_orig_node = node.input[0]

                g.consumedby[input_tensor_to_orig_node].append(new_node.name)
                """
                    12. update the prevnode's nextnodes
                    """
                if node.prevnodes:
                    prevnode = node.prevnodes[0]

                    prevnode.nextnodes.extend([new_node])

            zerothnextnode = node.nextnodes[0]
            node.nextnodes = [zerothnextnode]
            node.name = node_name

    g.graph_reorder_nodes()
    if save:
        g.save_model("PSF_v1.0_QReshape_dup.onnx", rawmodel=m.mproto)
    return g


def change_output_dtype(g):
    """
    Change the data type of output of Quantize and Dequantize layers according to zp and scale
    """
    nodes = g.nodemap.keys()
    for node_name in nodes:
        node = g.nodemap[node_name]
        if node.op_type == "QuantizeLinear":
            for input in node.input:
                if "z_p" in input or "zero_point" in input:
                    data_type = g.tensormap[input].numpy.dtype
                    # g.tensormap[input].numpy=g.tensormap[input].numpy.astype(np.int8)
                else:
                    data_type = np.int8
            g.tensormap[node.output[0]].dtype = data_type
        if node.op_type == "DequantizeLinear":
            for input in node.input:
                # if node.prevnodes==[] and g.tensormap[input].dtype!=np.int8 and 'scale' not in input and 'zero_point' not in input:
                #     g.tensormap[input].numpy=g.tensormap[input].numpy.astype(np.int8)

                # if "zp" in input or "zero_point" in input:
                #     data_type = g.tensormap[input].dtype
                # g.tensormap[input].numpy=g.tensormap[input].numpy.astype(np.int8)
                if "scale" in input:
                    data_type = g.tensormap[input].dtype
                else:
                    data_type = np.float32
            g.tensormap[node.output[0]].dtype = data_type
    return g


def removenode(g, op_type):
    rm_node_list = []
    for node in g.nodemap:
        if g.nodemap[node].op_type == op_type:
            rm_node_list.append(g.nodemap[node].name)

    for rm_node in rm_node_list:
        g.skip_node(rm_node)
        g.graph_reorder_nodes()
    return g


def loadmodel(input_path):
    m = onnx_tool.Model(input_path)
    g = m.graph
    g.graph_reorder_nodes()
    return m, g


def count_ops(g):
    # Utility to count op_types
    # Takes onnx_tool graph object
    # should load the model using loadmodel and pass g to this function
    # Return a dictionary
    op_count_dictionary = {}
    for node in g.nodemap:
        if g.nodemap[node].op_type in op_count_dictionary:
            op_count_dictionary[g.nodemap[node].op_type] += 1
        else:
            op_count_dictionary[g.nodemap[node].op_type] = 1
    return op_count_dictionary


def add_domain(g):
    # Utility to add domain name com.amd for each fused QOP

    ops = ["MatMul", "MatMulAdd", "MatMulAddGelu", "MHAGRPB", "LayerNorm", "SkipAdd"]
    for node_name in g.nodemap.keys():
        for op in ops:
            if "Q" + op in g.nodemap[node_name].op_type:
                node = g.nodemap[node_name]
                node.domain = "com.amd"
                flag = True

    return g


def get_node_names(g, op_type):
    return g.nodemap.keys()


def get_savepath(model_name, key=None):
    curr_path = os.getcwd()
    if key:
        if not os.path.exists(os.path.join(curr_path, model_name, key, "subgraphs")):
            os.makedirs(os.path.join(curr_path, model_name, key, "subgraphs"))
        save_path = os.path.join(curr_path, model_name, key, "subgraphs")
        if not os.path.exists(
            os.path.join(curr_path, model_name, key, "fused_subgraphs")
        ):
            os.makedirs(os.path.join(curr_path, model_name, key, "fused_subgraphs"))
        fuse_path = os.path.join(curr_path, model_name, key, "fused_subgraphs")
        return save_path, fuse_path
    else:
        if not os.path.exists(os.path.join(curr_path, model_name, "synth_models")):
            os.makedirs(os.path.join(curr_path, model_name, "synth_models"))
        return os.path.join(curr_path, model_name, "synth_models")


def get_layer_name_for_save(model_name, found_nodes, key, count=0):
    save_path, fuse_path = get_savepath(model_name, key)
    if "MatMul" in key:
        st = found_nodes[2]

        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "LayerNorm" in key:
        st = found_nodes[3]
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "MHAGRPB" in key:
        st = "MHAGRPB_Layer" + str(count)
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )
    if "SkipAdd" in key:
        st = found_nodes[2]
        st = st.replace("/", "_")
        return os.path.join(save_path, st + ".onnx"), os.path.join(
            fuse_path, "Fused_" + st + ".onnx"
        )


def check_datatypes(m, g, prompt_len, precision):
    # This function only works for a8w8
    # TODO add support for A16W8
    #   - Move the if precision  condition inside each optype
    verbose = False  # Set to True to enable prints
    dtype_list = []
    if precision == "a8w8":
        for n_m in g.nodemap.keys():
            layer_dict = {}
            node = g.nodemap[n_m]

            if "LayerNorm".lower() in node.op_type.lower():
                if g.tensormap[node.input[0]].dtype != np.uint16:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Input"
                    layer_dict["tensor name"] = node.input[0]
                    layer_dict["dtype"] = g.tensormap[node.input[0]].dtype
                    layer_dict["Expected dtype"] = "uint16"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

                if g.tensormap[node.output[0]].dtype != np.uint8:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Output"

                    layer_dict["tensor name"] = node.output[0]
                    layer_dict["dtype"] = g.tensormap[node.output[0]].dtype
                    layer_dict["Expected dtype"] = "uint8"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

            elif "MHAGRPB" in node.op_type:
                for inp in node.input[:3]:
                    if g.tensormap[inp].dtype != np.uint8:
                        layer_dict["Op Type"] = node.op_type
                        layer_dict["layer name"] = n_m
                        layer_dict["inp/op"] = "Input"

                        layer_dict["tensor name"] = inp
                        layer_dict["dtype"] = g.tensormap[inp].dtype
                        layer_dict["Expected dtype"] = "uint8"

                        dtype_list.append(layer_dict)
                        layer_dict = {}
                if g.tensormap[node.input[3]].dtype != np.uint16:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Input"

                    layer_dict["tensor name"] = inp
                    layer_dict["dtype"] = g.tensormap[inp].dtype
                    layer_dict["Expected dtype"] = "uint16"

                    dtype_list.append(layer_dict)
                    layer_dict = {}

                if g.tensormap[node.output[0]].dtype != np.uint8:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Output"

                    layer_dict["tensor name"] = node.output[0]
                    layer_dict["dtype"] = g.tensormap[node.output[0]].dtype
                    layer_dict["Expected dtype"] = "uint8"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

            elif "SkipAdd".lower() in node.op_type.lower():
                for inp in node.input[0:2]:
                    if g.tensormap[inp].dtype != np.uint8:
                        layer_dict["Op Type"] = node.op_type
                        layer_dict["layer name"] = n_m
                        layer_dict["inp/op"] = "Input"

                        layer_dict["tensor name"] = inp
                        layer_dict["dtype"] = g.tensormap[inp].dtype
                        layer_dict["Expected dtype"] = "uint8"

                        dtype_list.append(layer_dict)
                        layer_dict = {}
                if g.tensormap[node.output[0]].dtype != np.uint16:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Output"

                    layer_dict["tensor name"] = node.output[0]
                    layer_dict["dtype"] = g.tensormap[node.output[0]].dtype
                    layer_dict["Expected dtype"] = "uint16"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

            elif "MatMul".lower() in node.op_type.lower():
                inp = node.input[0]
                if g.tensormap[inp].dtype != np.uint8:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Input"

                    layer_dict["tensor name"] = inp
                    layer_dict["dtype"] = g.tensormap[inp].dtype
                    layer_dict["Expected dtype"] = "uint8"

                    dtype_list.append(layer_dict)
                    layer_dict = {}
                if g.tensormap[node.output[0]].dtype != np.uint8:
                    layer_dict["Op Type"] = node.op_type
                    layer_dict["layer name"] = n_m
                    layer_dict["inp/op"] = "Output"

                    layer_dict["tensor name"] = node.output[0]
                    layer_dict["dtype"] = g.tensormap[node.output[0]].dtype
                    layer_dict["Expected dtype"] = "uint8"
                    dtype_list.append(layer_dict)
                    layer_dict = {}

        if len(dtype_list):
            if verbose:
                # print(dtype_list)
                keys = dtype_list[0].keys()
                with open("unsupported_dtype.csv", "w", newline="") as output_file:
                    dict_writer = csv.DictWriter(output_file, keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(dtype_list)

            if verbose:
                print(
                    Fore.RED
                    + f"Model has unsupported data types, please refer to 'unsupported_dtype.csv' "
                )
                print(Fore.GREEN + f"Converting unsupported datatypes ...")

    # if force_dtype_change:
    g = update_output_shapes_dtypes(m, g, prompt_len, precision)
    return g


def update_output_shapes_dtypes(m, g, prompt_len, precision):
    for n_m in g.nodemap.keys():
        node = g.nodemap[n_m]
        if "LayerNorm".lower() in node.op_type.lower():
            # if g.tensormap[node.input[0]].dtype != np.uint16:
            if precision == "a8w8":
                g.tensormap[node.input[0]].dtype = "BFLOAT16"
                g.tensormap[node.output[0]].shape = g.tensormap[node.input[0]].shape
                for output in node.output:
                    g.tensormap[output].dtype = np.uint8
            # PSH and PSJ shoul go to else
            elif precision == "a16w8":
                g.tensormap[node.input[0]].dtype = "BFLOAT16"
                g.tensormap[node.output[0]].shape = g.tensormap[node.input[0]].shape
                for output in node.output:
                    g.tensormap[output].dtype = np.uint16

        elif "MHAGRPB" in node.op_type:
            if precision == "a8w8":
                g.tensormap[node.input[3]].dtype = "BFLOAT16"
                g.tensormap[node.output[0]].dtype = np.uint8
                g.tensormap[node.output[0]].shape = (1, prompt_len, 768)
            elif precision == "a16w8":
                g.tensormap[node.input[3]].dtype = "BFLOAT16"

                g.tensormap[node.output[0]].dtype = np.uint16
                g.tensormap[node.output[0]].shape = (1, prompt_len, 768)

        elif "SkipAdd".lower() in node.op_type.lower():
            if precision == "a8w8":
                for n_m in node.input:
                    g.tensormap[n_m].dtype = np.uint8
                g.tensormap[node.output[0]].dtype = "BFLOAT16"
                g.tensormap[node.output[0]].shape = (1, prompt_len, 768)
            elif precision == "a16w8":
                for n_m in node.input:
                    g.tensormap[n_m].dtype = np.uint16
                g.tensormap[node.output[0]].dtype = "BFLOAT16"
                g.tensormap[node.output[0]].shape = (1, prompt_len, 768)

        elif "MatMul".lower() in node.op_type.lower():
            if precision == "a8w8":
                g.tensormap[node.input[0]].dtype = np.uint8
                # print(g.tensormap[node.input[1]].shape[1])

                M = g.tensormap[node.input[0]].shape[1]
                N = g.tensormap[node.input[1]].shape[1]
                g.tensormap[node.output[0]].shape = (1, M, N)

                g.tensormap[node.output[0]].dtype = np.uint8
            elif precision == "a16w8":
                g.tensormap[node.input[0]].dtype = np.uint16
                # print(g.tensormap[node.input[1]].shape[1]
                M = g.tensormap[node.input[0]].shape[1]
                N = g.tensormap[node.input[1]].shape[1]
                g.tensormap[node.output[0]].shape = (1, M, N)
                g.tensormap[node.output[0]].dtype = np.uint16

    return g


def get_mha_inputs(g, Mha_input_dict):
    for n_m in g.nodemap.keys():
        if g.nodemap[n_m].domain == "com.amd":
            node = g.nodemap[n_m]
            # search for inputs of MHAGRPB (on the basis of matmul shapes)
            # If matmul add --- 2shapes
            #   --shape with 1152 N dim -> Query
            #   --Shape with 768 N dim -> Value
            # if MatMul only --1 shape
            #   -- shape with 1152 N dim-> Key
            if (
                "MatMulAdd".lower() in node.op_type.lower()
                and "Gelu".lower() not in node.op_type.lower()
            ):
                if (
                    len(node.nextnodes)
                    and "MHAGRPB".lower() in node.nextnodes[0].op_type.lower()
                ):
                    if node.nextnodes[0].name not in Mha_input_dict.keys():
                        Mha_input_dict[node.nextnodes[0].name] = {}
                    if g.tensormap[node.output[0]].shape[2] == 1152:
                        Mha_input_dict[node.nextnodes[0].name]["Q"] = node.output[0]
                    else:
                        Mha_input_dict[node.nextnodes[0].name]["V"] = node.output[0]

            elif (
                "MatMul".lower() in node.op_type.lower()
                and "gelu" not in node.op_type.lower()
                and "add" not in node.op_type.lower()
            ):
                if (
                    len(node.nextnodes) > 0
                    and "MHAGRPB".lower() in node.nextnodes[0].op_type.lower()
                ):
                    if node.nextnodes[0].name not in Mha_input_dict.keys():
                        Mha_input_dict[node.nextnodes[0].name] = {}
                    if g.tensormap[node.output[0]].shape[2] == 1152:
                        Mha_input_dict[node.nextnodes[0].name]["K"] = node.output[0]

                    for inp in node.input:
                        if len(g.tensormap[inp].shape) == 1:
                            actual_tensor_bias = g.tensormap[inp].numpy.data
                            pad_tensor_bias = np.zeros((1, g.tensormap[inp].shape[0]))
                            pad_tensor_bias[:, : actual_tensor_bias.shape[0]] = (
                                actual_tensor_bias
                            )
                            g.tensormap[inp] = onnx_tool.tensor.create_initial_Tensor(
                                inp,
                                pad_tensor_bias,
                            )
                            g.initials.append(inp)

    return g, Mha_input_dict


# def change_inputs(m, g,precision):
# #psh changes at Layernorm reducemean 1
# if len(g.nodemap)<150 and not gen_subgraphs:
#     if "413_DequantizeLinear" in g.nodemap.key() and "424_DequantizeLinear" in g.nodemap.keys():
#         node=g.nodemap["413_DequantizeLinear"]
#         node.output=[node.output[-1]]
#         matmul_input=node.output[0]
#         scale=node.input[-2]
#         zp=node.input[-1]
#         node=g.nodemap["424_DequantizeLinear"]
#         node.input[0]=matmul_input
#         node.input[1]=scale
#         node.input[2]=zp
#         g.graph_reorder_nodes()
#         g.save_model("PSH change.onnx")
#     # breakpoint()
def get_node_dtype(g, nodename):
    dtype_dict = {"a8w8": np.uint8, "a16w8": np.uint16}
    node = g.nodemap[nodename]
    if "matmul" in node.op_type.lower():
        inputs = node.input
        out_dtype = g.tensormap[inputs[-1]].dtype
        in_dtype = g.tensormap[inputs[0]].dtype
        # if out_dtype == in_dtype:
        #     pass
        # else:
        #     # breakpoint()
        #     print(
        #         "----------------------------------------------------------------------"
        #     )
        #     print(nodename)
        #     print(in_dtype)
        #     print(out_dtype)
        #     return out_dtype

    elif "MHAGRPB".lower() in node.op_type.lower():
        mha_input_dict = {}
        mha_input_dict = get_mha_inputs(g, mha_input_dict)
        inputs = node.input
        in_dtype = g.tensormap[inputs[0]].dtype

        out_dtype = g.tensormap[inputs[-1]].dtype
        # in_dtype_Q=g.tensormap[mha_input_dict[nodename]['Q']].dtype
        # in_dtype_K=g.tensormap[mha_input_dict[nodename]['K']].dtype
        # in_dtype_V=g.tensormap[mha_input_dict[nodename]['V']].dtype
        # # in_dtype_att=g.tensormap[inputs[0]].dtype
        # if out_dtype==in_dtype_Q:
        #     pass
        # else:
        #     # breakpoint()
        #     print("----------------------------------------------------------------------")
        #     print("Inpu and output datatype mismatch in the node")

        #     print(nodename)
        #     print(in_dtype)
        #     print(out_dtype)

        # breakpoint()
    elif "Layernorm".lower() in node.op_type.lower():
        out_dtype = g.tensormap[node.input[-1]].dtype
        in_dtype = g.tensormap[node.input[0]].dtype

    return out_dtype


def change_inputs(m, g, precision):
    # print(precision)

    # print("QDQ Params extraction for precision = "+precision)
    dtype_dict = {"a8w8": np.uint8, "a16w8": np.uint16}
    remove_list = []
    Mha_input_dict = {}
    graph_input = g.tensormap[g.input[0]].shape
    modified_graph_input = []
    prompt_len = graph_input[1]
    if prompt_len < 128:
        prompt_len_modified = 128
    else:
        prompt_len_modified = prompt_len
    # get a dictionary of MHA layers with respective Q,K,V input names
    g, Mha_input_dict = get_mha_inputs(g, Mha_input_dict)

    ## Below snippet adds QDQ params to nodes
    # Each optype has differnt QDQ coefficients that are packed into QDQ tensors and passed as input to the nod

    mha_count = 0
    for n_m in g.nodemap.keys():
        if g.nodemap[n_m].domain == "com.amd":
            # MatMulAdd
            if (
                "MatMulAdd".lower() in g.nodemap[n_m].op_type.lower()
                and "gelu" not in g.nodemap[n_m].op_type.lower()
            ):
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        # print(n_m)
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )
                            c += 1
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))

                if node_dtype == np.uint8:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = compute_qdq_coeff_matmul_bias(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[-5],
                        args[-4],
                        args[-3],
                        args[-2],
                        args[-1],
                    )
                    is_int16 = 0
                elif node_dtype == np.uint16:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[-5],
                        args[-4],
                        args[-3],
                        args[-2],
                        args[-1],
                    )
                    is_int16 = 1

                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = 0
                qdq_tensor[5] = 64
                qdq_tensor[6] = 64
                qdq_tensor[7] = shift_qb
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = matmul_shift
                qdq_tensor[10] = is_int16

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input
                node.input.pop(2)
            # MatMulAddGelu
            if "MatMulAddGelu".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                c = 0
                for inp in inputs:
                    if inp in g.initials:
                        # print(n_m)
                        if inp not in input_dict:
                            input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                        else:
                            input_dict[inp + str(c)] = np.asarray(
                                g.tensormap[inp].numpy
                            )

                args = list(input_dict.values())

                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))

                # print(len(inputs)) #To Debug length of inputs

                if node_dtype == np.uint8:
                    if len(inputs) == 19:
                        (
                            c0,
                            c1,
                            c2,
                            shift_qb,
                            shift_out,
                            matmul_shift,
                        ) = compute_qdq_coeff_matmul_bias(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[9],
                            args[10],
                            args[11],
                            args[12],
                            args[13],
                        )
                    else:
                        (
                            c0,
                            c1,
                            c2,
                            shift_qb,
                            shift_out,
                            matmul_shift,
                        ) = compute_qdq_coeff_matmul_bias(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[13],
                            args[14],
                            args[15],
                            args[16],
                            args[17],
                        )
                    is_int16 = 0
                elif node_dtype == np.uint16:
                    if len(inputs) == 19:
                        (
                            c0,
                            c1,
                            c2,
                            shift_qb,
                            shift_out,
                            matmul_shift,
                        ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[9],
                            args[10],
                            args[11],
                            args[12],
                            args[13],
                        )
                    else:
                        (
                            c0,
                            c1,
                            c2,
                            shift_qb,
                            shift_out,
                            matmul_shift,
                        ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[13],
                            args[14],
                            args[15],
                            args[16],
                            args[17],
                        )
                    is_int16 = 1

                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                # qdq_tensor[0] = 0
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = 0
                qdq_tensor[5] = 64
                qdq_tensor[6] = 64
                qdq_tensor[7] = shift_qb
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = matmul_shift
                qdq_tensor[10] = is_int16

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                gelu_qdq = EltwiseAdd(args[-4], args[-3], (1 / args[-2]), args[-1])
                c0_scale_a, c0_zp_a, c0_scale_b, c0_zp_b = gelu_qdq.cal_coeff()
                gelu_qdq_tensor = np.zeros((16)).astype(np.int32)
                gelu_qdq_tensor[0] = c0_zp_a
                gelu_qdq_tensor[1] = c0_scale_a
                gelu_qdq_tensor[2] = c0_zp_b
                gelu_qdq_tensor[3] = c0_scale_b
                gelu_qdq_tensor[4] = is_int16

                g.tensormap[n_m + "gelu_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "gelu_qdq_", gelu_qdq_tensor
                )
                g.initials.append(n_m + "gelu_qdq_")

                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                modified_input.append(n_m + "gelu_qdq_")
                node.input = modified_input
                node.input.pop(2)

            # MatMul
            if (
                "MatMul" in g.nodemap[n_m].op_type
                and "Add" not in g.nodemap[n_m].op_type
                # and g.nodemap[n_m].name != "MatMul_791"
            ):
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_dict = OrderedDict()
                for inp in inputs:
                    if inp in g.initials:
                        # print(n_m)
                        input_dict[inp] = np.asarray(g.tensormap[inp].numpy)
                args = list(input_dict.values())
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                if node_dtype == np.uint8:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = qdq_matmul_uint8_uint8_cstm(args)
                    is_int16 = 0
                elif node_dtype == np.uint16:
                    (
                        c0,
                        c1,
                        c2,
                        shift_qb,
                        shift_out,
                        matmul_shift,
                    ) = qdq_matmul_uint16_uint8_cstm(args)
                    is_int16 = 1

                g.tensormap[n_m + "_c0_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_c0_", c0
                )
                g.initials.append(n_m + "_c0_")
                qdq_tensor = np.zeros((16)).astype(np.int32)
                qdq_tensor.view(np.int64)[0] = 0
                qdq_tensor[2] = c1
                qdq_tensor[3] = c2
                qdq_tensor[4] = c2
                qdq_tensor[5] = 64
                qdq_tensor[6] = 64
                qdq_tensor[7] = shift_qb
                qdq_tensor[8] = shift_out
                qdq_tensor[9] = matmul_shift
                qdq_tensor[10] = is_int16
                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_c0_")
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

            # LayerNorm
            if "LayerNorm" in g.nodemap[n_m].op_type:
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []
                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)
                # TODO Pass appropriate scale and zp
                # Adding qparams as attrs

                convert_params_inps = []
                convert_params_ops = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)

                lrnlayer = LRN(1 / input_list[-2], input_list[-1])

                lrn_c0, lrn_c1 = lrnlayer.cal_coeff()
                lrn_qdq_tensor = np.zeros((16)).astype(np.int32)
                lrn_qdq_tensor[0] = lrn_c0
                lrn_qdq_tensor[1] = lrn_c1
                node_dtype = get_node_dtype(g, n_m)
                if node_dtype == np.uint8:
                    lrn_qdq_tensor[2] = 0
                else:
                    lrn_qdq_tensor[2] = 1
                node.set_attr("Node_Dtype", str(node_dtype))

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", lrn_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")
                scale_tensor = dequantize_bf16(
                    input_list[2], input_list[3], input_list[4]
                )

                beta_tensor = dequantize_bf16(
                    input_list[5], input_list[6], input_list[7]
                )
                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_qdq_")
                g.tensormap[modified_input[1]].numpy = scale_tensor

                g.tensormap[modified_input[2]].numpy = beta_tensor
                node.input = modified_input

            if "skipadd".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = []
                input_list = []
                for inp in inputs:
                    if inp in g.initials:
                        input_list.append(g.tensormap[inp].numpy)
                # Adding qparams as attrs
                convert_params_ops = []
                convert_params_inps = []
                convert_params_inps.append(input_list[0].item())
                convert_params_inps.append(input_list[1].item())
                convert_params_inps.append(input_list[2].item())
                convert_params_inps.append(input_list[3].item())
                convert_params_ops.append(input_list[-2].item())
                convert_params_ops.append(input_list[-1].item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)

                eltadd = EltwiseAdd(
                    input_list[0], input_list[1], input_list[2], input_list[3]
                )

                elt_c0, elt_c1, elt_c2, elt_c3 = eltadd.cal_coeff()
                elt_qdq_tensor = np.zeros((16)).astype(np.int32)
                elt_qdq_tensor[0] = elt_c0
                elt_qdq_tensor[1] = elt_c1
                elt_qdq_tensor[2] = elt_c2
                elt_qdq_tensor[3] = elt_c3

                g.tensormap[n_m + "_qdq_"] = onnx_tool.tensor.create_initial_Tensor(
                    n_m + "_qdq_", elt_qdq_tensor
                )
                g.initials.append(n_m + "_qdq_")

                for inp in inputs:
                    if (
                        ("scale" not in inp)
                        and ("zero_point" not in inp)
                        and ("ort" not in inp)
                    ):
                        modified_input.append(inp)
                modified_input.append(n_m + "_qdq_")
                node.input = modified_input

            if "MHAGRPB".lower() in g.nodemap[n_m].op_type.lower():
                node = g.nodemap[n_m]
                inputs = node.input
                modified_input = OrderedDict()
                if Mha_input_dict:
                    modified_input[0] = inputs[
                        inputs.index(Mha_input_dict[n_m]["Q"])
                    ]  # Query output

                    modified_input[1] = inputs[
                        inputs.index(Mha_input_dict[n_m]["K"])
                    ]  # Key

                    modified_input[2] = inputs[
                        inputs.index(Mha_input_dict[n_m]["V"])
                    ]  # value

                else:
                    # For subgraph, as we do not have matmuls to define the Q,K,V inputs we are relying on the input sequence
                    # input sequence is inturn decided by the way the order of nodenames in the pattern matching code
                    # order of nodenames in the pattern matching code
                    #     -- V is only one of size [1,prompt_len,768], so no issue
                    #     -- K,Q -- ambiguity occurs here as both of them have same shapes. but in all the patterns in pattern dictionary (k appears before Q)
                    # TODO: come up with beter solution for K,Q ambiguity (what if the order changes in upcomming new models)
                    index = 0
                    for i in node.input:
                        if g.tensormap[i].shape == [1, prompt_len, 768]:
                            v = i
                        elif (
                            g.tensormap[i].shape == [1, prompt_len, 1152] and index == 0
                        ):  # K appears first
                            k = i
                            index += 1
                        elif (
                            g.tensormap[i].shape == [1, prompt_len, 1152] and index == 1
                        ):  # q appears after K with same shape
                            q = i
                    modified_input[0] = q
                    modified_input[1] = k
                    modified_input[2] = v

                for inp in inputs:
                    if (
                        g.tensormap[inp].shape == [1, 1, 1, prompt_len]
                        or g.tensormap[inp].shape == (1, 1, 1, prompt_len)
                        or g.tensormap[inp].shape == [1, 1, 1, prompt_len_modified]
                        or g.tensormap[inp].shape == (1, 1, 1, prompt_len_modified)
                    ):
                        if prompt_len < 128:
                            g.tensormap[inp].shape[-1] = prompt_len_modified

                        modified_input[3] = inp
                        attention_mask_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        attention_mask_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                    # TODO how to remove this hardcoding below (98,8)
                    if g.tensormap[inp].shape == (96, 8) or g.tensormap[inp].shape == [
                        96,
                        8,
                    ]:
                        modified_input[4] = inp
                        g.tensormap[inp].numpy = g.tensormap[inp].numpy
                        grpb_matmul_wts = g.tensormap[inp].numpy
                        # TODO Check if the wts are uint8
                        grpb_matmul_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        grpb_matmul_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                    if g.tensormap[inp].shape == (
                        1,
                        12,
                        prompt_len,
                        prompt_len,
                    ) or g.tensormap[inp].shape == [
                        1,
                        12,
                        prompt_len_modified,
                        prompt_len,
                    ]:
                        modified_input[7] = inp + str(mha_count)
                        # TODO Check dtype
                        grpb_bias = g.tensormap[inp].numpy.data
                        grpb_bias = np.asarray(grpb_bias).reshape(
                            12, prompt_len, prompt_len
                        )
                        t1 = onnx_tool.tensor.create_initial_Tensor(
                            inp + str(mha_count),
                            grpb_bias,
                        )  # 12xprompt_lenxprompt_len
                        g.tensormap[inp + str(mha_count)] = t1
                        g.initials.append(inp + str(mha_count))
                        mha_count += 1
                        grpb_mul_ini_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        grpb_mul_ini_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                    # collect bias of size 8 from GRPB block the s,zp here
                    if g.tensormap[inp].shape == (8,) or g.tensormap[inp].shape == [
                        8,
                    ]:
                        # TODO Check dtype
                        grpb_matmul_bias = g.tensormap[inp].numpy

                        grpb_matmul_bias_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        grpb_matmul_bias_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                    if g.tensormap[inp].shape == (1, 12, 1, 1) or g.tensormap[
                        inp
                    ].shape == [1, 12, 1, 1]:
                        # TODO Check dtype
                        grpb_mul_1 = g.tensormap[inp].numpy

                        grpb_mul1_scale = g.tensormap[
                            inputs[inputs.index(inp) + 1]
                        ].numpy.astype(np.float32)
                        grpb_mul1_zp = g.tensormap[
                            inputs[inputs.index(inp) + 2]
                        ].numpy.astype(np.int32)

                key_scale = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[1]) + 1]].numpy
                ).astype(np.float32)
                key_zp = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[1]) + 2]].numpy
                ).astype(np.int32)
                query_scale = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[0]) + 1]].numpy
                ).astype(np.float32)
                query_zp = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[0]) + 2]].numpy
                ).astype(np.int32)
                v_scale = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[2]) + 1]].numpy
                ).astype(np.float32)
                v_zp = np.asarray(
                    g.tensormap[inputs[inputs.index(modified_input[2]) + 2]].numpy
                ).astype(np.int32)
                if not isinstance(node.attr["QKT_output_qparams"][0], str):
                    QKT_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["QKT_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    QKT_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["QKT_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    softmax_inp_scale = np.asarray(
                        g.tensormap[
                            node.attr["softmax_input_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    softmax_inp_zp = np.asarray(
                        g.tensormap[
                            node.attr["softmax_input_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    softmax_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["softmax_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    softmax_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["softmax_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    VSQKT_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_output_qparams"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    VSQKT_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["VSQKT_output_qparams"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    sigmoid_input_scale = np.asarray(
                        g.tensormap[
                            node.attr["sigmoid_params"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    sigmoid_input_zp = np.asarray(
                        g.tensormap[
                            node.attr["sigmoid_params"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    sigmoid_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["sigmoid_params"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    sigmoid_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["sigmoid_params"][3].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    sub_wts = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_sub_params"][0].decode("utf-8")
                        ].numpy
                    )
                    sub_scale = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_sub_params"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    sub_zp = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_sub_params"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    add_wts = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_add_params"][0].decode("utf-8")
                        ].numpy
                    )
                    add_scale = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_add_params"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    add_zp = np.asarray(
                        g.tensormap[
                            node.attr["GRPB_add_params"][2].decode("utf-8")
                        ].numpy
                    ).astype(np.int32)
                    # TODO Check dtype
                    div_wts = np.asarray(
                        g.tensormap[node.attr["div_params"][0].decode("utf-8")].numpy
                    )
                    div_scale = np.asarray(
                        g.tensormap[node.attr["div_params"][1].decode("utf-8")].numpy
                    ).astype(np.float32)
                    div_zp = np.asarray(
                        g.tensormap[node.attr["div_params"][2].decode("utf-8")].numpy
                    ).astype(np.int32)
                    grpb_matmul_output_scale = np.asarray(
                        g.tensormap[
                            node.attr["grpb_matmul_add_out_params"][0].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)
                    grpb_matmul_output_zp = np.asarray(
                        g.tensormap[
                            node.attr["grpb_matmul_add_out_params"][1].decode("utf-8")
                        ].numpy
                    ).astype(np.float32)

                else:
                    QKT_output_scale = np.asarray(
                        g.tensormap[node.attr["QKT_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    QKT_output_zp = np.asarray(
                        g.tensormap[node.attr["QKT_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    softmax_inp_scale = np.asarray(
                        g.tensormap[node.attr["softmax_input_qparams"][0]].numpy
                    ).astype(np.float32)
                    softmax_inp_zp = np.asarray(
                        g.tensormap[node.attr["softmax_input_qparams"][1]].numpy
                    ).astype(np.int32)
                    softmax_output_scale = np.asarray(
                        g.tensormap[node.attr["softmax_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    softmax_output_zp = np.asarray(
                        g.tensormap[node.attr["softmax_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    VSQKT_output_scale = np.asarray(
                        g.tensormap[node.attr["VSQKT_output_qparams"][0]].numpy
                    ).astype(np.float32)
                    VSQKT_output_zp = np.asarray(
                        g.tensormap[node.attr["VSQKT_output_qparams"][1]].numpy
                    ).astype(np.int32)
                    sigmoid_input_scale = np.asarray(
                        g.tensormap[node.attr["sigmoid_params"][0]].numpy
                    ).astype(np.float32)
                    sigmoid_input_zp = np.asarray(
                        g.tensormap[node.attr["sigmoid_params"][1]].numpy
                    ).astype(np.int32)
                    sigmoid_output_scale = np.asarray(
                        g.tensormap[node.attr["sigmoid_params"][2]].numpy
                    ).astype(np.float32)
                    sigmoid_output_zp = np.asarray(
                        g.tensormap[node.attr["sigmoid_params"][3]].numpy
                    ).astype(np.int32)
                    # TODO Check dtype
                    sub_wts = np.asarray(
                        g.tensormap[node.attr["GRPB_sub_params"][0]].numpy
                    )
                    sub_scale = np.asarray(
                        g.tensormap[node.attr["GRPB_sub_params"][1]].numpy
                    ).astype(np.float32)
                    sub_zp = np.asarray(
                        g.tensormap[node.attr["GRPB_sub_params"][2]].numpy
                    ).astype(np.int32)
                    # TODO Check dtype
                    add_wts = np.asarray(
                        g.tensormap[node.attr["GRPB_add_params"][0]].numpy
                    )
                    add_scale = np.asarray(
                        g.tensormap[node.attr["GRPB_add_params"][1]].numpy
                    ).astype(np.float32)
                    add_zp = np.asarray(
                        g.tensormap[node.attr["GRPB_add_params"][2]].numpy
                    ).astype(np.int32)
                    # TODO Check dtype
                    div_wts = np.asarray(g.tensormap[node.attr["div_params"][0]].numpy)
                    div_scale = np.asarray(
                        g.tensormap[node.attr["div_params"][1]].numpy
                    ).astype(np.float32)
                    div_zp = np.asarray(
                        g.tensormap[node.attr["div_params"][2]].numpy
                    ).astype(np.int32)
                    grpb_matmul_output_scale = np.asarray(
                        g.tensormap[node.attr["grpb_matmul_add_out_params"][0]].numpy
                    ).astype(np.float32)
                    grpb_matmul_output_zp = np.asarray(
                        g.tensormap[node.attr["grpb_matmul_add_out_params"][1]].numpy
                    ).astype(np.float32)
                # MHAGRPB qparam attribute
                convert_params_ops = []
                convert_params_inps = []
                convert_params_inps.append(query_scale.item())
                convert_params_inps.append(query_zp.item())
                convert_params_inps.append(key_scale.item())
                convert_params_inps.append(key_zp.item())
                convert_params_inps.append(v_scale.item())
                convert_params_inps.append(v_zp.item())
                convert_params_inps.append(attention_mask_scale.item())
                convert_params_inps.append(attention_mask_zp.item())
                convert_params_ops.append(g.tensormap[inputs[-2]].numpy.item())
                convert_params_ops.append(g.tensormap[inputs[-1]].numpy.item())
                node.set_attr("input_q_params", convert_params_inps)
                node.set_attr("output_q_params", convert_params_ops)
                node_dtype = get_node_dtype(g, n_m)
                node.set_attr("Node_dtype", str(node_dtype))
                if node_dtype == np.uint8:
                    coeff_qkt = qdq_act_matmul_uint8_uint8_cstm(
                        query_scale,
                        query_zp,
                        np.asarray(96).astype(np.int32),
                        key_scale,
                        key_zp,
                        QKT_output_scale,
                        QKT_output_zp,
                    )
                    coeff_smv = qdq_act_matmul_uint8_uint8_cstm(
                        softmax_output_scale,
                        softmax_output_zp,
                        np.asarray(512).astype(np.int32),
                        v_scale,
                        v_zp,
                        VSQKT_output_scale,
                        VSQKT_output_zp,
                    )
                    is_qkt_smv_int16 = 0
                    # print(prompt_len_modified)
                elif node_dtype == np.uint16:
                    # TODO : check change PSJ

                    coeff_qkt = qdq_act_matmul_uint16_uint16_cstm(
                        query_scale,
                        query_zp,
                        np.asarray(96).astype(np.int32),
                        key_scale,
                        key_zp,
                        QKT_output_scale,
                        QKT_output_zp,
                    )

                    coeff_smv = qdq_act_matmul_uint16_uint16_cstm(
                        softmax_output_scale,
                        softmax_output_zp,
                        np.asarray(prompt_len_modified).astype(np.int32),
                        v_scale,
                        v_zp,
                        VSQKT_output_scale,
                        VSQKT_output_zp,
                    )
                    is_qkt_smv_int16 = 1

                # breakpoint()

                div_val = (div_wts.astype(np.float32) - div_zp) * div_scale.astype(
                    np.float32
                )
                qdq_sm_in = LRN((QKT_output_scale / div_val), QKT_output_zp).cal_coeff()

                qdq_sm_out = LRN(
                    1 / softmax_output_scale, softmax_output_zp
                ).cal_coeff()

                qdq_params = mha_qdq_params_fill(
                    coeff_qkt, coeff_smv, qdq_sm_in, qdq_sm_out, is_qkt_smv_int16
                )
                if node_dtype == np.uint8:
                    (
                        c0_gate_linear,
                        c1_gate_linear,
                        c2_gate_linear,
                        shift_qb_gate_linear,
                        shift_out_gate_linear,
                        matmul_shift_date_linear,
                    ) = compute_qdq_coeff_matmul_bias(
                        query_scale,
                        query_zp,
                        grpb_matmul_wts,
                        grpb_matmul_scale,
                        grpb_matmul_zp,
                        grpb_matmul_bias,
                        grpb_matmul_bias_scale,
                        grpb_matmul_bias_zp,
                        grpb_matmul_output_scale,
                        grpb_matmul_output_zp,
                    )
                    is_grpb_int16 = 0
                elif node_dtype == np.uint16:
                    (
                        c0_gate_linear,
                        c1_gate_linear,
                        c2_gate_linear,
                        shift_qb_gate_linear,
                        shift_out_gate_linear,
                        matmul_shift_date_linear,
                    ) = dq_uint16A_uint8W_bias_matmul_q_param_gen(
                        query_scale,
                        query_zp,
                        grpb_matmul_wts,
                        grpb_matmul_scale,
                        grpb_matmul_zp,
                        grpb_matmul_bias,
                        grpb_matmul_bias_scale,
                        grpb_matmul_bias_zp,
                        grpb_matmul_output_scale,
                        grpb_matmul_output_zp,
                    )
                    is_grpb_int16 = 1

                coeff_grbp = [
                    c1_gate_linear,
                    c2_gate_linear,
                    shift_qb_gate_linear,
                    shift_out_gate_linear,
                    matmul_shift_date_linear,
                ]
                gprb_vec64 = grpb_qgprb_vec64_fill(
                    c0_gate_linear, coeff_qkt[0], coeff_smv[0]
                )

                gprb_vec32 = gprb_vec32_fill(
                    coeff_grbp,
                    grpb_matmul_output_scale,
                    grpb_matmul_output_zp,
                    grpb_mul_ini_scale,
                    grpb_mul_ini_zp,
                    grpb_mul_1,
                    grpb_mul1_scale,
                    grpb_mul1_zp,
                    sub_wts,
                    sub_scale,
                    sub_zp,
                    add_wts,
                    add_scale,
                    add_zp,
                    is_grpb_int16,
                )

                g.tensormap[n_m + "_mha_np_grpb_vec"] = (
                    onnx_tool.tensor.create_initial_Tensor(
                        n_m + "_mha_np_grpb_vec", gprb_vec64
                    )
                )
                # Key s,zp, query s,zp, qkt_out s,zp
                g.tensormap[n_m + "_mha_np_grpb_qdq"] = (
                    onnx_tool.tensor.create_initial_Tensor(
                        n_m + "_mha_np_grpb_qdq", gprb_vec32
                    )
                )
                g.tensormap[n_m + "_mha_np_qdq"] = (
                    onnx_tool.tensor.create_initial_Tensor(
                        n_m + "_mha_np_qdq", qdq_params
                    )
                )
                g.initials.append(n_m + "_mha_np_grpb_vec")
                g.initials.append(n_m + "_mha_np_grpb_qdq")
                g.initials.append(n_m + "_mha_np_qdq")
                modified_input[5] = n_m + "_mha_np_grpb_vec"
                modified_input[6] = n_m + "_mha_np_grpb_qdq"
                modified_input[8] = n_m + "_mha_np_qdq"

                modified_input_list = []

                for i in range(0, 9):
                    modified_input_list.append(modified_input[i])
                node.input = modified_input_list

    g = check_datatypes(m, g, prompt_len_modified, precision)
    const_inputs = []
    for n_m in g.nodemap.keys():
        if "Constant" in g.nodemap[n_m].op_type:
            remove_list.append(n_m)
            node = g.nodemap[n_m]
            g.tensormap[node.output[0]] = onnx_tool.tensor.Tensor(node.attr["value"])

            g.initials.append(node.output[0])

    for n_m in g.nodemap.keys():
        if g.nodemap[n_m].op_type in layername_dict.keys():
            g.nodemap[n_m].op_type = layername_dict[g.nodemap[n_m].op_type]
    return g


# def change_fused_subgraphs():

# if __name__ == "__main__":
#     file = "before_change_inputs.onnx"
#     m, g = loadmodel(file)
#     g = change_inputs(m, g)
