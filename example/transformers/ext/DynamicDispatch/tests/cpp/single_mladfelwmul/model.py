import os
import numpy as np
import onnx
from onnx import mapping
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
    make_tensor,
)
from onnx.checker import check_model
from dd_helper.optimizer import onnx_graph as ogm
from dd_helper.optimizer import fuse
import argparse
import struct
from functools import reduce

np.random.seed(0)

def convert_float_to_qint(in_f):
    ret = struct.unpack('I', struct.pack('f', in_f))[0]
    ret &= 0x7fffffff  # Remove sign bit
    return ret

def get_shift_from_int32_rep(rep):
    shift = 127 - (((rep >> 23) & 255) + 1) + (8 * struct.calcsize('i') - 2)
    return shift

class QuantParam:
    def __init__(self, ifm1_scale, ifm2_scale, ofm_scale, ifm1_zero_point, ifm2_zero_point, ofm_zero_point, tensor_sz):
        self.ifm1_scale = ifm1_scale
        self.ifm2_scale = ifm2_scale
        self.ofm_scale = ofm_scale
        self.ifm1_zero_point = ifm1_zero_point
        self.ifm2_zero_point = ifm2_zero_point
        self.ofm_zero_point = ofm_zero_point
        self.tensor_sz = tensor_sz

        self.coeff0 = None
        self.coeff1 = None
        self.c0_shift = None
        self.c1_shift = None

    def compute_qdq(self):
        C0 = self.ifm1_scale * self.ifm2_scale / self.ofm_scale
        c0_qint = convert_float_to_qint(C0)
        c0_shift = get_shift_from_int32_rep(c0_qint)
        self.c0_shift = c0_shift
        self.coeff0 = int(C0 * 2**self.c0_shift)
        C1 = C0 * self.ifm1_zero_point * self.ifm2_zero_point + self.ofm_zero_point
        c1_qint = convert_float_to_qint(C1)
        c1_shift = get_shift_from_int32_rep(c1_qint)
        self.c1_shift = c1_shift
        self.coeff1 = int(C1 * 2**self.c1_shift)
        return

    # Set QDQ Param for Kernel
    def get_params_array(self):
        params_array = np.zeros(22, dtype=np.int8)
        params_array[0:4] = np.array([self.tensor_sz], dtype=np.int32).view(np.int8)
        params_array[4:8] = np.array([self.coeff0], dtype=np.int32).view(np.int8)
        params_array[8:12] = np.array([self.coeff1], dtype=np.int32).view(np.int8)
        ifm1_zp = np.uint16(self.ifm1_zero_point)
        params_array[12] = ifm1_zp & 0xFF
        params_array[13] = (ifm1_zp >> 8) & 0xFF
        ifm2_zp = np.uint16(self.ifm2_zero_point)
        params_array[14] = ifm2_zp & 0xFF
        params_array[15] = (ifm2_zp >> 8) & 0xFF
        params_array[16] = self.c0_shift
        params_array[17] = self.c1_shift
        params_array[18] = 0
        params_array[19] = 0
        params_array[20] = (self.tensor_sz // (4096 * 8)) & 0xFF # num kernel iters
        params_array[21] = ((self.tensor_sz // (4096 * 8)) >> 8) & 0xFF

        return params_array

def create_mul_model(input_shapes, InT, WtT, OutT):
    ifm_shape = input_shapes[0]
    wgt_shape = input_shapes[1]

    tensor_sz = reduce(lambda x, y: x * y, ifm_shape)
    W = None
    wts_tsor = None

    #if wgt_shape[0] != ifm_shape[0] or not all(dim == 1 for dim in wgt_shape[1:]):
    #    raise ValueError(f"Invalid shapes: wgt_shape[0] must equal ifm_shape[0] and all elements in wgt_shape[1:] must be 1. Got wgt_shape={wgt_shape} and ifm_shape={ifm_shape}.")
    wts_np_dtype= mapping.TENSOR_TYPE_TO_NP_TYPE[WtT]
    wts = np.random.randint(low=0, high=255, size=wgt_shape).astype(wts_np_dtype)
    wts_tsor = make_tensor(f"W", WtT, wgt_shape, wts)

    X = make_tensor_value_info("X", InT, ifm_shape)
    Y = make_tensor_value_info("Y", OutT, ifm_shape)

    qdq_params = QuantParam(0.0008634580299258232, 0.0001138541119871661, 0.0008906692964956164, 41733, 19916, 42933, tensor_sz)
    qdq_params.compute_qdq()
    qdq = qdq_params.get_params_array()
    qdq_tsor = make_tensor(f"QDQ", onnx.TensorProto.INT8, [22], qdq)

    mul1 = make_node(
        name="mul1",
        op_type="Mladfelwmul",
        inputs=["X", "W", "QDQ"],
        outputs=["Y"],
        domain="com.amd",
    )
    W = make_tensor_value_info("W", WtT, wgt_shape)
    print(wgt_shape)
    graph = make_graph([mul1], "MUL", [X], [Y], initializer=[wts_tsor, qdq_tsor])

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", help="Dtype (a16)", required=True)

    args = parser.parse_args()
    dtype = args.dtype
    dir_name = "test_mladfelwmul"
    model_name = dir_name + "/mladfelwmul.onnx"
    json_name = dir_name + "/model_mladfelwmul_meta.json"

    if dtype == 'a16':
        onnx_model = create_mul_model(
        [[1, 512, 256, 256], [512, 1, 1]], onnx.TensorProto.UINT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16)

    os.makedirs(dir_name, exist_ok=True)

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{model_name}")

    meta_info = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *meta_info)
    print("JSON Metadata saved to", f"{json_name}")
