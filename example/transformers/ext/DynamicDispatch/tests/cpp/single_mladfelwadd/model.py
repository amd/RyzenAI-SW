import os
import numpy as np
import onnx
from onnx import helper
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
from functools import reduce

np.random.seed(42)
class QuantParam:
    def __init__(self, ifm1_scale, ifm2_scale, ofm_scale, ifm1_zero_point, ifm2_zero_point, ofm_zero_point, tensor_sz, bd):
        self.ifm1_scale = ifm1_scale
        self.ifm2_scale = ifm2_scale
        self.ofm_scale = ofm_scale
        self.ifm1_zero_point = ifm1_zero_point
        self.ifm2_zero_point = ifm2_zero_point
        self.ofm_zero_point = ofm_zero_point
        self.tensor_sz = tensor_sz

        self.ifm1_coeff = None
        self.ifm2_coeff = None
        self.zero_point_coeff = None
        self.ofm_shift = None
        self.ifm1_shift = None
        self.ifm2_shift = None
        self.zero_point_shift = None
        if (bd == "non_broadcast"):
            self.kernel_sv_size = 1024
            self.num_kernel_iters = np.uint16(self.tensor_sz / (4096 / 4 * 8))
        else:
            self.kernel_sv_size = 4096
            self.num_kernel_iters = np.uint16(self.tensor_sz / (4096 * 8))


    # Refer to this link for details:
    # https://gitenterprise.xilinx.com/AIELibs/mllib/blob/dev/internal/demo/win24/sd/kernels/python/operators/AddA16.py#L40
    def compute_qdq(self):
        ifm1_shift = np.floor(-np.log2(self.ifm1_scale / self.ofm_scale) + 31)
        ifm2_shift = np.floor(-np.log2(self.ifm2_scale / self.ofm_scale) + 31)
        signed_zp = self.ofm_zero_point - self.ifm1_scale * self.ifm1_zero_point / self.ofm_scale - self.ifm2_scale * self.ifm2_zero_point / self.ofm_scale
        zero_point_shift = np.floor(-np.log2(np.abs(signed_zp)) + 31)
        ofm_shift = max(ifm1_shift, ifm2_shift, zero_point_shift)

        self.ifm1_coeff = int(self.ifm1_scale / self.ofm_scale * 2**ifm1_shift)
        self.ifm2_coeff = int(self.ifm2_scale / self.ofm_scale * 2**ifm2_shift)
        self.zero_point_coeff = int(signed_zp * 2**zero_point_shift)
        self.ofm_shift = ofm_shift
        self.ifm1_shift = ofm_shift - ifm1_shift
        self.ifm2_shift = ofm_shift - ifm2_shift
        self.zero_point_shift = ofm_shift - zero_point_shift

    # Set QDQ Param for Kernel
    def get_params_array(self):
        self.compute_qdq()

        params_array = np.zeros(24, dtype=np.int8)
        params_array[0:4] = np.array([self.kernel_sv_size], dtype=np.int32).view(np.int8)
        params_array[20:22] = np.array([self.num_kernel_iters], dtype=np.uint16).view(np.int8)

        params_array[4:8] = np.array([self.ifm1_coeff], dtype=np.int32).view(np.int8)
        params_array[8:12] = np.array([self.ifm2_coeff], dtype=np.int32).view(np.int8)
        params_array[12:16] = np.array([self.zero_point_coeff], dtype=np.int32).view(np.int8)

        params_array[16] = np.int8(self.ofm_shift)
        params_array[17] = np.int8(self.ifm1_shift)
        params_array[18] = np.int8(self.ifm2_shift)
        params_array[19] = np.int8(self.zero_point_shift)

        params_array[22:24] = np.array([0], dtype=np.int16).view(np.int8)

        return params_array

def create_add_model(input_shapes, InT, WtT, OutT):
    ifm_shape = input_shapes[0]
    wgt_shape = input_shapes[1]

    # if first dimension is 1, remove it
    ifm_shape = ifm_shape[1:] if ifm_shape and ifm_shape[0] == 1 else ifm_shape
    wgt_shape = wgt_shape[1:] if wgt_shape and wgt_shape[0] == 1 else wgt_shape

    tensor_sz = reduce(lambda x, y: x * y, ifm_shape)
    W = None
    wts_tsor = None
    bd = "broadcasting"
    if ifm_shape == wgt_shape:
        W = make_tensor_value_info("W", WtT, wgt_shape)
        bd = "non_broadcast"
    else:
        if wgt_shape[0] != ifm_shape[0] or not all(dim == 1 for dim in wgt_shape[1:]):
            raise ValueError(f"Invalid shapes: wgt_shape[0] must equal ifm_shape[0] and all elements in wgt_shape[1:] must be 1. Got wgt_shape={wgt_shape} and ifm_shape={ifm_shape}.")
        wts_np_dtype= helper.tensor_dtype_to_np_dtype(WtT)
        wts = np.random.randint(low=0, high=255, size=wgt_shape).astype(wts_np_dtype)
        wts_tsor = make_tensor(f"W", WtT, wgt_shape, wts)

    X = make_tensor_value_info("X", InT, ifm_shape)
    Y = make_tensor_value_info("Y", OutT, ifm_shape)

    qdq_params = QuantParam(0.0008634580299258232, 0.0001138541119871661, 0.0008906692964956164, 41733, 19916, 42933, tensor_sz, bd)
    qdq = qdq_params.get_params_array()
    qdq_tsor =  make_tensor(f"QDQ", onnx.TensorProto.INT8, [24], qdq)

    add1 = make_node(
        name="add1",
        op_type="Mladfelwadd",
        inputs=["X", "W", "QDQ"],
        outputs=["Y"],
        domain="com.amd",
    )
    if ifm_shape == wgt_shape:
        graph = make_graph([add1], "ADD", [X, W], [Y], initializer=[qdq_tsor])
    else:
        graph = make_graph([add1], "ADD", [X], [Y], initializer=[wts_tsor, qdq_tsor])

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", help="Dtype (a16w16)", required=True)

    args = parser.parse_args()
    dtype = args.dtype
    dir_name = "test_mladfelwadd"
    model_name = dir_name + "/mladfelwadd.onnx"
    json_name = dir_name + "/model_mladfelwadd_meta.json"

    if dtype == 'a16w16':
        # onnx_model = create_add_model(
        # [[128, 256, 256],[128, 1, 1]], onnx.TensorProto.UINT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16)
        onnx_model = create_add_model(
        [[128, 512, 512],[128, 512, 512]], onnx.TensorProto.UINT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    os.makedirs(dir_name, exist_ok=True)

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{model_name}")

    meta_info = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *meta_info)
    print("JSON Metadata saved to", f"{json_name}")
