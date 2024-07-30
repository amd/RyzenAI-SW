import os
import numpy as np
import onnx
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
    make_tensor,
)
from onnx.checker import check_model
import onnxruntime
from ryzenai_dynamic_dispatch import onnx_graph as ogm
from ryzenai_dynamic_dispatch import fuse
import argparse

# from cal_coeff import MatMul
np.random.seed(42)


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_single_matmul_model(M, K, N, InT, WtT, OutT):
    X = make_tensor_value_info("X", InT, [M, K])
    Y = make_tensor_value_info("Y", WtT, [K, N])
    Z = make_tensor_value_info("Z", OutT, [M, N])

    SV_M = 16
    SV_K = 256
    SV_N = 8
    k_iter = K // SV_K
    C0 = 4
    C1 = 1
    C2 = 2
    C3 = 1
    shift_gemm_out = 1
    shift_qdq_out = 1
    np_qdq_params = np.zeros(16).astype(np.int32)
    np_qdq_params[0] = SV_M
    np_qdq_params[1] = SV_K
    np_qdq_params[2] = SV_N
    np_qdq_params[3] = k_iter
    np_qdq_params[4] = 0x2000
    np_qdq_params[5] = 0x4800
    np_qdq_params[6] = 0x3800
    np_qdq_params[7] = 0x3C00
    np_qdq_params[8] = 0x4000
    np_qdq_params[9] = 0x4400
    np_qdq_params[10] = C0 & 0xffffffff
    np_qdq_params[11] = (C0 >> 32) & 0xffffffff
    np_qdq_params[12] = C1
    np_qdq_params[13] = C2
    np_qdq_params[14] = C3
    np_qdq_params[15] = (shift_gemm_out | (shift_qdq_out << 16))

    qdq_params_tsor = make_tensor(
        f"qdq_params", onnx.TensorProto.INT32, np_qdq_params.shape, np_qdq_params
    )

    matmul = make_node(
        name="mladf_matmul",
        op_type="MLADFMATMULA16A16",
        inputs=["X", "Y", "qdq_params"],
        outputs=["Z"],
    )

    graph = make_graph(
        [matmul], "matmula16a16_fusion_rt", [X, Y], [Z], initializer=[qdq_params_tsor]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    # check_model(onnx_model)
    shape_inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    # check_model(shape_inferred_model)
    return shape_inferred_model, [np_qdq_params]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dtype", help="Dtype (a16a16/a16w8)", required=True)

    # args = parser.parse_args()
    # dtype = args.dtype
    dtype = "a16a16"
    dir_name = "test_mladfmatmul_" + str(dtype)
    model_name = dir_name + "/model_matmul.onnx"
    json_name = dir_name + "/model_matmul_meta.json"
    M, K, N = (4096, 512, 4096)
    onnx_model, wts = create_single_matmul_model(
    M, K, N, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16)


    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{json_name}", *metainfo
    )
    print("JSON Metadata saved to", f"{json_name}")
