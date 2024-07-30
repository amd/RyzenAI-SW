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
    wts = np.random.randint(low=0, high=4, size=(K, N)).astype(np.uint8)
    wts_tsor = make_tensor(f"W", WtT, wts.shape, wts)
    Z = make_tensor_value_info("Z", OutT, [M, N])

    SV_M = 16
    SV_K = 128
    SV_N = 16
    k_iter = K // SV_K
    shift_gemm_out = 1
    shift_qdq_out = 1
    np_kernel_params = np.zeros(16).astype(np.int32)
    np_kernel_params[0] = SV_M
    np_kernel_params[1] = SV_K
    np_kernel_params[2] = SV_N
    np_kernel_params[3] = 0x2000
    np_kernel_params[4] = 0x6000
    np_kernel_params[5] = 0x3800
    np_kernel_params[6] = k_iter
    np_kernel_params[7] = shift_qdq_out
    np_kernel_params[8] = shift_gemm_out

    kernel_params_tsor = make_tensor(
        f"kernel_params", onnx.TensorProto.INT32, np_kernel_params.shape, np_kernel_params
    )
    c0 = np.random.randint(0, 4, size=(1 * N)).astype(np.int64)
    qdq_c0_tsor = make_tensor(f"qdq_c0", onnx.TensorProto.INT64, c0.shape, c0)

    c1c2 = np.random.randint(0, 4, size=(1 * 2)).astype(np.int32)
    qdq_c1c2_tsor = make_tensor(f"qdq_c1c2", onnx.TensorProto.INT32, c1c2.shape, c1c2)

    matmul = make_node(
        name="mladf_matmul",
        op_type="MLADFMATMULA16W8",
        inputs=["X", "W", "qdq_c0", "qdq_c1c2", "kernel_params"],
        outputs=["Z"],
    )

    graph = make_graph(
        [matmul], "matmula16w8_fusion_rt", [X], [Z], initializer=[wts_tsor, qdq_c0_tsor, qdq_c1c2_tsor, kernel_params_tsor]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    # check_model(onnx_model)
    shape_inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    # check_model(shape_inferred_model)
    return shape_inferred_model


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dtype", help="Dtype (a16a16/a16w8)", required=True)

    # args = parser.parse_args()
    # dtype = args.dtype
    dtype = "a16w8"
    dir_name = "test_mladfmatmul_" + str(dtype)
    model_name = dir_name + "/model_matmul.onnx"
    json_name = dir_name + "/model_matmul_meta.json"
    M, K, N = (4096, 512, 512)
    onnx_model = create_single_matmul_model(
    M, K, N, onnx.TensorProto.UINT16, onnx.TensorProto.UINT8, onnx.TensorProto.UINT16)


    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{json_name}", *metainfo
    )
    print("JSON Metadata saved to", f"{json_name}")
