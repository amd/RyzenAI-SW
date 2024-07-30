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
    X = make_tensor_value_info("X", InT, [1, M, K])
    Y = make_tensor_value_info("Y", OutT, [1, M, N])

    # wts = np.load("tensor.npy")
    wts = np.random.randint(low=0, high=32, size=(K, N)).astype(np.uint8)
    # np_wts[...] = 1
    wts_tsor = make_tensor(f"W0", WtT, wts.shape, wts)

    # bias = np.random.uniform(-1, 1, wts.shape[1])

    c0 = np.random.randint(0, 32, size=(1 * N)).astype(np.int64)
    qdq_tsor = make_tensor(f"qdq", onnx.TensorProto.INT64, c0.shape, c0)

    # np_qdq_params = np.random.randint(-16, 16, size=(16)).astype(np.int32)
    np_qdq_params = np.zeros(16).astype(np.int32)
    np_qdq_params[0] = 0
    np_qdq_params[1] = 0  # c1
    np_qdq_params[2] = 10  # c2
    np_qdq_params[3] = 0
    np_qdq_params[4] = 32
    np_qdq_params[5] = 64
    np_qdq_params[6] = 0
    np_qdq_params[7] = 13

    qdq_params_tsor = make_tensor(
        f"qdq_params", onnx.TensorProto.INT32, np_qdq_params.shape, np_qdq_params
    )

    mul1 = make_node(
        name="mul1",
        op_type="MatMul",
        inputs=["X", "W0", "qdq", "qdq_params"],
        outputs=["Y"],
    )

    graph = make_graph(
        [mul1], "lr", [X], [Y], initializer=[wts_tsor, qdq_tsor, qdq_params_tsor]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    # check_model(onnx_model)
    shape_inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    # check_model(shape_inferred_model)
    return shape_inferred_model, [wts, c0, np_qdq_params]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", help="Dtype (a16w8/a8w8)", required=True)

    args = parser.parse_args()
    dtype = args.dtype
    dir_name = "test_matmul_" + str(dtype)
    model_name = dir_name + "/model_matmul1.onnx"
    json_name = dir_name + "/model_matmul1_meta_" + str(dtype) + ".json"
    K, N = (768, 1152)
    if dtype == "a16w8":
        M = 128
        onnx_model, wts = create_single_matmul_model(
        M, K, N, onnx.TensorProto.UINT16, onnx.TensorProto.UINT8, onnx.TensorProto.UINT16)
    else:
        M = 512
        onnx_model, wts = create_single_matmul_model(
        M, K, N, onnx.TensorProto.UINT8, onnx.TensorProto.UINT8, onnx.TensorProto.UINT8)


    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{json_name}", *metainfo
    )
    print("JSON Metadata saved to", f"{json_name}")
