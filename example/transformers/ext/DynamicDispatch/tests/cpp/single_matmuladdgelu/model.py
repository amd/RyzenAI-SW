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

np.random.seed(42)

QDQ_PARAMS_SIZE = 16


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_single_matmuladdgelu_model(M, K, N, InT, WtT, OutT):
    X = make_tensor_value_info("X", InT, [1, M, K])
    Y = make_tensor_value_info("Y", OutT, [1, M, N])

    np_wts = np.random.randint(low=-5, high=5, size=(K, N)).astype(np.uint8)
    wts_tsor = make_tensor(f"weights", onnx.TensorProto.UINT8, np_wts.shape, np_wts)

    np_qdq = np.random.randint(low=-5, high=5, size=(N)).astype(np.int64)
    qdq_tsor = make_tensor(f"qdq", onnx.TensorProto.INT64, np_qdq.shape, np_qdq)

    np_qdqparams = np.random.randint(low=-5, high=5, size=(QDQ_PARAMS_SIZE)).astype(
        np.int32
    )
    qdqparams_tsor = make_tensor(
        f"qdq_params", onnx.TensorProto.INT32, np_qdqparams.shape, np_qdqparams
    )

    np_geluqdqparams = np.random.randint(low=-5, high=5, size=(QDQ_PARAMS_SIZE)).astype(
        np.int32
    )
    geluqdqparams_tsor = make_tensor(
        f"gelu_qdq_params",
        onnx.TensorProto.INT32,
        np_geluqdqparams.shape,
        np_geluqdqparams,
    )

    mul1 = make_node(
        name="mul1",
        op_type="MatMulAddGelu",
        inputs=["X", "weights", "qdq", "qdq_params", "gelu_qdq_params"],
        outputs=["Y"],
    )

    graph = make_graph(
        [mul1],
        "lr",
        [X],
        [Y],
        initializer=[wts_tsor, qdq_tsor, qdqparams_tsor, geluqdqparams_tsor],
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model, [np_wts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", help="Dtype (a16w8/a8w8)", required=True)

    args = parser.parse_args()
    dtype = args.dtype
    dir_name = "test_matmuladdgelu_" + str(dtype)
    model_name = dir_name + "/model_matmuladdgelu.onnx"
    json_name = dir_name + "/model_matmuladdgelu_meta_" + str(dtype) + ".json"
    K, N = (768, 3072)
    if dtype == "a16w8":
        M = 128
        onnx_model, wts = create_single_matmuladdgelu_model(
        M, K, N, onnx.TensorProto.UINT16, onnx.TensorProto.UINT8, onnx.TensorProto.UINT16)
    else:
        M = 512
        onnx_model, wts = create_single_matmuladdgelu_model(
        M, K, N, onnx.TensorProto.UINT8, onnx.TensorProto.UINT8, onnx.TensorProto.UINT8)

    os.makedirs(dir_name, exist_ok=True)

    onnx.save(onnx_model, f"{model_name}")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{json_name}", *metainfo
    )
    print("JSON Metadata saved to", f"{json_name}")
