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
import coeff_compute
import argparse

np.random.seed(42)


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_add_model(M, K, InT, WtT, OutT):
    # breakpoint()
    X = make_tensor_value_info("X", InT, [1, M, K])
    W = make_tensor_value_info("W", InT, [1, M, K])
    Y = make_tensor_value_info("Y", OutT, [1, M, K])

    qdq = np.zeros(16).astype(np.int32)
    # qdq[:4] = [16384, 2, 16384, 2]
    coeffs = coeff_compute.EltwiseAdd(
        0.1527101695537567, 65497, 0.0000683120742905885, 32804
    ).cal_coeff()

    qdq[0] = 14979 #int(coeffs[0])
    qdq[1] = 4451 #int(coeffs[1])
    qdq[2] = 14674 #int(coeffs[2])
    qdq[3] = 1000 #int(coeffs[3])
    qdq_tsor = make_tensor(f"QDQ", onnx.TensorProto.INT32, qdq.shape, qdq)

    add1 = make_node(
        name="add1",
        op_type="Add",
        inputs=["X", "W", "QDQ"],
        outputs=["Y"],
        domain="com.amd",
    )

    graph = make_graph([add1], "ADD", [X, W], [Y], initializer=[qdq_tsor])
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", help="Dtype (a16w8/a8w8)", required=True)

    args = parser.parse_args()
    dtype = args.dtype
    dir_name = "test_add_" + str(dtype)
    model_name = dir_name + "/model_add.onnx"
    json_name = dir_name + "/model_add_meta_" + str(dtype) + ".json"
    D = 768
    if dtype == "a16w8":
        S = 128
        onnx_model = create_add_model(
        S, D, onnx.TensorProto.UINT16, onnx.TensorProto.INT8, onnx.TensorProto.BFLOAT16)
    else:
        S = 512
        onnx_model = create_add_model(
        S, D, onnx.TensorProto.UINT8, onnx.TensorProto.INT8, onnx.TensorProto.BFLOAT16)

    os.makedirs(dir_name, exist_ok=True)

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{model_name}")

    meta_info = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *meta_info)
    print("JSON Metadata saved to", f"{json_name}")
