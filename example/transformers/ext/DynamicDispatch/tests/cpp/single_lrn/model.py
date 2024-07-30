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


def create_lrn_model(S, D, InType, GammaType, BetaType, OutType):
    # breakpoint()
    input = make_tensor_value_info("input", InType, [1, S, D])
    out = make_tensor_value_info("out", OutType, [1, S, D])

    np_beta = np.random.randint(low=-17716, high=16256, size=(D)).astype(np.int16)
    # np_beta = np.ones((D)).astype(np.int16)
    beta_tensor = make_tensor(f"beta", BetaType, np_beta.shape, np_beta)

    np_gamma = np.random.randint(low=-17684, high=16255, size=(D)).astype(np.int16)
    # np_gamma = np.ones((D)).astype(np.int16)
    gamma_tensor = make_tensor(f"gamma", GammaType, np_beta.shape, np_gamma)

    qdq = np.zeros(16).astype(np.int32)
    qdq[0] = 15821
    qdq[1] = 129
    # coeffs = coeff_compute.LRN(0.1527101695537567, 65497).cal_coeff()
    qdq_tensor = make_tensor(f"QDQ", onnx.TensorProto.INT32, qdq.shape, qdq)

    mha = make_node(
        name="lrn",
        op_type="LayerNorm",
        inputs=["input", "gamma", "beta", "QDQ"],
        outputs=["out"],
        domain="com.amd",
    )

    graph = make_graph(
        [mha],
        "LRN",
        [input],
        [out],
        initializer=[gamma_tensor, beta_tensor, qdq_tensor],
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", help="Dtype (a16w8/a8w8)", required=True)

    args = parser.parse_args()
    dtype = args.dtype
    dir_name = "test_lrn_" + str(dtype)
    model_name = dir_name + "/model_lrn.onnx"
    json_name = dir_name + "/model_lrn_meta_" + str(dtype) + ".json"
    D = 768
    if dtype == "a16w8":
        S = 128
        onnx_model = create_lrn_model(
        S, D, onnx.TensorProto.BFLOAT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16)
    else:
        S = 512
        onnx_model = create_lrn_model(
        S, D, onnx.TensorProto.BFLOAT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT8)

    os.makedirs(dir_name, exist_ok=True)

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{model_name}")

    meta_info = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *meta_info)
    print("JSON Metadata saved to", f"{json_name}")
