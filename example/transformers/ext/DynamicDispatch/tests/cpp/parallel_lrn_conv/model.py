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
#import coeff_compute
import argparse

np.random.seed(42)


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_parallel_lrn_conv_model(S, D, InType, GammaType, BetaType, OutType):
    # breakpoint()
    lrn_in = make_tensor_value_info("lrn_in", InType, [1, S, D])
    lrn_out = make_tensor_value_info("lrn_out", OutType, [1, S, D])

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

    lrn = make_node(
        name="lrn",
        op_type="LayerNorm",
        inputs=["lrn_in", "gamma", "beta", "QDQ"],
        outputs=["lrn_out"],
        domain="com.amd",
    )

    testDataFolder = "tests\\cpp\\unit_tests\\testDataMladf\\psi_10008_3_512_512"
    Mi0, Mi1 = (14, 14)
    F0, F1 = (3, 3)
    K, N = (512, 512)
    Mo0, Mo1 = (14, 14)
    groupId = 512
    zeropoint = 10008

    conv_in = make_tensor_value_info("conv_in", onnx.TensorProto.UINT16, [1, Mi0*Mi1, K])
    conv_out = make_tensor_value_info("conv_out", onnx.TensorProto.UINT16, [1, Mo0*Mo1, N])

    np_wgts = np.fromfile(testDataFolder+"\\weight.const").view(np.uint8).astype(np.uint8)
    wgts_tensor = make_tensor(f"wgts", onnx.TensorProto.UINT8, [N, 1, F0, F1], np_wgts)


    conv = make_node(
        name="conv",
        op_type="QConv",
        inputs=["conv_in", "wgts"],
        outputs=["conv_out"],
        domain="com.amd",
    )
    attr = onnx.helper.make_attribute("group", 512)
    conv.attribute.append(attr)
    attr = onnx.helper.make_attribute("input_shape", [1, K, Mi0, Mi1])
    conv.attribute.append(attr)
    attr = onnx.helper.make_attribute("output_shape", [1, N, Mo0, Mo1])
    conv.attribute.append(attr)
    attr = onnx.helper.make_attribute("weight_shape", [N, 1, F0, F1])
    conv.attribute.append(attr)
    attr = onnx.helper.make_attribute("zero_point", 10008)
    conv.attribute.append(attr)


    graph = make_graph(
        [lrn, conv],
        "LRN_CONV",
        [lrn_in, conv_in],
        [lrn_out, conv_out],
        initializer=[gamma_tensor, beta_tensor, qdq_tensor, wgts_tensor],
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":

    dir_name = "test_parallel_lrn_conv"
    model_name = dir_name + "/model_lrn_conv.onnx"
    json_name = dir_name + "/model_lrn_conv_meta.json"
    S, D = (3136, 128)
    onnx_model = create_parallel_lrn_conv_model(
    S, D, onnx.TensorProto.BFLOAT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16, onnx.TensorProto.UINT16)

    os.makedirs(dir_name, exist_ok=True)

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{model_name}")

    meta_info = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *meta_info)
    print("JSON Metadata saved to", f"{json_name}")
