
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
from ryzenai_dynamic_dispatch import tune_graph
import struct

np.random.seed(42)

QDQ_CONFIG_0 = {"act_scale": 0.005941722076386213, "act_zp": 19793,
                "weight_scale": 0.0013358871219679713, "weight_zp": 115,
                "bias_scale": 0.00000793747040006565, "bias_zp": 0,
                "out_scale": 0.002246804302558303, "out_zp": 36260}

QDQ_CONFIG_1 = {"act_scale": 0.0005727267125621438, "act_zp": 41862,
                "weight_scale": 0.0051324679516255856, "weight_zp": 144,
                "bias_scale": 0.0000029395014280453324, "bias_zp": 0,
                "out_scale": 0.0006579186883755028, "out_zp": 28738}

QDQ_CONFIG_2 = {"act_scale": 0.00008783025259617716, "act_zp": 3170,
                "weight_scale": 0.004177127964794636, "weight_zp": 128,
                "bias_scale": 3.6687819715552905e-7, "bias_zp": 0,
                "out_scale": 0.0005834517651237547, "out_zp": 47867}

TEST_CONFIGS = {
    0: {"H": 128, "W": 128, "C_in": 256, "C_out": 512, "kernel_size": 1, "stride": 1, "qdq_config": QDQ_CONFIG_0},
    1: {"H": 64, "W": 64, "C_in": 4, "C_out": 512, "kernel_size": 3, "stride": 1, "qdq_config": QDQ_CONFIG_1},
    2: {"H": 64, "W": 64, "C_in": 512, "C_out": 512, "kernel_size": 3, "stride": 1, "qdq_config": QDQ_CONFIG_2}
}

def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)

ONNX_TO_NUMPY_DTYPE_MAP = {
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.UINT16: np.uint16,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.UINT32: np.uint32,
}

def create_conv_model(*,
    H, W, C_in, kernel_size, C_out, stride,
    InT, WeightT, BiasT, OutT,
    qdq_config
):

    # input format is [NHWC]
    ACT = make_tensor_value_info("ACT", InT, [1, H, W, C_in])

    # weights will take on shape that will be in ONNX model
    OVERRIDE_DATA = True

    if OVERRIDE_DATA:
        weight_file_path = "./bin/conv_case/weight_data.bin"
        wts = np.fromfile(weight_file_path, dtype=ONNX_TO_NUMPY_DTYPE_MAP[WeightT])
        wts = np.reshape(wts, (C_out, kernel_size, kernel_size, C_in))
        wts = np.transpose(wts, axes=[0, 3, 1, 2])

        bias_file_path = "./bin/conv_case/bias_data.bin"
        bias = np.fromfile(bias_file_path, dtype=ONNX_TO_NUMPY_DTYPE_MAP[BiasT])
        bias = np.reshape(bias, (C_out, 1, 1, 1))
    else:
        wts = np.random.randint(low=1, high=2, size=(C_out, C_in, kernel_size, kernel_size)).astype(ONNX_TO_NUMPY_DTYPE_MAP[WeightT])
        bias = np.random.randint(low=1, high=2, size=(C_out, 1, 1, 1)).astype(ONNX_TO_NUMPY_DTYPE_MAP[BiasT])

    WEIGHT = make_tensor("WEIGHT", WeightT, wts.shape, wts)
    BIAS = make_tensor("BIAS", BiasT, bias.shape, bias)

     # scale/zero-point for input/weights/bias/output -> at least 8 params
    ACT_SCALE = make_tensor("ACT_SCALE", onnx.TensorProto.FLOAT, [], [qdq_config["act_scale"]])
    ACT_ZERO_POINT = make_tensor("ACT_ZERO_POINT", InT, [], [qdq_config["act_zp"]])

    WEIGHT_SCALE = make_tensor("WEIGHT_SCALE", onnx.TensorProto.FLOAT, [], [qdq_config["weight_scale"]])
    WEIGHT_ZERO_POINT = make_tensor("WEIGHT_ZERO_POINT", WeightT, [], [qdq_config["weight_zp"]])

    BIAS_SCALE = make_tensor("BIAS_SCALE", onnx.TensorProto.FLOAT, [], [qdq_config["bias_scale"]])
    BIAS_ZERO_POINT = make_tensor("BIAS_ZERO_POINT", BiasT, [], [qdq_config["bias_zp"]])

    OUT_SCALE = make_tensor("OUT_SCALE", onnx.TensorProto.FLOAT, [], [qdq_config["out_scale"]])
    OUT_ZERO_POINT = make_tensor("OUT_ZERO_POINT", OutT, [], [qdq_config["out_zp"]])

    assert H % stride == 0, "H is not divisible by stride"
    assert W % stride == 0, "W is not divisible by stride"

    H_out = H // stride
    W_out = W // stride

    # output format is [NHWC]
    Y = make_tensor_value_info("Y", OutT, [1, H_out, W_out, C_out])

    # NOTE: for other ops, there seems to be an implicit assumption
    #       that bias will be absorbed by qdq tensor

    conv_0 = make_node(
        name="conv_0",
        op_type="xcom-conv2d",
        inputs=["ACT", "ACT_SCALE", "ACT_ZERO_POINT", "WEIGHT", "WEIGHT_SCALE", "WEIGHT_ZERO_POINT", "BIAS", "BIAS_SCALE", "BIAS_ZERO_POINT", "OUT_SCALE", "OUT_ZERO_POINT"],
        outputs=["Y"],
        domain="com.amd",
    )

    dilations_attr = onnx.helper.make_attribute("dilations", [1, 1])
    group_attr = onnx.helper.make_attribute("group", 1)
    input_shape_attr = onnx.helper.make_attribute("input_shape", [1, C_in, H, W])
    kernel_shape_attr = onnx.helper.make_attribute("kernel_shape", [kernel_size, kernel_size])
    output_shape_attr = onnx.helper.make_attribute("output_shape", [1, C_out, H_out, W_out])
    weight_shape_attr = onnx.helper.make_attribute("weight_shape", [C_out, C_in, kernel_size, kernel_size])
    pads_attr = onnx.helper.make_attribute("pads", [0, 0, 0, 0])
    strides_attr = onnx.helper.make_attribute("strides", [stride, stride])
    zero_point_attr = onnx.helper.make_attribute("zero_point", qdq_config["act_zp"])
    input_format_attr = onnx.helper.make_attribute("input_format", "NHWC")
    conv_0.attribute.extend([dilations_attr, group_attr, input_format_attr, input_shape_attr, kernel_shape_attr, output_shape_attr, weight_shape_attr, pads_attr, strides_attr, zero_point_attr])

    graph = make_graph([conv_0], "conv", [ACT], [Y], initializer=[ACT_SCALE, ACT_ZERO_POINT, WEIGHT, WEIGHT_SCALE, WEIGHT_ZERO_POINT, BIAS, BIAS_SCALE, BIAS_ZERO_POINT, OUT_SCALE, OUT_ZERO_POINT])

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=int, help="test config", default=0)

    args = parser.parse_args()
    config = args.config
    dir_name = f"test_xcom_conv2d_{config}"
    model_name = dir_name + f"/model_xcom_conv_{config}.onnx"
    json_name = dir_name + f"/model_xcom_conv_{config}_meta.json"

    #data formats
    InT = onnx.TensorProto.UINT16
    weightT = onnx.TensorProto.UINT8
    BiasT = onnx.TensorProto.INT32
    OutT = onnx.TensorProto.UINT16

    assert config < len(TEST_CONFIGS), f"Unsupported test config {config}"

    test_config = TEST_CONFIGS[config]

    onnx_model = create_conv_model(H=test_config["H"],
                                   W=test_config["W"],
                                   C_in=test_config["C_in"],
                                   kernel_size=test_config["kernel_size"],
                                   C_out=test_config["C_out"],
                                   stride=test_config["stride"],
                                   InT=InT,
                                   WeightT=weightT,
                                   BiasT=BiasT,
                                   OutT=OutT,
                                   qdq_config=test_config["qdq_config"])

    os.makedirs(dir_name, exist_ok=True)

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{model_name}")

    meta_info = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *meta_info)
    print("JSON Metadata saved to", f"{json_name}")
