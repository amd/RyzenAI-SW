
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
from ryzenai_dynamic_dispatch import tune_graph

np.random.seed(42)


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_square_model(M, K, InT, OutT):
    # breakpoint()
    X = make_tensor_value_info("X", InT, [1, M, K])
    Y = make_tensor_value_info("Y", OutT, [1, M, K])

    sq1 = make_node(
        name="sq1",
        op_type="square",
        inputs=["X"],
        outputs=["Y"],
        domain="com.amd",
    )

    graph = make_graph([sq1], "square", [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model

def create_cube_model(M, K, InT, OutT):
    # breakpoint()
    X = make_tensor_value_info("X", InT, [1, M, K])
    Y = make_tensor_value_info("Y", OutT, [1, M, K])

    cube1 = make_node(
        name="cube1",
        op_type="cube",
        inputs=["X"],
        outputs=["Y"],
        domain="com.amd",
    )

    graph = make_graph([cube1], "cube", [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model

def create_square_cube_model(M, K, InT, OutT):
    # breakpoint()
    X = make_tensor_value_info("X", InT, [1, M, K])
    Y0 = make_tensor_value_info("Y0", OutT, [1, M, K])
    Y = make_tensor_value_info("Y", OutT, [1, M, K])

    sq1 = make_node(
        name="sq1",
        op_type="square",
        inputs=["X"],
        outputs=["Y0"],
        domain="com.amd",
    )

    cube1 = make_node(
        name="cube1",
        op_type="cube",
        inputs=["Y0"],
        outputs=["Y"],
        domain="com.amd",
    )

    graph = make_graph([sq1, cube1], "square_cube", [X], [Y], value_info=[Y0])
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model

def create_sqaure_cube_fork_model(M, K, InT, OutT):
    # breakpoint()
    X = make_tensor_value_info("X", InT, [1, M, K])
    Y0 = make_tensor_value_info("Y0", OutT, [1, M, K])
    Y = make_tensor_value_info("Y", OutT, [1, M, K])
    Y1 = make_tensor_value_info("Y1", OutT, [1, M, K])

    sq1 = make_node(
        name="sq1",
        op_type="square",
        inputs=["X"],
        outputs=["Y0"],
        domain="com.amd",
    )

    cube1 = make_node(
        name="cube1",
        op_type="cube",
        inputs=["Y0"],
        outputs=["Y"],
        domain="com.amd",
    )

    sq2 = make_node(
        name="sq2",
        op_type="square",
        inputs=["Y0"],
        outputs=["Y1"],
        domain="com.amd",
    )

    graph = make_graph([sq1, cube1, sq2], "square_cube_square", [X], [Y, Y1], value_info=[Y0])
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])

    return onnx_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="different graphs [0/1/2/3]", type=int, default=3)

    args = parser.parse_args()
    case = args.case
    dir_name = f"test_pm_swap"
    model_name = dir_name + f"/model_pm_swap_{case}.onnx"
    json_name = dir_name + f"/model_pm_swap_{case}_meta.json"
    H = 1
    W = 32
    tune_graph.__HW_MIN_SIZE = W

    if case == 0:
        onnx_model = create_square_model(H, W,
                                    onnx.TensorProto.INT32,
                                    onnx.TensorProto.INT32)
    elif case == 1:
        onnx_model = create_cube_model(H, W,
                                    onnx.TensorProto.INT32,
                                    onnx.TensorProto.INT32)
    elif case == 2:
        onnx_model = create_square_cube_model(H, W,
                                    onnx.TensorProto.INT32,
                                    onnx.TensorProto.INT32)
    elif case == 3:
        onnx_model = create_sqaure_cube_fork_model(H, W,
                                    onnx.TensorProto.INT32,
                                    onnx.TensorProto.INT32)
    else:
        assert False, "Invalid test case!"

    os.makedirs(dir_name, exist_ok=True)

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model, f"{model_name}")

    meta_info = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(f"{json_name}", *meta_info)
    print("JSON Metadata saved to", f"{json_name}")
