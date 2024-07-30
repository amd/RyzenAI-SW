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


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


# Y = X*Sigmoid(X))
def create_silu_model(M, N, InT, OutT):
    X = make_tensor_value_info("X", InT, [1, M, N])
    Y = make_tensor_value_info("Y", OutT, [1, M, N])

    silu = make_node(
        name="silu",
        op_type="SILU",
        inputs=["X"],
        outputs=["Y"],
        domain="amd.com",
    )

    graph = make_graph(
        [silu], "lr", [X], [Y], initializer=[]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    M, N = (1, 11008)
    dir_name = "test_silu_abf16"
    os.makedirs(dir_name, exist_ok=True)

    onnx_model = create_silu_model(
        M, N, onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16
    )
    onnx.save(onnx_model, f"{dir_name}/model_silu.onnx")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/model_silu_meta.json", *metainfo
    )
    print("JSON Metadata saved to", f"{dir_name}/model_silu_meta.json")
