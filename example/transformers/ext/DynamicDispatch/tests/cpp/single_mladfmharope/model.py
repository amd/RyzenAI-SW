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


# Z = X*Y
def create_mladfmharope_model(B, M, N, LhsT, RhsT, OutT):
    X = make_tensor_value_info("X", LhsT, [B, M, N])
    Y = make_tensor_value_info("Y", RhsT, [2, M, N])
    Z = make_tensor_value_info("Z", OutT, [B, M, N])

    mladfmharope = make_node(
        name="mladfmharope",
        op_type="MLADFMHAROPE",
        inputs=["X", "Y"],
        outputs=["Z"],
        domain="amd.com",
    )

    graph = make_graph(
        [mladfmharope], "mladfmharope", [X,Y], [Z], initializer=[]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    B, M, N = (32, 4096, 128)
    dir_name = "test_mladfmharope_abf16"
    os.makedirs(dir_name, exist_ok=True)

    onnx_model = create_mladfmharope_model(
        B, M, N, onnx.TensorProto.BFLOAT16,onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16
    )
    onnx.save(onnx_model, f"{dir_name}/model_mladfmharope.onnx")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/model_mladfmharope_meta.json", *metainfo
    )
    print("JSON Metadata saved to", f"{dir_name}/model_mladfmharope_meta.json")
