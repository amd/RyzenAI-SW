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

# from cal_coeff import MatMul
np.random.seed(42)


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_parallel_matmul_model(M, K, N, InT, WtT, OutT):
    XA = make_tensor_value_info("XA", InT, [1, M, K])
    XB = make_tensor_value_info("XB", InT, [1, M, K])
    YA = make_tensor_value_info("YA", OutT, [1, M, N])
    YB = make_tensor_value_info("YB", OutT, [1, M, N])

    # wts = np.load("tensor.npy")
    wts = np.random.randint(low=0, high=32, size=(K, N)).astype(np.uint8)
    # np_wts[...] = 1
    wtsa_tsor = make_tensor(f"WA0", WtT, wts.shape, wts)
    wtsb_tsor = make_tensor(f"WB0", WtT, wts.shape, wts)

    # bias = np.random.uniform(-1, 1, wts.shape[1])

    c0 = np.random.randint(0, 32, size=(1 * N)).astype(np.int64)
    qdqa_tsor = make_tensor(f"qdqa", onnx.TensorProto.INT64, c0.shape, c0)
    qdqb_tsor = make_tensor(f"qdqb", onnx.TensorProto.INT64, c0.shape, c0)

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

    qdqa_params_tsor = make_tensor(
        f"qdqa_params", onnx.TensorProto.INT32, np_qdq_params.shape, np_qdq_params
    )
    qdqb_params_tsor = make_tensor(
        f"qdqb_params", onnx.TensorProto.INT32, np_qdq_params.shape, np_qdq_params
    )

    mula = make_node(
        name="mula",
        op_type="MatMul",
        inputs=["XA", "WA0", "qdqa", "qdqa_params"],
        outputs=["YA"],
    )
    mulb = make_node(
        name="mulb",
        op_type="MatMul",
        inputs=["XB", "WB0", "qdqb", "qdqb_params"],
        outputs=["YB"],
    )

    graph = make_graph(
        [mula, mulb], "lr", [XA, XB], [YA, YB], initializer=[wtsa_tsor, wtsb_tsor, qdqa_tsor, qdqb_tsor, qdqa_params_tsor, qdqb_params_tsor]
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    # check_model(onnx_model)
    shape_inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    # check_model(shape_inferred_model)
    return shape_inferred_model, [wts, c0, np_qdq_params]


if __name__ == "__main__":
    M, K, N = (512, 768, 1152)
    dir_name = "test_parallel_matmul_a8w8"
    os.makedirs(dir_name, exist_ok=True)

    onnx_model_i16, wts = create_parallel_matmul_model(
        M, K, N, onnx.TensorProto.UINT8, onnx.TensorProto.UINT8, onnx.TensorProto.UINT8
    )
    onnx.save(onnx_model_i16, f"{dir_name}/model_parallel_matmul.onnx")
    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model_i16), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/model_parallel_matmul_meta.json", *metainfo
    )
    print("JSON Metadata saved to", f"{dir_name}/model_parallel_matmul_meta.json")
