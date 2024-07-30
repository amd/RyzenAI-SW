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

np.random.seed(42)


def mha_qdq():
    grpb_qdq_params = np.zeros(16).astype(np.int32)

    grpb_qdq_params[0] = 0
    grpb_qdq_params[1] = 0
    grpb_qdq_params[2] = 1
    grpb_qdq_params[3] = 0
    grpb_qdq_params[4] = 32
    grpb_qdq_params[5] = 8
    grpb_qdq_params[6] = 2
    grpb_qdq_params[7] = 8
    grpb_qdq_params[8] = f2bf(np.asarray(0.0125).astype(np.float32))
    grpb_qdq_params[9] = 28
    grpb_qdq_params[10] = f2bf(np.asarray(0.5).astype(np.float32))
    grpb_qdq_params[11] = 8
    grpb_qdq_params[12] = f2bf(np.asarray(4.0).astype(np.float32))
    grpb_qdq_params[13] = f2bf(np.asarray(3.0).astype(np.float32))
    grpb_qdq_params[14] = f2bf(np.asarray(2.0).astype(np.float32))

    qdq_params = np.zeros(96).astype(np.int32)

    qry_subv_rows = 32
    qry_subv_cols = 96
    key_subv_rows = 64
    key_subv_rows_int16 = 16
    key_subv_cols = 96
    val_subv_rows = 64
    val_subv_cols = 64
    out_subv_rows = 32
    out_subv_cols = 64

    gprb_rows = 96
    gprb_cols = 8

    num_qdq_nodes = 6
    QDQparam_size = 16

    GPRB_buf_size = 1024
    c0 = 1
    C1 = 1
    C2 = 1
    C3 = 1
    # int32* C0 = (int32*)aie_qdq
    SQb = 0
    Sout = 8

    # QKT
    qdq_params[(16 * 0) + 0] = c0
    qdq_params[(16 * 0) + 1] = C1
    qdq_params[(16 * 0) + 2] = C2
    qdq_params[(16 * 0) + 3] = C3
    qdq_params[(16 * 0) + 4] = qry_subv_rows
    qdq_params[(16 * 0) + 5] = key_subv_rows
    qdq_params[(16 * 0) + 6] = SQb
    qdq_params[(16 * 0) + 7] = Sout

    # SM *V
    qdq_params[(16 * 1) + 0] = c0
    qdq_params[(16 * 1) + 1] = C1
    qdq_params[(16 * 1) + 2] = C2
    qdq_params[(16 * 1) + 3] = C3
    qdq_params[(16 * 1) + 4] = qry_subv_rows
    qdq_params[(16 * 1) + 5] = val_subv_cols
    qdq_params[(16 * 1) + 6] = SQb
    qdq_params[(16 * 1) + 7] = Sout

    # DQ before SM
    qdq_params[(16 * 2) + 0] = 126
    qdq_params[(16 * 2) + 1] = f2bf(np.asarray(0.45).astype(np.float32))

    # Q after SM
    qdq_params[(16 * 3) + 0] = 0
    qdq_params[(16 * 3) + 1] = f2bf(np.asarray(0.003921568).astype(np.float32))
    return grpb_qdq_params, qdq_params


def f2bf(data, bits=16):
    xf = (
        data.astype(np.float32).getfield(np.int32) & ~(2 ** (32 - bits) - 1)
    ).getfield(np.float32)
    x32 = xf.view(np.uint32)
    x16 = (x32 >> 16).astype(np.uint16)
    return x16


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


def create_mha_model(
    St, Di, S, D, QueryProto, KeyProto, ValueProto, WgtProto, OutProto
):
    # breakpoint()
    query = make_tensor_value_info("query", QueryProto, [1, St, Di])
    key = make_tensor_value_info("key", KeyProto, [1, St, Di])
    val = make_tensor_value_info("val", ValueProto, [1, S, D])
    out = make_tensor_value_info("out", OutProto, [1, S, D])
    mask = make_tensor_value_info("mask", onnx.TensorProto.UINT16, [1, 1, 1, S])

    np_wgt = np.random.randint(low=0, high=16, size=(96, 8)).astype(np.uint8)
    wgt_tsor = make_tensor(f"wgts", WgtProto, np_wgt.shape, np_wgt)

    np_grpb_vec_64b = np.ones(15).astype(np.int64)
    grpb_vec_tsor_64b = make_tensor(
        f"grpb_vec_64b", onnx.TensorProto.INT64, np_grpb_vec_64b.shape, np_grpb_vec_64b
    )

    np_grpb_vec_32b = np.ones(32).astype(np.int32)
    grpb_vec_tsor_32b = make_tensor(
        f"grpb_vec_32b", onnx.TensorProto.INT32, np_grpb_vec_32b.shape, np_grpb_vec_32b
    )

    np_bias = np.random.randint(low=0, high=255, size=(12, St, S)).astype(np.uint8)
    bias_tsor = make_tensor(f"bias", WgtProto, np_bias.shape, np_bias)

    np_qdq = np.random.randint(low=-5, high=5, size=(6 * 16)).astype(np.int32)
    qdq_tsor = make_tensor(f"qdq", onnx.TensorProto.INT32, np_qdq.shape, np_qdq)

    mha = make_node(
        name="mha",
        op_type="MHAGRPB",
        inputs=[
            "query",
            "key",
            "val",
            "mask",
            "wgts",
            "grpb_vec_64b",
            "grpb_vec_32b",
            "bias",
            "qdq",
        ],
        outputs=["out"],
        domain="com.amd",
    )

    graph = make_graph(
        [mha],
        "MHAGRPB",
        [query, key, val, mask],
        [out],
        initializer=[
            wgt_tsor,
            grpb_vec_tsor_64b,
            grpb_vec_tsor_32b,
            bias_tsor,
            qdq_tsor,
        ],
    )
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    # check_model(onnx_model)
    # shape_inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    # check_model(shape_inferred_model)
    return onnx_model


if __name__ == "__main__":
    St, Di, S, D = (512, 1152, 512, 768)
    dir_name = "test_mha"
    os.makedirs(dir_name, exist_ok=True)
    onnx_model_i16 = create_mha_model(
        St,
        Di,
        S,
        D,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT8,
    )
    onnx_model_i16.opset_import.append(onnx.helper.make_opsetid("com.amd", 1))
    onnx.save(onnx_model_i16, f"{dir_name}/mha.onnx")

    op_list, new_tensors, tensor_map, aux_info = fuse.prepare_metadata(
        ogm.ONNXGraph(onnx_model_i16), dir_name
    )
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/mha.onnx.json", op_list, new_tensors, tensor_map, aux_info
    )
    print("JSON Metadata saved to", f"{dir_name}/mha.onnx.json")
