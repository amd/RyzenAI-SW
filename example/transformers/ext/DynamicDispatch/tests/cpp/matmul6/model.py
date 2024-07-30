import sys
import onnx
from onnx import numpy_helper
from onnx import helper
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
)
from onnx.checker import check_model

import os

from ryzenai_dynamic_dispatch import onnx_graph as ogm
from ryzenai_dynamic_dispatch import fuse
import csv


def load_from_csv(csv_file_name):
    with open(csv_file_name, "r") as f:
        rd = csv.DictReader(f)
        res = [row for row in rd]
    return res


def Unit_matmul_model(
    num_heads,
    i,
    input_node,
    dic,
    layer_name,
    prev_layer,
    M,
    K,
    N,
    inp_dtype,
    wts_dtype,
    out_dtype,
    new_head=0,
):

    dic["initializers"]["layer_" + str(i) + "_" + layer_name + "_" + "matmul1_wts"] = (
        numpy_helper.from_array(
            np.random.uniform(low=0, high=255, size=(K, N)).astype(wts_dtype),
            name="layer_" + str(i) + "_" + layer_name + "_" + "matmul1_wts",
        )
    )

    # This will become input to next layer
    dic["inputs"]["layer_" + str(i) + "_" + layer_name + "_" + "matmul1_out"] = (
        make_tensor_value_info(
            "layer_" + str(i) + "_" + layer_name + "_" + "matmul1_out",
            out_dtype,
            [1, M, N],
        )
    )

    if (layer_name == "Q" or layer_name == "K") or (
        layer_name == "Out" and i == num_heads
    ):
        dic["outputs"]["layer_" + str(i) + "_" + layer_name + "_" + "matmul1_out"] = (
            dic["inputs"]["layer_" + str(i) + "_" + layer_name + "_" + "matmul1_out"]
        )

    if new_head:
        input_str = "layer_" + str(i - 1) + "_" + prev_layer
    else:
        input_str = "layer_" + str(i) + "_" + prev_layer

    dic["nodes"]["layer_" + str(i) + "_" + layer_name + "_" + "matmul1"] = (
        onnx.helper.make_node(
            "MatMul",  # op name
            inputs=[
                dic["inputs"][input_str + "_" + "matmul1_out"].name,
                "layer_" + str(i) + "_" + layer_name + "_" + "matmul1_wts",
            ],
            outputs=["layer_" + str(i) + "_" + layer_name + "_" + "matmul1_out"],
            name="layer_" + str(i) + "_" + layer_name + "_" + "matmul_1",
        )
    )
    # print(dic["nodes"]["layer_" + str(i) + "_" + layer_name + "_" + "matmul1"])
    return dic


def Create_unit_tests(excel_sheet, num_heads):
    # shapes_df=pd.read_excel(excel_sheet)
    shapes_df = load_from_csv(excel_sheet)
    dic = {}
    dic["nodes"] = {}
    dic["inputs"] = {}
    dic["outputs"] = {}
    dic["initializers"] = {}
    dic["inputs"]["layer_0_input_matmul1_out"] = model_input = make_tensor_value_info(
        "input", TensorProto.INT8, [1, 512, 768]
    )
    inp_dtype = TensorProto.INT8
    wts_dtype = np.int8
    out_dtype = TensorProto.INT8
    for i in range(1, num_heads + 1):
        for index, row in enumerate(shapes_df):  # .iterrows():
            if i == 1:
                prev_layer = "input"
            else:
                prev_layer = "Out"

            if row["Name"] == "Q":

                dic = Unit_matmul_model(
                    num_heads,
                    i,
                    model_input,
                    dic,
                    "Q",
                    prev_layer,
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=1,
                )
            elif row["Name"] == "K":
                dic = Unit_matmul_model(
                    num_heads,
                    i,
                    model_input,
                    dic,
                    "K",
                    prev_layer,
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=1,
                )
            elif row["Name"] == "V":

                dic = Unit_matmul_model(
                    num_heads,
                    i,
                    model_input,
                    dic,
                    "V",
                    prev_layer,
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=1,
                )
            elif row["Name"] == "Proj":
                dic = Unit_matmul_model(
                    num_heads,
                    i,
                    model_input,
                    dic,
                    "Att_Proj",
                    "V",
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=0,
                )

            elif row["Name"] == "Inter":
                dic = Unit_matmul_model(
                    num_heads,
                    i,
                    model_input,
                    dic,
                    "Inter",
                    "Att_Proj",
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=0,
                )

            elif row["Name"] == "Out":
                dic = Unit_matmul_model(
                    num_heads,
                    i,
                    model_input,
                    dic,
                    "Out",
                    "Inter",
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=0,
                )

    graph = make_graph(
        list(dic["nodes"].values()),
        "12 head 6 Matmuls",
        inputs=[model_input],
        outputs=list(dic["outputs"].values()),
    )
    # print(dic['initializers'])
    graph.initializer.extend(list(dic["initializers"].values()))
    model = make_model(graph)
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    model = onnx.shape_inference.infer_shapes(model=model)
    # onnx.save_model(model,"Matmuls_6x12_for_fusion.onnx")
    return model


if __name__ == "__main__":
    num_heads = int(sys.argv[1])
    dir_name = "test_matmul6"
    os.makedirs(dir_name, exist_ok=True)
    onnx_model = Create_unit_tests(
        os.getenv("DOD_ROOT") + f"/tests/cpp/matmul6/PSF_shapes.csv", num_heads
    )
    onnx.save_model(onnx_model, f"{dir_name}/Matmuls_6x{num_heads}_for_fusion.onnx")

    op_list, new_tensors, tensor_map = fuse.prepare_metadata(
        ogm.ONNXGraph(onnx_model), dir_name
    )
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/Matmuls_6x{num_heads}_for_fusion.onnx.json",
        op_list,
        new_tensors,
        tensor_map,
    )
    # print("JSON Metadata :\n", json_str)
