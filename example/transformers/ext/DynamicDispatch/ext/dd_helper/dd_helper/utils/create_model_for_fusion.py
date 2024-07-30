##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import os, sys
import numpy as np
import onnx

from dd_helper.optimizer.utils import *

# import onnx_graph as ogm
# import fuse
import csv
import argparse
from unit_models import *


def create_heads(
    shapes_df,
    num_heads,
    dic,
    St,
    Di,
    S,
    D,
    gamma_dtype,
    beta_dtype,
    inp_dtype,
    wts_dtype,
    out_dtype,
    matmul_add=False,
    matmul_add_gelu=False,
    add_skip=False,
):
    for i in range(1, num_heads + 1):
        # for index,row in shapes_df.iterrows():
        for index, row in enumerate(shapes_df):
            if i == 1:
                prev_layer = "input"
            else:
                if add_skip:
                    prev_layer = "Add_Skip-2"
                else:
                    prev_layer = "Out"

            dic = unit_layernorm_node(
                num_heads,
                i,
                dic,
                "LayerNorm-1",
                prev_layer,
                St,
                Di,
                S,
                D,
                gamma_dtype,
                beta_dtype,
                new_head=1,
            )

            if row["Name"] == "Q":
                dic = unit_matmul_model(
                    num_heads,
                    i,
                    dic,
                    "Q",
                    "LayerNorm-1",
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=0,
                    matmul_add=matmul_add,
                )
            elif row["Name"] == "K":
                dic = unit_matmul_model(
                    num_heads,
                    i,
                    dic,
                    "K",
                    "LayerNorm-1",
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=0,
                )
            elif row["Name"] == "V":
                dic = unit_matmul_model(
                    num_heads,
                    i,
                    dic,
                    "V",
                    "LayerNorm-1",
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=0,
                    matmul_add=matmul_add,
                )

                dic = unit_mha_node(
                    i,
                    dic,
                    St,
                    Di,
                    S,
                    D,
                    onnx.TensorProto.INT16,
                    onnx.TensorProto.INT16,
                    onnx.TensorProto.INT16,
                    onnx.TensorProto.INT16,
                    onnx.TensorProto.INT16,
                )

            elif row["Name"] == "Proj":
                dic = unit_matmul_model(
                    num_heads,
                    i,
                    dic,
                    "Projection_FC",
                    "MHA",
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=0,
                    matmul_add=matmul_add,
                )

                if add_skip:
                    dic = unit_add_skip_node(
                        num_heads,
                        i,
                        dic,
                        "Add_Skip-1",
                        "LayerNorm-1",
                        "Projection_FC",
                        int(row["M"]),
                        int(row["K"]),
                        int(row["N"]),
                        inp_dtype,
                        wts_dtype,
                        out_dtype,
                    )
                    dic = unit_layernorm_node(
                        num_heads,
                        i,
                        dic,
                        "LayerNorm-2",
                        "Add_Skip-1",
                        St,
                        Di,
                        S,
                        D,
                        gamma_dtype,
                        beta_dtype,
                        new_head=0,
                    )
                else:
                    dic = unit_layernorm_node(
                        num_heads,
                        i,
                        dic,
                        "LayerNorm-2",
                        "Projection_FC",
                        St,
                        Di,
                        S,
                        D,
                        gamma_dtype,
                        beta_dtype,
                        new_head=0,
                    )

            elif row["Name"] == "Inter":
                dic = unit_matmul_model(
                    num_heads,
                    i,
                    dic,
                    "Inter",
                    "LayerNorm-2",
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    inp_dtype,
                    wts_dtype,
                    out_dtype,
                    new_head=0,
                    matmul_add=True,
                    gelu=matmul_add_gelu,
                )

            elif row["Name"] == "Out":
                dic = unit_matmul_model(
                    num_heads,
                    i,
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
                    matmul_add=matmul_add,
                )
                if add_skip:
                    dic = unit_add_skip_node(
                        num_heads,
                        i,
                        dic,
                        "Add_Skip-2",
                        "LayerNorm-2",
                        "Out",
                        int(row["M"]),
                        int(row["K"]),
                        int(row["N"]),
                        inp_dtype,
                        wts_dtype,
                        out_dtype,
                    )
    if add_skip:
        prev_layer = "Add_Skip-2"
    else:
        prev_layer = "Out"
    dic = unit_layernorm_node(
        num_heads + 1,
        i + 1,
        dic,
        "LayerNorm-3",
        prev_layer,
        St,
        Di,
        S,
        D,
        gamma_dtype,
        beta_dtype,
        new_head=1,
        output=False,
    )
    return dic, i


def create_PSx_graph(
    model_name,
    excel_sheet,
    num_heads=12,
    matmul_add=False,
    matmul_add_gelu=False,
    add_skip=False,
):
    # import pandas as pd
    # shapes_df=pd.read_excel(excel_sheet)
    if model_name == "PSH":
        num_heads = 4
    shapes_df = load_from_csv(excel_sheet)
    dic, model_input = initialize_model()
    inp_dtype = onnx.TensorProto.INT16
    wts_dtype = np.int8
    out_dtype = onnx.TensorProto.INT16
    St, Di, S, D = (512, 1152, 512, 768)
    gamma_dtype = np.int16
    beta_dtype = np.int16

    dic, i = create_heads(
        shapes_df,
        num_heads,
        dic,
        St,
        Di,
        S,
        D,
        gamma_dtype,
        beta_dtype,
        inp_dtype,
        wts_dtype,
        out_dtype,
        matmul_add=matmul_add,
        matmul_add_gelu=matmul_add_gelu,
        add_skip=add_skip,
    )

    if model_name == "PSH":
        dic = unit_matmul_model(
            num_heads + 1,
            i + 1,
            dic,
            "Logits_FC",
            "LayerNorm-3",
            512,
            768,
            128,
            inp_dtype,
            wts_dtype,
            out_dtype,
            new_head=0,
            output=True,
        )
        for j in range(5):
            curr_layer = "GEMM+" + str(j)
            dic = unit_matmul_model(
                num_heads + 1,
                i + 1,
                dic,
                curr_layer,
                "LayerNorm-3",
                512,
                768,
                128,
                inp_dtype,
                wts_dtype,
                out_dtype,
                new_head=0,
                output=True,
            )
    if model_name == "PSK":
        # to add a node for mul,reducesum fusion op
        dic = unit_matmul_model(
            num_heads + 1,
            i + 1,
            dic,
            "Gemm-1",
            "LayerNorm-3",
            512,
            768,
            768,
            inp_dtype,
            wts_dtype,
            out_dtype,
            new_head=0,
            output=False,
        )
        dic = unit_matmul_model(
            num_heads + 1,
            i + 1,
            dic,
            "Gemm-2",
            "Gemm-1",
            512,
            768,
            768,
            inp_dtype,
            wts_dtype,
            out_dtype,
            new_head=0,
            output=True,
        )

    if model_name == "PSL" or model_name == "PSF":
        dic = unit_matmul_model(
            num_heads + 1,
            i + 1,
            dic,
            "Logits_FC",
            "LayerNorm-3",
            512,
            768,
            128,
            inp_dtype,
            wts_dtype,
            out_dtype,
            new_head=0,
            matmul_add=matmul_add,
            output=True,
        )

    filename, model = dict_to_model(model_name, dic, model_input, num_heads)

    return filename, model


if __name__ == "__main__":
    # num_heads = int(sys.argv[1])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heads",
        default=12,
        type=int,
        help="Number of heads(layer) in the Model, default=12",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        help="Model to replicate, will generate full model based on the model_name",
        default="PSF",
        choices=["PSF", "PSK", "PSL", "PSH"],
        required=False,
    )
    parser.add_argument(
        "--layer_name",
        help="LayerName to create unit test ( will create unit test with default matmul shapes of [512,768,768])",
        default="ALL",
        choices=[
            "LayerNorm",
            "MatMul",
            "MatMulAdd",
            "MatMulAddGelu",
            "MHAGRPB",
            "AddSkip",
            "ALL",
        ],
        required=False,
    )

    parser.add_argument(
        "--excel_sheet",
        help="CSV path which contain Name of the layers and MKN shapes of the layers",
        default="./shapes.csv",
        required=False,
    )

    args = parser.parse_args()
    num_heads = args.heads

    excel_sheet = args.excel_sheet
    models = []

    """
    For single unit  node
    pass the layername as param, support layers=[LayerNorm,MatMul,MatMul_Add,MatMul_Add_Gelu] TODO MHA layer (WIP)
    """

    if args.layer_name == "ALL":
        layers = [
            "LayerNorm",
            "MatMul",
            "MatMulAdd",
            "MatMulAddGelu",
            "MHAGRPB",
            "AddSkip",
        ]
        for layer in layers:
            filename, onnx_model = create_unit_node(layer)
            models.append(filename)
    else:
        filename, onnx_model = create_unit_node(args.layer_name)

    """
    For Full Model
    """
    filename, model = create_PSx_graph(
        args.model_name,
        excel_sheet,
        args.heads,
        matmul_add=False,
        matmul_add_gelu=True,
        add_skip=True,
    )

    # for onnx_filename in models:
    #     print("Working on ", onnx_filename)
    #     onnx_model = onnx.load(onnx_filename)
    #     op_list, new_tensors, tensor_map = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model))
    #     json_str = fuse.save_tensors_to_json(f"{onnx_filename}.json", op_list, new_tensors, tensor_map)
    #     # print('JSON Metadata :\n', json_str)
