##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import os, csv
import onnx
import numpy as np
from onnx import numpy_helper
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
)

from dd_helper.optimizer.utils import *


def initialize_model():
    dic = {}
    dic["nodes"] = {}
    dic["inputs"] = {}
    dic["outputs"] = {}
    dic["initializers"] = {}
    dic["inputs"]["layer_0_input_out"] = model_input = make_tensor_value_info(
        "input", TensorProto.INT16, [1, 512, 768]
    )
    return dic, model_input


def dict_to_model(model_name, dic, model_input, num_heads):
    save_path = get_savepath(model_name)
    graph = make_graph(
        list(dic["nodes"].values()),
        model_name,
        inputs=[model_input],
        outputs=list(dic["outputs"].values()),
    )
    graph.initializer.extend(list(dic["initializers"].values()))
    model = make_model(graph)
    filename = os.path.join(save_path, f"{model_name}_{num_heads}heads_for_fusion.onnx")
    onnx.save_model(model, filename)
    update_output_shapes_dtypes(filename)
    return filename, model


def load_from_csv(csv_file_name):
    with open(csv_file_name, "r") as f:
        rd = csv.DictReader(f)
        res = [row for row in rd]
    return res


def update_output_shapes_dtypes(model_name):
    m, g = loadmodel(model_name)

    for n_m in g.nodemap.keys():
        node = g.nodemap[n_m]
        if node.op_type == "LayerNorm":
            g.tensormap[node.output[0]].shape = (1, 512, 768)
            g.tensormap[node.output[0]].dtype = np.int16

        if "MatMul".lower() in node.op_type.lower():
            M = g.tensormap[node.input[0]].shape[1]

            N = g.tensormap[node.input[1]].shape[1]
            g.tensormap[node.output[0]].shape = (1, M, N)
            g.tensormap[node.output[0]].dtpye = np.int16
        elif node.op_type == "MHAGRPB":
            g.tensormap[node.output[0]].dtype = np.int16
            g.tensormap[node.output[0]].shape = (1, 512, 768)
        elif node.op_type.lower() == "ADD".lower():
            g.tensormap[node.output[0]].dtype = np.int16
            g.tensormap[node.output[0]].shape = g.tensormap[node.input[0]].shape

    g.save_model(model_name, rawmodel=m.mproto)


def unit_add_skip_node(
    num_heads,
    i,
    dic,
    layer_name,
    prev_layer1,
    prev_layer2,
    M,
    K,
    N,
    inp_dtype,
    wts_dtype,
    out_dtype,
    new_head=0,
    output=False,
):
    node_optype = "ADD"
    node_output = "layer_" + str(i) + "_" + layer_name + "_out"

    dic["inputs"][node_output] = make_tensor_value_info(
        node_output, out_dtype, [1, M, N]
    )

    node_input1 = "layer_" + str(i) + "_" + prev_layer1 + "_out"
    node_input2 = "layer_" + str(i) + "_" + prev_layer2 + "_out"

    node_inputs = [node_input1, node_input2]
    dic["nodes"]["layer_" + str(i) + "_" + layer_name + "_skip_add"] = (
        onnx.helper.make_node(
            node_optype,  # op name
            inputs=node_inputs,
            outputs=[node_output],
            name="layer_" + str(i) + "_" + layer_name + "_skip_add",
            domain="com.amd",
        )
    )
    print(dic["inputs"])
    return dic


def unit_matmul_model(
    num_heads,
    i,
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
    matmul_add=False,
    gelu=False,
    output=False,
):
    # initializer (wts)
    # print(dic['initializers'])
    weights_tensor = "layer_" + str(i) + "_" + layer_name + "_matmul_wts"
    node_output = "layer_" + str(i) + "_" + layer_name + "_out"

    dic["initializers"][weights_tensor] = numpy_helper.from_array(
        np.random.uniform(low=0, high=255, size=(K, N)).astype(wts_dtype),
        name=weights_tensor,
    )

    # This will become input to next layer
    # output tensor

    dic["inputs"][node_output] = make_tensor_value_info(
        node_output, out_dtype, [1, M, N]
    )

    # if (layer_name=='Q' or layer_name=='K') or (layer_name=='Out' and i==num_heads ):
    if output:
        dic["outputs"][node_output] = dic["inputs"][node_output]

    if new_head:
        input_str = "layer_" + str(i - 1) + "_" + prev_layer
    else:
        input_str = "layer_" + str(i) + "_" + prev_layer

    if matmul_add:
        dic["initializers"]["layer_" + str(i) + "_" + layer_name + "_add_bias"] = (
            numpy_helper.from_array(
                np.random.uniform(low=0, high=255, size=(N)).astype(np.int16),
                name="layer_" + str(i) + "_" + layer_name + "_add_bias",
            )
        )
        if gelu:
            node_optype = "MatMulAddGelu"
        else:
            node_optype = "MatMulAdd"
        node_inputs = [
            dic["inputs"][input_str + "_out"].name,
            weights_tensor,
            "layer_" + str(i) + "_" + layer_name + "_add_bias",
        ]

    else:
        node_optype = "MatMul"
        if gelu:
            node_optype = "MatMulAddGelu"
        node_inputs = [dic["inputs"][input_str + "_out"].name, weights_tensor]

    dic["nodes"]["layer_" + str(i) + "_" + layer_name + "_matmul"] = (
        onnx.helper.make_node(
            node_optype,  # op name
            inputs=node_inputs,
            outputs=[node_output],
            name="layer_" + str(i) + "_" + layer_name + "_matmul",
            domain="com.amd",
        )
    )
    # print(dic['nodes']['layer_'+str(i)+'_'+layer_name+'_matmul1'])
    return dic


def unit_mha_node(
    i,
    dic,
    St,
    Di,
    S,
    D,
    QueryProto,
    KeyProto,
    ValueProto,
    MaskProto,
    OutProto,
    output=False,
):
    # print(dic[])

    query = dic["inputs"]["layer_" + str(i) + "_Q_out"]
    key = dic["inputs"]["layer_" + str(i) + "_K_out"]
    val = dic["inputs"]["layer_" + str(i) + "_V_out"]
    grpb_wts = "layer_" + str(i) + "_GRPB_wts"
    mha_mask = "layer_" + str(i) + "_MHA_mask"
    mha_out = "layer_" + str(i) + "_MHA_out"
    grpb_bias = "layer_" + str(i) + "_GRPB_bias"
    # mask = make_tensor_value_info('mask', MaskProto, [1, S])
    dic["inputs"][mha_out] = make_tensor_value_info(
        mha_out, onnx.TensorProto.INT16, [1, S, D]
    )

    # np_wts = [np.random.randint(low=-5, high=5, size=(K, N)).astype(np.int8) for i in range(2)]
    np_mask = np.ones((1, S)).astype(np.int16)

    # dic['initializers'][grpb_wts] = numpy_helper.from_array(np.random.uniform(low=0, high=255, size=(1,512)).astype(np.int8),name=grpb_wts)

    dic["initializers"][mha_mask] = numpy_helper.from_array(
        np.random.uniform(low=0, high=255, size=(1, S)).astype(np.int16), name=mha_mask
    )

    dic["initializers"][grpb_bias] = numpy_helper.from_array(
        np.random.uniform(low=0, high=255, size=(1, S)).astype(np.int16), name=grpb_bias
    )

    # numpy_helper.from_array(np.random.uniform(low=0, high=255, size=(K,N)).astype(np.uint8)
    dic["nodes"]["layer_" + str(i) + "_MHAGRPB"] = make_node(
        name="layer_" + str(i) + "_MHAGRPB",
        op_type="MHAGRPB",
        inputs=[query.name, key.name, val.name, mha_mask, grpb_bias],
        outputs=[mha_out],
        domain="com.amd",
    )
    if output == True:
        dic["outputs"][mha_out] = dic["inputs"][mha_out]

    return dic


def unit_layernorm_node(
    num_heads,
    i,
    dic,
    layer_name,
    prev_layer,
    St,
    Di,
    S,
    D,
    gamma_dtype,
    beta_dtype,
    new_head=0,
    output=False,
):
    # inpit-512x768
    # 768-gamma
    # 768-beta
    node_output = "layer_" + str(i) + "_" + layer_name + "_out"
    gamma = "layer_" + str(i) + "_" + layer_name + "_gamma"
    beta = "layer_" + str(i) + "_" + layer_name + "_beta"

    dic["initializers"][gamma] = numpy_helper.from_array(
        np.random.uniform(low=0, high=255, size=(768)).astype(gamma_dtype), name=gamma
    )
    dic["initializers"][beta] = numpy_helper.from_array(
        np.random.uniform(low=0, high=255, size=(768)).astype(beta_dtype), name=beta
    )
    dic["inputs"][node_output] = make_tensor_value_info(
        node_output, TensorProto.INT16, [1, S, D]
    )

    # print(node_output)
    if output:
        dic["outputs"][node_output] = dic["inputs"][node_output]

    if new_head:
        input_str = "layer_" + str(i - 1) + "_" + prev_layer

    else:
        input_str = "layer_" + str(i) + "_" + prev_layer

    print(input_str + "_out")
    # breakpoint()

    dic["nodes"]["layer_" + str(i) + "_" + layer_name + "_LayerNorm"] = make_node(
        name="layer_" + str(i) + "_" + layer_name + "_LayerNorm",
        op_type="LayerNorm",
        inputs=[
            dic["inputs"][input_str + "_out"].name,
            gamma,
            beta,
        ],
        outputs=[node_output],
        domain="com.amd",
    )
    return dic


def create_unit_node(
    layer_name,
    inp_dtype=TensorProto.INT16,
    wts_dtype=np.int8,
    out_dtype=TensorProto.INT16,
    gamma_dtype=np.int16,
    beta_dtype=np.int16,
):
    print(layer_name)
    dic, model_input = initialize_model()
    St, Di, S, D = (512, 1152, 512, 768)
    M, K, N = (512, 768, 768)
    i = 1
    num_heads = 1
    prev_layer = "input"
    if layer_name == "LayerNorm":
        dic = unit_layernorm_node(
            num_heads,
            i,
            dic,
            "LayerNorm",
            prev_layer,
            St,
            Di,
            S,
            D,
            gamma_dtype,
            beta_dtype,
            new_head=1,
            output=True,
        )
        inputs = [model_input]
    elif layer_name == "MatMul":
        dic = unit_matmul_model(
            num_heads,
            i,
            dic,
            "unit",
            prev_layer,
            M,
            K,
            N,
            inp_dtype,
            wts_dtype,
            out_dtype,
            new_head=1,
            matmul_add=False,
            output=True,
        )
        inputs = [model_input]
    elif layer_name == "MatMulAdd":
        dic = unit_matmul_model(
            num_heads,
            i,
            dic,
            "unit",
            prev_layer,
            M,
            K,
            N,
            inp_dtype,
            wts_dtype,
            out_dtype,
            new_head=1,
            matmul_add=True,
            output=True,
        )
        inputs = [model_input]
    elif layer_name == "MatMulAddGelu":
        M, K, N = (512, 768, 3072)
        dic = unit_matmul_model(
            num_heads,
            i,
            dic,
            "unit",
            prev_layer,
            M,
            K,
            N,
            inp_dtype,
            wts_dtype,
            out_dtype,
            new_head=1,
            matmul_add=True,
            gelu=True,
            output=True,
        )
        inputs = [model_input]
    elif layer_name == "MHAGRPB":
        del dic["inputs"]["layer_0_input_out"]
        dic["inputs"]["layer_" + str(i) + "_Q_out"] = make_tensor_value_info(
            "layer_" + str(i) + "_Q_out", TensorProto.INT16, [1, 512, 1152]
        )
        dic["inputs"]["layer_" + str(i) + "_K_out"] = make_tensor_value_info(
            "layer_" + str(i) + "_K_out", TensorProto.INT16, [1, 512, 1152]
        )
        dic["inputs"]["layer_" + str(i) + "_V_out"] = make_tensor_value_info(
            "layer_" + str(i) + "_V_out", TensorProto.INT16, [1, 512, 768]
        )
        # TODO
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
            output=True,
        )
        dic["inputs"].pop("layer_" + str(i) + "_MHA_out")
        inputs = dic["inputs"].values()

    elif layer_name == "AddSkip":
        del dic["inputs"]["layer_0_input_out"]
        dic["inputs"]["layer_" + str(i) + "_LayerNorm_out"] = make_tensor_value_info(
            "layer_" + str(i) + "_LayerNorm_out", TensorProto.INT16, [1, 512, 768]
        )
        dic["inputs"]["layer_" + str(i) + "_MatMul_out"] = make_tensor_value_info(
            "layer_" + str(i) + "_MatMul_out", TensorProto.INT16, [1, 512, 768]
        )
        dic = unit_add_skip_node(
            num_heads,
            i,
            dic,
            "unit_add_skip",
            "LayerNorm",
            "MatMul",
            1,
            512,
            768,
            inp_dtype,
            wts_dtype,
            out_dtype,
            new_head=1,
            output=True,
        )
        # dic['inputs'].pop('layer_'+str(i)+'_unit_add_skip')
        inputs = dic["inputs"].values()

    graph = make_graph(
        list(dic["nodes"].values()),
        f"single_{layer_name}_node",
        inputs=inputs,
        outputs=list(dic["outputs"].values()),
    )
    # print(dic['initializers'])
    graph.initializer.extend(list(dic["initializers"].values()))
    model = make_model(graph)
    filename = get_savepath("Unit_models")
    onnx.save_model(model, os.path.join(filename, f"{layer_name}.onnx"))
    update_output_shapes_dtypes(os.path.join(filename, f"{layer_name}.onnx"))
    return filename, model


if __name__ == "__main__":
    layers = ["LayerNorm", "MatMul", "MatMulAdd", "MatMulAddGelu", "MHAGRPB", "AddSkip"]
    # layers=[]
    for layer in layers:
        filename, onnx_model = create_unit_node(layer)
