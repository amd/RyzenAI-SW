##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import argparse
import copy
import json

import onnx
from onnx import helper, TensorProto

from tabulate import tabulate
from colorama import init, Fore
from pathlib import Path

# ONNX tool
from .. import onnx_tool
from ..onnx_tool import Graph
from ..onnx_tool.fusion import *
from ..onnx_tool.fusion import create_descs_from_nodenames, FusionPattern

# Fusion utils
from .utils import *

# from .fuse import *
init(autoreset=True)

# This file contains only fusioncode


def remove_const_nodes(input_model_path):
    sgmodel = onnx.load(input_model_path)
    constant_nodes = {}
    for node in sgmodel.graph.node:
        if node.op_type == "Constant":
            # Store constant node output as (type, data)
            const_data = [data for data in node.attribute[0].t.int64_data]
            constant_nodes[node.output[0]] = (onnx.TensorProto.INT64, const_data)

    names = []
    for node in sgmodel.graph.node:
        if node.op_type == "ReduceSum" or node.op_type == "Unsqueeze":
            for name in node.input:
                if name in constant_nodes.keys():
                    names.append(name)
    for name in names:
        type_name, data = constant_nodes[name]
        if len(data) == 1:
            sgmodel.graph.initializer.append(
                onnx.helper.make_tensor(name, type_name, dims=[1], vals=data)
            )

    sgmodel = onnx.shape_inference.infer_shapes(sgmodel)
    # onnx.checker.check_model(sgmodel)
    onnx.save(sgmodel, input_model_path)


def get_tuple_list(dictionary):
    l_t = []
    total = 0
    # l_t.append(["OP_TYPE","COUNT"])
    for key in dictionary.keys():
        l_t.append([key, dictionary[key]])
        total += dictionary[key]
    l_t.append(["Total", total])
    return l_t


def enable_pad(m, g, prompt_len, pad_prompt_len):
    # Pad initializers
    # pad_prompt_len=512
    for tns in g.initials:
        tensor = g.tensormap[tns]
        if tensor.numpy.shape == [1, prompt_len, 768] or tensor.numpy.shape == (
            1,
            prompt_len,
            768,
        ):
            izr_1_name = tensor.name
            actual_tensor_izr1 = g.tensormap[izr_1_name].numpy.data
            padded_tensor_izr1 = np.zeros((1, pad_prompt_len, 768)).astype(np.uint16)
            padded_tensor_izr1[:, : actual_tensor_izr1.shape[1], :] = actual_tensor_izr1

            g.tensormap[izr_1_name] = onnx_tool.tensor.create_initial_Tensor(
                izr_1_name,
                padded_tensor_izr1,
            )
            g.initials.append(izr_1_name)
        if tensor.numpy.shape == [
            1,
            12,
            prompt_len,
            prompt_len,
        ] or tensor.numpy.shape == (1, 12, prompt_len, prompt_len):
            izr_2_name = tensor.name
            actual_tensor_izr2 = g.tensormap[izr_2_name].numpy.data
            padded_tensor_izr2 = np.zeros(
                (1, 12, pad_prompt_len, pad_prompt_len)
            ).astype(np.uint16)
            padded_tensor_izr2[
                :, :, : actual_tensor_izr2.shape[2], : actual_tensor_izr2.shape[3]
            ] = actual_tensor_izr2
            g.tensormap[izr_2_name] = onnx_tool.tensor.create_initial_Tensor(
                izr_2_name,
                padded_tensor_izr2,
            )
            g.initials.append(izr_2_name)

    # Change Shapes for all dynamic inputs
    for tns in g.tensormap:
        tensor = g.tensormap[tns]
        if tns not in g.initials:
            if len(tensor.shape) >= 2:
                if type(tensor.shape) is tuple:
                    tensor.shape = list(tensor.shape)
                    for i in range(len(tensor.shape)):
                        if tensor.shape[i] == prompt_len:
                            tensor.shape[i] = pad_prompt_len
                elif type(tensor.shape) is list:
                    for i in range(len(tensor.shape)):
                        if tensor.shape[i] == prompt_len:
                            tensor.shape[i] = pad_prompt_len
    for n_m in g.nodemap:
        node = g.nodemap[n_m]
        if node.op_type == "Reshape":
            shape_tensor = g.tensormap[node.input[1]]
            new_numpy_tensor = np.zeros(shape_tensor.numpy.shape).astype(np.int64)
            for i in range(len(shape_tensor.numpy)):
                if shape_tensor.numpy[i] == prompt_len:
                    new_numpy_tensor[i] = pad_prompt_len
                else:
                    new_numpy_tensor[i] = shape_tensor.numpy[i]
            shape_tensor.numpy = new_numpy_tensor
    g.graph_reorder_nodes()
    return g


def get_precision_from_xclbin(xclbin):
    if "a16w8" in xclbin:
        return "a16w8"
    elif "a8w8" in xclbin:
        return "a8w8"


"""
fuse_layer :
- Create subgraphs for all the available patterns (gen_subgraphs == True)
- Fuse the total model with all available patterns
- counts ops before and after fusion
- Important : Make sure that in dynamic_dispatch_patterns, bigger pattern comes first
    - Example: QMHAGRPB, QMatMulAddGelu, QMatMulADD, QMatMul, QLayerNorm, QSkipAdd
"""


def fuse_layers(
    input_onnx_file,
    output_onnx_file=None,
    xclbin=None,
    gen_subgraphs=False,
    verbose=False,
):
    if xclbin:
        precision = get_precision_from_xclbin(xclbin)

    # Replace constant nodes as initializers
    remove_const_nodes(input_onnx_file)

    # Model dir/path
    model_dir, model_name = os.path.split(input_onnx_file)
    model_name = model_name.replace(".onnx", "")
    # Output path
    if not output_onnx_file:
        output_path = os.path.join(os.getcwd(), model_name)
        os.makedirs(os.path.join(output_path), exist_ok=True)
        output_onnx_file = os.path.join(output_path, model_name + ".fused.onnx")
    else:
        output_path = Path(output_onnx_file).parent

    # Original model
    m, g = loadmodel(input_onnx_file)

    # First Remove the extra QDQ from the
    rm_node_list = ["424_convert_QuantizeLinear", "424_convert_DequantizeLinear"]
    for rm_node in rm_node_list:
        if rm_node in g.nodemap.keys():
            g.skip_node(rm_node)
            g.graph_reorder_nodes()

    # Shape infer
    for n_m in g.nodemap.keys():
        node = g.nodemap[n_m]
        if "quantizelinear" in node.op_type.lower():
            g.tensormap[node.output[0]].shape = g.tensormap[node.input[0]].shape

    # Count Ops
    op_count_dictionary = count_ops(g)

    # Change dtype
    g = change_output_dtype(g)

    # Duplicate DQ layer if required
    g = duplicate_layer(m, g, "DequantizeLinear")

    # Re-order nodes
    g.graph_reorder_nodes()

    # Keep a copy of original graph
    original_tensormap = copy.deepcopy(g.tensormap)

    graph_input = g.tensormap[g.input[0]].shape
    modified_graph_input = []
    prompt_len = graph_input[1]
    # pad model to 128 if prompt len<128
    # TODO:
    # This should be a seperate utility in future, as currently we are only working on attention based models this will be used,
    # in future we may encounter models where we may need to change some other dimension
    if not gen_subgraphs:
        if prompt_len < 128:
            pad_prompt_len = 128
            g = enable_pad(m, g, prompt_len, pad_prompt_len)

    # Get all supported subgraph patterns
    # Contains QMHAGRPB, QMatMuL, QMaTMuLAdd, QMaTMuLAddGelu,
    # QLAayerNorm, QSkipAdd(elementwise add), found in PSF, PSH, PSJ models
    from .dynamic_dispatch_subgraphs_2 import patterns as fuse_patterns

    # Find and fuse each subgraph
    flags = {}
    for key in fuse_patterns.keys():
        flags[key] = False
        for fuse_pattern in fuse_patterns[key]:
            # Descs = create_descs_from_nodenames(g, dict3[key][0])
            if flags[key] == True:
                print(
                    'Fused "{key}" Pattern already. Skipping {key} - { fuse_patterns[key].index(fuse_pattern) } pattern'
                )
                continue
            if verbose:
                print(
                    "Pattern Key: {}, Pattern Length: {}".format(key, len(fuse_pattern))
                )
            try:
                Descs = fuse_pattern
                # if key not in pattern_descs:
                #     pattern_descs[key] = [Descs]
                # if gen_subgraphs and 'LayerNorm' in key and len(Descs)>5:
                #     continue
                pattern = FusionPattern(Descs)
                subgraphs = pattern.search_pattern(g)
                if verbose:
                    print(
                        Fore.LIGHTYELLOW_EX
                        + f"Number of patterns found with {key}Pattern =",
                        len(subgraphs),
                    )
                count = 0
                for nodes in subgraphs:
                    if "QMatMul" in key:
                        en_fuse, mm_out_shape = check_if_wts_matmul(
                            g, nodes, original_tensormap
                        )
                        if en_fuse:
                            flag = True  # to make sure we dont find a pattern of matmul nodes that are already fused in other matmul patterns
                            for name in nodes:
                                if name not in g.nodemap:
                                    flag = False
                            if flag == True:
                                if gen_subgraphs:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)
                                    save_name, fuse_name = get_layer_name_for_save(
                                        model_name, nodes, key
                                    )
                                    k.save_model(save_name, rawmodel=m.mproto)
                                    k.fuse_subgraph_node_names(
                                        nodes, key, nodes[0], True
                                    )
                                    k = add_domain(k)
                                    k.save_model(fuse_name, rawmodel=m.mproto)
                                    m_k, k = loadmodel(fuse_name)
                                    # if "Add" not in key:
                                    k = change_inputs(m_k, k, precision)
                                    k.save_model(fuse_name, rawmodel=m.mproto)

                                else:
                                    k = Graph(
                                        g.get_onnxgraph_by_nodenames(nodes),
                                        onnx_tool.utils.ModelConfig({}),
                                    )
                                    node_outputs = []
                                    for n in nodes:
                                        node_outputs.extend(k.nodemap[n].output)

                                g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                                g.nodemap[nodes[0]].set_attr(
                                    "output_shape",
                                    mm_out_shape,
                                )
                                flags[key] == True

                    elif "QMHAGRPB" in key:
                        (
                            QKT_input_qparams,
                            QKT_output_qparams,
                            VSQKT_input_qparams,
                            VSQKT_output_qparams,
                            softmax_input_qparams,
                            softmax_output_qparams,
                            sub_scale,
                            grpb_add_params,
                            sigmoid_params,
                            div_params,
                            grpb_matmul_add_out_params,
                        ) = calc_q_params(g, nodes)
                        if gen_subgraphs:
                            k = Graph(
                                g.get_onnxgraph_by_nodenames(nodes),
                                onnx_tool.utils.ModelConfig({}),
                            )
                            save_name, fuse_name = get_layer_name_for_save(
                                model_name, nodes, key, count
                            )
                            node_outputs = []
                            for n in nodes:
                                node_outputs.extend(k.nodemap[n].output)
                            k.save_model(
                                save_name,
                                rawmodel=m.mproto,
                            )

                            k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            k.attr = {}
                            k.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            k.nodemap[nodes[0]].set_attr(
                                "QKT_input_qparams", QKT_input_qparams
                            )
                            k.nodemap[nodes[0]].set_attr(
                                "QKT_output_qparams", QKT_output_qparams
                            )
                            k.nodemap[nodes[0]].set_attr(
                                "VSQKT_input_qparams", VSQKT_input_qparams
                            )
                            k.nodemap[nodes[0]].set_attr(
                                "VSQKT_output_qparams", VSQKT_output_qparams
                            )
                            k.nodemap[nodes[0]].set_attr(
                                "softmax_input_qparams", softmax_input_qparams
                            )
                            k.nodemap[nodes[0]].set_attr(
                                "softmax_output_qparams", softmax_output_qparams
                            )
                            k.nodemap[nodes[0]].set_attr("GRPB_sub_params", sub_scale)
                            k.nodemap[nodes[0]].set_attr(
                                "GRPB_add_params", grpb_add_params
                            )
                            k.nodemap[nodes[0]].set_attr(
                                "sigmoid_params", sigmoid_params
                            )
                            k.nodemap[nodes[0]].set_attr("div_params", div_params)

                            k.nodemap[nodes[0]].set_attr(
                                "grpb_matmul_add_out_params", grpb_matmul_add_out_params
                            )

                            k = add_domain(k)
                            k.save_model(fuse_name, rawmodel=m.mproto)
                            m_k, k = loadmodel(fuse_name)
                            k = change_inputs(m_k, k, precision)
                            k.save_model(fuse_name, rawmodel=m.mproto)
                            count += 1

                        else:
                            k = Graph(
                                g.get_onnxgraph_by_nodenames(nodes),
                                onnx_tool.utils.ModelConfig({}),
                            )
                            node_outputs = []
                            for n in nodes:
                                node_outputs.extend(k.nodemap[n].output)

                        g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                        g.attr = {}
                        g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                        g.nodemap[nodes[0]].set_attr(
                            "output_shape", original_tensormap[node_outputs[-1]].shape
                        )
                        g.nodemap[nodes[0]].set_attr(
                            "QKT_input_qparams", QKT_input_qparams
                        )
                        g.nodemap[nodes[0]].set_attr(
                            "QKT_output_qparams", QKT_output_qparams
                        )
                        g.nodemap[nodes[0]].set_attr(
                            "VSQKT_input_qparams", VSQKT_input_qparams
                        )
                        g.nodemap[nodes[0]].set_attr(
                            "VSQKT_output_qparams", VSQKT_output_qparams
                        )
                        g.nodemap[nodes[0]].set_attr(
                            "softmax_input_qparams", softmax_input_qparams
                        )
                        g.nodemap[nodes[0]].set_attr(
                            "softmax_output_qparams", softmax_output_qparams
                        )
                        g.nodemap[nodes[0]].set_attr("GRPB_sub_params", sub_scale)
                        g.nodemap[nodes[0]].set_attr("GRPB_add_params", grpb_add_params)
                        g.nodemap[nodes[0]].set_attr("sigmoid_params", sigmoid_params)
                        g.nodemap[nodes[0]].set_attr("div_params", div_params)
                        g.nodemap[nodes[0]].set_attr(
                            "grpb_matmul_add_out_params", grpb_matmul_add_out_params
                        )

                        flags[key] == True

                    elif "LayerNorm" in key:
                        if "LayerNormalization_fused_ReduceMean_8" in nodes:
                            if len(nodes) > 5:
                                nodes = nodes[:-2]
                        if gen_subgraphs:
                            k = Graph(
                                g.get_onnxgraph_by_nodenames(nodes),
                                onnx_tool.utils.ModelConfig({}),
                            )
                            node_outputs = []
                            for n in nodes:
                                node_outputs.extend(k.nodemap[n].output)
                            save_name, fuse_name = get_layer_name_for_save(
                                model_name, nodes, key
                            )
                            k.save_model(save_name, rawmodel=m.mproto)
                            k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            k = add_domain(k)
                            k.save_model(fuse_name, rawmodel=m.mproto)
                            m_k, k = loadmodel(fuse_name)
                            k = change_inputs(m_k, k, precision)
                            k.save_model(fuse_name, rawmodel=m.mproto)

                        else:
                            k = Graph(
                                g.get_onnxgraph_by_nodenames(nodes),
                                onnx_tool.utils.ModelConfig({}),
                            )
                            node_outputs = []
                            for n in nodes:
                                node_outputs.extend(k.nodemap[n].output)

                        g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                        g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                        g.nodemap[nodes[0]].set_attr(
                            "output_shape", original_tensormap[node_outputs[-1]].shape
                        )
                        flags[key] == True

                    elif "SkipAdd" in key:
                        fuse, nodes = check_if_wts_add(g, nodes)
                        if fuse:
                            if gen_subgraphs:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)
                                save_name, fuse_name = get_layer_name_for_save(
                                    model_name, nodes, key
                                )
                                # change k.input = addnode.input
                                k.save_model(save_name, rawmodel=m.mproto)
                                k.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                                k = add_domain(k)
                                k.save_model(fuse_name, rawmodel=m.mproto)
                                m_k, k = loadmodel(fuse_name)
                                k = change_inputs(m_k, k, precision)
                                k.save_model(fuse_name, rawmodel=m.mproto)

                            else:
                                k = Graph(
                                    g.get_onnxgraph_by_nodenames(nodes),
                                    onnx_tool.utils.ModelConfig({}),
                                )
                                node_outputs = []
                                if (
                                    "Add_290" in nodes
                                ):  # Hardcoding for the unique case in PSH TODO : Remove this change and make it generic
                                    node_outputs.extend(
                                        [
                                            "424_convert_QuantizeLinear_Output",
                                            "424_convert_DequantizeLinear_Output",
                                        ]
                                    )

                                for n in nodes:
                                    node_outputs.extend(k.nodemap[n].output)

                            g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
                            g.nodemap[nodes[0]].set_attr("nodes", node_outputs)
                            g.nodemap[nodes[0]].set_attr(
                                "output_shape",
                                original_tensormap[node_outputs[-1]].shape,
                            )
                            flags[key] == True

            except Exception as error:
                if verbose:
                    print(
                        Fore.RED
                        + f"Pattern is not found with {key}-{fuse_patterns[key].index(fuse_pattern)} pattern, going to next avaiable pattern of {key} "
                    )
                    print(error)
                pass

    if verbose:
        print(Fore.LIGHTYELLOW_EX + "Fused Pattern =", key)
    if gen_subgraphs:
        print(
            "- "
            + Fore.CYAN
            + "Saving Subgraphs in: "
            + Fore.RESET
            + "{}".format(model_name)
        )

    # Count ops
    op_count_dictionary_f = count_ops(g)

    # Add domain
    g = add_domain(g)

    if verbose:
        g.save_model("before_change_inputs.onnx")

    # Change inputs
    g = change_inputs(m, g, precision)

    g.save_model(output_onnx_file, rawmodel=m.mproto)  # ADD file
    # inames,onames= generate_transactions(output_onnx_file)

    if verbose:
        print(
            Fore.LIGHTYELLOW_EX + "Opcount before fusion (original model)\n" + "-" * 40
        )
        print(
            Fore.CYAN
            + tabulate(
                get_tuple_list(op_count_dictionary),
                headers=["OP_TYPE", "COUNT"],
                tablefmt="github",
            )
        )
        print(Fore.LIGHTYELLOW_EX + "Opcount after fusion (Fused model)\n" + "-" * 40)
        print(
            Fore.CYAN
            + tabulate(
                get_tuple_list(op_count_dictionary_f),
                headers=["OP_TYPE", "COUNT"],
                tablefmt="github",
            )
        )

    if gen_subgraphs:
        print(
            "- "
            + Fore.CYAN
            + "Saving Fused Subgraphs in: "
            + Fore.RESET
            + "{}\n".format(model_name)
        )


def create_patterns(
    model_name,
    psf_a16_model=None,
    psf_a8_model=None,
    psh_a16_model=None,
    psh_a8_model=None,
):
    pattern_dict = {}
    m, g = loadmodel(psf_a8_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # g.save_model("PSF_w8a8.onnx")
    for key in dict_PSF_a8.keys():
        descs = create_descs_from_nodenames(g, dict_PSF_a8[key])
        if key in pattern_dict.keys():
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]
        # pattern=FusionPattern(descs)
        # subgraphs=pattern.search_pattern(g)
        # for subgraph in subgraphs:
        #     # print(subgraph)
        #     g.fuse_subgraph_node_names(subgraph, key, subgraph[0], True)

    m, g = loadmodel(psf_a16_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # pattern_dict={}
    for key in dict_PSF_a16.keys():
        descs = create_descs_from_nodenames(g, dict_PSF_a16[key])
        if key in pattern_dict:
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]

    m, g = loadmodel(psh_a16_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # pattern_dict={}
    for key in dict_PSH_a16.keys():
        descs = create_descs_from_nodenames(g, dict_PSH_a16[key])
        if key in pattern_dict:
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]

    m, g = loadmodel(psh_a8_model)
    g = change_output_dtype(g)
    g = duplicate_layer(m, g, "DequantizeLinear")
    # pattern_dict={}
    g.save_model("PSH_a8_afterdequant.onnx")
    for key in dict_PSH_a8.keys():
        descs = create_descs_from_nodenames(g, dict_PSH_a8[key])
        if key in pattern_dict:
            pattern_dict[key].append(descs)
        else:
            pattern_dict[key] = [descs]

    with open("dynamic_dispatch_subgraphs_2.py", "w") as f:
        f.write("patterns = ")
        json.dump(pattern_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model_path",
        help="input onnx model path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_model_path", help="model output path", type=str, required=False
    )

    parser.add_argument(
        "--onnx_checker", help="To validate exported onnx model", action="store_true"
    )
    parser.add_argument(
        "--model_name",
        help="To validate exported onnx model",
        default="PSF",
    )
    parser.add_argument(
        "--gen_subgraphs",
        help="Enable subgraph generation",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument("--xclbin", help="XCLBIN file name", default="", required=True)
    parser.add_argument(
        "--verbose",
        help="Enable debug prints",
        default=False,
        required=False,
        action="store_true",
    )
    # Parse args
    args = parser.parse_args()
    # Display args
    print("\n- " + Fore.CYAN + "Arguments" + Fore.RESET + "\n" + "-" * 80)
    for k, v in vars(args).items():
        print("  {:<20}: {}".format(k, v))
    print("-" * 80)

    fuse_layers(
        args.input_model_path,
        args.output_model_path,
        xclbin=args.xclbin,
        gen_subgraphs=args.gen_subgraphs,
        verbose=args.verbose,
    )
