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
import json
import time
import sys

np.random.seed(42)

from . import onnx_graph as ogm

TENSOR_PACK_ALIGNMENT = 4  # Bytes


def np_ref(X, A, B):
    return (X @ A) @ B


def align_to_next(n, A):
    """Align the 'n' to a multiple of 'A'
    new_n >= n
    new_n % A = 0
    """
    return ((n + A - 1) // A) * A


def pack_tensors(tensors, alignment=1):
    """Given a list of tensors, create a new tensor with accumulated size.
    Return the new tensor and list of each input tensor to offset in output tensor
    alignment refers to required alignment for each input tensor in the new tensor.
    """
    buffer_size = 0
    res = {}

    for tensor in tensors:
        if isinstance(tensor, onnx.onnx_ml_pb2.TensorProto):
            parse_fn = ogm.parseTensorProto
        else:
            parse_fn = ogm.parseTensorGraphNode

        new_offset = buffer_size
        tensor_info = parse_fn(tensor)
        res[tensor.name] = tensor_info
        res[tensor.name]["offset"] = new_offset
        buffer_size += tensor_info["size_in_bytes"]
        buffer_size = align_to_next(buffer_size, alignment)

    return buffer_size, res


def __assign_xrtkernel_argid(buffer_packs):
    """
    This function assigns an arg_id to each buffer_pack.
    The arg_id is the index of buffer_pack sent into xrt::kernel() call.
    Any change in order will cause UB.
    Current order - input:0, output:1, scratch:2, const: 3
    """
    for i, buffer_pack in enumerate(buffer_packs):
        buffer_pack["xrt_arg_id"] = i
    return buffer_packs


def combine_dicts(dicts, throw_on_duplicates=True):
    """
    Combine multiplle dicts to one
    Throw exception if duplicate keys
    """
    new_dict = {}
    for dict_ in dicts:
        for key, value in dict_.items():
            if throw_on_duplicates and key in new_dict:
                raise Exception(f"Found duplicate key: {key}. Dicts are {dicts}")
            new_dict[key] = value
    return new_dict


def prepare_tensor_maps(tensor_map):
    """
    Reformat the raw data to more structured data for next layer
    tensor_map = {tensor_name -> [tensor size, sub_bo_dict]}
    """
    # Create a mapping from fused_tensor_name -> [buf_sz, arg_id]
    new_tensors = {}
    for i, (key, [buf_sz, tensors]) in enumerate(tensor_map.items()):
        new_tensors[key] = {
            "buffer_size": buf_sz,
            "xrt_arg_id": i,
            "packed_tensors": list(tensors.keys()),
        }

    # Create a mapping from orig_tensor_name -> [fused_tensor_name, arg_id, offset] map
    new_tensors_map = {}
    for new_tensor_label, [_buf_sz, subbos] in tensor_map.items():
        for old_tensor_label, tensor_info in subbos.items():
            if old_tensor_label not in new_tensors_map:
                new_tensors_map[old_tensor_label] = {
                    "packed_buffer_label": new_tensor_label,
                    "xrt_arg_id": new_tensors[new_tensor_label]["xrt_arg_id"],
                    **tensor_info,
                }
            else:
                raise Exception(
                    f"Found duplicate key: {key}. Dicts are {new_tensors_map}"
                )

    return new_tensors, new_tensors_map


def prepare_metadata(graph: ogm.ONNXGraph, tmp_dir: str = ".", prefix: str = ""):
    input_tensors = graph.getPrimaryInputs()
    input_size, input_pack = pack_tensors(input_tensors, TENSOR_PACK_ALIGNMENT)

    output_tensors = graph.getPrimaryOutputs()
    output_size, output_pack = pack_tensors(output_tensors, TENSOR_PACK_ALIGNMENT)

    scratch_tensors = graph.getIntermediateTensors()
    scratch_size, scratch_pack = pack_tensors(scratch_tensors, TENSOR_PACK_ALIGNMENT)

    const_tensors = graph.getConstTensors()
    const_size, const_pack = pack_tensors(const_tensors, TENSOR_PACK_ALIGNMENT)

    # IMP : The order of keys below dictates the xrt buffer args of fused kernel
    new_tensors, new_tensor_map = prepare_tensor_maps(
        {
            "in": [input_size, input_pack],
            "out": [output_size, output_pack],
            "scratch": [scratch_size, scratch_pack],
            "const": [const_size, const_pack],
            "super_instr": [0, {}],
        }
    )

    # Dump const tensors to files and update the tensor map
    const_file_info = graph.writeConsts(const_tensors, tmp_dir, prefix)
    for key, (filename, filesz) in const_file_info.items():
        new_tensor_map[key].update({"file_name": filename, "file_size": filesz})

    # Get the order of ops to be executed.
    op_names = graph.topologicalSortOpsviaONNX()
    op_list = []
    for op_name in op_names:
        op = graph.nodev[op_name]
        op_type = op.op.op_type
        op_args = [node.name for node in op.inputs + op.outputs]
        op_attrs = ogm.extract_op_attrs(op.op)
        op_list.append(
            {"name": op_name, "type": op_type, "args": op_args, "attrs": op_attrs}
        )

    # Large testcase
    # tmp_sz, tmp_off = pack_tensors(input_tensors+output_tensors+scratch_tensors+const_tensors)
    # print("TMP : ", tmp_sz, tmp_off)
    # new_tensors, new_tensor_map = prepare_tensor_maps({'all':[tmp_sz, tmp_off]})

    return op_list, new_tensors, new_tensor_map, graph.aux_info


def save_tensors_to_json(filename, op_list, new_tensors, tensor_map, aux_info):
    data = {
        "op_list": op_list,
        "fused_tensors": new_tensors,
        "tensor_map": tensor_map,
        "aux_info": aux_info,
    }
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=2)
    # print(f"Saved metadata to {filename}")
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    model_path = sys.argv[1]
    model = onnx.load(model_path)
    model_dir, model_filename = os.path.split(model_path)
    tmp_dir_name = os.path.join(model_dir, model.graph.name)
    os.makedirs(tmp_dir_name, exist_ok=True)

    onnx_graph = ogm.ONNXGraph(model)
    op_list, new_tensors, tensor_map, aux_info = prepare_metadata(
        onnx_graph, tmp_dir=tmp_dir_name
    )

    # print("new tensors : ", new_tensors, '\n')
    # print("tensor map : ", tensor_map, '\n')
    # print("op_list : ", op_list, '\n')
    json_str = save_tensors_to_json(
        tmp_dir_name + "/" + model_filename + ".json",
        op_list,
        new_tensors,
        tensor_map,
        aux_info,
    )

    print(f"JSON saved to {os.path.join(tmp_dir_name, model_filename)}.json")
