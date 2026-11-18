#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import json
import onnx
import logging
import platform
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Optional, Union
import onnxruntime as ort
import numpy as np
from onnx import helper, numpy_helper

from olive.passes import Pass 
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.model import ONNXModelHandler


def load_model(path):
    return onnx.load(path)

def save_model(model, path):
    onnx.save(model, path)

def initializer_to_array(initializer):
    return numpy_helper.to_array(initializer)

def array_to_initializer(arr, name):
    return numpy_helper.from_array(arr, name=name)

def find_node_by_output_name(nodes, output_name):
    for n in nodes:
        if output_name in n.output:
            return n
    return None

def is_scalar_array(arr):
    return np.asarray(arr).size == 1

def squeeze_to_vector_if_channel_like(arr, target_len=None):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if (target_len is None) or (arr.shape[0] == target_len):
            return arr
        return None
    non1_dims = [d for d in arr.shape if d != 1]
    if len(non1_dims) == 1:
        squeezed = arr.reshape(non1_dims[0])
        if (target_len is None) or (squeezed.shape[0] == target_len):
            return squeezed
    return None

# ----------------------
# Pattern detection & folding
# ----------------------

def fold_pattern_in_graph(model):
    graph = model.graph

    # Build fast lookup for Constant nodes (output_name -> numpy array)
    const_node_values = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_node_values[node.output[0]] = numpy_helper.to_array(attr.t)

    # Name -> initializer proto mapping for graph.initializer
    name_to_init = {init.name: init for init in graph.initializer}

    nodes = graph.node[:]  # working copy list
    changed = False

    # iterate over transpose nodes (helps target correct upstream chain)
    for transpose_node in list(nodes):
        if transpose_node.op_type != "Transpose":
            continue

        transpose_input = transpose_node.input[0]
        transpose_output = transpose_node.output[0]

        # Step 1: Add upstream
        add_node = find_node_by_output_name(nodes, transpose_input)
        if add_node is None or add_node.op_type != "Add":
            continue

        if len(add_node.input) != 2:
            print(f"[1] Add doesn't have 2 inputs: {add_node.name or '(unnamed)'} - skipping")
            continue

        matmul_input_name = None
        bias_init_name = None
        for inp in add_node.input:
            if inp in name_to_init or inp in const_node_values:
                bias_init_name = inp
            else:
                matmul_input_name = inp

        if bias_init_name is None or matmul_input_name is None:
            print(f"[2] Add inputs not (matmul_out, const_bias): {add_node.name or '(unnamed)'} - skipping")
            continue

        matmul_node = find_node_by_output_name(nodes, matmul_input_name)
        if matmul_node is None or matmul_node.op_type != "MatMul":
            print(f"[3] Upstream of Add is not MatMul: {add_node.name or '(unnamed)'} - skipping")
            continue

        if len(matmul_node.input) != 2:
            print(f"[4] MatMul has not exactly 2 inputs: {matmul_node.name or '(unnamed)'} - skipping")
            continue

        # Identify weight initializer and data input name
        weight_init_name = None
        data_input_name = None
        for inp in matmul_node.input:
            if inp in name_to_init or inp in const_node_values:
                weight_init_name = inp
            else:
                data_input_name = inp

        if weight_init_name is None or data_input_name is None:
            print(f"[5] MatMul inputs not (data, weightConst): {matmul_node.name or '(unnamed)'} - skipping")
            continue

        # Check pre-mul (scalar) before MatMul
        data_node = find_node_by_output_name(nodes, data_input_name)
        pre_mul_node = None
        pre_mul_scalar_name = None

        if data_node is not None and data_node.op_type == "Mul":
            mul_scalar_init = None
            mul_other_input_name = None
            for inp in data_node.input:
                const_val = None
                if inp in name_to_init:
                    const_val = numpy_helper.to_array(name_to_init[inp])
                elif inp in const_node_values:
                    const_val = const_node_values[inp]

                if const_val is not None and is_scalar_array(const_val):
                    mul_scalar_init = inp
                else:
                    mul_other_input_name = inp

            if mul_scalar_init is not None and mul_other_input_name is not None:
                pre_mul_node = data_node
                pre_mul_scalar_name = mul_scalar_init
                data_input_name = mul_other_input_name
            else:
                print(f"[6] Mul feeding MatMul is not scalar multiply: {data_node.name or '(unnamed)'} - skipping")
                continue
        else:
            pre_mul_node = None
            pre_mul_scalar_name = None

        # Find post-transpose Mul (per-channel)
        post_mul_node = None
        for n in nodes:
            if transpose_output in n.input and n.op_type == "Mul":
                post_mul_node = n
                break
        if post_mul_node is None:
            print(f"[7] No post-Transpose Mul consuming '{transpose_output}' - skipping")
            continue

        # Identify constant input of post_mul_node
        post_mul_init_name = None
        post_mul_other_input = None
        for inp in post_mul_node.input:
            if inp in name_to_init or inp in const_node_values:
                post_mul_init_name = inp
            else:
                post_mul_other_input = inp

        if post_mul_init_name is None:
            print(f"[8] Post-Transpose Mul has no constant input: {post_mul_node.name or '(unnamed)'} - skipping")
            continue

        if post_mul_other_input is None or post_mul_other_input != transpose_output:
            print(f"[9] Post-Mul does not consume Transpose output as non-const input - skipping")
            continue

        # Helper to get arrays from initializer or Constant node
        def get_const_arr(name):
            if name in name_to_init:
                return numpy_helper.to_array(name_to_init[name])
            if name in const_node_values:
                return np.asarray(const_node_values[name])
            return None

        W_arr = get_const_arr(weight_init_name)
        b_arr = get_const_arr(bias_init_name)
        A_arr = get_const_arr(post_mul_init_name)
        B0_arr = get_const_arr(pre_mul_scalar_name) if pre_mul_scalar_name is not None else np.array(1.0, dtype=np.float32)

        if any(x is None for x in (W_arr, b_arr, A_arr)):
            print(f"[10] Missing initializer arrays for W/b/A - skipping")
            continue

        if W_arr.ndim != 2:
            print(f"[11] Weight {weight_init_name} not 2D (shape {W_arr.shape}) - skipping")
            continue
        c_dim, m_dim = W_arr.shape

        b_vec = squeeze_to_vector_if_channel_like(b_arr, target_len=m_dim)
        if b_vec is None:
            print(f"[12] Bias {bias_init_name} shape {b_arr.shape} not broadcastable to (m={m_dim}) - skipping")
            continue

        A_vec = squeeze_to_vector_if_channel_like(A_arr, target_len=m_dim)
        if A_vec is None:
            print(f"[13] Post-mul A {post_mul_init_name} shape {A_arr.shape} not per-channel (m={m_dim}) - skipping")
            continue

        if not is_scalar_array(B0_arr):
            print(f"[14] Pre-mul scalar {pre_mul_scalar_name} is not scalar (shape {np.asarray(B0_arr).shape}) - skipping")
            continue

        # compute folded constants
        B0_val = float(np.asarray(B0_arr).reshape(-1)[0])
        scale_vec = (B0_val * A_vec).astype(W_arr.dtype)
        W_folded = W_arr * scale_vec[None, :]
        b_folded = (A_vec * b_vec).astype(b_vec.dtype)

        # create unique new initializer names
        new_W_name = weight_init_name + "_folded"
        new_b_name = bias_init_name + "_folded"
        idx = 0
        while new_W_name in name_to_init:
            idx += 1
            new_W_name = f"{weight_init_name}_folded_{idx}"
        idx2 = 0
        while new_b_name in name_to_init:
            idx2 += 1
            new_b_name = f"{bias_init_name}_folded_{idx2}"

        # add new initializers
        new_W_init = array_to_initializer(W_folded, new_W_name)
        new_b_init = array_to_initializer(b_folded, new_b_name)
        graph.initializer.extend([new_W_init, new_b_init])
        name_to_init[new_W_name] = new_W_init
        name_to_init[new_b_name] = new_b_init

        # build new MatMul and Add nodes
        matmul_out = matmul_node.output[0] + "_folded"
        new_matmul = helper.make_node(
            "MatMul",
            inputs=[data_input_name, new_W_name],
            outputs=[matmul_out],
            name=(matmul_node.name or "MatMul") + "_folded"
        )

        new_add_out = add_node.output[0] + "_folded"
        new_add = helper.make_node(
            "Add",
            inputs=[matmul_out, new_b_name],
            outputs=[new_add_out],
            name=(add_node.name or "Add") + "_folded"
        )

        # insert new nodes before the transpose node (match by transpose output)
        insert_pos = None
        for i, n in enumerate(graph.node):
            if transpose_output in n.output:
                insert_pos = i
                break
        if insert_pos is None:
            insert_pos = len(graph.node)
        graph.node.insert(insert_pos, new_matmul)
        graph.node.insert(insert_pos + 1, new_add)

        # reconnect transpose to take new_add_out as input (match by transpose output)
        for n in graph.node:
            if transpose_output in n.output:
                for ii, inp in enumerate(n.input):
                    if inp == add_node.output[0]:
                        n.input[ii] = new_add_out
                break

        # Rewire downstream consumers of post_mul_node output to use transpose_output
        # If downstream nodes consumed the post-mul output, they must now consume the transpose output
        post_mul_outputs = list(post_mul_node.output)
        # replace inputs in all nodes and also graph outputs if needed
        for consumer in graph.node:
            for i, inp in enumerate(consumer.input):
                if inp in post_mul_outputs:
                    consumer.input[i] = transpose_output

        # Update graph.output names that match post_mul outputs
        for g_out in graph.output:
            if g_out.name in post_mul_outputs:
                # replace graph output name to point to transpose output
                g_out.name = transpose_output

        # Remove old nodes by outputs (safe)
        outputs_to_remove = set()
        for old in (matmul_node, add_node, post_mul_node, pre_mul_node):
            if old is None:
                continue
            for out in old.output:
                outputs_to_remove.add(out)

        new_node_list = []
        for n in graph.node:
            if any(out in outputs_to_remove for out in n.output):
                continue
            new_node_list.append(n)

        graph.ClearField('node')
        graph.node.extend(new_node_list)

        # update local lists/maps
        nodes = graph.node[:]
        name_to_init = {init.name: init for init in graph.initializer}

        changed = True
        print(f"Folded pattern around transpose (output='{transpose_output}'):")
        print(f"  weight '{weight_init_name}' -> '{new_W_name}' (shape {W_folded.shape})")
        print(f"  bias   '{bias_init_name}' -> '{new_b_name}' (shape {b_folded.shape})")
        if pre_mul_scalar_name:
            print(f"  folded scalar '{pre_mul_scalar_name}' (value {B0_val})")
        print(f"  folded post-mul '{post_mul_init_name}' (shape {np.asarray(A_arr).shape})")
        print("  inserted MatMul+Add and reconnected Transpose + downstream consumers.")
        print("-------------------------------------------------------------")

    return changed

class FuseConvMul(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                required=True,
                description="file name of the folded model, written to tmp_name",
            ),
            "tmp_name": PassConfigParam(
                type_=str,
                default="tmp_olive_custom",
                description="temp dir name, created in working directory if it is relative path",
            ),
        }

    def _run_for_config(self, model: ONNXModelHandler, config, output_model_path):
        tmp_dir_path = self._create_tmp_dir(config.tmp_name)    
        tmp_model_path = str(tmp_dir_path / config.model_name)

        ori_model_path = model.model_path
        model = load_model(ori_model_path)
        changed = fold_pattern_in_graph(model)
        assert changed, "no MatMul-Add-Transpose-Mul pattern found to fold"
        inferred = onnx.shape_inference.infer_shapes(model)
        return model_proto_to_olive_model(inferred, tmp_model_path, config)

    def _create_tmp_dir(self, tmp_name):
        cwd = os.getcwd()
        dir_path = Path(os.path.join(cwd, tmp_name))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path