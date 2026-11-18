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
from pathlib import Path
from argparse import Namespace
from typing import Optional, Union

from olive.passes import Pass 
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.model import ONNXModelHandler

class SplitModel(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "trunk_model_name": PassConfigParam(
                type_=str,
                required=True,
                description="file name of the trunk model (graph up to the split tensor), written to tmp_name; this is the pass output",
            ),
            "tail_model_name": PassConfigParam(
                type_=str,
                required=True,
                description="file name of the tail model (rest of the graph), written to tmp_name",
            ),
            "split_node_name": PassConfigParam(
                type_=str,
                required=True,
                description="name of the node whose first output is the split tensor",
            ),
            "tmp_name": PassConfigParam(
                type_=str,
                default="tmp_olive_custom",
                description="temp dir name, created in working directory if it is relative path",
            ),
        }

    def _run_for_config(self, model: ONNXModelHandler, config, output_model_path):
        ori_model_path = model.model_path
        tmp_dir_path = self._create_tmp_dir(config.tmp_name)    
        trunk_model_path = str(tmp_dir_path / config.trunk_model_name)
        tail_model_path = str(tmp_dir_path / config.tail_model_name)
        split_node_name = config.split_node_name

        trunk_model, tail_model = self._split_model_v2(ori_model_path, split_node_name)

        onnx.save(tail_model, tail_model_path)
        return model_proto_to_olive_model(trunk_model, trunk_model_path, config)

    def _reachable(self, graph, tensor_name):
        reachable = {tensor_name} 
        nodes = []

        changed = True
        while changed:
            changed = False
            for node in graph.node:
                if node in nodes:
                    continue
                for out_name in node.output:
                    if out_name in reachable:
                        nodes.append(node)
                        reachable.update(node.input)
                        changed = True
        return nodes, reachable

    def _remove_unused_initializers(self, graph):
        used_tensors = set()

        for node in graph.node:
            for inp in node.input:
                used_tensors.add(inp)

        for out in graph.output:
            used_tensors.add(out.name)

        unused = []
        for init in graph.initializer:
            if init.name not in used_tensors:
                unused.append(init)

        for u in unused:
            print(f"Removing unused initializer: {u.name}")
            graph.initializer.remove(u)

    def _split_model_v2(self, ori_model_path, split_node_name):
        model = onnx.load(ori_model_path)
        graph = model.graph

        split_tensor = None
        for node in graph.node:
            if node.name == split_node_name:
                split_tensor = node.output[0]
                break

        nodes_A, _ = self._reachable(model.graph, split_tensor)
        model_A = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes_A,
                name="model_part_A",
                inputs=graph.input,
                outputs=[onnx.helper.make_tensor_value_info(split_tensor, onnx.TensorProto.FLOAT, None)],
                initializer=graph.initializer
            )
        )
        self._remove_unused_initializers(model_A.graph)
        self._sync_opset_version(model_A, model)
        model_A.ir_version = model.ir_version

        nodes_B = []
        for node in graph.node:
            if node not in nodes_A:
                nodes_B.append(node)

        model_B = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes_B,
                name="model_part_B",
                inputs=[onnx.helper.make_tensor_value_info(split_tensor, onnx.TensorProto.FLOAT, None)],
                outputs=graph.output,
                initializer=graph.initializer
            )
        )

        self._remove_unused_initializers(model_A.graph)
        self._sync_opset_version(model_B, model)
        model_B.ir_version = model.ir_version

        return model_A, model_B

    def _create_tmp_dir(self, tmp_name):
        cwd = os.getcwd()
        dir_path = Path(os.path.join(cwd, tmp_name))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _sync_opset_version(self, target_model, source_model):
        target_model.opset_import.clear()
        for opset in source_model.opset_import:
            target_model.opset_import.append(
                onnx.helper.make_operatorsetid(opset.domain, opset.version)
            )
