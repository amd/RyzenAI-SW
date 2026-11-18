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

class MergeModel(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                required=True,
                description="file name of the merged model, written to tmp_name",
            ),
            "tail_model_name": PassConfigParam(
                type_=str,
                required=True,
                description="file name of the tail model in tmp_name, merged after the input (trunk) model",
            ),
            "tmp_name": PassConfigParam(
                type_=str,
                default="tmp_olive_custom",
                description="temp dir name, created in working directory if it is relative path",
            ),
        }

    def _run_for_config(self, model: ONNXModelHandler, config, output_model_path):
        trunk_model_path = model.model_path
        tmp_dir_path = self._create_tmp_dir(config.tmp_name)    
        tail_model_path = str(tmp_dir_path / config.tail_model_name)
        merged_model_path = str(tmp_dir_path / config.model_name)

        merged_model = self._merge_model(trunk_model_path, tail_model_path)

        return model_proto_to_olive_model(merged_model, merged_model_path, config)

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

    def _merge_model(self, trunk_model_path, tail_model_path):

        A = onnx.load(trunk_model_path)
        B = onnx.load(tail_model_path)

        merged_nodes = list(A.graph.node) + list(B.graph.node)

        a_list = list(A.graph.initializer)
        b_list = list(B.graph.initializer)
        init_dict = {}
        for init in a_list:
            init_dict[init.name] = init
        for init in b_list:
            if init.name not in init_dict:
                init_dict[init.name] = init
        merged_initializers = list(init_dict.values())

        merged_graph = onnx.helper.make_graph(
            merged_nodes,
            name="merged_model",
            inputs=A.graph.input,
            outputs=B.graph.output,
            initializer=merged_initializers
        )
        merged_model = onnx.helper.make_model(merged_graph)
        self._sync_opset_version(merged_model, A, B)
        merged_model.ir_version = A.ir_version if A.ir_version > B.ir_version else B.ir_version
        self._remove_unused_initializers(merged_model.graph)
        return merged_model

    def _create_tmp_dir(self, tmp_name):
        cwd = os.getcwd()
        dir_path = Path(os.path.join(cwd, tmp_name))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _sync_opset_version(self, target_model, source_model_0, source_model_1):
        target_model.opset_import.clear()
        for opset in source_model_0.opset_import:
            target_model.opset_import.append(
                onnx.helper.make_operatorsetid(opset.domain, opset.version)
            )
        for opset in source_model_1.opset_import:
            target_model.opset_import.append(
                onnx.helper.make_operatorsetid(opset.domain, opset.version)
            )
        return
