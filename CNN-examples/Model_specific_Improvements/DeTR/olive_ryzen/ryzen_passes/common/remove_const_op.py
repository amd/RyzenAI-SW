#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import json
import logging
import platform
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Optional, Union
import onnx

from olive.passes import Pass 
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.model import ONNXModelHandler

class RemoveConstOp(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                required=True,
                description="model path for model-without-const-op",
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

        model_path = model.model_path
        onnx_model = onnx.load(model_path)
        onnx_model = self._constant_to_initializer(onnx_model)
        return model_proto_to_olive_model(onnx_model, tmp_model_path, config)

    def _create_tmp_dir(self, tmp_name):
        cwd = os.getcwd()
        dir_path = Path(os.path.join(cwd, tmp_name))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _constant_to_initializer(self, model):
        graph = model.graph
        new_initializers = []
        remove_nodes = []
    
        for node in graph.node:
            if node.op_type == "Constant":
                const_output_name = node.output[0]
                value_attr = next((attr for attr in node.attribute if attr.name == "value"), None)
                if value_attr is None:
                    continue
    
                tensor = value_attr.t
                tensor.name = const_output_name
    
                new_initializers.append(tensor)
                remove_nodes.append(node)
    
        for node in remove_nodes:
            graph.node.remove(node)
    
        init_dict = {}
        for init in graph.initializer:
            init_dict[init.name] = init
        for init in new_initializers:
            init_dict[init.name] = init
        graph.initializer.clear()
        graph.initializer.extend(list(init_dict.values()))
    
        input_names = [i.name for i in graph.input]
        for t in new_initializers:
            if t.name in input_names:
                graph.input.remove(next(i for i in graph.input if i.name == t.name))
    
        return model