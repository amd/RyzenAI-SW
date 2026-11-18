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
import onnxruntime as ort
import onnx

from olive.passes import Pass 
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.model import ONNXModelHandler


class ONNXRuntimeOptimize(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                required=True,
                description="file name of the ORT-optimized model, written to tmp_name",
            ),
            "tmp_name": PassConfigParam(
                type_=str,
                default="tmp_olive_custom",
                description="temp dir name, created in working directory if it is relative path",
            ),
        }

    def _run_for_config(self, model: ONNXModelHandler, config, output_model_path):
        tmp_dir_path = self._create_tmp_dir(config.tmp_name)    
        opt_model_path = str(tmp_dir_path / config.model_name)

        ori_model_path = model.model_path
        self._onnxruntime_optimize(ori_model_path, opt_model_path)
        onnx_model = onnx.load(opt_model_path)
        return model_proto_to_olive_model(onnx_model, opt_model_path, config)

    def _create_tmp_dir(self, tmp_name):
        cwd = os.getcwd()
        dir_path = Path(os.path.join(cwd, tmp_name))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _onnxruntime_optimize(self, ori_model_path, opt_model_path):
        sess_option = ort.SessionOptions()
        sess_option.optimized_model_filepath = opt_model_path
        sess_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        kwargs = {}
        kwargs["disabled_optimizers"] = ["ConstantSharing", "MatMulScaleFusion", "MatMulAddFusion"]
        _ = ort.InferenceSession(ori_model_path, sess_option, providers=["CPUExecutionProvider"], **kwargs)
        return
