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
import numpy as np
from pathlib import Path
from argparse import Namespace
from typing import Optional, Union

from olive.passes import Pass 
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.model import ONNXModelHandler

class RedirectModel(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                required=True,
                description="file name of a model previously saved in tmp_name (by the store pass); it is loaded and becomes the pass output",
            ),
            "tmp_name": PassConfigParam(
                type_=str,
                default="tmp_olive_custom",
                description="temp dir name, created in working directory if it is relative path",
            ),
        }

    def _run_for_config(self, model: ONNXModelHandler, config, output_model_path):

        src_model_path = model.model_path
        tmp_dir_path = self._create_tmp_dir(config.tmp_name)    
        dst_model_path = str(tmp_dir_path / config.model_name)

        model = onnx.load(dst_model_path)
        return model_proto_to_olive_model(model, dst_model_path, config)

    def _create_tmp_dir(self, tmp_name):
        cwd = os.getcwd()
        dir_path = Path(os.path.join(cwd, tmp_name))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
