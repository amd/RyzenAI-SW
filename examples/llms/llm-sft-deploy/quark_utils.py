# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
import json
from accelerate.utils.modeling import find_tied_parameters
from torch import nn
from transformers import AutoTokenizer
from quark.torch.quantization.config.config import Config
from quark.torch.export.config.config import ExporterConfig
from quark.torch import ModelExporter
from quark.torch.export.api import ModelImporter
from safetensors.torch import load_file as safe_load_file
import sys
import os

PT_WEIGHTS_NAME = "model_state_dict.pth"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
def import_hf_model(model: nn.Module, model_info_dir: str):
    '''
    Load the model file, perform preprocessing and post-processing, load weights into the model.
    '''
    print("Start importing hf_format quantized model ...")
    importer = ModelImporter(model_info_dir=model_info_dir)
    model_config = importer.get_model_config()
    model_state_dict = _load_hf_state_dict(model_info_dir)
    model = importer.import_model(model, model_config, model_state_dict)
    _untie_parameters(model, model_state_dict)
    model.load_state_dict(model_state_dict)
    print("hf_format quantized model imported successfully.")
    return model

def _load_hf_state_dict(model_info_dir: str) -> Dict[str, torch.Tensor]:
    '''
    Load the state dict from safetensor file by load_file of safetensors.torch.
    '''
    model_state_dict: Dict[str, torch.Tensor] = {}
    safetensors_dir = Path(model_info_dir)
    safetensors_path = safetensors_dir / SAFE_WEIGHTS_NAME
    safetensors_index_path = safetensors_dir / SAFE_WEIGHTS_INDEX_NAME
    if safetensors_path.exists():
        model_state_dict = safe_load_file(str(safetensors_path))
    # is_shard
    elif safetensors_index_path.exists():
        with open(str(safetensors_index_path), "r") as file:
            safetensors_indices = json.load(file)
        safetensors_files = [value for _, value in safetensors_indices["weight_map"].items()]
        safetensors_files = list(set(safetensors_files))
        for filename in safetensors_files:
            filepath = safetensors_dir / filename
            model_state_dict.update(safe_load_file(str(filepath)))
    else:
        raise FileNotFoundError(f"Neither {str(safetensors_path)} nor {str(safetensors_index_path)} were found. Please check that the model path specified {str(safetensors_dir)} is correct.")
    return model_state_dict

def _untie_parameters(model: nn.Module, model_state_dict: Dict[str, Any]) -> None:
    '''
    Some parameters share weights, such as embedding and lm_head, and when exporting with `PretrainedModel.save_pretrained`
    only one of them will be exported, so need to copy the parameters.
    '''
    # TODO: Only embedding for now, need to solve other cases, such as encoder-decoder tied
    tied_param_groups = find_tied_parameters(model)
    if len(tied_param_groups) > 0:
        if len(tied_param_groups) > 1 or "lm_head.weight" not in tied_param_groups[0]:
            raise ValueError(
                f"Your have tied_param_groups: {tied_param_groups}, temporarily does not support the case where tied_param is not 'lm_head and embedding'"
            )
        missing_key: List[str] = []
        tied_param_value: Optional[torch.Tensor] = None
        for tied_param_name in tied_param_groups[0]:
            if tied_param_name in model_state_dict.keys():
                tied_param_value = model_state_dict[tied_param_name]
            else:
                missing_key.append(tied_param_name)
        if tied_param_value is not None:
            for tied_param_key in missing_key:
                model_state_dict[tied_param_key] = tied_param_value
        else:
            raise ValueError("Cannot assign a value to tied_params because tied_param_value is None")
