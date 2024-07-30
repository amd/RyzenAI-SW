#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#
import argparse
import os
import shutil
import sys

import onnx
import torch
from onnxruntime.quantization.matmul_4bits_quantizer import (
    GPTQWeightOnlyQuantConfig,
    MatMul4BitsQuantizer,
    RTNWeightOnlyQuantConfig,
)
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import (
    AutoOptimizationConfig,
    ORTModelForCausalLM,
    ORTOptimizer,
)
from optimum.onnxruntime.configuration import OptimizationConfig

from transformers import AutoConfig

from .chatglm_config.chat_glm import CustomChatGLM2OnnxConfig
from .qwen_config.qwen import CustomQwenOnnxConfig


def check_onnx_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if ".onnx" in filename or ".onnx.data" in filename:
            print(
                f"\nAn ONNX file is already present in the path {directory}. Please clean the directory"
            )
            sys.exit(1)


def copy_required_files(input_dir: str, output_dir: str):
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not ".onnx" in filename:
            shutil.copy(filepath, os.path.join(output_dir, filename))


from optimizer import optimize_model


class AttentionMaskFormat:
    # Build 1D mask indice (sequence length). It requires right side padding! Recommended for BERT model to get best performance.
    MaskIndexEnd = 0

    # For experiment only. Do not use it in production.
    MaskIndexEndAndStart = 1

    # Raw attention mask with 0 means padding (or no attention) and 1 otherwise.
    AttentionMask = 2

    # No attention mask
    NoMask = 3


class FusionOptions:
    """Options of fusion in graph optimization"""

    def __init__(self, model_type):
        self.enable_gelu = True
        self.enable_layer_norm = True
        self.enable_attention = False
        self.enable_rotary_embeddings = True

        # Use MultiHeadAttention instead of Attention operator. The difference:
        # (1) Attention has merged weights for Q/K/V projection, which might be faster in some cases since 3 MatMul is
        #     merged into one.
        # (2) Attention could only handle self attention; MultiHeadAttention could handle both self and cross attention.
        self.use_multi_head_attention = False
        self.disable_multi_head_attention_bias = False

        self.enable_skip_layer_norm = True
        self.enable_embed_layer_norm = True
        self.enable_bias_skip_layer_norm = True
        self.enable_bias_gelu = True
        self.enable_gelu_approximation = False
        self.enable_qordered_matmul = True

        self.enable_shape_inference = True
        self.enable_gemm_fast_gelu = False
        self.group_norm_channels_last = True

        if model_type == "clip":
            self.enable_embed_layer_norm = False

        # Set default to sequence length for BERT model to use fused attention to speed up.
        # Note that embed layer normalization will convert 2D mask to 1D when mask type is MaskIndexEnd.
        self.attention_mask_format = AttentionMaskFormat.AttentionMask
        if model_type == "bert":
            self.attention_mask_format = AttentionMaskFormat.MaskIndexEnd
        elif model_type == "vit":
            self.attention_mask_format = AttentionMaskFormat.NoMask

        self.attention_op_type = None

        # options for stable diffusion
        if model_type in ["unet", "vae", "clip"]:
            self.enable_nhwc_conv = True
            self.enable_group_norm = True
            self.enable_skip_group_norm = True
            self.enable_bias_splitgelu = True
            self.enable_packed_qkv = True
            self.enable_packed_kv = True
            self.enable_bias_add = True


class ONNXLLMConverter:
    def __init__(self, model_name, optimization_level=2):
        """
        Initialize the ONNX LLM Converter.

        Args:
        - model_name :  LLM model name
        - export_dir: The folder to save the exported ONNX model.
        - optimization_level: The level of optimization to apply (default is 2).
        """
        self.model_name = model_name
        self.optimization_level = optimization_level
        self.config = AutoConfig.from_pretrained(model_name)
        export_dir = ""
        optim_dir = ""

    def export(self, model_output_dir):  # Export the model and optimize if enabled
        """
        Convert the language model to ONNX format.
        """
        self.export_dir = os.path.join(model_output_dir, "fp32")

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        # Check if onnx model is already present in the output directory
        check_onnx_files(self.export_dir)

        if "chatglm" in self.model_name:
            onnx_config = CustomChatGLM2OnnxConfig(
                config=self.config,
                task="text-generation",
                use_past_in_inputs=False,
            )
            onnx_config_with_past = CustomChatGLM2OnnxConfig(
                self.config, task="text-generation", use_past=True
            )

            custom_onnx_configs = {
                "model": onnx_config,
            }
        else:
            custom_onnx_configs = None

        try:
            main_export(
                self.model_name,
                output=self.export_dir,
                task="text-generation-with-past",
                trust_remote_code=True,
                custom_onnx_configs=custom_onnx_configs,
                no_post_process=True,
                opset=15,
            )
        except Exception as e:
            print(f"failed export with error {e}")

    def optimize(self, inp_model_path, model_output_dir, opt_level, only_onnxruntime):

        # Optimize the exported float model

        self.optim_dir = os.path.join(model_output_dir, "fp32_optm")
        exported_path = ""
        if not os.path.exists(self.optim_dir):
            os.makedirs(self.optim_dir)
        else:
            check_onnx_files(self.optim_dir)

        if inp_model_path:
            exported_path = inp_model_path
        else:
            exported_path = os.path.join(self.export_dir, "model.onnx")

        if not os.path.exists(exported_path):
            print(f"\nExported model does not exist in path {exported_path}\n")
            sys.exit(1)

        optimization_options = FusionOptions("gpt2")

        try:
            model_opt = optimize_model(
                exported_path,
                model_type="gpt2",
                num_heads=self.config.num_attention_heads,
                hidden_size=self.config.hidden_size,
                opt_level=opt_level,
                optimization_options=optimization_options,
                only_onnxruntime=only_onnxruntime,
            )
        except Exception as e:
            print(f"failed optimize with error {e}")

        if not os.path.exists(self.optim_dir):
            os.makedirs(self.optim_dir)
        optim_path = os.path.join(self.optim_dir, "model.onnx")
        model_opt.save_model_to_file(optim_path, use_external_data_format=True)

        if inp_model_path:
            copy_required_files(os.path.dirname(inp_model_path), self.optim_dir)
        else:
            copy_required_files(self.export_dir, self.optim_dir)

        print(f"Optimized ONNX model saved to {optim_path}")

    def quantize(
        self,
        inp_model_path,
        model_output_dir,
        group_size,
        optimize=True,
        algo_config=None,
        accuracy_level=0,
        symmetric=True,
    ):

        quantized_model_dir = os.path.join(model_output_dir, "quant")
        model_path = ""
        if not os.path.exists(quantized_model_dir):
            os.makedirs(quantized_model_dir)
        else:
            check_onnx_files(quantized_model_dir)

        if inp_model_path:
            model_path = inp_model_path
        elif optimize == False:
            model_path = os.path.join(self.export_dir, "model.onnx")
        else:
            model_path = os.path.join(self.optim_dir, "model.onnx")

        if not os.path.exists(model_path):
            print(f"\nModel does not exist in path {model_path}\n")
            sys.exit(1)

        # Load the ONNX model
        onnx_model = onnx.load(model_path, load_external_data=True)
        quantized_model_path = os.path.join(quantized_model_dir, "model.onnx")

        # Quantize the model
        try:
            quant = MatMul4BitsQuantizer(
                onnx_model,
                group_size,
                symmetric,
                accuracy_level=accuracy_level,
                algo_config=algo_config,
            )
        except Exception as e:
            print(f"failed optimize with error {e}")

        quant.process()

        # Save the quantized model
        quant.model.save_model_to_file(
            quantized_model_path, use_external_data_format=True
        )

        if inp_model_path:
            copy_required_files(os.path.dirname(inp_model_path), quantized_model_dir)
        elif optimize == False:
            copy_required_files(self.export_dir, quantized_model_dir)
        else:
            copy_required_files(self.optim_dir, quantized_model_dir)

        print(f"Quantized model saved to {quantized_model_path}")
