import copy
import gc
import os
import time
from typing import Dict, List, Optional

import qlinear
import torch
from modeling_chatglm3_amd import SelfAttention

# AWQ
from qmodule import WQLinear
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.phi.modeling_phi import PhiAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention


class TransformConfig:
    def __init__(
        self,
        flash_attention_plus,
        fast_mlp,
        fast_attention,
        precision,
        model_name,
        target,
        w_bit: int = 4,
        group_size: int = 128,
        profilegemm=False,
        profile_layer=False,
        mhaops=None,
    ):
        self.flash_attention_plus = flash_attention_plus
        self.fast_mlp = fast_mlp
        self.fast_attention = fast_attention
        self.precision = precision
        self.target = target
        self.model_name = model_name  # args.model_name
        self.w_bit = w_bit
        self.group_size = group_size
        self.profilegemm = profilegemm
        self.profile_layer = profile_layer
        self.mhaops = mhaops


class RyzenAILLMEngine:
    node_count = 0
    supported_models = {
        "flash_attention_plus": [
            "opt-125m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
            "opt-30b",
            "llama-2-7b",
            "llama-2-7b-chat",
            "llama-2-13b",
            "llama-2-13b-chat",
            "CodeLlama-7b-hf",
            "code-llama-2-7b",
            "Qwen1.5-7B",
            "Qwen1.5-7B-Chat",
            "chatglm3-6b",
            "Mistral-7B-v0.1",
            "phi-2",
            "Meta-Llama-3-8B-Instruct",
        ],
        "fast_mlp": [
            "llama-2-7b",
            "llama-2-7b-chat",
            "llama-2-13b",
            "llama-2-13b-chat",
            "CodeLlama-7b-hf",
            "code-llama-2-7b",
            "Mistral-7B-v0.1",
        ],
        "fast_attention": [
            "llama-2-7b",
            "llama-2-7b-chat",
        ],
    }

    @classmethod
    def replace_node(cls, model, old_node, new_node, new_node_args, new_node_kwargs={}):
        cls.node_count = 0

        def _replace(
            module, name, old_node, new_node, new_node_args, new_node_kwargs={}
        ):
            for attr_str in dir(module):
                try:
                    target_attr = getattr(module, attr_str)
                    if type(target_attr) == old_node:
                        _old = target_attr
                        _new = new_node(*new_node_args, **new_node_kwargs)
                        if (
                            _old._get_name() == "DynamicQuantizedLinear"
                        ):  # Replaced by qlinear.QLinear
                            _new.in_features = _old.in_features
                            _new.out_features = _old.out_features
                            _new.weight_bias = _old._packed_params._weight_bias()
                            _new.quantize_weights()
                            del _old
                        elif (
                            _old.__class__.__name__ == "Linear"
                        ):  # Replaced by qlinear.QLinearPerGrp
                            _new.in_features = _old.in_features
                            _new.out_features = _old.out_features
                            _new.bias = _old.bias
                            _new.weight = _old.weight
                            del _old
                        elif (
                            _old.__class__.__name__ == "WQLinear"
                        ):  # Replaced by qlinear.QLinearPerGrp
                            _new.in_features = _old.in_features
                            _new.out_features = _old.out_features
                            _new.bias = _old.bias
                            _new.w_bit = _old.w_bit
                            _new.group_size = _old.group_size
                            _new.qweight = _old.qweight
                            _new.qzeros = _old.qzeros
                            _new.scales = _old.scales
                            del _old
                            gc.collect()
                        elif _old.__class__.__name__ == "Softmax":  # experimental
                            _new.dim = _old.dim
                        elif (
                            _old.__class__.__name__ == "OPTAttention"
                        ):  # Replaced by OPTFlashAttentionPlus
                            _new.head_dim = _old.head_dim
                            _new.is_decoder = _old.is_decoder
                            _new.q_proj = copy.deepcopy(_old.q_proj)
                            _new.k_proj = copy.deepcopy(_old.k_proj)
                            _new.v_proj = copy.deepcopy(_old.v_proj)
                            _new.out_proj = copy.deepcopy(_old.out_proj)
                            del _old
                            _new.init_faplus()
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "LlamaAttention"
                        ):  # Replaced by LlamaFlashAttentionPlus or LlamaFastAttention
                            _new.config = _old.config
                            _new.layer_idx = _old.layer_idx
                            _new.q_proj = copy.deepcopy(_old.q_proj)
                            _new.k_proj = copy.deepcopy(_old.k_proj)
                            _new.v_proj = copy.deepcopy(_old.v_proj)
                            _new.o_proj = copy.deepcopy(_old.o_proj)
                            del _old
                            _new.init_faplus()
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "Qwen2Attention"
                        ):  # Replaced by Qwen2FlashAttentionPlus
                            _new.config = _old.config
                            _new.layer_idx = _old.layer_idx
                            _new.q_proj = copy.deepcopy(_old.q_proj)
                            _new.k_proj = copy.deepcopy(_old.k_proj)
                            _new.v_proj = copy.deepcopy(_old.v_proj)
                            _new.o_proj = copy.deepcopy(_old.o_proj)
                            del _old
                            _new.init_faplus()
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "MistralAttention"
                        ):  # Replaced by MistralFlashAttentionPlus
                            _new.config = _old.config
                            _new.layer_idx = _old.layer_idx
                            _new.q_proj = copy.deepcopy(_old.q_proj)
                            _new.k_proj = copy.deepcopy(_old.k_proj)
                            _new.v_proj = copy.deepcopy(_old.v_proj)
                            _new.o_proj = copy.deepcopy(_old.o_proj)
                            del _old
                            _new.init_faplus()
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "PhiAttention"
                        ):  # Replaced by PhiFlashAttentionPlus
                            _new.config = _old.config
                            _new.layer_idx = _old.layer_idx
                            _new.q_proj = copy.deepcopy(_old.q_proj)
                            _new.k_proj = copy.deepcopy(_old.k_proj)
                            _new.v_proj = copy.deepcopy(_old.v_proj)
                            _new.dense = copy.deepcopy(_old.dense)
                            del _old
                            _new.init_faplus()
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "LlamaMLP"
                        ):  # Replaced by LlamaFastMLP
                            _new.gate_proj = copy.deepcopy(_old.gate_proj)
                            _new.up_proj = copy.deepcopy(_old.up_proj)
                            _new.down_proj = copy.deepcopy(_old.down_proj)
                            _new.act_fn = _old.act_fn
                            del _old
                            _new.init_fastmlp()
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "SelfAttention"
                        ):  # Replaced by SelfAttention in chatglm3-6b
                            _new.layer_number = _old.layer_number
                            _new.qkv_hidden_size = copy.deepcopy(_old.qkv_hidden_size)
                            _new.query_key_value = copy.deepcopy(_old.query_key_value)
                            _new.dense = copy.deepcopy(_old.dense)
                            del _old
                            gc.collect()
                        else:
                            pass
                        setattr(module, attr_str, _new)
                        cls.node_count += 1
                except Exception as e:
                    print(
                        f"[RyzenAILLMEngine] replace_node: Exception encountered with: {attr_str}!!"
                    )
                    print(f"[RyzenAILLMEngine] Exception: {repr(e)}")
                    raise SystemExit

            for name, immediate_child_module in module.named_children():
                _replace(
                    immediate_child_module,
                    name,
                    old_node,
                    new_node,
                    new_node_args,
                    new_node_kwargs,
                )

        print(
            f"[RyzenAILLMEngine] Model transformation: Replacing {old_node} layers with {new_node} ..."
        )
        _replace(model, "model", old_node, new_node, new_node_args, new_node_kwargs)
        print(
            f"[RyzenAILLMEngine] Model transformation done!: Replaced {cls.node_count} {old_node} layers with {new_node}."
        )

    @classmethod
    def qualify(cls, model: torch.nn.Module, user_requested: Dict) -> bool:
        available_opts = {}
        for mode in user_requested.keys():
            available_opts[mode] = False
            if user_requested[mode]:
                for m in cls.supported_models[mode]:
                    if model.model_name in m:
                        available_opts[mode] = True

        ok_to_proceed = True
        for mode in user_requested.keys():
            if ((user_requested[mode] == True) and (available_opts[mode] == True)) or (
                user_requested[mode] == False
            ):
                ok_to_proceed = ok_to_proceed and True
            else:
                ok_to_proceed = False

        # print(f"[RyzenAILLMEngine] user_requested: {user_requested}")
        # print(f"[RyzenAILLMEngine] available_opts: {available_opts}")
        # print(f"[RyzenAILLMEngine] model_name: {model.model_name}")
        return ok_to_proceed

    @classmethod
    def transform(cls, model: torch.nn.Module, transform_confg: TransformConfig):
        user_requested = {
            "flash_attention_plus": transform_confg.flash_attention_plus,
            "fast_mlp": transform_confg.fast_mlp,
            "fast_attention": transform_confg.fast_attention,
        }
        print(f"[RyzenAILLMEngine] Checking for available optimizations ... ")
        ok_to_proceed = cls.qualify(model, user_requested)
        if ok_to_proceed == False:
            print(
                f"[RyzenAILLMEngine] Optimizations not available for this; run without optimizations ... exiting ... !!!"
            )
            raise SystemExit

        ## Flash Attention and Attention optimizations
        if user_requested["flash_attention_plus"]:
            if "opt" in model.model_name:
                from opt_flash_attention import OPTFlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                    "model_name": transform_confg.model_name,
                }
                cls.replace_node(
                    model, OPTAttention, OPTFlashAttentionPlus, node_args, node_kwargs
                )
            elif ("llama" in model.model_name) or ("Llama" in model.model_name):
                from llama_flash_attention import LlamaFlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "model_name": transform_confg.model_name,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model,
                    LlamaAttention,
                    LlamaFlashAttentionPlus,
                    node_args,
                    node_kwargs,
                )
            elif "Qwen" in model.model_name:
                from qwen2_flash_attention import Qwen2FlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model,
                    Qwen2Attention,
                    Qwen2FlashAttentionPlus,
                    node_args,
                    node_kwargs,
                )
            elif "chatglm3" in model.model_name:
                from chatglm3_flash_attention import ChatGLM3FlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "layer_number": 0,
                    "model_name": transform_confg.model_name,
                }
                cls.replace_node(
                    model,
                    SelfAttention,
                    ChatGLM3FlashAttentionPlus,
                    node_args,
                    node_kwargs,
                )
            elif "Mistral" in model.model_name:
                from mistral_flash_attention import MistralFlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model,
                    MistralAttention,
                    MistralFlashAttentionPlus,
                    node_args,
                    node_kwargs,
                )
            elif "phi" in model.model_name:
                from phi_flash_attention import PhiFlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model, PhiAttention, PhiFlashAttentionPlus, node_args, node_kwargs
                )

        if user_requested["fast_attention"]:
            if ("llama" in model.model_name) or ("Llama" in model.model_name):
                from llama_fast_attention import LlamaFastAttention

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "model_name": transform_confg.model_name,
                    "precision": transform_confg.precision,
                    "profile": transform_confg.profile_layer,
                    "mhaops": transform_confg.mhaops,
                }
                cls.replace_node(
                    model,
                    LlamaAttention,
                    LlamaFastAttention,
                    node_args,
                    node_kwargs,
                )

        if user_requested["fast_mlp"]:
            if (transform_confg.precision == "w4abf16") and (
                transform_confg.target == "aie"
            ):
                if ("llama" in model.model_name) or ("Llama" in model.model_name):
                    from llama_fast_mlp_npu import LlamaFastMLP

                    node_args = ()
                    node_kwargs = {"precision": transform_confg.precision}
                    cls.replace_node(
                        model, LlamaMLP, LlamaFastMLP, node_args, node_kwargs
                    )

        if transform_confg.precision == "w4abf16":
            cls.replace_node(
                model,
                WQLinear,
                qlinear.QLinearPerGrp,
                (),
                {
                    "device": "cpu",
                    "w_bit": transform_confg.w_bit,
                    "group_size": transform_confg.group_size,
                },
            )

        print(model)
        if transform_confg.precision == "w4abf16":
            from collections import defaultdict

            import onnxruntime as ort

            for n, m in model.named_modules():
                if isinstance(m, qlinear.QLinearPerGrp):
                    print(f"[RyzenAILLMEngine] Preparing weights of layer : {n}")
                    if transform_confg.target == "npugpu":
                        if "up_proj" in n:
                            m.device = transform_confg.target
                        else:
                            m.device = "aie"
                    else:
                        m.device = transform_confg.target
                    m.quantize_weights()
                    if ("up_proj" in n) and (transform_confg.target == "npugpu"):
                        m.device = transform_confg.target
                        ortfile = model.model_name + "_onnx_params_fp16\\" + n + ".onnx"
                        m.session = ort.InferenceSession(
                            ortfile,
                            providers=["DmlExecutionProvider"],  # DmlExecutionProvider
                        )
                        print(f" {n}: {ortfile} : {os.path.exists(ortfile)}")

            model = model.to(torch.bfloat16)
        else:
            if transform_confg.target == "aie":
                node_args = ()
                node_kwargs = {
                    "device": transform_confg.target,
                    "quant_mode": transform_confg.precision,
                    "profiler": transform_confg.profilegemm,
                }
                cls.replace_node(
                    model,
                    torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                    qlinear.QLinear,
                    node_args,
                    node_kwargs,
                )
        gc.collect()
        return model
