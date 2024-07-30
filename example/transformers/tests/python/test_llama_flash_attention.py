#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy

import pytest
import qlinear
import torch
from llama_flash_attention import LlamaFlashAttentionPlus
from qmodule import WQLinear
from ryzenai_llm_engine import RyzenAILLMEngine
from test_utils import (
    awq_processing,
    generate_attention_test_input,
    pergrp_processing,
    to_skip,
)

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

torch.manual_seed(0)


llama_attention_shapes = [
    ("llama-2-wts-hf/7B", 1, 1),
    ("llama-2-wts-hf/7B", 1, 128),
    ("llama-2-wts-hf/7B", 1, 512),
    ("llama-2-wts-hf/7B", 1, 717),
    ("llama-2-wts-hf/7B", 1, 2000),
    ("llama-2-wts-hf/7B", 1, 2545),
    ("llama-2-wts-hf/7B", 1, 4000),
    ("Meta-Llama-3-8B-Instruct", 1, 1),
    ("Meta-Llama-3-8B-Instruct", 1, 128),
    ("Meta-Llama-3-8B-Instruct", 1, 512),
    ("Meta-Llama-3-8B-Instruct", 1, 717),
    ("Meta-Llama-3-8B-Instruct", 1, 2000),
    ("Meta-Llama-3-8B-Instruct", 1, 2545),
    ("Meta-Llama-3-8B-Instruct", 1, 4000),
    ("Meta-Llama-3-8B-Instruct", 1, 7531),
    ("Meta-Llama-3-8B-Instruct", 1, 8100),
    # ("llama-2-wts-hf/7B", 4, 1   ),
    # ("llama-2-wts-hf/7B", 4, 128 ),
    # ("llama-2-wts-hf/7B", 4, 512 ),
    # ("llama-2-wts-hf/7B", 4, 717 ),
    # ("llama-2-wts-hf/7B", 4, 2000),
    # ("llama-2-wts-hf/7B", 4, 2545),
    # ("llama-2-wts-hf/7B", 4, 4000),
    # ("Meta-Llama-3-8B-Instruct", 4, 1   ),
    # ("Meta-Llama-3-8B-Instruct", 4, 128 ),
    # ("Meta-Llama-3-8B-Instruct", 4, 512 ),
    # ("Meta-Llama-3-8B-Instruct", 4, 717 ),
    # ("Meta-Llama-3-8B-Instruct", 4, 2000),
    # ("Meta-Llama-3-8B-Instruct", 4, 2545),
    # ("Meta-Llama-3-8B-Instruct", 4, 4000),
    # ("Meta-Llama-3-8B-Instruct", 4, 7531),
    # ("Meta-Llama-3-8B-Instruct", 4, 8100),
]


@pytest.mark.parametrize("llama_attention_shape", llama_attention_shapes)
@pytest.mark.parametrize("device", ["aie"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_llama_flash_attention(llama_attention_shape, quant_mode, w_bit, device, dtype):
    to_skip(device, dtype, quant_mode)
    model_name, batch_size, sequence_length = llama_attention_shape

    config = LlamaConfig()  # Use default param
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    is_llama2 = "2" in model_name
    attn = LlamaAttention(
        config=config,
        layer_idx=0,
    )
    attn_fa = LlamaFlashAttentionPlus(
        config=config,
        layer_idx=0,
        precision=quant_mode,
        model_name=model_name,
    )

    if quant_mode == "w4abf16":
        # Convert before replication, as only separate projections can be copied
        if is_llama2:
            # Linear -> WQLinear
            awq_processing(attn, w_bit)
        else:  # llama3
            # Linear -> QLinearPerGrp
            pergrp_processing(attn, w_bit)

    # Copy separate QKV projections (Linear or WQLinear)
    attn_fa.k_proj = copy.deepcopy(attn.k_proj)
    attn_fa.v_proj = copy.deepcopy(attn.v_proj)
    attn_fa.q_proj = copy.deepcopy(attn.q_proj)
    attn_fa.o_proj = copy.deepcopy(attn.o_proj)

    if quant_mode == "none":
        attn = attn.to(dtype)
        attn_fa = attn_fa.to(dtype)
    elif quant_mode == "w4abf16":
        # Merge QKV projections (WQLinear)
        attn_fa.init_faplus()

        if is_llama2:
            RyzenAILLMEngine.replace_node(
                attn,
                WQLinear,
                qlinear.QLinearPerGrp,
                (),
                {"device": device, "w_bit": w_bit, "group_size": 128},
            )
            RyzenAILLMEngine.replace_node(
                attn_fa,
                WQLinear,
                qlinear.QLinearPerGrp,
                (),
                {"device": device, "w_bit": w_bit, "group_size": 128},
            )
        else:
            attn.q_proj.device = device
            attn.k_proj.device = device
            attn.v_proj.device = device
            attn.o_proj.device = device
            attn_fa.qkv_proj.device = device
            attn_fa.o_proj.device = device

        # Post processing for QLinearPerGrp
        attn.q_proj.quantize_weights()
        attn.k_proj.quantize_weights()
        attn.v_proj.quantize_weights()
        attn.o_proj.quantize_weights()
        attn_fa.qkv_proj.quantize_weights()
        attn_fa.o_proj.quantize_weights()
    else:  # w8a8 / w8a16
        # Linear -> torch.ao.nn.quantized.dynamic.modules.linear.Linear
        torch.ao.quantization.quantize_dynamic(
            attn, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
        torch.ao.quantization.quantize_dynamic(
            attn_fa, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )

        # Merge QKV projections (DynamicQuantizedLinear)
        attn_fa.init_faplus()

        if device == "aie":
            node_args = ()
            node_kwargs = {
                "device": "aie",
                "quant_mode": quant_mode,
                "profiler": False,
            }
            RyzenAILLMEngine.replace_node(
                attn,
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear.QLinear,
                node_args,
                node_kwargs,
            )
            RyzenAILLMEngine.replace_node(
                attn_fa,
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear.QLinear,
                node_args,
                node_kwargs,
            )

    attn.eval()
    attn_fa.eval()
    # print(attn)
    # print(attn_fa)
    # print("==========")

    hidden_states, attention_mask, position_ids, past_key_value_vanilla = (
        generate_attention_test_input(
            batch_size,
            num_attention_heads,
            sequence_length,
            hidden_size,
            kv_cache_type="cache",
            attn_type="llama",
            has_mask=True,
            dtype=dtype,
        )
    )
    past_key_value_flash = copy.deepcopy(past_key_value_vanilla)

    output_vanilla, _, _ = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value_vanilla,
    )

    output_flash, _, _ = attn_fa(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value_flash,
    )

    assert torch.allclose(output_flash, output_vanilla, atol=1e-2)
