#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy

import pytest
import qlinear
import torch
from qmodule import WQLinear
from qwen2_flash_attention import Qwen2FlashAttentionPlus
from ryzenai_llm_engine import RyzenAILLMEngine
from test_utils import awq_processing, generate_attention_test_input, to_skip

from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

torch.manual_seed(0)


qwen2_attention_shapes = [
    ("Qwen/Qwen1.5-7B-Chat", 1, 1),
    ("Qwen/Qwen1.5-7B-Chat", 1, 128),
    ("Qwen/Qwen1.5-7B-Chat", 1, 512),
    ("Qwen/Qwen1.5-7B-Chat", 1, 717),
    ("Qwen/Qwen1.5-7B-Chat", 1, 2000),
    ("Qwen/Qwen1.5-7B-Chat", 1, 2545),
    ("Qwen/Qwen1.5-7B-Chat", 1, 4000),
]


@pytest.mark.parametrize("qwen2_attention_shape", qwen2_attention_shapes)
@pytest.mark.parametrize("device", ["aie"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])  # Only AIE and bf16 for now
def test_qwen2_flash_attention(qwen2_attention_shape, quant_mode, w_bit, device, dtype):
    to_skip(device, dtype, quant_mode)
    model_name, batch_size, sequence_length = qwen2_attention_shape

    config = Qwen2Config()  # Use default param
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    attn = Qwen2Attention(
        config=config,
        layer_idx=0,
    )
    attn_fa = Qwen2FlashAttentionPlus(
        config=config,
        layer_idx=0,
        model_name=model_name,
    )

    # Linear -> WQLinear
    # Convert before replication, as only separate projections can be copied
    awq_processing(attn, w_bit)

    # Copy separate QKV projections (Linear or WQLinear)
    attn_fa.k_proj = copy.deepcopy(attn.k_proj)
    attn_fa.v_proj = copy.deepcopy(attn.v_proj)
    attn_fa.q_proj = copy.deepcopy(attn.q_proj)
    attn_fa.o_proj = copy.deepcopy(attn.o_proj)

    RyzenAILLMEngine.replace_node(
        attn,
        WQLinear,
        qlinear.QLinearPerGrp,
        (),
        {"device": device, "w_bit": w_bit, "group_size": 128},
    )
    # Post processing for QLinearPerGrp
    attn.q_proj.quantize_weights()
    attn.k_proj.quantize_weights()
    attn.v_proj.quantize_weights()
    attn.o_proj.quantize_weights()

    # Merge QKV projections (WQLinear)
    attn_fa.init_faplus()

    RyzenAILLMEngine.replace_node(
        attn_fa,
        WQLinear,
        qlinear.QLinearPerGrp,
        (),
        {"device": device, "w_bit": w_bit, "group_size": 128},
    )
    # Post processing for QLinearPerGrp
    attn_fa.qkv_proj.quantize_weights()
    attn_fa.o_proj.quantize_weights()

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
