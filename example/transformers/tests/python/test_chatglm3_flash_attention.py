#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy
import os
import sys
import time

import pytest
import torch
from test_utils import generate_attention_test_input, pergrp_processing, to_skip

sys.path.append(os.getenv("PYTORCH_AIE_PATH") + "/models/llm/")
from chatglm3.configuration_chatglm import ChatGLMConfig
from chatglm3.modeling_chatglm3_amd import SelfAttention
from chatglm3_flash_attention import ChatGLM3FlashAttentionPlus

torch.manual_seed(0)


chatglm3_attention_shapes = [
    ("THUDM/chatglm3-6b", 1, 1),
    ("THUDM/chatglm3-6b", 1, 128),
    ("THUDM/chatglm3-6b", 1, 512),
    ("THUDM/chatglm3-6b", 1, 717),
    ("THUDM/chatglm3-6b", 1, 2000),
    ("THUDM/chatglm3-6b", 1, 2545),
    ("THUDM/chatglm3-6b", 1, 4000),
    # ("THUDM/chatglm3-6b", 4, 1   ),
    # ("THUDM/chatglm3-6b", 4, 128 ),
    # ("THUDM/chatglm3-6b", 4, 512 ),
    # ("THUDM/chatglm3-6b", 4, 717 ),
    # ("THUDM/chatglm3-6b", 4, 2000),
    # ("THUDM/chatglm3-6b", 4, 2545),
    # ("THUDM/chatglm3-6b", 4, 4000),
]


@pytest.mark.parametrize("chatglm3_attention_shape", chatglm3_attention_shapes)
@pytest.mark.parametrize("device", ["cpu", "aie"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.quant_combo_skip
def test_chatglm3_flash_attention(
    chatglm3_attention_shape, quant_mode, w_bit, device, dtype
):
    to_skip(device, dtype, quant_mode)
    model_name, batch_size, sequence_length = chatglm3_attention_shape

    config = ChatGLMConfig()
    # config = ChatGLMConfig.from_pretrained(model_name)
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads

    attn = SelfAttention(config=config, layer_number=27)
    attn_fa = ChatGLM3FlashAttentionPlus(
        config=config,
        layer_number=27,
        dtype=dtype,
    )
    if quant_mode == "w4abf16":
        # Linear -> QLinearPerGrp
        # Convert before replication, as only separate projections can be copied
        pergrp_processing(attn, w_bit)

    # Copy params
    attn_fa.qkv_hidden_size = copy.deepcopy(attn.qkv_hidden_size)
    attn_fa.query_key_value = copy.deepcopy(attn.query_key_value)
    attn_fa.dense = copy.deepcopy(attn.dense)

    if quant_mode == "none":
        attn = attn.to(dtype)
        attn_fa = attn_fa.to(dtype)
    if quant_mode == "w4abf16":
        # Post processing for QLinearPerGrp
        attn.query_key_value.device = device
        attn.dense.device = device
        attn.query_key_value.quantize_weights()
        attn.dense.quantize_weights()

        # Post processing for QLinearPerGrp
        attn_fa.query_key_value.device = device
        attn_fa.dense.device = device
        attn_fa.query_key_value.quantize_weights()
        attn_fa.dense.quantize_weights()

    attn.eval()
    attn_fa.eval()
    # print(attn)
    # print(attn_fa)

    att_time = 0
    att_fa_time = 0

    hidden_states, attention_mask, _, kv_cache = generate_attention_test_input(
        batch_size,
        num_attention_heads,
        sequence_length,
        hidden_size,
        gqa=config.multi_query_attention,
        kv_heads=(
            config.multi_query_group_num
            if config.multi_query_attention
            else num_attention_heads
        ),
        attn_type="opt",
        has_mask=False,
        dtype=dtype,
    )
    hidden_states = hidden_states.transpose(0, 1)
    rotary_pos_emb = torch.rand(sequence_length, batch_size, num_attention_heads, 2).to(
        dtype
    )
    kv_cache_flash = copy.deepcopy(kv_cache)

    attn_fa_start_time = time.perf_counter()
    output_flash = attn_fa(
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=kv_cache_flash,
        use_cache=True,
    )
    attn_fa_end_time = time.perf_counter()
    att_fa_time += attn_fa_end_time - attn_fa_start_time

    if kv_cache is not None:
        # Permute KV cache for L-first format
        kv_cache = (
            kv_cache[0].permute(2, 0, 1, 3).contiguous(),
            kv_cache[1].permute(2, 0, 1, 3).contiguous(),
        )
    attn_start_time = time.perf_counter()
    output_vanilla = attn(
        hidden_states, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=True
    )
    attn_end_time = time.perf_counter()
    att_time += attn_end_time - attn_start_time

    print(f"Time of attn:")
    print(att_time)
    print(f"Time of attn_fa:")
    print(att_fa_time)

    assert torch.allclose(output_flash[0], output_vanilla[0], atol=1e-2)
