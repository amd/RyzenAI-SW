#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy

import pytest
import torch
from mistral_flash_attention import MistralFlashAttentionPlus
from test_utils import generate_attention_test_input, pergrp_processing, to_skip

from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralAttention

torch.manual_seed(0)


mistral_attention_shapes = [
    ("mistralai/Mistral-7B-v0.1", 1, 1),
    ("mistralai/Mistral-7B-v0.1", 1, 128),
    ("mistralai/Mistral-7B-v0.1", 1, 512),
    ("mistralai/Mistral-7B-v0.1", 1, 717),
    ("mistralai/Mistral-7B-v0.1", 1, 2000),
    ("mistralai/Mistral-7B-v0.1", 1, 2545),
    ("mistralai/Mistral-7B-v0.1", 1, 4000),
    # ("mistralai/Mistral-7B-v0.1", 4, 1   ),
    # ("mistralai/Mistral-7B-v0.1", 4, 128 ),
    # ("mistralai/Mistral-7B-v0.1", 4, 512 ),
    # ("mistralai/Mistral-7B-v0.1", 4, 717 ),
    # ("mistralai/Mistral-7B-v0.1", 4, 2000),
    # ("mistralai/Mistral-7B-v0.1", 4, 2545),
    # ("mistralai/Mistral-7B-v0.1", 4, 4000),
]


@pytest.mark.parametrize("mistral_attention_shape", mistral_attention_shapes)
@pytest.mark.parametrize("device", ["aie"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.quant_combo_skip
def test_mistral_flash_attention(
    mistral_attention_shape, device, dtype, quant_mode, w_bit
):
    to_skip(device, dtype, quant_mode)
    model_name, batch_size, sequence_length = mistral_attention_shape

    config = MistralConfig.from_pretrained(model_name)
    attn = MistralAttention(
        config=config,
        layer_idx=0,
    )
    attn_fa = MistralFlashAttentionPlus(
        config=config,
        layer_idx=0,
        model_name=model_name,
    )
    embedding_dim = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads

    # Linear -> QLinearPerGrp
    # Convert before replication, as only separate projections can be copied
    pergrp_processing(attn, w_bit)

    # Copy separate QKV projections (QLinearPerGrp)
    attn_fa.k_proj = copy.deepcopy(attn.k_proj)
    attn_fa.v_proj = copy.deepcopy(attn.v_proj)
    attn_fa.q_proj = copy.deepcopy(attn.q_proj)
    attn_fa.o_proj = copy.deepcopy(attn.o_proj)

    # Post processing for QLinearPerGrp
    attn.q_proj.device = device
    attn.k_proj.device = device
    attn.v_proj.device = device
    attn.o_proj.device = device
    attn.q_proj.quantize_weights()
    attn.k_proj.quantize_weights()
    attn.v_proj.quantize_weights()
    attn.o_proj.quantize_weights()

    # Merge QKV projections (QLinearPerGrp)
    attn_fa.init_faplus()

    # Post processing for QLinearPerGrp
    attn_fa.qkv_proj.device = device
    attn_fa.o_proj.device = device
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
            embedding_dim,
            kv_cache_type="cache",
            gqa=(num_attention_heads != num_key_value_heads),
            kv_heads=num_key_value_heads,
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
