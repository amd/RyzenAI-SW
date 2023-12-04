#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import pytest, json
from transformers import LlamaConfig
from llama_flash_attention import LlamaFlashAttention
from utils import generate_attention_test_input

llama_attention_shapes = [
    ("llama-2-wts-hf/7B", 32, 4096,  1, 512 ),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 1024),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 1536),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 2000),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 3000),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 4000),
    ("llama-2-wts-hf/7B", 32, 4096,  4, 512 ),
    ("llama-2-wts-hf/7B", 32, 4096,  4, 1024),
    ("llama-2-wts-hf/7B", 32, 4096,  4, 1536),
    ("llama-2-wts-hf/7B", 32, 4096,  4, 2000),
    ("llama-2-wts-hf/7B", 32, 4096,  4, 3000),
    ("llama-2-wts-hf/7B", 32, 4096,  4, 4000),
]

@pytest.mark.parametrize("llama_attention_shape", llama_attention_shapes)
def test_llama_flash_attention(llama_attention_shape):
    model_name, _, embedding_dim, batch_size, sequence_length = llama_attention_shape
    config = LlamaConfig()
    attn = LlamaFlashAttention(
        config,
        llama_name=model_name,
        flash_config_path="../../ops/python/llama_flash_attention_config.json",
    )
    attn.eval()
    # print(attn)

    hidden_states, attention_mask, position_ids = generate_attention_test_input(
        batch_size, sequence_length, embedding_dim,
        attn_type="llama", has_mask=True, dtype=torch.float32)

    output_vanilla, _, _ = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        force_vanilla=True,
    )

    output_flash, _, _ = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        force_vanilla=False,
    )

    assert torch.allclose(output_vanilla, output_flash, atol=1e-4)
