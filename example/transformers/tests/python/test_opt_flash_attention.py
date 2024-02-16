#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import pytest
from opt_flash_attention import OPTFlashAttention
from utils import generate_attention_test_input

opt_attention_shapes = [
    ("facebook/opt-125m", 12, 768,  1, 512 ),
    ("facebook/opt-125m", 12, 768,  1, 1024),
    ("facebook/opt-125m", 12, 768,  1, 1536),
    ("facebook/opt-125m", 12, 768,  1, 2048),
    ("facebook/opt-125m", 12, 768,  4, 512 ),
    ("facebook/opt-125m", 12, 768,  4, 1024),
    ("facebook/opt-125m", 12, 768,  4, 1536),
    ("facebook/opt-125m", 12, 768,  4, 2048),
    ("facebook/opt-350m", 16, 1024, 1, 512 ),
    ("facebook/opt-350m", 16, 1024, 1, 1024),
    ("facebook/opt-350m", 16, 1024, 1, 1536),
    ("facebook/opt-350m", 16, 1024, 1, 2048),
    ("facebook/opt-350m", 16, 1024, 4, 512 ),
    ("facebook/opt-350m", 16, 1024, 4, 1024),
    ("facebook/opt-350m", 16, 1024, 4, 1536),
    ("facebook/opt-350m", 16, 1024, 4, 2048),
    ("facebook/opt-1.3b", 32, 2048, 1, 512 ),
    ("facebook/opt-1.3b", 32, 2048, 1, 1024),
    ("facebook/opt-1.3b", 32, 2048, 1, 1536),
    ("facebook/opt-1.3b", 32, 2048, 1, 2048),
    ("facebook/opt-1.3b", 32, 2048, 4, 512 ),
    ("facebook/opt-1.3b", 32, 2048, 4, 1024),
    ("facebook/opt-1.3b", 32, 2048, 4, 1536),
    ("facebook/opt-1.3b", 32, 2048, 4, 2048),
    ("facebook/opt-2.7b", 32, 2560, 1, 512 ),
    ("facebook/opt-2.7b", 32, 2560, 1, 1024),
    ("facebook/opt-2.7b", 32, 2560, 1, 1536),
    ("facebook/opt-2.7b", 32, 2560, 1, 2048),
    ("facebook/opt-2.7b", 32, 2560, 4, 512 ),
    ("facebook/opt-2.7b", 32, 2560, 4, 1024),
    ("facebook/opt-2.7b", 32, 2560, 4, 1536),
    ("facebook/opt-2.7b", 32, 2560, 4, 2048),
    ("facebook/opt-6.7b", 32, 4096, 1, 512 ),
    ("facebook/opt-6.7b", 32, 4096, 1, 1024),
    ("facebook/opt-6.7b", 32, 4096, 1, 1536),
    ("facebook/opt-6.7b", 32, 4096, 1, 2048),
    ("facebook/opt-6.7b", 32, 4096, 4, 512 ),
    ("facebook/opt-6.7b", 32, 4096, 4, 1024),
    ("facebook/opt-6.7b", 32, 4096, 4, 1536),
    ("facebook/opt-6.7b", 32, 4096, 4, 2048),
    ("facebook/opt-13b",  40, 5120, 1, 512 ),
    ("facebook/opt-13b",  40, 5120, 1, 1024),
    ("facebook/opt-13b",  40, 5120, 1, 1536),
    ("facebook/opt-13b",  40, 5120, 1, 2048),
    ("facebook/opt-13b",  40, 5120, 4, 512 ),
    ("facebook/opt-13b",  40, 5120, 4, 1024),
    ("facebook/opt-13b",  40, 5120, 4, 1536),
    ("facebook/opt-13b",  40, 5120, 4, 2048),
]

@pytest.mark.parametrize("opt_attention_shape", opt_attention_shapes)
def test_opt_flash_attention(opt_attention_shape):
    model_name, num_attention_heads, embedding_dim, batch_size, sequence_length = opt_attention_shape
    attn = OPTFlashAttention(
        embed_dim=embedding_dim,
        num_heads=num_attention_heads,
        dropout=0.0,
        is_decoder=True,
        bias=True,
        opt_name=model_name,
        flash_config_path="../../ops/python/opt_flash_attention_config.json",
    )
    attn.eval()
    # print(attn)

    hidden_states, attention_mask, _ = generate_attention_test_input(
        batch_size, sequence_length, embedding_dim,
        attn_type="opt", has_mask=True, dtype=torch.float32)
    
    output_vanilla, _, _ = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        force_vanilla=True,
    )

    output_flash, _, _ = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        force_vanilla=False,
    )

    assert torch.allclose(output_vanilla, output_flash, atol=1e-4)
