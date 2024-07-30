#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)

import llama_fast_attention
import ryzenai_torch_cpp


def golden(kv_seq_len, query_states, key_states, position_ids):
    cos, sin = llama_fast_attention.LlamaRotaryEmbeddingLocal.forward(
        seq_len=kv_seq_len
    )
    query_states, key_states = llama_fast_attention.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    return query_states, key_states


@pytest.mark.parametrize("m", [128, 256, 512, 1024, 2048])
def test_ryzenai_torch_cpp_rope(m):

    query_states = torch.rand((1, 32, m, 128)).to(torch.bfloat16)
    key_states = torch.rand((1, 32, m, 128)).to(torch.bfloat16)
    kv_seq_len = m
    position_ids = torch.zeros((1, m))
    for i in range(m):
        position_ids[0, i] = i

    position_ids = position_ids.to(torch.int64)
    query_states_ref, key_states_ref = golden(
        kv_seq_len, query_states, key_states, position_ids
    )

    # placeholder
    query_states_npu, key_states_npu = golden(
        kv_seq_len, query_states, key_states, position_ids
    )

    res_q = torch.allclose(query_states_ref, query_states_npu)
    res_k = torch.allclose(key_states_ref, key_states_npu)

    # placeholder
    op = ryzenai_torch_cpp.aie_rope_torch()
    y = op.execute(query_states_ref)

    result = (res_q == True) and (res_k == True)
    result = result and torch.allclose(query_states_ref, y)
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")
    assert result == True
