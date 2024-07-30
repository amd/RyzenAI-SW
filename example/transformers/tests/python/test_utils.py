import copy

import pytest
import qlinear
import torch
from qmodule import WQLinear
from quantizer import pseudo_quantize_tensor


def to_skip(device, dtype, quant_mode):
    if quant_mode == "w4abf16" and dtype == torch.float32:
        pytest.skip(
            reason="w4abf16 only supports bf16 inputs and workloads on AIE/CPU. (bf16 @ int4 = fp32 -> bf16)"
        )
    elif quant_mode.startswith("w8a") and dtype == torch.bfloat16:
        pytest.skip(
            reason="QLinear only support fp32 inputs and workloads on AIE/CPU. (.numpy() doesn't accept bf16)"
        )
    elif device == "aie" and quant_mode == "none":
        pytest.skip(reason="Only quantized workloads on AIE.")

    assert True


def generate_attention_test_input(
    b,
    H,
    L,
    D,
    kv_cache_length=128,
    kv_cache_type="tuple",
    gqa=False,
    kv_heads=-1,
    attn_type="opt",
    has_mask=True,
    dtype=torch.float32,
):
    hidden_states = torch.rand(b, L, D)
    hidden_states = hidden_states.to(dtype)

    past_key_value = None
    attention_mask = None
    position_ids = None

    # KV cache for token phase
    Lx = L
    kv_heads = kv_heads if gqa else H
    if L == 1:
        key_states = torch.rand((b, kv_heads, kv_cache_length, D // H), dtype=dtype)
        value_states = torch.rand((b, kv_heads, kv_cache_length, D // H), dtype=dtype)
        if kv_cache_type == "tuple":
            past_key_value = (key_states, value_states)
        else:  # Cache
            from transformers.cache_utils import DynamicCache

            past_key_value = DynamicCache()
            past_key_value.update(key_states, value_states, 0)
        Lx += kv_cache_length

    if has_mask:
        attention_mask = torch.rand(b, 1, L, Lx) * 1.0e-2
        attention_mask = attention_mask.to(dtype)
    if attn_type == "llama":
        position_ids = torch.randint(0, L, (b, L)).to(torch.long)

    return hidden_states, attention_mask, position_ids, past_key_value


def awq_processing(model, w_bit):
    q_config = {
        "zero_point": True,
        "q_group_size": 128,
    }

    def f(module):
        module.weight.data, scales, zeros = pseudo_quantize_tensor(
            module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
        )
        return WQLinear.from_linear(
            module, w_bit, q_config["q_group_size"], False, scales, zeros
        )

    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                setattr(model, n, f(m).to("cpu"))


def pergrp_processing(model, w_bit):
    def f(module):
        m = qlinear.QLinearPerGrp(
            module.in_features,
            module.out_features,
            module.bias is not None,
            "cpu",
            w_bit,
            128,  # group_size
        )
        m.weight = copy.deepcopy(module.weight)
        if module.bias is not None:
            m.bias = copy.deepcopy(module.bias)
        return m

    # Prevent any possible bias in Linear that introduces grad_fn and crashes copy.deepcopy()
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                setattr(model, n, f(m).to("cpu"))
