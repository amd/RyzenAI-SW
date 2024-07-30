#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy

import pytest
import qlinear
import torch
from opt_flash_attention import OPTFlashAttentionPlus
from qmodule import WQLinear
from ryzenai_llm_engine import RyzenAILLMEngine
from test_utils import awq_processing, generate_attention_test_input, to_skip

from transformers import OPTConfig
from transformers.models.opt.modeling_opt import OPTAttention

torch.manual_seed(0)


opt_attention_shapes = [
    ("facebook/opt-125m", 1, 1),
    ("facebook/opt-125m", 1, 128),
    ("facebook/opt-125m", 1, 512),
    ("facebook/opt-125m", 1, 717),
    ("facebook/opt-125m", 1, 1801),
    ("facebook/opt-125m", 1, 2048),
    ("facebook/opt-1.3b", 1, 1),
    ("facebook/opt-1.3b", 1, 128),
    ("facebook/opt-1.3b", 1, 512),
    ("facebook/opt-1.3b", 1, 717),
    ("facebook/opt-1.3b", 1, 1801),
    ("facebook/opt-1.3b", 1, 2048),
    ("facebook/opt-13b", 1, 1),
    ("facebook/opt-13b", 1, 128),
    ("facebook/opt-13b", 1, 512),
    ("facebook/opt-13b", 1, 717),
    ("facebook/opt-13b", 1, 1801),
    ("facebook/opt-13b", 1, 2048),
    # ("facebook/opt-125m", 4, 1   ),
    # ("facebook/opt-125m", 4, 128 ),
    # ("facebook/opt-125m", 4, 512 ),
    # ("facebook/opt-125m", 4, 717 ),
    # ("facebook/opt-125m", 4, 1024),
    # ("facebook/opt-125m", 4, 1536),
    # ("facebook/opt-125m", 4, 1801),
    # ("facebook/opt-125m", 4, 2048),
    # ("facebook/opt-1.3b", 4, 1   ),
    # ("facebook/opt-1.3b", 4, 128 ),
    # ("facebook/opt-1.3b", 4, 512 ),
    # ("facebook/opt-1.3b", 4, 717 ),
    # ("facebook/opt-1.3b", 4, 1024),
    # ("facebook/opt-1.3b", 4, 1536),
    # ("facebook/opt-1.3b", 4, 1801),
    # ("facebook/opt-1.3b", 4, 2048),
    # ("facebook/opt-13b",  4, 1   ),
    # ("facebook/opt-13b",  4, 128 ),
    # ("facebook/opt-13b",  4, 512 ),
    # ("facebook/opt-13b",  4, 717 ),
    # ("facebook/opt-13b",  4, 1024),
    # ("facebook/opt-13b",  4, 1536),
    # ("facebook/opt-13b",  4, 1801),
    # ("facebook/opt-13b",  4, 2048),
]


@pytest.mark.parametrize("opt_attention_shape", opt_attention_shapes)
@pytest.mark.parametrize("device", ["aie"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.quant_combo_skip
def test_opt_flash_attention(opt_attention_shape, quant_mode, w_bit, device, dtype):
    to_skip(device, dtype, quant_mode)
    model_name, batch_size, sequence_length = opt_attention_shape

    config = OPTConfig.from_pretrained(model_name)
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    attn = OPTAttention(
        config=config,
        is_decoder=True,
    )
    attn_fa = OPTFlashAttentionPlus(
        config=config,
        is_decoder=True,
        precision=quant_mode,
        model_name=model_name,
    )
    if quant_mode == "w4abf16":
        # Linear -> WQLinear
        # Convert before replication, as only separate projections can be copied
        awq_processing(attn, w_bit)

    # Copy params, including separate QKV projections (Linear or WQLinear)
    attn_fa.embed_dim = attn.embed_dim
    attn_fa.num_heads = attn.num_heads
    attn_fa.dropout = attn.dropout
    attn_fa.head_dim = attn.embed_dim // attn.num_heads
    attn_fa.scaling = attn.head_dim**-0.5
    attn_fa.is_decoder = attn.is_decoder
    attn_fa.k_proj = copy.deepcopy(attn.k_proj)
    attn_fa.v_proj = copy.deepcopy(attn.v_proj)
    attn_fa.q_proj = copy.deepcopy(attn.q_proj)
    attn_fa.out_proj = copy.deepcopy(attn.out_proj)

    if quant_mode == "none":
        attn = attn.to(dtype)
        attn_fa = attn_fa.to(dtype)
    elif quant_mode == "w4abf16":
        # import pdb; pdb.set_trace()
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
        attn.out_proj.quantize_weights()

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
        attn_fa.out_proj.quantize_weights()
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

    hidden_states, attention_mask, _, past_key_value = generate_attention_test_input(
        batch_size,
        num_attention_heads,
        sequence_length,
        hidden_size,
        attn_type="opt",
        has_mask=True,
        dtype=dtype,
    )

    output_vanilla, _, _ = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
    )

    output_flash, _, _ = attn_fa(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
    )

    assert torch.allclose(output_flash, output_vanilla, atol=1e-2)
