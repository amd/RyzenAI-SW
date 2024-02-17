#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import pytest
import builtins
import os
import sys
from transformers import LlamaConfig
from utils import Utils, generate_attention_test_input, replicate_llama_attention_params
from quantizer import pseudo_quantize_tensor
from qmodule import WQLinear
import qlinear

sys.path.append(os.getenv("PYTORCH_AIE_PATH") + "/models/llama2/")


llama_attention_shapes = [
    ("llama-2-wts-hf/7B", 32, 4096,  1, 1   ),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 512 ),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 1024),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 1536),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 2000),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 3000),
    ("llama-2-wts-hf/7B", 32, 4096,  1, 4000),
    # ("llama-2-wts-hf/7B", 32, 4096,  4, 1   ), # B > 1 to be fixed, same below
    # ("llama-2-wts-hf/7B", 32, 4096,  4, 512 ),
    # ("llama-2-wts-hf/7B", 32, 4096,  4, 1024),
    # ("llama-2-wts-hf/7B", 32, 4096,  4, 1536),
    # ("llama-2-wts-hf/7B", 32, 4096,  4, 2000),
    # ("llama-2-wts-hf/7B", 32, 4096,  4, 3000),
    # ("llama-2-wts-hf/7B", 32, 4096,  4, 4000),
]


params = [
    # ("cpu", torch.float32,  None,   "v0"),
    # ("cpu", torch.float32,  "w8a8", "v0"),
    # ("cpu", torch.float32,  "awq",  "v0"), # buggy, QlinearPerGrp doesn't have forward_dict for cpu and aie_emu
    ("cpu", torch.bfloat16, None,   "v0"),
    # ("cpu", torch.bfloat16, "awq",  "v0"), # buggy, QlinearPerGrp doesn't have forward_dict for cpu and aie_emu
    ("aie", torch.bfloat16, "awq",  "v0"),
    # ("aie", torch.float32,  "w8a8", "v0"), # good if runs alone; throws either "IOCTL_KIPUDRV_HW_CTX failed: Element not found." or "Windows fatal exception: access violation" when runs with another aie test.

    # # ("cpu", torch.float32,  None,   "v1"),
    # # ("cpu", torch.float32,  "w8a8", "v1"),
    # # ("cpu", torch.float32,  "awq",  "v1"), # buggy, QlinearPerGrp doesn't have forward_dict for cpu and aie_emu
    ("cpu", torch.bfloat16, None,   "v1"),
    # # ("cpu", torch.bfloat16, "awq",  "v1"), # buggy, QlinearPerGrp doesn't have forward_dict for cpu and aie_emu
    ("aie", torch.bfloat16, "awq",  "v1"),
    # # ("aie", torch.float32,  "w8a8", "v1"), # good if runs alone; throws either "IOCTL_KIPUDRV_HW_CTX failed: Element not found." or "Windows fatal exception: access violation" when runs with another aie test.
]


def awq_processing(model):
    q_config = {
        "zero_point": True,
        "q_group_size": 128,
    }

    def f(module):
        module.weight.data, scales, zeros = pseudo_quantize_tensor(module.weight.data, n_bit=3, get_scale_zp=True, **q_config)
        return WQLinear.from_linear(module, 3, q_config['q_group_size'], False, scales, zeros)

    model.q_proj = f(model.q_proj).to("cpu")
    model.k_proj = f(model.k_proj).to("cpu")
    model.v_proj = f(model.v_proj).to("cpu")
    model.o_proj = f(model.o_proj).to("cpu")
    return


@pytest.mark.parametrize("llama_attention_shape", llama_attention_shapes)
@pytest.mark.parametrize("amdllama", [False, True])
@pytest.mark.parametrize("param", params)
def test_llama_flash_attention(llama_attention_shape, amdllama, param):
    builtins.amdllama = amdllama
    from llama_flash_attention import LlamaFlashAttention
    if hasattr(builtins, "amdllama") and builtins.amdllama:
        from modeling_llama_amd import LlamaAttention
    else:
        from transformers.models.llama.modeling_llama import LlamaAttention

    model_name, _, embedding_dim, batch_size, sequence_length = llama_attention_shape
    device, dtype, quant_mode, impl = param

    config = LlamaConfig() # Use default param
    attn = LlamaAttention(config=config)
    attn_fa = LlamaFlashAttention(
        config=config,
        llama_name=model_name,
        flash_config_path="../../ops/python/llama_flash_attention_config.json",
        device=device,
        impl=impl,
        quant_mode=quant_mode,
        dtype=dtype
    )

    if quant_mode == "awq":
        # Linear -> WQLinear
        # Convert before replication, as only separate projections can be copied
        awq_processing(attn)

    # Copy params, including separate QKV projections (Linear or WQLinear)
    replicate_llama_attention_params(attn, attn_fa)

    if quant_mode is None:
        attn = attn.to(dtype)
        attn_fa = attn_fa.to(dtype)
    elif quant_mode == "awq":
        Utils.replace_node( attn,
                            WQLinear,
                            qlinear.QLinearPerGrp,
                            (), {'device': 'cpu', 'w_bit': 3, 'group_size':128})
        # Post processing for QLinearPerGrp
        attn.q_proj.device = device
        attn.q_proj.quantize_weights()
        attn.k_proj.device = device
        attn.k_proj.quantize_weights()
        attn.v_proj.device = device
        attn.v_proj.quantize_weights()
        attn.o_proj.device = device
        attn.o_proj.quantize_weights()

        # Merge QKV projections (WQLinear)
        # Vanilla never merge
        attn_fa.initialize_quant_fa()

        Utils.replace_node( attn_fa,
                            WQLinear,
                            qlinear.QLinearPerGrp,
                            (), {'device': 'cpu', 'w_bit': 3, 'group_size':128})
        # Post processing for QLinearPerGrp
        attn_fa.qkv_proj.device = device
        attn_fa.qkv_proj.quantize_weights()
        attn_fa.o_proj.device = device
        attn_fa.o_proj.quantize_weights()
    else:
        # Linear -> torch.ao.nn.quantized.dynamic.modules.linear.Linear
        torch.ao.quantization.quantize_dynamic(attn, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        torch.ao.quantization.quantize_dynamic(attn_fa, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)

        # Merge QKV projections (DynamicQuantizedLinear)
        # Vanilla never merge
        attn_fa.initialize_quant_fa()

        if device == "aie":
            node_args = ()
            node_kwargs = {
                'device': 'aie',
                'quant_mode': 'w8a8',
                'profiler': False,
                'kernel_x_shape': (8, 4096),
                'kernel_y_shape': (4096, 4096),
                'impl': impl
            }
            Utils.replace_node( attn, 
                                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                                qlinear.QLinear,
                                node_args, node_kwargs)
            Utils.replace_node( attn_fa, 
                                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                                qlinear.QLinear,
                                node_args, node_kwargs)
    attn.eval()
    attn_fa.eval()
    # print(attn)
    # print(attn_fa)

    hidden_states, attention_mask, position_ids, past_key_value = generate_attention_test_input(
        batch_size, 32, sequence_length, embedding_dim,
        attn_type="llama", has_mask=True, dtype=dtype)

    output_vanilla, _, _ = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=True,
    )

    output_flash, _, _ = attn_fa(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=True,
    )

    assert torch.allclose(output_flash, output_vanilla, atol=1e-2)
