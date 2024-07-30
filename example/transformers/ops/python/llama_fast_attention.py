#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import gc
import json
import math
import sys
import time
from typing import Optional, Tuple

import ryzenai_torch_cpp
import torch
from ryzenai_attention import *
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbeddingLocal:
    dim = 128
    max_position_embeddings = 4096  # self.max_position_embeddings
    base = 10000
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).to(torch.bfloat16) / dim))
    t = torch.arange(max_position_embeddings, device="cpu", dtype=torch.bfloat16)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    emb_cos = emb.cos()
    emb_sin = emb.sin()

    @classmethod
    def forward(cls, seq_len=None):
        return (
            cls.emb_cos[:seq_len],
            cls.emb_sin[:seq_len],
        )


mha_npu = ryzenai_torch_cpp.aie_mha_npu_torch()


class LlamaFastAttention(RyzenAIAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        precision: str = "w4abf16",
        model_name: str = "llama-2-7b-chat",
        flash_config_path: str = "",
        profile: bool = False,
        mhaops: str = "all",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.precision = precision
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = (
            4096  # config.max_position_embeddings (2048) is not enough
        )
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = torch.tensor(self.head_dim**-0.5)
        self.profile = profile

        def create_timer():
            start_time = None

            def TIC():
                nonlocal start_time
                start_time = time.perf_counter()

            def TOC(name):
                elapsed_time = time.perf_counter() - start_time
                print(f"{name}: {elapsed_time:.6f}")

            return TIC, TOC

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        def nullfun():
            pass

        def nullfun2(name):
            pass

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        if self.precision == "w4abf16":
            self.matmul_qkv = self.matmul_qkv_w4abf16
            self.dtype = torch.bfloat16
        elif self.precision.startswith("w8"):
            self.matmul_qkv = self.matmul_qkv_w8
            self.dtype = torch.float32

        # For FA
        self.model_name = model_name
        self.torchcpu = ryzenai_torch_cpp.cpu_mha()
        self.scaling_bmm = 1 / math.sqrt(self.head_dim)
        """
        self.softmax_mladf = ryzenai_torch_cpp.aie_softmax_torch()
        self.bmm1_mladf = ryzenai_torch_cpp.aie_bmm_torch(True)
        self.bmm2_mladf = ryzenai_torch_cpp.aie_bmm_torch(False)
        """

        if self.profile:
            self.TIC, self.TOC = create_timer()

        else:
            self.TIC = nullfun
            self.TOC = nullfun2
        self.mhaops = mhaops

    def init_faplus(self):
        if self.precision == "w4abf16":
            self.merge_qkv_w4abf16(self.q_proj, self.k_proj, self.v_proj)
            del self.q_proj, self.k_proj, self.v_proj
        elif self.precision.startswith("w8"):
            self.merge_qkv_w8(self.q_proj, self.k_proj, self.v_proj, self.precision)
            del self.q_proj, self.k_proj, self.v_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,  # Dummy for now.
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        self.TIC()
        bsz, q_len, _ = hidden_states.size()

        # QKV projections
        query_states, key_states, value_states = self.matmul_qkv(hidden_states)
        self.TOC(" llama fast attention QKV")
        self.TIC()
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        self.TOC(" llama fast attention reshape")
        self.TIC()
        cos, sin = LlamaRotaryEmbeddingLocal.forward(seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        self.TOC(" llama fast attention Rope")
        self.TIC()
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )
        self.TOC(" llama fast attention KV cache update")
        self.TIC()
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        self.TOC(" llama fast attention repeat_kv")
        # self.TIC()
        if q_len == 2048:
            """
            if self.mhaops == "pytorchmha":  # CPU vanilla
                attn_weights = (
                    torch.matmul(query_states, key_states.transpose(2, 3))
                    * self.scaling_bmm
                )
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=3, dtype=torch.float32
                ).to(torch.bfloat16)
                attn_output = torch.matmul(attn_weights, value_states)
            """
            if self.mhaops == "all":  # bmm softmax bmm on NPU
                """
                attn_weights = (
                    self.bmm1_mladf.execute(
                        query_states[0].contiguous(), key_states[0].contiguous()
                    )
                    * self.scaling_bmm
                )
                attn_weights = self.softmax_mladf.execute(
                    attn_weights, attention_mask[0]
                )
                attn_output = self.bmm2_mladf.execute(
                    attn_weights.contiguous(), value_states[0].contiguous()
                )
                attn_output = attn_output.unsqueeze(0)
                """
                self.TIC()
                attn_output = mha_npu.execute(
                    query_states[0].contiguous(),
                    key_states[0].contiguous(),
                    value_states[0].contiguous(),
                    attention_mask[0].contiguous(),
                )
                self.TOC(" llama fast attention MHa")

                attn_output = attn_output.unsqueeze(0)

                """
                elif self.mhaops == "softmax":  # only softmax on NPU
                    attn_weights = (
                        torch.matmul(query_states, key_states.transpose(2, 3))
                        * self.scaling_bmm
                    )
                    attn_weights = self.softmax_mladf.execute(
                        attn_weights[0], attention_mask[0]
                    )
                    attn_weights = attn_weights.unsqueeze(0)
                    attn_output = torch.matmul(attn_weights, value_states)
                elif self.mhaops == "bmm1":  # only bmm1 on NPU
                    attn_weights = (
                        self.bmm1_mladf.execute(
                            query_states[0].contiguous(), key_states[0].contiguous()
                        )
                        * self.scaling_bmm
                    )
                    attn_weights = attn_weights.unsqueeze(0)
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                    attn_weights = torch.nn.functional.softmax(
                        attn_weights, dim=3, dtype=torch.float32
                    ).to(torch.bfloat16)
                    attn_output = torch.matmul(attn_weights, value_states)
                elif self.mhaops == "bmm2":  # bmm2 on NPU
                    attn_weights = (
                        torch.matmul(query_states, key_states.transpose(2, 3))
                        * self.scaling_bmm
                    )
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                    attn_weights = torch.nn.functional.softmax(
                        attn_weights, dim=3, dtype=torch.float32
                    ).to(torch.bfloat16)
                    attn_output = self.bmm2_mladf.execute(
                        attn_weights[0].contiguous(), value_states[0].contiguous()
                    )
                    attn_output = attn_output.unsqueeze(0)
                """
            elif self.mhaops == "libtorchflat":  # CPU fused flat attn
                attn_output = self.torchcpu.mha_top(
                    query_states, key_states, value_states, attention_mask
                )
            else:
                print(
                    f"\n\n [LlamaFastAttention] Encountered unexpected setting for ops"
                )
                raise SystemExit
        else:
            attn_output = self.torchcpu.mha_top(
                query_states, key_states, value_states, attention_mask
            )
        # self.TOC(" llama fast attention mha")
        self.TIC()

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        self.TOC(" llama fast attention transpose reshape")
        self.TIC()
        attn_output = self.o_proj(attn_output)
        self.TOC(" llama fast attention output proj")
        return attn_output, None, past_key_value
