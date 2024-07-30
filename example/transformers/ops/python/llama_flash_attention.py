#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import json
from typing import Optional, Tuple

import torch
from ryzenai_attention import *

from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


class LlamaFlashAttentionPlus(RyzenAIAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        precision: str = "w4abf16",
        model_name: str = "llama-2-7b-chat",
        flash_config_path: str = "",
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
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self._init_rope()

        if self.precision == "w4abf16":
            self.matmul_qkv = self.matmul_qkv_w4abf16
            self.dtype = torch.bfloat16
        elif self.precision.startswith("w8"):
            self.matmul_qkv = self.matmul_qkv_w8
            self.dtype = torch.float32

        # For FA
        self.model_name = model_name
        self.flash_config = (
            json.load(open(flash_config_path)) if flash_config_path != "" else None
        )
        self.seq_len = -1
        self.kv_head_indices = torch.repeat_interleave(
            torch.arange(0, self.num_key_value_heads), self.num_key_value_groups
        ).tolist()
        self.tensor_allocated = False

    def _init_tensors(self, bsz, seq_len, lh=-1, lq=-1, lk=-1):
        # Only init when not allocated
        if self.tensor_allocated:
            return

        super()._init_tensors(bsz, seq_len, lh, lq, lk)
        self.q2 = torch.zeros(
            (bsz * self.lh, self.lq, self.head_dim), dtype=self.dtype
        )  # Rotate_half Q: B * lh * lq * d
        self.k2 = torch.zeros(
            (bsz * self.lh, self.lk, self.head_dim), dtype=self.dtype
        )  # Rotate_half K: B * lh * lk * d

    def _del_tensors(self):
        if not self.tensor_allocated:
            return

        super()._del_tensors()
        del self.q2
        del self.k2

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

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
        bsz, q_len, _ = hidden_states.size()
        use_fa = q_len >= FLASH_ATTENTION_SEQUENCE_LENGTH

        # QKV projections
        query_states, key_states, value_states = self.matmul_qkv(hidden_states)

        # RoPE
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # 3D -> 3D
        if use_fa:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            attn_output = self.flash_mha_rope(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
            )

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
                _ = past_key_value.update(
                    RyzenAIAttention.reshape_BLD_to_BHLd(
                        key_states, bsz, -1, q_len, self.head_dim
                    ),
                    RyzenAIAttention.reshape_BLD_to_BHLd(
                        value_states, bsz, -1, q_len, self.head_dim
                    ),
                    self.layer_idx,
                    cache_kwargs,
                )  # No need to return
        else:
            query_states = RyzenAIAttention.reshape_BLD_to_BHLd(
                query_states, bsz, -1, q_len, self.head_dim
            )
            key_states = RyzenAIAttention.reshape_BLD_to_BHLd(
                key_states, bsz, -1, q_len, self.head_dim
            )
            value_states = RyzenAIAttention.reshape_BLD_to_BHLd(
                value_states, bsz, -1, q_len, self.head_dim
            )

            if q_len == 1:
                query_states, key_states = self.apply_rotary_pos_emb_token_fast(
                    query_states, key_states, position_ids
                )
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )

            if past_key_value is not None:
                cache_kwargs = None  # OK for now, {"sin": sin, "cos": cos} for Streaming LLM (SinkCache)
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = self.mha_default(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
            )
            attn_output = RyzenAIAttention.reshape_BHLd_to_BLD(
                attn_output, bsz, q_len, self.hidden_size
            )

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
