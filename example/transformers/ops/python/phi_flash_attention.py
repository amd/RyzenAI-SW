#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import json
from typing import Optional, Tuple

import torch
from ryzenai_attention import *

from transformers import PhiConfig
from transformers.cache_utils import Cache
from transformers.models.phi.modeling_phi import (
    PhiDynamicNTKScalingRotaryEmbedding,
    PhiLinearScalingRotaryEmbedding,
    PhiRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


class PhiFlashAttentionPlus(RyzenAIAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(
        self,
        config: PhiConfig,
        layer_idx: Optional[int] = None,
        precision: str = "w4abf16",
        model_name: str = "microsoft/phi-2",
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
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
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
        self.dense = None
        self._init_rope()

        if self.precision == "w4abf16":
            self.matmul_qkv = self.matmul_qkv_w4abf16
            self.dtype = torch.bfloat16
        else:
            print(f"Not supported with w8a8 ... exitting ...!!\n\n")
            raise SystemExit

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = torch.nn.LayerNorm(
                config.hidden_size // self.num_heads,
                eps=config.layer_norm_eps,
                elementwise_affine=True,
            )
            self.k_layernorm = torch.nn.LayerNorm(
                config.hidden_size // self.num_heads,
                eps=config.layer_norm_eps,
                elementwise_affine=True,
            )

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
        # fp32 for Q and K and all related tensors
        self.qk = torch.ones(
            (self.bsz * self.lh, self.lq, self.lk), dtype=torch.float32
        )
        self.q = torch.zeros(
            (self.bsz * self.lh, self.lq, self.head_dim), dtype=torch.float32
        )  # B * lh * lq * d
        self.k = torch.zeros(
            (self.bsz * self.lh, self.lk, self.head_dim), dtype=torch.float32
        )  # B * lh * lk * d
        self.max = torch.zeros((2, self.bsz * self.lh, self.lq), dtype=torch.float32)
        self.sum = torch.zeros((2, self.bsz * self.lh, self.lq), dtype=torch.float32)
        self.tmp = torch.zeros((self.bsz * self.lh, self.lq), dtype=torch.float32)
        self.mask = torch.zeros((self.bsz, self.lq, self.lk), dtype=torch.float32)
        self.tile_min = (
            torch.ones(self.mask.shape, dtype=torch.float32) * MASK_TILE_MIN_VAL
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = PhiRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = PhiLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = PhiDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        use_fa = q_len >= FLASH_ATTENTION_SEQUENCE_LENGTH
        query_states, key_states, value_states = self.matmul_qkv(hidden_states)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        query_states = RyzenAIAttention.reshape_BLD_to_BHLd(
            query_states, bsz, -1, q_len, self.head_dim
        )
        key_states = RyzenAIAttention.reshape_BLD_to_BHLd(
            key_states, bsz, -1, q_len, self.head_dim
        )
        value_states = RyzenAIAttention.reshape_BLD_to_BHLd(
            value_states, bsz, -1, q_len, self.head_dim
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, position_ids
        )

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_emb.dim,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 3D -> 3D
        if use_fa:
            # TODO: 3D -> 3D if RoPE can be merged.
            attn_output = self.flash_mha(
                RyzenAIAttention.reshape_BHLd_to_BLD(
                    query_states.to(torch.float32), bsz, q_len, -1
                ),
                RyzenAIAttention.reshape_BHLd_to_BLD(
                    key_states.to(torch.float32), bsz, q_len, -1
                ),
                RyzenAIAttention.reshape_BHLd_to_BLD(value_states, bsz, q_len, -1),
                attention_mask=attention_mask,
            )
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = self.mha_default(
                query_states.to(torch.float32),
                key_states.to(torch.float32),
                value_states,
                attention_mask=attention_mask,
            )
            attn_output = RyzenAIAttention.reshape_BHLd_to_BLD(
                attn_output, bsz, -1, self.hidden_size
            )

        # Output projection
        attn_output = self.dense(attn_output)

        return attn_output, attn_output, past_key_value
