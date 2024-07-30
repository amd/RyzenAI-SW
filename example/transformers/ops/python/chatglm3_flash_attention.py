#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import json
import os
import sys
from typing import Optional

import torch
from ryzenai_attention import *

sys.path.append(os.getenv("PYTORCH_AIE_PATH") + "/models/llm/chatglm3/")
from configuration_chatglm import ChatGLMConfig
from modeling_chatglm3_amd import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ChatGLM3FlashAttentionPlus(RyzenAIAttention):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: ChatGLMConfig,
        layer_number,
        model_name: Optional[str] = "THUDM/chatglm3-6b",
        flash_config_path: Optional[str] = "",
        dtype="bfloat16",
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.hidden_size = config.kv_channels * config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.head_dim = config.kv_channels
        self.multi_query_attention = config.multi_query_attention
        self.num_key_value_heads = (
            config.multi_query_group_num
            if config.multi_query_attention
            else self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        if dtype == "bfloat16" or dtype == "torch.bfloat16" or dtype == torch.bfloat16:
            self.dtype = torch.bfloat16
        else:
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

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        hidden_states = hidden_states.transpose(0, 1)
        bsz, q_len, _ = hidden_states.size()
        use_fa = q_len >= FLASH_ATTENTION_SEQUENCE_LENGTH

        mixed_x_states = self.query_key_value(hidden_states).to(self.dtype)
        (query_states, key_states, value_states) = mixed_x_states.split(
            [
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=-1,
        )

        query_states = RyzenAIAttention.reshape_BLD_to_BHLd(
            query_states, bsz, -1, q_len, self.head_dim
        )
        key_states = RyzenAIAttention.reshape_BLD_to_BHLd(
            key_states, bsz, -1, q_len, self.head_dim
        )
        value_states = RyzenAIAttention.reshape_BLD_to_BHLd(
            value_states, bsz, -1, q_len, self.head_dim
        )

        # RoPE
        if rotary_pos_emb is not None:
            query_states = apply_rotary_pos_emb(
                query_states, rotary_pos_emb.transpose(0, 1)
            )
            key_states = apply_rotary_pos_emb(
                key_states, rotary_pos_emb.transpose(0, 1)
            )

        if (
            attention_mask is None and True
        ):  # q_len == k_len holds as otherwise QKV projections shouldn't merge
            attention_mask = torch.ones(bsz, 1, q_len, q_len, dtype=self.dtype) * NEGINF
            attention_mask.triu_(diagonal=1)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_states = torch.cat((cache_k, key_states), dim=-2)
            value_states = torch.cat((cache_v, value_states), dim=-2)
        if use_cache:
            kv_cache = (key_states, value_states)
        else:
            kv_cache = None

        # 4D -> 3D
        if use_fa:
            # TODO: 3D -> 3D if RoPE can be merged.
            attn_output = self.flash_mha(
                RyzenAIAttention.reshape_BHLd_to_BLD(query_states, bsz, q_len, -1),
                RyzenAIAttention.reshape_BHLd_to_BLD(key_states, bsz, q_len, -1),
                RyzenAIAttention.reshape_BHLd_to_BLD(value_states, bsz, q_len, -1),
                attention_mask=attention_mask,
            )
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = self.mha_default(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
            )
            attn_output = RyzenAIAttention.reshape_BHLd_to_BLD(
                attn_output, bsz, -1, self.hidden_size
            )

        attn_output = self.dense(attn_output).to(self.dtype).transpose(0, 1)

        return attn_output, kv_cache
