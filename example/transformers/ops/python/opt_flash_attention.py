#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import json
import logging
from typing import Optional, Tuple

import torch
from ryzenai_attention import *

from transformers.models.opt.configuration_opt import OPTConfig


class OPTFlashAttentionPlus(RyzenAIAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        precision: str = "w4abf16",
        model_name: str = "facebook/opt-1.3b",
        flash_config_path: str = "",
        **kwargs,
    ):
        super().__init__()
        self.config = config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
            val = None
            if fn_arg_name in kwargs:
                logging.warning(
                    "Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38."
                    " Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument(
            "hidden_size", config, "embed_dim", kwargs
        )
        self.num_heads = _handle_deprecated_argument(
            "num_attention_heads", config, "num_heads", kwargs
        )
        self.dropout = _handle_deprecated_argument(
            "attention_dropout", config, "dropout", kwargs
        )
        self.enable_bias = _handle_deprecated_argument(
            "enable_bias", config, "bias", kwargs
        )

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True
        self.hidden_size = self.embed_dim

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.precision = precision

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
        self.kv_head_indices = torch.arange(0, self.num_heads).tolist()
        self.tensor_allocated = False

    def init_faplus(self):
        if self.precision == "w4abf16":
            self.merge_qkv_w4abf16(self.q_proj, self.k_proj, self.v_proj)
            del self.q_proj, self.k_proj, self.v_proj
        elif self.precision.startswith("w8"):
            self.merge_qkv_w8(self.q_proj, self.k_proj, self.v_proj, self.precision)
            del self.q_proj, self.k_proj, self.v_proj

    def mha_default(self, query_states, key_states, value_states, **kwargs):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.baddbmm(
            (
                kwargs["attention_mask"].view(bsz, q_len, -1)
                if kwargs["attention_mask"] is not None
                else torch.zeros((1,), dtype=query_states.dtype)
            ),
            query_states.view(bsz * num_heads, -1, head_dim),
            key_states.view(bsz * num_heads, -1, head_dim).transpose(-1, -2),
            alpha=self.scaling,
        )

        if kwargs["attention_mask"] is not None:
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(
                    torch.finfo(attn_weights.dtype).min, device=attn_weights.device
                ),
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        if kwargs["layer_head_mask"] is not None:
            attn_weights *= kwargs["layer_head_mask"].view(1, -1, 1, 1)
        attn_output = torch.matmul(
            attn_weights.view(bsz * num_heads, q_len, -1), value_states
        )

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()
        use_fa = tgt_len >= FLASH_ATTENTION_SEQUENCE_LENGTH

        # QKV projections
        query_states, key_states, value_states = self.matmul_qkv(hidden_states)

        # 3D -> 3D
        if use_fa:
            attn_output = self.flash_mha(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
            )

            key_states = RyzenAIAttention.reshape_BLD_to_BHLd(
                key_states, bsz, -1, tgt_len, self.head_dim
            )
            value_states = RyzenAIAttention.reshape_BLD_to_BHLd(
                value_states, bsz, -1, tgt_len, self.head_dim
            )
            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            if self.is_decoder:
                past_key_value = (key_states, value_states)
        else:
            query_states = RyzenAIAttention.reshape_BLD_to_BHLd(
                query_states, bsz, -1, tgt_len, self.head_dim
            )
            key_states = RyzenAIAttention.reshape_BLD_to_BHLd(
                key_states, bsz, -1, tgt_len, self.head_dim
            )
            value_states = RyzenAIAttention.reshape_BLD_to_BHLd(
                value_states, bsz, -1, tgt_len, self.head_dim
            )

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            if self.is_decoder:
                past_key_value = (key_states, value_states)

            attn_output = self.mha_default(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                tgt_len=tgt_len,
            )
            attn_output = RyzenAIAttention.reshape_BHLd_to_BLD(
                attn_output, bsz, tgt_len, self.embed_dim
            )

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value
