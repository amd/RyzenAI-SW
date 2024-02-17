#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import torch.nn.functional as F
from transformers import LlamaConfig
import builtins
from modeling_llama_amd import LlamaAttention
from typing import Optional, Tuple
from utils import get_fa_tile_heuristic, get_aiegemm
import numpy as np 
import json
import math
from collections import defaultdict

FLASH_ATTENTION_SEQUENCE_LENGTH = 512
LLAMA_MASK_TILE_MIN_VAL = -60 # To prevent exp_() from lagging due to big absolute input values. Might vary with different models and precisions (currently -60 works for all test cases).
LLAMA_NEGINF = -1e38


def rotate_half_inplace(x, out):
    """Rotates half the hidden dims of the input."""
    s = x.shape[-1]
    s2 = s // 2
    x1 = x[..., :s2]
    x2 = x[..., s2:]
    torch.cat((x2, x1), dim=-1, out=out)
    out[..., :(s-s2)].neg_()


def get_first_occurences(l):
    last = -1
    d = {}
    for idx, x in enumerate(l):
        if x != last:
            d[x] = idx
            last = x
    return d


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, start=0, end=None, seq_len=None, position_ids=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        if end is None:
            end = seq_len

        if position_ids is None:
            return (
                self.cos_cached[:, :, start:end, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, start:end, ...].to(dtype=x.dtype),
            )
        else:
            return (
                (self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype).squeeze(1).squeeze(0))[position_ids].unsqueeze(1)[:, :, start:end, ...],
                (self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype).squeeze(1).squeeze(0))[position_ids].unsqueeze(1)[:, :, start:end, ...],
            )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaFlashAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        llama_name: Optional[str] = "meta-llama/Llama-2-7b",
        flash_config_path: Optional[str] = None,
        device: str = "cpu",
        max_new_tokens: int = 11,
        impl: str = 'v0',
        quant_mode = None,
        dtype = "float32", # str or torch.dtype
    ):
        super().__init__(config=config)
        print("Using HuggingFace Llama Attention in prefill phase when sequence length is less than {}, and flash attention when greater.".format(FLASH_ATTENTION_SEQUENCE_LENGTH))
        self.scaling = self.head_dim**-0.5
        self.llama_name = llama_name
        self.flash_config = json.load(open(flash_config_path)) if flash_config_path is not None else None
        self.bsz = -1
        self.q_len = -1
        self.lh = -1
        self.lq = -1
        self.lk = -1
        # e.g.
        # nkvh = 32, nkvg = 1: 0...31
        # nkvh = 8, nkvg = 4: 0000,1111...7777
        self.kv_head_indices = torch.repeat_interleave(torch.arange(0, self.num_key_value_heads), self.num_key_value_groups).tolist()
        self._init_rope()

        if dtype == "bfloat16" or dtype == "torch.bfloat16" or dtype == torch.bfloat16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # For model saving using pickle (can't pickle lambda)
        self.forward_funcs = defaultdict(self._forward_prefill_attn_wrapper)
        self.forward_funcs[1] = self._forward_llama_attn

        # Ditto
        self.projection_funcs = defaultdict(self._qkv_projections_merged_wrapper)
        self.projection_funcs[None] = self._qkv_projections_separated
        self.projection_funcs_token_phase = defaultdict(self._qkv_projections_merged_token_phase_wrapper)
        self.projection_funcs_token_phase[None] = self._qkv_projections_separated

        self.max_new_tokens = max_new_tokens
        self.quant_mode = quant_mode
        self.device = device
        self.impl = impl

        # Indice of the current new token (in token phase)
        self.current_new_token_idx = 0

        # Prefill/token markers
        self.is_prefill = False
        self.prefill_tensor_allocated = False
        self.token_tensor_allocated   = False
        self.denom = torch.sqrt(torch.tensor(self.head_dim))

    def _forward_prefill_attn_wrapper(self):
        return self._forward_llama_attn_prefill #_forward_flash_attn
    
    def _qkv_projections_merged_wrapper(self):
        return self._qkv_projections_merged
    
    def _qkv_projections_merged_token_phase_wrapper(self):
        return self._qkv_projections_merged_token_phase

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def initialize_quant_fa_awq(self):
        """
            Merge 3 WQLinear objects into 1
        """

        from qmodule import WQLinear
        self.q_shape = (self.q_proj.in_features, self.q_proj.out_features)
        self.qkv_proj = WQLinear(
            self.q_proj.w_bit,
            self.q_proj.group_size,
            self.q_proj.in_features,
            self.q_proj.out_features*3,
            False,
            self.q_proj.qweight.device)
        self.qkv_proj.qweight = torch.empty(self.q_proj.qweight.shape[0] * 3, self.q_proj.qweight.shape[1], dtype=torch.int8,     device=self.q_proj.qweight.device)
        self.qkv_proj.qzeros  = torch.empty(self.q_proj.qzeros.shape[0] * 3,  self.q_proj.qzeros.shape[1],  dtype=torch.int8,     device=self.q_proj.qzeros.device)
        self.qkv_proj.scales  = torch.empty(self.q_proj.scales.shape[0] * 3,  self.q_proj.scales.shape[1],  dtype=torch.bfloat16, device=self.q_proj.scales.device)

        self.qkv_proj.qweight[:self.q_proj.out_features,                            :].copy_(self.q_proj.qweight)
        self.qkv_proj.qweight[ self.q_proj.out_features:self.q_proj.out_features*2, :].copy_(self.k_proj.qweight)
        self.qkv_proj.qweight[ self.q_proj.out_features*2:,                         :].copy_(self.v_proj.qweight)
        self.qkv_proj.qzeros[ :self.q_proj.out_features,                            :].copy_(self.q_proj.qzeros)
        self.qkv_proj.qzeros[  self.q_proj.out_features:self.q_proj.out_features*2, :].copy_(self.k_proj.qzeros)
        self.qkv_proj.qzeros[  self.q_proj.out_features*2:,                         :].copy_(self.v_proj.qzeros)
        self.qkv_proj.scales[ :self.q_proj.out_features,                            :].copy_(self.q_proj.scales)
        self.qkv_proj.scales[  self.q_proj.out_features:self.q_proj.out_features*2, :].copy_(self.k_proj.scales)
        self.qkv_proj.scales[  self.q_proj.out_features*2:,                         :].copy_(self.v_proj.scales)
        if self.q_proj.bias:
            self.qkv_proj.bias = torch.empty((self.q_proj.bias.shape[0] * 3,), dtype=torch.float16, device=self.q_proj.bias.device)
            self.qkv_proj.bias[:self.q_proj.out_features,                          ].copy_(self.q_proj.bias)
            self.qkv_proj.bias[ self.q_proj.out_features:self.q_proj.out_features*2].copy_(self.k_proj.bias)
            self.qkv_proj.bias[ self.q_proj.out_features*2:,                       ].copy_(self.v_proj.bias)

        self.qkv_states_fp   = np.zeros((1, 1, self.q_shape[1]*3), dtype=np.float32)
        self.qkv_states_torch= torch.from_numpy(self.qkv_states_fp)
        self.query_states_np = np.empty((1, 1, self.q_shape[1]), dtype=np.float32)
        self.key_states_np   = np.empty((1, 1, self.q_shape[1]), dtype=np.float32)
        self.value_states_np = np.empty((1, 1, self.q_shape[1]), dtype=np.float32)

        del self.q_proj, self.k_proj, self.v_proj

    def initialize_quant_fa_qlinear(self):
        """
            Merge 3 DynamicQuantizedLinear objects into 1
        """

        self.q_shape = (self.q_proj.in_features, self.q_proj.out_features)
        self.q_proj_wb = self.q_proj._packed_params._weight_bias()
        self.k_proj_wb = self.k_proj._packed_params._weight_bias()
        self.v_proj_wb = self.v_proj._packed_params._weight_bias()
        self.qkv_proj = torch.empty(self.q_proj.in_features, self.q_proj.out_features*3)
        self.qkv_proj[:, :self.q_proj.out_features]                            = torch.int_repr(self.q_proj_wb[0]).transpose(0, 1)
        self.qkv_proj[:,  self.q_proj.out_features:self.q_proj.out_features*2] = torch.int_repr(self.k_proj_wb[0]).transpose(0, 1)
        self.qkv_proj[:,  self.q_proj.out_features*2:]                         = torch.int_repr(self.v_proj_wb[0]).transpose(0, 1)
        self.qkv_proj = self.qkv_proj.to(self.dtype)

        self.q_proj_scale = self.q_proj_wb[0].q_scale()
        self.k_proj_scale = self.k_proj_wb[0].q_scale()
        self.v_proj_scale = self.v_proj_wb[0].q_scale()
        self.q_proj_bias = None if self.q_proj_wb[1] is None else self.q_proj_wb[1].data
        self.k_proj_bias = None if self.k_proj_wb[1] is None else self.k_proj_wb[1].data
        self.v_proj_bias = None if self.v_proj_wb[1] is None else self.v_proj_wb[1].data

        del self.q_proj, self.k_proj, self.v_proj, self.q_proj_wb, self.k_proj_wb, self.v_proj_wb

        if self.device == "aie":
            self.aiegemm = get_aiegemm(self.impl)
            if self.dtype == torch.bfloat16:
                self.aiegemm.initialize_weights(np.ascontiguousarray(self.qkv_proj.to(torch.float32).numpy().astype(np.int8)))
            else:
                self.aiegemm.initialize_weights(np.ascontiguousarray(self.qkv_proj.numpy().astype(np.int8)))
        else:
            self.qkv_proj_np = self.qkv_proj.to(torch.float32).numpy()
        self.states_fp = np.zeros((1,1,self.q_shape[1]*3), dtype=np.float32)
        self.states_token = np.zeros((1,self.q_shape[1]*3), dtype=np.int32)
        self.x_abs = np.empty((1, self.q_shape[1]), dtype=np.float32)
        self.x_round = np.empty((1,self.q_shape[1]), dtype=np.float32)
        self.x_scaled = np.empty((1,self.q_shape[1]), dtype=np.float32)
        self.x_clip  = np.empty((1,self.q_shape[1]), dtype=np.float32)
        self.x_max = np.array(1.0, dtype=np.float32)
        self.x_scale = np.array(1.0, dtype=np.float32)
        #input("Enter a key")

        self.bsz_token, self.q_len_token = 1, 1
        self.proj_shape_token = (self.bsz_token * self.num_heads, -1, self.head_dim)
        self.q_proj_bias_np = None if self.q_proj_bias is None else self.q_proj_bias.to(torch.float32).numpy()
        self.k_proj_bias_np = None if self.k_proj_bias is None else self.k_proj_bias.to(torch.float32).numpy()
        self.v_proj_bias_np = None if self.v_proj_bias is None else self.v_proj_bias.to(torch.float32).numpy()

        self.query_states_np = np.empty((1, 1, self.q_shape[1]), dtype=np.float32)
        self.key_states_np = np.empty((1, 1, self.q_shape[1]), dtype=np.float32)
        self.value_states_np = np.empty((1, 1, self.q_shape[1]), dtype=np.float32)

    def initialize_quant_fa(self):
        if self.quant_mode == "awq":
            self.initialize_quant_fa_awq()
        elif self.quant_mode is not None:
            self.initialize_quant_fa_qlinear()

    def _reshape_to_bhld_non_contiguous(self, tensor: torch.Tensor, seq_len: int, bsz: int, lh: int = -1):
        lh = self.num_heads if lh == -1 else lh
        return tensor.view(bsz, seq_len, lh, self.head_dim).transpose(1, 2)

    def _init_prefill_tensors(self, bsz, q_len, lh, lq, lk):
        self.bsz = bsz
        self.q_len = q_len
        self._init_tiling_factors(lh, lq, lk)
        self.attn_weights = torch.zeros((bsz, q_len, self.hidden_size), dtype=self.dtype)
        self.qk = torch.zeros((bsz * self.lh, self.lq, self.lk), dtype=self.dtype)
        self.qk_indices = torch.zeros((bsz * self.lh, self.lq), dtype=torch.long)
        self.qkv = torch.zeros((bsz, self.lq, self.lh, self.head_dim), dtype=self.dtype)   # B * lq * lh * d
        self.q =   torch.zeros((bsz, self.lh, self.lq, self.head_dim), dtype=self.dtype)   # B * lh * lq * d
        self.k =   torch.zeros((bsz, self.lh, self.lk, self.head_dim), dtype=self.dtype)   # B * lh * lk * d
        self.v =   torch.zeros((bsz, self.lh, self.lk, self.head_dim), dtype=self.dtype)   # B * lh * lk * d
        self.q2 =  torch.zeros((bsz, self.lh, self.lq, self.head_dim), dtype=self.dtype)   # Rotate_half Q: B * lh * lq * d
        self.k2 =  torch.zeros((bsz, self.lh, self.lk, self.head_dim), dtype=self.dtype)   # Rotate_half K: B * lh * lk * d
        self.bmm2_out = torch.zeros((bsz * self.lh, self.lq, self.head_dim), dtype=self.dtype)
        self.max = [torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype), torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype)]
        self.sum = [torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype), torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype)]
        self.tmp = torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype)
        self.mask = torch.zeros((bsz, 1, self.lq, self.lk), dtype=self.dtype)
        self.tile_min = torch.ones(self.mask.shape, dtype=self.dtype) * LLAMA_MASK_TILE_MIN_VAL
        self.prefill_tensor_allocated = True

    def _del_prefill_tensors(self):
        if not self.prefill_tensor_allocated:
            return
        del self.attn_weights
        del self.qk
        del self.qk_indices
        del self.qkv
        del self.q
        del self.k
        del self.v
        del self.q2
        del self.k2
        del self.bmm2_out
        del self.max
        del self.sum
        del self.tmp
        del self.mask
        del self.tile_min
        self.prefill_tensor_allocated = False

    def _init_token_tensors(self, past_key_value):
        # Pre-allocate past key/value states
        bsz, num_heads, _, embed_dim = past_key_value[0].shape
        self.total_len = self.q_len + self.max_new_tokens
        dtype = torch.bfloat16 # Why? past_key_value[0].dtype
        # self.past_key_states   = torch.zeros((bsz, num_heads, self.total_len, embed_dim), dtype=dtype)
        # self.past_value_states = torch.zeros((bsz, num_heads, self.total_len, embed_dim), dtype=dtype)
        # self.past_key_states[:, :, :self.q_len, :].copy_(past_key_value[0].to(dtype))
        # self.past_value_states[:, :, :self.q_len, :].copy_(past_key_value[1].to(dtype))

        # self.token_attn_weights = torch.zeros((bsz * num_heads, 1, self.total_len), dtype=dtype)
        # self.weights_min = torch.tensor(torch.finfo(self.token_attn_weights.dtype).min, dtype=dtype)
        # self.attn_probs = torch.zeros((bsz * num_heads, 1, self.total_len), dtype=dtype)
        # self.attn_output = torch.zeros((bsz * num_heads, 1, embed_dim), dtype=dtype)

        self.token_tensor_allocated = True

    def _del_token_tensors(self):
        if self.token_tensor_allocated:
            # del self.past_key_states
            # del self.past_value_states

            # del self.token_attn_weights
            # del self.weights_min
            # del self.attn_probs
            # del self.attn_output

            self.token_tensor_allocated = False

    def _init_tiling_factors(self, lh, lq, lk):
        key = "{}_{}_{}_{}".format(self.llama_name, self.bsz, self.q_len, str(self.dtype))
        if (lh == -1 or lq == -1 or lk == -1): # Tile factors not provided
            if self.flash_config is None or key not in self.flash_config.keys():
                self.lh = 8 if self.num_heads % 8 == 0 else 6
                self.lq, self.lq_remainder = get_fa_tile_heuristic(self.q_len)
                self.lk, self.lk_remainder = get_fa_tile_heuristic(self.q_len)
            else:
                self.lh, self.lq, self.lk = self.flash_config[key]
                self.lq_remainder, self.lk_remainder = self.lq, self.lk
        else: # Use external factors
            self.lh, self.lq, self.lk = lh, lq, lk
            self.lq_remainder, self.lk_remainder = self.lq, self.lk
        assert self.num_heads % self.lh == 0, "Head numbers is not divisible by lh!"

    def _qkv_projections_separated(self, hidden_states):
        query_states  = self.q_proj(hidden_states)
        query_states *= self.scaling
        key_states    = self.k_proj(hidden_states)
        value_states  = self.v_proj(hidden_states) # (B, L, D2)
        return query_states, key_states, value_states
    
    def _qkv_projections_merged(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        if self.is_prefill:
            if self.quant_mode == "awq":
                hidden_states = hidden_states[0].view(torch.int16).numpy()
                states = torch.zeros((1, hidden_states.shape[0], self.q_shape[1] * 3)) # TODO: B > 1
                self.qkv_proj.aiegemm.execute(hidden_states, states)
            else:
                states = hidden_states @ self.qkv_proj

            query_states  = states[:, :, :self.q_shape[1]]
            query_states *= self.scaling
            key_states    = states[:, :, self.q_shape[1]:self.q_shape[1]*2]
            value_states  = states[:, :, self.q_shape[1]*2:]

            if self.quant_mode != "awq":
                query_states *= self.q_proj_scale
                if self.q_proj_bias is not None:
                    query_states += self.q_proj_bias
                key_states *= self.k_proj_scale
                if self.k_proj_bias is not None:
                    key_states += self.k_proj_bias
                value_states *= self.v_proj_scale
                if self.v_proj_bias is not None:
                    value_states += self.v_proj_bias

        else: # numpy
            hidden_states = hidden_states[0].to(torch.float32).numpy()
            if self.quant_mode == "awq":
                self.qkv_states_fp.fill(0)
                self.qkv_proj.aiegemm.execute(hidden_states, self.qkv_states_fp[0])

                self.query_states_np = self.qkv_states_fp[:, :, :self.q_shape[1]]
                self.key_states_np = self.qkv_states_fp[:, :, self.q_shape[1]:self.q_shape[1]*2]
                self.value_states_np = self.qkv_states_fp[:, :, self.q_shape[1]*2:]

            else: # "w8a8" etc
                self.states_token.fill(0)
                np.abs(hidden_states, out=self.x_abs)
                np.divide(np.max(self.x_abs), 128.0, out=self.x_scale)
                np.divide(hidden_states, self.x_scale, out=self.x_scaled)
                np.clip(np.round(self.x_scaled, out=self.x_round), -128, 127, out=self.x_clip)

                if self.device == "aie":
                    self.aiegemm.execute(self.x_clip.astype(np.int8, copy=False), self.states_token)
                    np.multiply(self.states_token.astype(np.float32, copy=False), self.x_scale, out=self.states_fp[0])
                else:
                    np.matmul(self.x_clip, self.qkv_proj_np, out=self.states_fp[0])
                    self.states_fp *= self.x_scale

                self.query_states_np = self.states_fp[:, :, :self.q_shape[1]] * self.q_proj_scale
                if self.q_proj_bias_np is not None:
                    self.query_states_np += self.q_proj_bias_np
                self.query_states_np *= self.scaling

                self.key_states_np = self.states_fp[:, :, self.q_shape[1]:self.q_shape[1]*2] * self.k_proj_scale
                if self.k_proj_bias_np is not None:
                    self.key_states_np += self.k_proj_bias_np
                
                self.value_states_np = self.states_fp[:, :, self.q_shape[1]*2:] * self.v_proj_scale
                if self.v_proj_bias_np is not None:
                    self.value_states_np += self.v_proj_bias_np

            query_states = self._reshape_to_bhld_non_contiguous(torch.from_numpy(self.query_states_np).to(torch.bfloat16), q_len, bsz, self.num_heads)
            key_states =   self._reshape_to_bhld_non_contiguous(  torch.from_numpy(self.key_states_np).to(torch.bfloat16), q_len, bsz, self.num_key_value_heads)
            value_states = self._reshape_to_bhld_non_contiguous(torch.from_numpy(self.value_states_np).to(torch.bfloat16), q_len, bsz, self.num_key_value_heads)

        return query_states, key_states, value_states

    def _qkv_projections_merged_token_phase(self, hidden_states):
        self.qkv_proj.aiegemm.execute(hidden_states[0].view(torch.int16).numpy(), self.qkv_states_fp[0])
        self.qkv_states_torch = torch.from_numpy(self.qkv_states_fp).to(torch.bfloat16)

        query_states = self.qkv_states_torch[:, :, :self.q_shape[1]]
        key_states = self.qkv_states_torch[:, :, self.q_shape[1]:self.q_shape[1]*2]
        value_states = self.qkv_states_torch[:, :, self.q_shape[1]*2:]

        return query_states, key_states, value_states

    def _forward_llama_attn_prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        force_vanilla = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        #import pdb; pdb.set_trace()
        aieout = np.zeros((1, q_len, self.q_shape[1]*3), dtype=np.float32)
        #print(f"{hidden_states.shape} {aieout.shape} ")
        self.qkv_proj.aiegemm.execute(hidden_states[0].view(torch.int16).numpy(), aieout[0])
        self.qkv_states_torch = torch.from_numpy(aieout).to(torch.bfloat16)

        query_states = self.qkv_states_torch[:, :, :self.q_shape[1]]
        key_states = self.qkv_states_torch[:, :, self.q_shape[1]:self.q_shape[1]*2]
        value_states = self.qkv_states_torch[:, :, self.q_shape[1]*2:]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        #print(f"Test 3")
        return attn_output, attn_weights, past_key_value

    def _forward_flash_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # B * L
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        force_vanilla: Optional[bool] = False,
        lh: Optional[int] = -1,
        lq: Optional[int] = -1,
        lk: Optional[int] = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # Only run vanilla attention when no quantization
        #if self.quant_mode is None and (force_vanilla or q_len < FLASH_ATTENTION_SEQUENCE_LENGTH): # 512 by default
        return self._forward_llama_attn_prefill(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        # Init tensors at the beginning of prefill
        if not self.is_prefill:
            self._init_prefill_tensors(bsz, q_len, lh, lq, lk)
            self._del_token_tensors()
            self.is_prefill = True
        lh, lq, lk = self.lh, self.lq, self.lk
        lqr, lkr = self.lq_remainder, self.lk_remainder
        if self.config.pretraining_tp > 1:
            assert self.device != "aie", "Not supported for now." # Disable for AIE
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            query_states *= self.scaling

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            # QKV projections
            query_states, key_states, value_states = self.projection_funcs[self.quant_mode](hidden_states)
        # Q: (B, L, H * d), K/V: (B, L, kvh * d)

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len, position_ids=position_ids)
        lq_steps = (q_len + lq - 1) // lq
        lk_steps = (q_len + lk - 1) // lk

        for h in range(self.num_heads // lh):
            hh = (h*lh*self.head_dim)
            h_ind = self.kv_head_indices[(h*lh):(h*lh+lh)]
            kh_ind_occ = get_first_occurences(h_ind) # GQA: K/V head indices and their first occurences in this tile

            for i in range(lq_steps):
                self.max[0].fill_(LLAMA_NEGINF)
                self.max[1].fill_(LLAMA_NEGINF)
                self.sum[0].zero_()
                self.sum[1].zero_()
                self.qkv.zero_()
                ii = (i*lq)
                ii_end = min(ii+lq, q_len)
                lq_tile = (lqr if i == lq_steps-1 else lq)

                # Q and its rotate_half
                self.q[:, :, :(ii_end-ii), :].copy_(self._reshape_to_bhld_non_contiguous(query_states[:, ii:ii_end, hh:(hh+lh*self.head_dim)].detach(), -1, bsz, lh))

                # The next 4 lines are equivalent to apply_rotary_pos_emb(q, k, cos, sin, position_ids) FOR Q (partial length)
                rotate_half_inplace(self.q, self.q2)
                self.q2[:, :, :(ii_end-ii), :] *= sin[:, :, ii:ii_end, :]
                self.q[:, :, :(ii_end-ii), :] *= cos[:, :, ii:ii_end, :]
                self.q += self.q2 # Only up to (ii_end-ii) is valid for the last tile

                # Update Q for past_key_value
                if use_cache:
                    query_states[:, ii:ii_end, hh:(hh+lh*self.head_dim)] = self.q.view(bsz, -1, lh * self.head_dim)[:, :(ii_end-ii), :]

                for j in range(lk_steps):
                    current = j % 2
                    last = 1 - (j % 2)
                    jj = (j*lk)
                    jj_end = min(jj+lk, q_len)
                    lk_tile = (lkr if j == lk_steps-1 else lk)

                    # (bsz, lq/lk, lh * d) -> (bsz, lh, lq/lk, d)
                    # Slicing using slice objects can reduce memory consumption. TODO: Try it.
                    self.v[:, :, :(jj_end-jj), :].copy_(self._reshape_to_bhld_non_contiguous(value_states.view(bsz, q_len, -1, self.head_dim)[:, jj:jj_end, h_ind, :].detach(), -1, bsz, lh))

                    # K and its rotate_half
                    self.k[:, :, :(jj_end-jj), :].copy_(self._reshape_to_bhld_non_contiguous(  key_states.view(bsz, q_len, -1, self.head_dim)[:, jj:jj_end, h_ind, :].detach(), -1, bsz, lh))

                    # The next 4 lines are equivalent to apply_rotary_pos_emb(q, k, cos, sin, position_ids) FOR K (partial length)
                    rotate_half_inplace(self.k, self.k2)
                    self.k2[:, :, :(jj_end-jj), :] *= sin[:, :, jj:jj_end, :]
                    self.k[:, :, :(jj_end-jj), :] *= cos[:, :, jj:jj_end, :]
                    self.k += self.k2

                    # Update K for past_key_value (only update once when finished) [VISIT THIS LINE IF GQA GOES WRONG LATER!]
                    if use_cache and i == lq_steps - 1:
                        for ind, occ in kh_ind_occ.items():
                            key_states[:, jj:jj_end, ind*self.head_dim:((ind+1)*self.head_dim)] = self.k[:, occ, :, :].view(bsz, -1, self.head_dim)[:, :(jj_end-jj), :]

                    # bmm1 and mask
                    torch.bmm(
                        self.q.view(bsz * lh, -1, self.head_dim)[:, :lq_tile, :],
                        self.k.view(bsz * lh, -1, self.head_dim)[:, :lk_tile, :].transpose(-1, -2),
                        out=self.qk[:, :lq_tile, :lk_tile]) # (bsz * lh, lq, lk)
                    if attention_mask is not None:
                        self.mask[:, :, :(ii_end-ii), :(jj_end-jj)].copy_(attention_mask[:, :, ii:ii_end, jj:jj_end].detach())
                        self.qk = self.qk.view(bsz, -1, lq, lk)
                        self.qk.add_(self.mask)
                        torch.max(self.qk, self.tile_min, out=self.qk)
                        self.qk = self.qk.view(-1, lq, lk)

                    # m^j
                    torch.max(self.qk[:, :, :(jj_end-jj)], -1, out=(self.tmp, self.qk_indices))
                    torch.max(self.tmp, self.max[last], out=self.max[current]) # m^j = max(m^(j-1), rowmax(S^j))

                    # P^j
                    self.qk.sub_(self.max[current].view(bsz * lh, -1, 1))
                    self.qk.exp_() # P^j = e^(S^j - m^j)

                    # l^j
                    torch.sum(self.qk[:, :, :(jj_end-jj)], dim=-1, out=self.sum[current]) # rowsum(P^j)
                    torch.bmm(
                        self.qk[:, :lq_tile, :lk_tile],
                        self.v.view(bsz * lh, -1, self.head_dim)[:, :lk_tile, :],
                        out=self.bmm2_out[:, :lq_tile, :]) # P^j * V_j, (bsz * lh, lq, d)
                    torch.sub(self.max[last], self.max[current], out=self.tmp)
                    self.tmp.exp_() # self.tmp = e^(m^(j-1) - m^j)
                    self.sum[last] *= self.tmp
                    self.sum[current] += self.sum[last] # l^j = e^(m^(j-1) - m^j) * l^(j-1) + rowsum(P^j)

                    # O^j
                    self.qkv *= self.tmp.view(bsz, lh, -1, 1).transpose(1, 2)
                    self.qkv += self.bmm2_out.view(bsz, lh, lq, self.head_dim).transpose(1, 2) # O^j = e^(m^(j-1) - m^j) * O^(j-1) + P^j * V_j

                self.qkv.div_(self.sum[current].view(bsz, lh, -1, 1).transpose(1, 2))
                self.attn_weights[:, ii:ii_end, hh:(hh+lh*self.head_dim)] = self.qkv.view(bsz, lq, lh * self.head_dim)[:, :(ii_end-ii), :]

        if self.config.pretraining_tp > 1:
            attn_output = self.attn_weights.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(self.attn_weights)

        if use_cache:
            key_states   = self._reshape_to_bhld_non_contiguous(key_states,   -1, bsz, self.num_key_value_heads) # (B, H, L, d)
            value_states = self._reshape_to_bhld_non_contiguous(value_states, -1, bsz, self.num_key_value_heads) # (B, H, L, d)
            if past_key_value is not None:
                key_states =   torch.cat([past_key_value[0], key_states  ], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        return attn_output, None, past_key_value

    def _forward_llama_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # B * L
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        force_vanilla: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len = 1, 1 

        self.is_prefill = False
        query_states, key_states, value_states = self.projection_funcs_token_phase[self.quant_mode](hidden_states)
        query_states = query_states.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        key_states = torch.cat([past_key_value[0].to(torch.bfloat16), key_states], dim=2)
        value_states = torch.cat([past_key_value[1].to(torch.bfloat16), value_states], dim=2)
        past_key_value = (key_states, value_states) 

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / self.denom
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # B * L
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        force_vanilla: Optional[bool] = False,
    ):
        attn_output, attn_weights_reshaped, past_key_value = self.forward_funcs[hidden_states.shape[1]]( 
                                                                        hidden_states,
                                                                        attention_mask,
                                                                        position_ids,
                                                                        past_key_value,
                                                                        output_attentions,
                                                                        use_cache,
                                                                        padding_mask,
                                                                        force_vanilla)
        return attn_output, attn_weights_reshaped, past_key_value
