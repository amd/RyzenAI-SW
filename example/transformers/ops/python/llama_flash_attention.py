#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import torch.nn.functional as F
from transformers import LlamaConfig
import builtins
if hasattr(builtins, "amdllama") and builtins.amdllama:
    from modeling_llama_amd import LlamaAttention
else:
    from transformers.models.llama.modeling_llama import LlamaAttention
from typing import Optional, Tuple
import json

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
    ):
        super().__init__(config=config)
        print("Using HuggingFace Llama Attention when sequence length is less than {}, and flash attention when greater.".format(FLASH_ATTENTION_SEQUENCE_LENGTH))
        self.scaling = self.head_dim**-0.5
        self.llama_name = llama_name
        self.flash_config = json.load(open(flash_config_path)) if flash_config_path is not None else None
        self.bsz = -1
        self.q_len = -1
        self.lh = -1
        self.lq = -1
        self.lk = -1
        self.dtype = torch.float32
        # e.g.
        # nkvh = 32, nkvg = 1: 0...31
        # nkvh = 8, nkvg = 4: 0000,1111...7777
        self.kv_head_indices = torch.repeat_interleave(torch.arange(0, self.num_key_value_heads), self.num_key_value_groups).tolist()
        self._init_rope()

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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def _shape_no_contiguous(self, tensor: torch.Tensor, seq_len: int, bsz: int, lh: int):
        return tensor.view(bsz, seq_len, lh, self.head_dim).transpose(1, 2)

    def _init_tensors(self, bsz, q_len, lh, lq, lk, dtype):
        self.bsz = bsz
        self.q_len = q_len
        self.dtype = dtype
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

    def _del_tensors(self):
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

    def _init_tiling_factors(self, lh, lq, lk):
        key = "{}_{}_{}_{}".format(self.llama_name, self.bsz, self.q_len, str(self.dtype))
        if (lh == -1 or lq == -1 or lk == -1): # Tile factors not provided
            if self.flash_config is None or key not in self.flash_config.keys():
                self.lh = 8 if self.num_heads % 8 == 0 else 6
                self.lq = self.q_len // 2
                self.lk = self.q_len // 2
            else:
                self.lh, self.lq, self.lk = self.flash_config[key]
        else: # Use external factors
            self.lh, self.lq, self.lk = lh, lq, lk
        assert self.q_len % self.lq == 0 and self.q_len % self.lk == 0, "Seq length is not divisible by lq and lk!"
        assert self.num_heads % self.lh == 0, "Head numbers is not divisible by lh!"

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
        lh: Optional[int] = -1,
        lq: Optional[int] = -1,
        lk: Optional[int] = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if force_vanilla or q_len < FLASH_ATTENTION_SEQUENCE_LENGTH: # 512 by default
            return LlamaAttention.forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        dtype = hidden_states.dtype
        self._init_tensors(bsz, q_len, lh, lq, lk, dtype)
        lh, lq, lk = self.lh, self.lq, self.lk
        if self.config.pretraining_tp > 1:
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
            query_states = self.q_proj(hidden_states)
            query_states *= self.scaling
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        # Q: (B, L, H * d), K/V: (B, L, kvh * d)

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len, position_ids=position_ids)

        for h in range(self.num_heads // lh):
            hh = (h*lh*self.head_dim)
            h_ind = self.kv_head_indices[(h*lh):(h*lh+lh)]
            kh_ind_occ = get_first_occurences(h_ind) # GQA: K/V head indices and their first occurences in this tile
            
            for i in range(q_len // lq):
                self.max[0].fill_(LLAMA_NEGINF)
                self.max[1].fill_(LLAMA_NEGINF)
                self.sum[0].zero_()
                self.sum[1].zero_()
                self.qkv.zero_()
                ii = (i*lq)

                # Q and its rotate_half
                self.q.copy_(self._shape_no_contiguous(query_states[:, ii:(ii+lq), hh:(hh+lh*self.head_dim)].detach(), -1, bsz, lh))
                # The next 4 lines are equivalent to apply_rotary_pos_emb(q, k, cos, sin, position_ids) FOR Q (partial length)
                rotate_half_inplace(self.q, self.q2)
                self.q2 *= sin[:, :, ii:(ii+lq), :]
                self.q *= cos[:, :, ii:(ii+lq), :]
                self.q += self.q2
                # Update Q for past_key_value
                if use_cache:
                    query_states[:, ii:(ii+lq), hh:(hh+lh*self.head_dim)] = self.q.view(bsz, -1, lh * self.head_dim)

                for j in range(q_len // lk):
                    current = j % 2
                    last = 1 - (j % 2)
                    jj = (j*lk)
                    # (bsz, lq/lk, lh * d) -> (bsz, lh, lq/lk, d)
                    # Slicing using slice objects can reduce memory consumption. TODO: Try it.
                    self.v.copy_(self._shape_no_contiguous(value_states.view(bsz, q_len, -1, self.head_dim)[:, jj:(jj+lk), h_ind, :].detach(), -1, bsz, lh))

                    # K and its rotate_half
                    self.k.copy_(self._shape_no_contiguous(  key_states.view(bsz, q_len, -1, self.head_dim)[:, jj:(jj+lk), h_ind, :].detach(), -1, bsz, lh))
                    # The next 4 lines are equivalent to apply_rotary_pos_emb(q, k, cos, sin, position_ids) FOR K (partial length)
                    rotate_half_inplace(self.k, self.k2)
                    self.k2 *= sin[:, :, jj:(jj+lk), :]
                    self.k *= cos[:, :, jj:(jj+lk), :]
                    self.k += self.k2
                    # Update K for past_key_value (only update once when finished) [VISIT THIS LINE IF GQA GOES WRONG LATER!]
                    if use_cache and i == (q_len // lq) - 1:
                        for ind, occ in kh_ind_occ.items():
                            key_states[:, jj:(jj+lk), ind*self.head_dim:((ind+1)*self.head_dim)] = self.k[:, occ, :, :].view(bsz, -1, self.head_dim)

                    # bmm1 and mask
                    torch.bmm(self.q.view(bsz * lh, -1, self.head_dim), self.k.view(bsz * lh, -1, self.head_dim).transpose(-1, -2), out=self.qk)
                    if attention_mask is not None:
                        self.mask.copy_(attention_mask[:, :, ii:(ii+lq), jj:(jj+lk)].detach())
                        self.qk = self.qk.view(bsz, -1, lq, lk)
                        self.qk.add_(self.mask)
                        torch.max(self.qk, self.tile_min, out=self.qk)
                        self.qk = self.qk.view(-1, lq, lk)

                    # m^j
                    torch.max(self.qk, -1, out=(self.tmp, self.qk_indices))
                    torch.max(self.tmp, self.max[last], out=self.max[current]) # m^j = max(m^(j-1), rowmax(S^j))

                    # P^j
                    self.qk.sub_(self.max[current].view(bsz * lh, -1, 1))
                    self.qk.exp_() # P^j = e^(S^j - m^j)

                    # l^j
                    torch.sum(self.qk, dim=-1, out=self.sum[current]) # rowsum(P^j)
                    torch.bmm(self.qk, self.v.view(bsz * lh, -1, self.head_dim), out=self.bmm2_out) # P^j * V_j
                    torch.sub(self.max[last], self.max[current], out=self.tmp)
                    self.tmp.exp_() # self.tmp = e^(m^(j-1) - m^j)
                    self.sum[last] *= self.tmp
                    self.sum[current] += self.sum[last] # l^j = e^(m^(j-1) - m^j) * l^(j-1) + rowsum(P^j)

                    # O^j
                    self.qkv *= self.tmp.view(bsz, lh, -1, 1).transpose(1, 2)
                    self.qkv += self.bmm2_out.view(bsz, lh, lq, self.head_dim).transpose(1, 2) # O^j = e^(m^(j-1) - m^j) * O^(j-1) + P^j * V_j

                self.qkv.div_(self.sum[current].view(bsz, lh, -1, 1).transpose(1, 2))
                self.attn_weights[:, ii:(ii+lq), hh:(hh+lh*self.head_dim)] = self.qkv.view(bsz, lq, lh * self.head_dim)

        if self.config.pretraining_tp > 1:
            attn_output = self.attn_weights.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(self.attn_weights)

        if use_cache:
            key_states   = self._shape_no_contiguous(key_states,   -1, bsz, self.num_key_value_heads) # (B, H, L, d)
            value_states = self._shape_no_contiguous(value_states, -1, bsz, self.num_key_value_heads) # (B, H, L, d)
            if past_key_value is not None:
                key_states =   torch.cat([past_key_value[0], key_states  ], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None
        self._del_tensors()

        return attn_output, None, past_key_value
