#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import builtins
if hasattr(builtins, "amdopt") and builtins.amdopt:
    from modeling_opt_amd import OPTAttention
else:
    from transformers.models.opt.modeling_opt import OPTAttention
from typing import Optional, Tuple
import json

FLASH_ATTENTION_SEQUENCE_LENGTH = 512
OPT_MASK_TILE_MIN_VAL = -60 # To prevent exp_() from lagging due to big absolute input values. Might vary with different models and precisions (currently -60 works for all test cases).
OPT_NEGINF = -1e38


class OPTFlashAttention(OPTAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        opt_name: Optional[str] = "facebook/opt-1.3b",
        flash_config_path: Optional[str] = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            bias=bias
        )
        print("Using HuggingFace OPT Attention when sequence length is less than {}, and flash attention when greater.".format(FLASH_ATTENTION_SEQUENCE_LENGTH))
        self.opt_name = opt_name
        self.config = json.load(open(flash_config_path)) if flash_config_path is not None else None
        self.bsz = -1
        self.tgt_len = -1
        self.lh = -1
        self.lq = -1
        self.lk = -1
        self.dtype = torch.float32

    def _shape_no_contiguous(self, tensor: torch.Tensor, seq_len: int, bsz: int, lh: int):
        return tensor.view(bsz, seq_len, lh, self.head_dim).transpose(1, 2)

    def _init_tensors(self, bsz, tgt_len, lh, lq, lk, dtype):
        self.bsz = bsz
        self.tgt_len = tgt_len
        self.dtype = dtype
        self._init_tiling_factors(lh, lq, lk)
        self.attn_weights = torch.zeros((bsz, tgt_len, self.embed_dim), dtype=self.dtype)
        self.qk = torch.zeros((bsz * self.lh, self.lq, self.lk), dtype=self.dtype)
        self.qk_indices = torch.zeros((bsz * self.lh, self.lq), dtype=torch.long)
        self.qkv = torch.zeros((bsz, self.lq, self.lh, self.head_dim), dtype=self.dtype)   # B * lq * lh * d
        self.q =   torch.zeros((bsz, self.lh, self.lq, self.head_dim), dtype=self.dtype)   # B * lh * lq * d
        self.k =   torch.zeros((bsz, self.lh, self.lk, self.head_dim), dtype=self.dtype)   # B * lh * lk * d
        self.v =   torch.zeros((bsz, self.lh, self.lk, self.head_dim), dtype=self.dtype)   # B * lh * lk * d
        self.bmm2_out = torch.zeros((bsz * self.lh, self.lq, self.head_dim), dtype=self.dtype)
        self.max = [torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype), torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype)]
        self.sum = [torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype), torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype)]
        self.tmp = torch.zeros((bsz * self.lh, self.lq), dtype=self.dtype)
        self.mask = torch.zeros((bsz, 1, self.lq, self.lk), dtype=self.dtype)
        self.tile_min = torch.ones(self.mask.shape, dtype=self.dtype) * OPT_MASK_TILE_MIN_VAL

    def _del_tensors(self):
        del self.attn_weights 
        del self.qk 
        del self.qk_indices 
        del self.qkv 
        del self.q 
        del self.k 
        del self.v 
        del self.bmm2_out 
        del self.max 
        del self.sum 
        del self.tmp 
        del self.mask 
        del self.tile_min

    def _init_tiling_factors(self, lh, lq, lk):
        key = "{}_{}_{}_{}".format(self.opt_name, self.bsz, self.tgt_len, str(self.dtype))
        if (lh == -1 or lq == -1 or lk == -1): # Tile factors not provided
            if self.config is None or key not in self.config.keys():
                self.lh = 8 if self.num_heads % 8 == 0 else 6
                self.lq = self.tgt_len // 2
                self.lk = self.tgt_len // 2
            else:
                self.lh, self.lq, self.lk = self.config[key]
        else: # Use external factors
            self.lh, self.lq, self.lk = lh, lq, lk
        assert self.tgt_len % self.lq == 0 and self.tgt_len % self.lk == 0, "Seq length is not divisible by lq and lk!"
        assert self.num_heads % self.lh == 0, "Head numbers is not divisible by lh!"

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        force_vanilla: Optional[bool] = False,
        lh: Optional[int] = -1,
        lq: Optional[int] = -1,
        lk: Optional[int] = -1,
    ) -> Tuple[torch.Tensor, int, int]:
        bsz, tgt_len, _ = hidden_states.size()
        if force_vanilla or tgt_len < FLASH_ATTENTION_SEQUENCE_LENGTH: # 512 by default
            return OPTAttention.forward(
                self,
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions
            )

        dtype = hidden_states.dtype
        self._init_tensors(bsz, tgt_len, lh, lq, lk, dtype)
        lh, lq, lk = self.lh, self.lq, self.lk
        query_states = self.q_proj(hidden_states)
        query_states *= self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states) # (B, L, D2)
        past_key_value = None
        if self.is_decoder:
            past_key_value = (self._shape_no_contiguous(key_states, -1, bsz, self.num_heads), self._shape_no_contiguous(value_states, -1, bsz, self.num_heads))

        for h in range(self.num_heads // lh):
            hh = (h*lh*self.head_dim)
            for i in range(tgt_len // lq):
                self.max[0].fill_(OPT_NEGINF)
                self.max[1].fill_(OPT_NEGINF)
                self.sum[0].zero_()
                self.sum[1].zero_()
                self.qkv.zero_()
                ii = (i*lq)
                # (bsz, lq, lh * d) -> (bsz, lh, lq, d)
                self.q.copy_(self._shape_no_contiguous(query_states[:, ii:(ii+lq), hh:(hh+lh*self.head_dim)].detach(), -1, bsz, lh))
                for j in range(tgt_len // lk):
                    current = j % 2
                    last = 1 - (j % 2)
                    jj = (j*lk)
                    # (bsz, lk, lh * d) -> (bsz, lh, lk, d)
                    self.k.copy_(self._shape_no_contiguous(  key_states[:, jj:(jj+lk), hh:(hh+lh*self.head_dim)].detach(), -1, bsz, lh))
                    self.v.copy_(self._shape_no_contiguous(value_states[:, jj:(jj+lk), hh:(hh+lh*self.head_dim)].detach(), -1, bsz, lh))

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

        attn_output = self.out_proj(self.attn_weights)
        self._del_tensors()

        return attn_output, None, past_key_value