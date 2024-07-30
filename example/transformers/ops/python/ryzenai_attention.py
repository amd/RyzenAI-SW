#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import numpy as np
import qlinear
import RyzenAI
import torch

# AWQ
from qmodule import WQLinear

FLASH_ATTENTION_SEQUENCE_LENGTH = 4096
MASK_TILE_MIN_VAL = (
    -60
)  # To prevent exp_() from lagging due to big absolute input values. Might vary with different models and precisions (currently -60 works for all test cases).
NEGINF = -1e38


class RyzenAIAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.qkv_proj = None

    def merge_qkv_w4abf16(self, q, k, v):
        self.has_bias = q.bias is not None
        self.q_shape = (q.in_features, q.out_features)
        self.k_shape = (k.in_features, k.out_features)
        self.v_shape = (v.in_features, v.out_features)
        assert (
            self.q_shape[0] == self.k_shape[0] == self.v_shape[0]
        ), "QKV in_features don't match!"
        assert type(q) == type(k) == type(v), "QKV types don't match!"

        self.in_features = self.q_shape[0]
        self.out_features = self.q_shape[1] + self.k_shape[1] + self.v_shape[1]
        self.q_plus_k_out_features = self.q_shape[1] + self.k_shape[1]

        if isinstance(q, WQLinear):  # AWQ
            self.qkv_proj = WQLinear(
                q.w_bit,
                q.group_size,
                self.in_features,
                self.out_features,
                self.has_bias,
                "cpu",
            )

            self.qkv_proj.qweight = torch.empty(
                q.qweight.shape[0] + k.qweight.shape[0] + v.qweight.shape[0],
                q.qweight.shape[1],
                dtype=torch.int8,
                device=q.qweight.device,
            )
            self.qkv_proj.qzeros = torch.empty(
                q.qzeros.shape[0] + k.qzeros.shape[0] + v.qzeros.shape[0],
                q.qzeros.shape[1],
                dtype=torch.int8,
                device=q.qzeros.device,
            )
            self.qkv_proj.scales = torch.empty(
                q.scales.shape[0] + k.scales.shape[0] + v.scales.shape[0],
                q.scales.shape[1],
                dtype=torch.bfloat16,
                device=q.scales.device,
            )
            self.qkv_proj.qweight[: self.q_shape[1], :].copy_(q.qweight)
            self.qkv_proj.qweight[
                self.q_shape[1] : self.q_plus_k_out_features, :
            ].copy_(k.qweight)
            self.qkv_proj.qweight[self.q_plus_k_out_features :, :].copy_(v.qweight)
            self.qkv_proj.qzeros[: self.q_shape[1], :].copy_(q.qzeros)
            self.qkv_proj.qzeros[self.q_shape[1] : self.q_plus_k_out_features, :].copy_(
                k.qzeros
            )
            self.qkv_proj.qzeros[self.q_plus_k_out_features :, :].copy_(v.qzeros)
            self.qkv_proj.scales[: self.q_shape[1], :].copy_(q.scales)
            self.qkv_proj.scales[self.q_shape[1] : self.q_plus_k_out_features, :].copy_(
                k.scales
            )
            self.qkv_proj.scales[self.q_plus_k_out_features :, :].copy_(v.scales)
        else:  # PerGrp: QKV as QLinearPerGrp
            self.qkv_proj = qlinear.QLinearPerGrp(
                self.in_features,
                self.out_features,
                self.has_bias,
                "cpu",
                q.w_bit,
                q.group_size,
            )

            self.qkv_proj.weight = torch.empty(
                self.out_features,
                self.in_features,
                dtype=q.weight.dtype,
                device=q.weight.device,
            )
            self.qkv_proj.weight[: self.q_shape[1], :].copy_(q.weight)
            self.qkv_proj.weight[self.q_shape[1] : self.q_plus_k_out_features, :].copy_(
                k.weight
            )
            self.qkv_proj.weight[self.q_plus_k_out_features :, :].copy_(v.weight)

        if self.has_bias:
            self.qkv_proj.bias = torch.empty(
                (self.out_features,), dtype=q.bias.dtype, device=q.bias.device
            )
            self.qkv_proj.bias[: self.q_shape[1]].copy_(q.bias)
            self.qkv_proj.bias[self.q_shape[1] : self.q_plus_k_out_features].copy_(
                k.bias
            )
            self.qkv_proj.bias[self.q_plus_k_out_features :].copy_(v.bias)

        self.qkv_states_torch = torch.empty(
            (1, 1, self.out_features), dtype=torch.bfloat16
        )

    def merge_qkv_w8(self, q, k, v, precision):
        self.q_shape = (q.in_features, q.out_features)
        self.k_shape = (k.in_features, k.out_features)
        self.v_shape = (v.in_features, v.out_features)
        assert (
            self.q_shape[0] == self.k_shape[0] == self.v_shape[0]
        ), "QKV in_features don't match!"

        self.in_features = self.q_shape[0]
        self.out_features = self.q_shape[1] + self.k_shape[1] + self.v_shape[1]
        self.q_plus_k_out_features = self.q_shape[1] + self.k_shape[1]

        q_wb = q._packed_params._weight_bias()
        k_wb = k._packed_params._weight_bias()
        v_wb = v._packed_params._weight_bias()

        self.qkv_proj = torch.empty(
            self.in_features, self.out_features, dtype=torch.int8
        )
        self.qkv_proj[:, : self.q_shape[1]] = torch.int_repr(q_wb[0]).transpose(0, 1)
        self.qkv_proj[:, self.q_shape[1] : self.q_plus_k_out_features] = torch.int_repr(
            k_wb[0]
        ).transpose(0, 1)
        self.qkv_proj[:, self.q_plus_k_out_features :] = torch.int_repr(
            v_wb[0]
        ).transpose(0, 1)

        self.q_w_scale = q_wb[0].q_scale()
        self.k_w_scale = k_wb[0].q_scale()
        self.v_w_scale = v_wb[0].q_scale()
        self.q_bias = None if q_wb[1] is None else q_wb[1].data
        self.k_bias = None if k_wb[1] is None else k_wb[1].data
        self.v_bias = None if v_wb[1] is None else v_wb[1].data

        qkv_np_int8 = self.qkv_proj.to(torch.float32).numpy().astype(np.int8)

        if precision == "w8a8":
            self.aiegemm = RyzenAI.qlinear_2_a8w8acc32("int8", "int8", "int32")
            self.dyn_min, self.dyn_max = -128.0, 127.0
            self.dyn_denom = self.dyn_max + 1
            self.accum_type = np.int32
            self.act_type = np.int8
        else:  # w8a16 for STX
            self.aiegemm = RyzenAI.qlinear_2_a16w8acc64("int16", "int8", "int64")
            self.dyn_min, self.dyn_max = -32768.0, 32767.0
            self.dyn_denom = self.dyn_max + 1
            self.accum_type = np.int64
            self.act_type = np.int16

        self.aiegemm.initialize_weights(np.ascontiguousarray(qkv_np_int8))

    def matmul_qkv_w4abf16(self, hidden_states):
        """
        Returns 3D QKV torch tensors of bfloat16
        """
        # GEMM has to be 2D, bias included
        hidden_states = hidden_states[0].view(torch.int16)
        seq_len = hidden_states.shape[0]
        if seq_len == 1:
            self.qkv_proj.aiegemm.execute(
                hidden_states, self.qkv_states_torch.view(torch.int16)
            )
            states = self.qkv_states_torch
        else:
            states = torch.empty((1, seq_len, self.out_features), dtype=torch.bfloat16)
            self.qkv_proj.aiegemm.execute(hidden_states, states.view(torch.int16))

        query_states = states[:, :, : self.q_shape[1]].clone()
        key_states = states[:, :, self.q_shape[1] : self.q_plus_k_out_features].clone()
        value_states = states[:, :, self.q_plus_k_out_features :].clone()

        return query_states, key_states, value_states

    def matmul_qkv_w8(self, hidden_states):
        """
        Returns 3D QKV torch tensors of float32 (w8ax doesn't support bfloat16)
        """
        hidden_states = hidden_states[0].to(torch.float32).numpy()
        np.fabs(hidden_states, out=hidden_states)
        x_scale = np.divide(np.max(hidden_states), self.dyn_denom)
        np.divide(hidden_states, x_scale, out=hidden_states)
        np.clip(
            np.round(hidden_states, out=hidden_states),
            self.dyn_min,
            self.dyn_max,
            out=hidden_states,
        )
        c = np.zeros((hidden_states.shape[0], self.out_features), dtype=self.accum_type)
        self.aiegemm.execute(hidden_states.astype(self.act_type, copy=False), c)
        states = c.astype(np.float32, copy=False)
        states = torch.from_numpy(states).unsqueeze(0)
        states *= x_scale

        query_states = states[:, :, : self.q_shape[1]]
        key_states = states[:, :, self.q_shape[1] : self.q_plus_k_out_features]
        value_states = states[:, :, self.q_plus_k_out_features :]

        query_states *= self.q_w_scale
        if self.q_bias is not None:
            query_states += self.q_bias
        key_states *= self.k_w_scale
        if self.k_bias is not None:
            key_states += self.k_bias
        value_states *= self.v_w_scale
        if self.v_bias is not None:
            value_states += self.v_bias

        return query_states, key_states, value_states

    def mha_default(self, query_states, key_states, value_states, **kwargs):
        """vanilla attention that supports any seq len
        this code should be identical to what is int OPT/Llama, any model that has MHA
        initial version of this functions must be on CPU with bfloat16 precision

        overload this function in child class if needed
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.baddbmm(
            (
                kwargs["attention_mask"].view(bsz, q_len, -1).to(query_states.dtype)
                if kwargs["attention_mask"] is not None
                else torch.zeros((1,), dtype=query_states.dtype)
            ),
            query_states.view(bsz * num_heads, -1, head_dim),
            key_states.view(bsz * num_heads, -1, head_dim).transpose(-1, -2),
            alpha=self.scaling,
        )

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(value_states.dtype)
        attn_output = torch.matmul(
            attn_weights.view(bsz * num_heads, q_len, -1), value_states
        )

        return attn_output

    def flash_mha_rope(self, query_states, key_states, value_states, **kwargs):
        bsz, tgt_len, _ = key_states.shape
        self._init_tensors(bsz, tgt_len)
        attn_output = self.compute_fmha_rope(
            query_states, key_states, value_states, **kwargs
        )
        self._del_tensors()
        return attn_output

    def flash_mha(self, query_states, key_states, value_states, **kwargs):
        bsz, tgt_len, _ = key_states.shape
        self._init_tensors(bsz, tgt_len)
        attn_output = self.compute_fmha(
            query_states, key_states, value_states, **kwargs
        )
        self._del_tensors()
        return attn_output

    def _init_tensors(self, bsz, seq_len, lh=-1, lq=-1, lk=-1):
        # Only init when not allocated
        if self.tensor_allocated:
            return
        self.bsz = bsz
        self.seq_len = seq_len
        self._init_tiling_factors(lh, lq, lk)
        self.attn_weights = torch.zeros(
            (self.bsz, seq_len, self.hidden_size), dtype=self.dtype
        )
        self.qk = torch.zeros((self.bsz * self.lh, self.lq, self.lk), dtype=self.dtype)
        self.qk_indices = torch.zeros((self.bsz * self.lh, self.lq), dtype=torch.long)
        self.qkv = torch.zeros(
            (self.bsz * self.lh, self.lq, self.head_dim), dtype=self.dtype
        )  # B * lh * lq * d
        self.q = torch.zeros(
            (self.bsz * self.lh, self.lq, self.head_dim), dtype=self.dtype
        )  # B * lh * lq * d
        self.k = torch.zeros(
            (self.bsz * self.lh, self.lk, self.head_dim), dtype=self.dtype
        )  # B * lh * lk * d
        self.v = torch.zeros(
            (self.bsz * self.lh, self.lk, self.head_dim), dtype=self.dtype
        )  # B * lh * lk * d
        self.max = torch.zeros((2, self.bsz * self.lh, self.lq), dtype=self.dtype)
        self.sum = torch.zeros((2, self.bsz * self.lh, self.lq), dtype=self.dtype)
        self.tmp = torch.zeros((self.bsz * self.lh, self.lq), dtype=self.dtype)
        self.mask = torch.zeros((self.bsz, self.lq, self.lk), dtype=self.dtype)
        self.tile_min = (
            torch.ones(self.mask.shape, dtype=self.dtype) * MASK_TILE_MIN_VAL
        )
        self.tensor_allocated = True

    def _del_tensors(self):
        if not self.tensor_allocated:
            return
        del self.attn_weights
        del self.qk
        del self.qk_indices
        del self.qkv
        del self.q
        del self.k
        del self.v
        del self.max
        del self.sum
        del self.tmp
        del self.mask
        del self.tile_min
        self.tensor_allocated = False

    def _init_tiling_factors(self, lh=-1, lq=-1, lk=-1):
        key = "{}_{}_{}_{}".format(
            self.model_name, self.bsz, self.seq_len, str(self.dtype)
        )
        if lq == -1 or lk == -1:  # Tile factors not provided
            if self.flash_config is None or key not in self.flash_config.keys():
                # self.lh = 8 if self.num_heads % 8 == 0 else 6
                self.lh = self.num_heads // 2
                self.lq, self.lq_remainder = RyzenAIAttention.get_fa_tile_heuristic(
                    self.seq_len
                )
                self.lk, self.lk_remainder = RyzenAIAttention.get_fa_tile_heuristic(
                    self.seq_len
                )
            else:
                self.lh, self.lq, self.lk = self.flash_config[key]
                self.lq_remainder, self.lk_remainder = self.lq, self.lk
        else:  # Use external factors
            self.lh, self.lq, self.lk = lh, lq, lk
            self.lq_remainder, self.lk_remainder = self.lq, self.lk
        assert self.num_heads % self.lh == 0, "Head numbers is not divisible by lh!"

    def _bmm_softmax_bmm(
        self,
        lq_tile,
        lk_tile,
        current,
        last,
    ):
        # bmm1 and mask
        torch.baddbmm(
            self.mask[:, :lq_tile, :lk_tile],
            self.q[:, :lq_tile, :],
            self.k[:, :lk_tile, :].transpose(-1, -2),
            alpha=self.scaling,
            out=self.qk[:, :lq_tile, :lk_tile],
        )  # (bsz * lh, lq, lk)
        torch.max(self.qk, self.tile_min, out=self.qk)

        # m^j
        torch.max(
            self.qk[:, :, :lk_tile], -1, out=(self.tmp, self.qk_indices)
        )  # :lk_tile valid
        torch.max(
            self.tmp, self.max[last], out=self.max[current]
        )  # m^j = max(m^(j-1), rowmax(S^j))

        # P^j
        self.qk.sub_(self.max[current].unsqueeze(-1))
        self.qk.exp_()  # P^j = e^(S^j - m^j)

        # l^j
        torch.sum(
            self.qk[:, :, :lk_tile], dim=-1, out=self.sum[current]
        )  # rowsum(P^j), :lk_tile valid
        torch.sub(self.max[last], self.max[current], out=self.tmp)
        self.tmp.exp_()  # self.tmp = e^(m^(j-1) - m^j)

        # torch.addcmul(a, b, c, out=d): d = a + b * c
        # l^j = e^(m^(j-1) - m^j) * l^(j-1) + rowsum(P^j)
        torch.addcmul(
            self.sum[current], self.sum[last], self.tmp, out=self.sum[current]
        )

        # bmm2
        self.qkv *= self.tmp.view(self.bsz * self.lh, -1, 1)
        torch.baddbmm(
            self.qkv[:, :lq_tile, :],
            self.qk.to(self.v.dtype)[
                :, :lq_tile, :lk_tile
            ],  # bmm2 in designated type (Phi-2 support)
            self.v[:, :lk_tile, :],
            out=self.qkv[:, :lq_tile, :],
        )  # P^j * V_j, (bsz * lh, lq, d)

    def compute_fmha_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        **kwargs,
    ):
        self.lqr, self.lkr = self.lq_remainder, self.lk_remainder
        self.lq_steps = (self.seq_len + self.lq - 1) // self.lq
        self.lk_steps = (self.seq_len + self.lk - 1) // self.lk

        # For convenience
        bsz, seq_len = self.bsz, self.seq_len
        lh, lq, lk = self.lh, self.lq, self.lk
        lqr, lkr = self.lqr, self.lkr
        lq_steps, lk_steps = self.lq_steps, self.lk_steps
        # TODO: cache position goes to here if needed.

        # First part of apply_rotary_pos_emb, the rest fused into FA
        cos = kwargs["cos"][kwargs["position_ids"]]
        sin = kwargs["sin"][kwargs["position_ids"]]

        for h in range(self.num_heads // lh):
            hh = h * lh * self.head_dim
            h_ind = self.kv_head_indices[(h * lh) : (h * lh + lh)]
            kh_ind_occ = RyzenAIAttention.get_first_occurences(
                h_ind
            )  # GQA: K/V head indices and their first occurences in this tile

            for i in range(lq_steps):
                self.max.fill_(NEGINF)
                self.sum.zero_()
                self.qkv.zero_()
                lq_tile = lqr if i == lq_steps - 1 else lq
                ii = i * lq
                ii_end = ii + lq_tile

                # Q and its rotate_half
                self.q[:, :(lq_tile), :].copy_(
                    self.reshape_BLD_to_BH_Ld(
                        query_states[
                            :, ii:ii_end, hh : (hh + lh * self.head_dim)
                        ].detach(),
                        bsz,
                        lh,
                        -1,
                        self.head_dim,
                    )
                )

                # The next 4 lines are equivalent to apply_rotary_pos_emb(q, k, cos, sin, position_ids) FOR Q (partial length)
                RyzenAIAttention.rotate_half_inplace(self.q, self.q2)
                self.q2[:, :(lq_tile), :] *= sin[:, ii:ii_end, :]
                self.q[:, :(lq_tile), :] *= cos[:, ii:ii_end, :]
                self.q += self.q2  # Only up to (ii_end-ii) is valid for the last tile

                for j in range(lk_steps):
                    current = j % 2
                    last = 1 - (j % 2)
                    lk_tile = lkr if j == lk_steps - 1 else lk
                    jj = j * lk
                    jj_end = jj + lk_tile

                    # (bsz, lq/lk, lh * d) -> (bsz, lh, lq/lk, d)
                    # Slicing using slice objects can reduce memory consumption. TODO: Try it.
                    self.v[:, :(lk_tile), :].copy_(
                        RyzenAIAttention.reshape_BLD_to_BH_Ld(
                            value_states.view(bsz, seq_len, -1, self.head_dim)[
                                :, jj:jj_end, h_ind, :
                            ].detach(),
                            bsz,
                            lh,
                            -1,
                            self.head_dim,
                        )
                    )

                    # K and its rotate_half
                    self.k[:, :(lk_tile), :].copy_(
                        RyzenAIAttention.reshape_BLD_to_BH_Ld(
                            key_states.view(bsz, seq_len, -1, self.head_dim)[
                                :, jj:jj_end, h_ind, :
                            ].detach(),
                            bsz,
                            lh,
                            -1,
                            self.head_dim,
                        )
                    )

                    # The next 4 lines are equivalent to apply_rotary_pos_emb(q, k, cos, sin, position_ids) FOR K (partial length)
                    RyzenAIAttention.rotate_half_inplace(self.k, self.k2)
                    self.k2[:, :(lk_tile), :] *= sin[:, jj:jj_end, :]
                    self.k[:, :(lk_tile), :] *= cos[:, jj:jj_end, :]
                    self.k += self.k2

                    # Update K for past_key_value (only update once when finished) [VISIT THIS LINE IF GQA GOES WRONG LATER!]
                    if i == lq_steps - 1:
                        for ind, occ in kh_ind_occ.items():
                            key_states[
                                :,
                                jj:jj_end,
                                ind * self.head_dim : ((ind + 1) * self.head_dim),
                            ] = self.k[occ, :, :].view(bsz, -1, self.head_dim)[
                                :, :(lk_tile), :
                            ]  # Only works for self.k with bsz = 1

                    if kwargs["attention_mask"] is not None:
                        self.mask[:, :lq_tile, :lk_tile].copy_(
                            kwargs["attention_mask"][:, :, ii:ii_end, jj:jj_end]
                            .squeeze(1)
                            .detach()
                        )  # (bsz, 1, lq, lk) -> (bsz, lq, lk)

                    # Core FMHA
                    self._bmm_softmax_bmm(lq_tile, lk_tile, current, last)

                self.qkv.div_(self.sum[current].view(bsz * lh, -1, 1))
                self.attn_weights[:, ii:ii_end, hh : (hh + lh * self.head_dim)] = (
                    self.qkv.view(bsz, lh, -1, self.head_dim)
                    .transpose(1, 2)
                    .reshape(bsz, -1, lh * self.head_dim)[:, :(lq_tile), :]
                )

        return self.attn_weights

    # This function is for flash MHA computation of OPT and ChatGLM3-6D. It supports GQA while it does't have RoPE like compute_fmha_rope for Llama2.
    # Reason: apply_rotary_pos_emb for ChatGLM3-6D (1st and 2nd quarters cos/sin + interleaved + concat with 2nd half)
    #         is completely different from that of Llama2 (1st and 2nd halves cos/sin). Also, GQA can be conveniently degraded
    #         to vanilla self-attention.
    def compute_fmha(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        **kwargs,
    ):
        self.lqr, self.lkr = self.lq_remainder, self.lk_remainder
        self.lq_steps = (self.seq_len + self.lq - 1) // self.lq
        self.lk_steps = (self.seq_len + self.lk - 1) // self.lk

        # For convenience
        bsz, seq_len = self.bsz, self.seq_len
        lh, lq, lk = self.lh, self.lq, self.lk
        lqr, lkr = self.lqr, self.lkr
        lq_steps, lk_steps = self.lq_steps, self.lk_steps

        for h in range(self.num_heads // lh):
            hh = h * lh * self.head_dim
            h_ind = self.kv_head_indices[(h * lh) : (h * lh + lh)]

            for i in range(lq_steps):
                self.max.fill_(NEGINF)
                self.sum.zero_()
                self.qkv.zero_()
                lq_tile = lqr if i == lq_steps - 1 else lq
                ii = i * lq
                ii_end = ii + lq_tile

                # Q and its rotate_half
                self.q[:, :(lq_tile), :].copy_(
                    self.reshape_BLD_to_BH_Ld(
                        query_states[
                            :, ii:ii_end, hh : (hh + lh * self.head_dim)
                        ].detach(),
                        bsz,
                        lh,
                        -1,
                        self.head_dim,
                    )
                )

                for j in range(lk_steps):
                    current = j % 2
                    last = 1 - (j % 2)
                    lk_tile = lkr if j == lk_steps - 1 else lk
                    jj = j * lk
                    jj_end = jj + lk_tile

                    # (bsz, lq/lk, lh * d) -> (bsz, lh, lq/lk, d)
                    # Slicing using slice objects can reduce memory consumption. TODO: Try it.
                    self.v[:, :(lk_tile), :].copy_(
                        RyzenAIAttention.reshape_BLD_to_BH_Ld(
                            value_states.view(bsz, seq_len, -1, self.head_dim)[
                                :, jj:jj_end, h_ind, :
                            ].detach(),
                            bsz,
                            lh,
                            -1,
                            self.head_dim,
                        )
                    )

                    # K and its rotate_half
                    self.k[:, :(lk_tile), :].copy_(
                        RyzenAIAttention.reshape_BLD_to_BH_Ld(
                            key_states.view(bsz, seq_len, -1, self.head_dim)[
                                :, jj:jj_end, h_ind, :
                            ].detach(),
                            bsz,
                            lh,
                            -1,
                            self.head_dim,
                        )
                    )

                    if kwargs["attention_mask"] is not None:
                        self.mask[:, :lq_tile, :lk_tile].copy_(
                            kwargs["attention_mask"][:, :, ii:ii_end, jj:jj_end]
                            .squeeze(1)
                            .detach()
                        )  # (bsz, 1, lq, lk) -> (bsz, lq, lk)

                    # Core FMHA
                    self._bmm_softmax_bmm(lq_tile, lk_tile, current, last)

                self.qkv.div_(self.sum[current].view(bsz * lh, -1, 1))
                self.attn_weights[:, ii:ii_end, hh : (hh + lh * self.head_dim)] = (
                    self.qkv.view(bsz, lh, -1, self.head_dim)
                    .transpose(1, 2)
                    .reshape(bsz, -1, lh * self.head_dim)[:, :(lq_tile), :]
                )

        return self.attn_weights

    def apply_rotary_pos_emb_token_fast(self, q, k, position_ids):
        cos, sin = (
            self.rotary_emb.cos_cached[position_ids].to(q.dtype),
            self.rotary_emb.sin_cached[position_ids].to(q.dtype),
        )
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def get_fa_tile_heuristic(cls, l: int):
        # # Fixed tiles
        # if l <= TILE_MEDIUM:
        #     return TILE_SMALL, TILE_SMALL
        # elif TILE_MEDIUM < l <= TILE_MEDIUM + TILE_SMALL:
        #     return TILE_MEDIUM, TILE_SMALL
        # elif TILE_MEDIUM + TILE_SMALL < l <= TILE_LARGE:
        #     return TILE_MEDIUM, TILE_MEDIUM
        # elif TILE_LARGE < l <= TILE_LARGE + TILE_MEDIUM:
        #     _, x = self.get_lqlk_heuristic(l - TILE_MEDIUM)
        #     return TILE_MEDIUM, x
        # elif TILE_LARGE + TILE_MEDIUM < l <= TILE_LARGE * 2:
        #     return TILE_LARGE, TILE_LARGE
        # else:
        #     _, x = self.get_lqlk_heuristic(l - TILE_LARGE)
        #     return TILE_LARGE, x

        # Tmp solution for simply /2
        x = (l + 1) // 2
        return x, l - x

    @classmethod
    def reshape_BLD_to_BHLd(cls, tensor: torch.Tensor, B: int, H: int, L: int, d: int):
        """
        BLD -> BLHd -> BHLd
        """
        return tensor.view(B, L, H, d).transpose(1, 2)

    @classmethod
    def reshape_BHLd_to_BLD(cls, tensor: torch.Tensor, B: int, L: int, D: int):
        """
        BHLd -> BLHd -> BLD
        """
        return tensor.transpose(1, 2).reshape(B, L, D)

    @classmethod
    def reshape_BLD_to_BH_Ld(cls, tensor: torch.Tensor, B: int, H: int, L: int, d: int):
        """
        BLD -> BLHd -> BH_Ld
        """
        return tensor.view(B, L, H, d).transpose(1, 2).reshape(B * H, L, d)

    @classmethod
    def rotate_half_inplace(cls, x: torch.Tensor, out: torch.Tensor):
        """Rotates half the hidden dims of the input."""
        s = x.shape[-1]
        s2 = s // 2
        x1 = x[..., :s2]
        x2 = x[..., s2:]
        torch.cat((x2, x1), dim=-1, out=out)
        out[..., : (s - s2)].neg_()

    @classmethod
    def get_first_occurences(cls, l: int):
        last = -1
        d = {}
        for idx, x in enumerate(l):
            if x != last:
                d[x] = idx
                last = x
        return d
