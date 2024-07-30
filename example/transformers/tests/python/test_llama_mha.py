#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#


import copy
import math

import pytest
import torch

torch.manual_seed(0)
import time

import ryzenai_torch_cpp


def test_mha_llama2():

    def mha_golden(query_states, key_states, value_states, attention_mask):
        scaling = 1 / math.sqrt(128)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=3, dtype=torch.float32
        ).to(torch.bfloat16)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=0.1, training=False)
        attn_output = torch.matmul(attn_weights, value_states)

        # dont it for now
        # attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output

    proj_shape = (1, 32, 2048, 128)
    mask_shape = (1, 1, 2048, 2048)

    query_states = torch.rand(proj_shape).to(torch.bfloat16)
    key_states = torch.rand(proj_shape).to(torch.bfloat16)
    value_states = torch.rand(proj_shape).to(torch.bfloat16)

    attention_mask = torch.rand(mask_shape).to(torch.bfloat16)

    scaling = torch.tensor(1 / math.sqrt(128))

    t0 = time.perf_counter()

    print(
        f"shapes: {query_states.shape} {key_states.shape} {value_states.shape} {attention_mask.shape}"
    )
    print(query_states.is_contiguous())
    print(key_states.is_contiguous())
    print(value_states.is_contiguous())

    y0 = mha_golden(query_states, key_states, value_states, attention_mask)
    print(f"pytorch mha reference:  {time.perf_counter()-t0}")

    tcpp = ryzenai_torch_cpp.cpu_mha()

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y1 = tcpp.mha_tensorized(query_states, key_states, value_states, attention_mask)
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha tensorized: {y1.shape} {tarr}")

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y2 = tcpp.mha_multithread(
            query_states, key_states, value_states, attention_mask
        )
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha multithread: {tarr}")

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y3 = tcpp.mha_flat(query_states, key_states, value_states, attention_mask)
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha flat: {tarr}")

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y4 = tcpp.mha_flat_multithread(
            query_states, key_states, value_states, attention_mask
        )
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha flat multithread:  {tarr}")

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y5 = tcpp.mha_top(query_states, key_states, value_states, attention_mask)
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha top:  {tarr}")

    # with transpose: # ([1, 2048, 32, 128])
    # without transpose: # ([1, 32, 2048, 128])
    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y1[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha tensorized: {i}: {err.min()}, {err.max()}")

    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y2[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha multithread: {i}: {err.min()}, {err.max()}")

    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y3[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha flat: {i}: {err.min()}, {err.max()}")

    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y4[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha flat multithread: {i}: {err.min()}, {err.max()}")

    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y5[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha top: {i}: {err.min()}, {err.max()}")

    assert torch.allclose(y0, y5, atol=0.008)


@pytest.mark.skip(reason="this is for win24 mha")
def test_mha_win24():

    num_heads = 1
    embed_size = 4096
    head_dim = 512

    def mha_golden(query_states, key_states, value_states, attention_mask):
        scaling = 1 / math.sqrt(head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=3, dtype=torch.float32
        ).to(torch.bfloat16)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=0.1, training=False)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    proj_shape = (1, num_heads, embed_size, head_dim)
    mask_shape = (1, num_heads, embed_size, embed_size)

    query_states = torch.rand(proj_shape).to(torch.bfloat16)
    key_states = torch.rand(proj_shape).to(torch.bfloat16)
    value_states = torch.rand(proj_shape).to(torch.bfloat16)

    attention_mask = torch.rand(mask_shape).to(torch.bfloat16)

    scaling = torch.tensor(1 / math.sqrt(head_dim))

    t0 = time.perf_counter()

    print(
        f"shapes: {query_states.shape} {key_states.shape} {value_states.shape} {attention_mask.shape}"
    )
    print(query_states.is_contiguous())
    print(key_states.is_contiguous())
    print(value_states.is_contiguous())

    y0 = mha_golden(query_states, key_states, value_states, attention_mask)
    print(f"pytorch mha reference:  {time.perf_counter()-t0}")

    tcpp = ryzenai_torch_cpp.cpu_mha()  # num_heads, embed_size, head_dim)

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y1 = tcpp.mha_tensorized(query_states, key_states, value_states, attention_mask)
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha tensorized: {y1.shape} {tarr}")

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y2 = tcpp.mha_multithread(
            query_states, key_states, value_states, attention_mask
        )
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha multithread: {tarr}")

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y3 = tcpp.mha_flat(query_states, key_states, value_states, attention_mask)
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha flat: {tarr}")

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y4 = tcpp.mha_flat_multithread(
            query_states, key_states, value_states, attention_mask
        )
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha flat multithread:  {tarr}")

    tarr = []
    for i in range(5):
        t0 = time.perf_counter()
        y5 = tcpp.mha_top(query_states, key_states, value_states, attention_mask)
        tarr.append(time.perf_counter() - t0)
    print(f"libtorch mha top:  {tarr}")

    # with transpose: # ([1, 2048, 32, 128])
    # without transpose: # ([1, 32, 2048, 128])
    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y1[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha tensorized: {i}: {err.min()}, {err.max()}")

    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y2[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha multithread: {i}: {err.min()}, {err.max()}")

    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y3[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha flat: {i}: {err.min()}, {err.max()}")

    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y4[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha flat multithread: {i}: {err.min()}, {err.max()}")

    for i in range(y0.shape[1]):
        err = y0[0, i, :, :] - y5[0, i, :, :]
        err /= y0[0, i, :, :]
        err = err.abs()
        print(f"libtorch mha top: {i}: {err.min()}, {err.max()}")

    assert torch.allclose(y0, y5, atol=0.008)
