# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixed-order MLA kernels adapted to RTP's packed queries and KV pages.

Adapted from vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6,
vllm/model_executor/determinism/batch_invariant.py (bmm_kernel) and
vllm/v1/attention/ops/triton_decode_attention.py (_fwd_grouped_kernel_stage1).

The upstream decode kernel was adapted from SGLang (9f635ea50de920aa507f486daafba26a5b837574),
which was originally adapted from LightLLM (96353e868a840db4d103138caf15ed9dbea8c186).
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _head_matmul(
    query,
    weight,
    output,
    rows,
    heads: tl.constexpr,
    in_dim: tl.constexpr,
    out_dim: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    weight_head_stride: tl.constexpr,
    weight_in_stride: tl.constexpr,
    weight_out_stride: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    head = tl.program_id(0)
    row = tl.program_id(1) * block_m + tl.arange(0, block_m)
    column = tl.program_id(2) * block_n + tl.arange(0, block_n)
    reduction = tl.arange(0, block_k)
    q_base = query + row.to(tl.int64) * query_row_stride + head * query_head_stride
    w_base = weight + head.to(tl.int64) * weight_head_stride
    accumulator = tl.zeros((block_m, block_n), tl.float32)
    for tile in range(tl.cdiv(in_dim, block_k)):
        k = tile * block_k + reduction
        q = tl.load(
            q_base[:, None] + k[None, :],
            (row[:, None] < rows) & (k[None, :] < in_dim),
            other=0,
        )
        w = tl.load(
            w_base
            + k[:, None] * weight_in_stride
            + column[None, :] * weight_out_stride,
            (k[:, None] < in_dim) & (column[None, :] < out_dim),
            other=0,
        )
        accumulator = tl.dot(q, w, accumulator, input_precision="ieee")
    out_base = output + (row.to(tl.int64) * heads + head) * out_dim
    tl.store(
        out_base[:, None] + column[None, :],
        accumulator,
        (row[:, None] < rows) & (column[None, :] < out_dim),
    )


def head_matmul(query: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply each query head by its own [in_dim, out_dim] weight."""
    rows, heads, in_dim = query.shape
    out_dim = weight.size(2)
    if (
        weight.shape[:2] != (heads, in_dim)
        or query.dtype != weight.dtype
        or query.stride(2) != 1
    ):
        raise ValueError("unsupported per-head matmul tensors")
    output = query.new_empty((rows, heads, out_dim))
    if rows:
        _head_matmul[(heads, triton.cdiv(rows, 16), triton.cdiv(out_dim, 64))](
            query,
            weight,
            output,
            rows,
            heads,
            in_dim,
            out_dim,
            query.stride(0),
            query.stride(1),
            *weight.stride(),
            16,
            64,
            32,
            num_warps=4,
        )
    return output


def cache_slots(layout: torch.Tensor, page_size: int) -> torch.Tensor:
    positions, pages = layout[:, 3], layout[:, 4]
    return torch.where(
        positions >= 0, pages.to(torch.int64) * page_size + positions % page_size, -1
    )


@triton.jit(do_not_specialize=["table_stride"])
def _mla_attention(
    query,
    query_rope,
    cache,
    cache_rope,
    cache_scales,
    block_table,
    layout,
    topk_indices,
    output,
    table_stride,
    heads: tl.constexpr,
    rank: tl.constexpr,
    rope_dim: tl.constexpr,
    query_rope_row_stride: tl.constexpr,
    query_rope_head_stride: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    rope_page_stride: tl.constexpr,
    rope_token_stride: tl.constexpr,
    scale_page_stride: tl.constexpr,
    scale_token_stride: tl.constexpr,
    page_size: tl.constexpr,
    scale: tl.constexpr,
    paged: tl.constexpr,
    fp8_cache: tl.constexpr,
    topk_size: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
    block_rank: tl.constexpr,
    block_rope: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * block_h + tl.arange(0, block_h)
    dim = tl.arange(0, block_rank)
    valid_output = (head[:, None] < heads) & (dim[None, :] < rank)
    output_ptr = output + (token * heads + head[:, None]) * rank + dim[None, :]
    position = tl.load(layout + token * 5 + 3)
    if position < 0:
        tl.store(output_ptr, 0, valid_output)
        return

    sequence = tl.load(layout + token * 5).to(tl.int64)
    start = tl.load(layout + token * 5 + 1)
    if topk_size:
        length = tl.minimum(position + 1, topk_size)
    elif paged:
        length = position + 1
    else:
        length = token - start + 1
    q = tl.load(
        query + (token * heads + head[:, None]) * rank + dim[None, :],
        valid_output,
        other=0,
    )
    if block_rope:
        rope_dim_idx = tl.arange(0, block_rope)
        q_rope = tl.load(
            query_rope
            + token * query_rope_row_stride
            + head[:, None] * query_rope_head_stride
            + rope_dim_idx[None, :],
            (head[:, None] < heads) & (rope_dim_idx[None, :] < rope_dim),
            other=0,
        )
    maximum = tl.full((block_h,), float("-inf"), tl.float32)
    total = tl.zeros((block_h,), tl.float32)
    acc = tl.zeros((block_h, block_rank), tl.float32)
    for tile in tl.range(0, length, block_n):
        index = tile + tl.arange(0, block_n)
        valid = index < length
        if topk_size:
            index = tl.load(topk_indices + token * topk_size + index, valid, other=-1)
            valid = valid & (index >= 0) & (index <= position)
        if paged:
            page = tl.load(
                block_table + sequence * table_stride + index // page_size,
                valid,
                other=0,
            ).to(tl.int64)
            valid = valid & (page > 0)
            row = index % page_size
        else:
            page = tl.full((block_n,), 0, tl.int64)
            row = start + index
        latent_offset = page * cache_page_stride + row * cache_token_stride
        k = tl.load(
            cache + latent_offset[None, :] + dim[:, None],
            (dim[:, None] < rank) & valid[None, :],
            other=0,
        )
        if fp8_cache:
            k = k.to(tl.float8e4nv, bitcast=True)
            scale_offset = page * scale_page_stride + row * scale_token_stride
            k_scale = tl.load(
                cache_scales + scale_offset[None, :] + dim[:, None] // 128,
                (dim[:, None] < rank) & valid[None, :],
                other=0,
            )
            k = (k.to(tl.float32) * k_scale).to(q.dtype)
        score = tl.dot(q, k)
        if block_rope:
            rope_offset = page * rope_page_stride + row * rope_token_stride
            k_rope = tl.load(
                cache_rope + rope_offset[None, :] + rope_dim_idx[:, None],
                (rope_dim_idx[:, None] < rope_dim) & valid[None, :],
                other=0,
            )
            score += tl.dot(q_rope, k_rope)
        score = tl.where(valid[None, :], score * scale, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        # Give empty tiles a finite subtraction base, retaining the running max.
        subtraction_base = tl.where(next_max == float("-inf"), 0.0, next_max)
        correction = tl.exp(maximum - subtraction_base)
        probability = tl.exp(score - subtraction_base[:, None])
        total = total * correction + tl.sum(probability, 1)
        acc = acc * correction[:, None] + tl.dot(probability.to(k.dtype), tl.trans(k))
        maximum = next_max
    tl.store(output_ptr, acc / tl.where(total > 0, total, 1.0)[:, None], valid_output)


def attention(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    cache: torch.Tensor,
    layout: torch.Tensor,
    pages: torch.Tensor | None,
    *,
    scale: float,
    fp8_cache: bool = False,
    topk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply absorbed MLA with the same causal reduction for each query token."""
    rows, heads, rank = query.shape
    rope_dim = query_rope.size(-1)
    cache_scales = None
    if fp8_cache:
        # RTP fp8_ds_mla is [512 FP8 values, 4 FP32 scales, 64 base-dtype RoPE].
        packed_cache = cache.view(torch.uint8)
        cache = packed_cache[..., :rank]
        rope_start = rank + rank // 128 * 4
        cache_scales = packed_cache[..., rank:rope_start].view(torch.float32)
        cache_rope = packed_cache[..., rope_start:].view(query.dtype)
    else:
        cache_rope = cache[..., rank:]
    output = torch.empty_like(query)
    if rows:
        _mla_attention[(rows, triton.cdiv(heads, 16))](
            query,
            query_rope,
            cache,
            cache_rope,
            cache_scales,
            pages,
            layout,
            topk_indices,
            output,
            pages.stride(0) if pages is not None else 0,
            heads,
            rank,
            rope_dim,
            query_rope.stride(0),
            query_rope.stride(1),
            cache.stride(0),
            cache.stride(1),
            cache_rope.stride(0),
            cache_rope.stride(1),
            cache_scales.stride(0) if cache_scales is not None else 0,
            cache_scales.stride(1) if cache_scales is not None else 0,
            cache.size(1),
            scale,
            pages is not None,
            fp8_cache,
            topk_indices.size(1) if topk_indices is not None else 0,
            16,
            32,
            max(32, triton.next_power_of_2(rank)),
            max(32, triton.next_power_of_2(rope_dim)) if rope_dim else 0,
            num_warps=4,
        )
    return output
