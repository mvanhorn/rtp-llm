# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA kernels with a fixed reduction order for each token.

Persistent GEMM and 2D attention's online softmax are adapted from vLLM
db9527a46873454610df6dbedf79a36d6bf1a7f6 (v0.31.0):
vllm/model_executor/determinism/batch_invariant.py and
vllm/v1/attention/ops/triton_{unified_attention,attention_helpers}.py.
"""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from .batch_invariant_nvfp4 import nvfp4_activation, nvfp4_weight


@triton.jit(do_not_specialize=["rows", "x_scale_row_stride", "x_scale_col_stride"])
def _linear(
    x,
    weight,
    bias,
    output,
    rows,
    cols: tl.constexpr,
    inner: tl.constexpr,
    weight_row_stride: tl.constexpr,
    weight_col_stride: tl.constexpr,
    bias_stride: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_sms: tl.constexpr,
    x_scales=None,
    weight_scales=None,
    x_scale_row_stride=0,
    x_scale_col_stride=0,
    weight_scale_row_stride: tl.constexpr = 0,
    weight_scale_col_stride: tl.constexpr = 0,
    packed_scales: tl.constexpr = False,
    nvfp4_input_scale=None,
    nvfp4_weight_scale=None,
):
    # vLLM's persistent, grouped tile traversal. Worker assignment may change
    # with M, while every tile still reduces the complete K dimension.
    tiles_m = tl.cdiv(rows, block_m)
    tiles_n = tl.cdiv(cols, block_n)
    group_m: tl.constexpr = 8
    tiles_per_group = group_m * tiles_n
    k = tl.arange(0, block_k)
    if nvfp4_input_scale is not None:
        global_input_scale = tl.load(nvfp4_input_scale)
        global_scale = global_input_scale * tl.load(nvfp4_weight_scale)
    for tile_id in tl.range(tl.program_id(0), tiles_m * tiles_n, num_sms, flatten=True):
        first_m = tile_id // tiles_per_group * group_m
        group_width = tl.minimum(tiles_m - first_m, group_m)
        tile_in_group = tile_id % tiles_per_group
        pid_m = first_m + tile_in_group % group_width
        pid_n = tile_in_group // group_width
        m = pid_m.to(tl.int64) * block_m + tl.arange(0, block_m)
        n = pid_n.to(tl.int64) * block_n + tl.arange(0, block_n)
        acc = tl.zeros((block_m, block_n), tl.float32)
        for tile in range(tl.cdiv(inner, block_k)):
            index = tile * block_k + k
            a = tl.load(
                x + m[:, None] * inner + index[None, :],
                (m[:, None] < rows) & (index[None, :] < inner),
                other=0.0,
            )
            valid_weight = (n[None, :] < cols) & (index[:, None] < inner)
            if nvfp4_input_scale is not None:
                a = nvfp4_activation(a, global_input_scale, block_m, block_k)
                b = nvfp4_weight(
                    weight,
                    weight_scales,
                    index,
                    n,
                    weight_row_stride,
                    weight_col_stride,
                    weight_scale_row_stride,
                    weight_scale_col_stride,
                    valid_weight,
                )
            else:
                b = tl.load(
                    weight
                    + n[None, :] * weight_row_stride
                    + index[:, None] * weight_col_stride,
                    valid_weight,
                    other=0.0,
                )
            if x_scales is not None:
                scale_col = tile // 4 if packed_scales else tile
                scale_row = n if packed_scales else n // 128
                sx = tl.load(
                    x_scales + m * x_scale_row_stride + scale_col * x_scale_col_stride,
                    m < rows,
                    other=0,
                )
                sw = tl.load(
                    weight_scales
                    + scale_row * weight_scale_row_stride
                    + scale_col * weight_scale_col_stride,
                    n < cols,
                    other=0,
                )
                if packed_scales:
                    shift = (tile % 4) * 8
                    sx = (((sx >> shift) & 0xFF) << 23).to(tl.float32, bitcast=True)
                    sw = (((sw >> shift) & 0xFF) << 23).to(tl.float32, bitcast=True)
                partial = tl.dot(a.to(tl.float16), b.to(tl.float16))
                acc += partial * (sx[:, None] * sw[None, :])
            else:
                if a.dtype == tl.float8e4nv:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)
                acc = tl.dot(a, b, acc, input_precision="ieee")
        if nvfp4_input_scale is not None:
            acc *= global_scale
        if has_bias:
            acc += tl.load(bias + n * bias_stride, n < cols, other=0)[None, :]
        tl.store(
            output + m[:, None] * cols + n[None, :],
            acc,
            (m[:, None] < rows) & (n[None, :] < cols),
        )


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply an [out_features, in_features] weight to independent input rows."""
    if not x.is_cuda or weight.device != x.device:
        raise ValueError(
            "batch-invariant linear requires tensors on the same CUDA device"
        )
    if x.dtype != weight.dtype or x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("batch-invariant linear requires matching FP16, BF16 or FP32")
    inner = weight.size(1)
    if inner <= 0 or x.shape[-1] != inner:
        raise ValueError("batch-invariant linear input and weight dimensions differ")
    return _launch_linear(x, weight, bias, out_dtype or x.dtype)


def _launch_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    x_scales: torch.Tensor | None = None,
    weight_scales: torch.Tensor | None = None,
    nvfp4_input_scale: torch.Tensor | None = None,
    nvfp4_weight_scale: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    cols, inner = weight.shape
    if nvfp4_input_scale is not None:
        inner *= 2
    if bias is not None:
        if bias.numel() != cols or bias.device != x.device:
            raise ValueError("batch-invariant linear bias must match the weight output")
        bias = bias.reshape(cols)
    rows = x.numel() // inner
    output = torch.empty((*x.shape[:-1], cols), device=x.device, dtype=out_dtype)
    if rows and cols:
        block_m, block_n = 16, 64
        block_k = group_size if x_scales is not None else 32
        scale_options = {}
        if weight_scales is not None:
            scale_options.update(
                {
                    "weight_scales": weight_scales,
                    "weight_scale_row_stride": weight_scales.stride(0),
                    "weight_scale_col_stride": weight_scales.stride(1),
                    "packed_scales": weight_scales.dtype == torch.int32,
                }
            )
        if x_scales is not None:
            scale_options.update(
                {
                    "x_scales": x_scales,
                    "x_scale_row_stride": x_scales.stride(0),
                    "x_scale_col_stride": x_scales.stride(1),
                }
            )
        if nvfp4_input_scale is not None:
            scale_options.update(
                {
                    "nvfp4_input_scale": nvfp4_input_scale,
                    "nvfp4_weight_scale": nvfp4_weight_scale,
                }
            )
        with torch.cuda.device(x.device) if x.is_cuda else nullcontext():
            num_sms = (
                torch.cuda.get_device_properties(x.device).multi_processor_count
                if x.is_cuda
                else 1
            )
            tiles = triton.cdiv(rows, block_m) * triton.cdiv(cols, block_n)
            _linear[(min(tiles, num_sms),)](
                x.contiguous(),
                weight,
                bias,
                output,
                rows,
                cols,
                inner,
                weight.stride(0),
                weight.stride(1),
                bias.stride(0) if bias is not None else 0,
                bias is not None,
                block_m,
                block_n,
                block_k,
                num_sms,
                num_warps=4,
                **scale_options,
            )
    return output


def fp8_per_tensor_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # Static scaling is shared by all rows. Dynamic scaling must derive its
    # maximum from each token independently.
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if weight_scale.numel() != 1 or (
        input_scale is not None and input_scale.numel() != 1
    ):
        raise ValueError("per-tensor FP8 linear requires scalar checkpoint scales")
    quantized, input_scale = quantize_fp8_rows(rows, input_scale)
    # FP8 values are exactly representable in FP16. Convert inside the tile
    # where Triton supports FP8 loads, and keep the FP32 accumulator.
    if torch.cuda.get_device_capability(x.device) >= (8, 9):
        output = _launch_linear(quantized, weight, None, torch.float32)
    else:
        output = linear(quantized.half(), weight.half(), out_dtype=torch.float32)
    output *= input_scale * weight_scale
    if bias is not None:
        output += bias
    return output.to(out_dtype or x.dtype).reshape(*x.shape[:-1], weight.size(0))


def quantize_fp8_rows(
    rows: torch.Tensor, input_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_scale is None:
        input_scale = (
            rows.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 448
        )
    else:
        input_scale = input_scale.reshape(1, 1).expand(rows.size(0), 1)
    return (rows.float() / input_scale).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ), input_scale


def quantize_fp8_blocks(
    rows: torch.Tensor, packed: bool, group_size: int = 128, *, eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )
    from rtp_llm.ops.compute_ops import per_token_group_quant_fp8_v2

    rows = rows.contiguous()
    if group_size not in (32, 128) or rows.size(1) % group_size:
        raise ValueError(
            "batch-invariant block FP8 requires K divisible by its group size"
        )
    quantized = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    scales = create_per_token_group_quant_fp8_output_scale(
        rows.shape,
        rows.device,
        group_size,
        column_major_scales=packed,
        scale_tma_aligned=packed,
        scale_ue8m0=packed,
    )
    # Use the same quantizer for every batch size. Each scale sees one token.
    if rows.size(0):
        per_token_group_quant_fp8_v2(
            rows, quantized, scales, group_size, eps, -448.0, 448.0, packed, False, None
        )
    return quantized, scales


def fp8_per_block_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    group_size: int = 128,
) -> torch.Tensor:
    if x.shape[-1] != weight.size(1):
        raise ValueError("batch-invariant linear input and weight dimensions differ")
    rows = x.reshape(-1, x.shape[-1])
    quantized, scales = quantize_fp8_blocks(
        rows, weight_scales.dtype == torch.int32, group_size
    )
    output = _launch_linear(
        quantized, weight, bias, x.dtype, scales, weight_scales, group_size=group_size
    )
    return output.reshape(*x.shape[:-1], weight.size(0))


def lm_head(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None
) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise ValueError("batch-invariant FP8 LM head requires a weight scale")
        return fp8_per_tensor_linear(x, weight, scale, out_dtype=torch.float32)
    return linear(x.to(weight.dtype), weight, out_dtype=torch.float32)


@triton.jit
def _rms_norm(
    x,
    weight,
    output,
    size: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, block)
    values = tl.load(x + row * size + col, col < size, other=0).to(tl.float32)
    gamma = tl.load(weight + col, col < size, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(values * values, 0) / size + eps)
    tl.store(output + row * size + col, values * scale * gamma, col < size)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.contiguous()
    if x.numel() == 0:
        return x
    size = x.size(-1)
    output = torch.empty_like(x)
    _rms_norm[(x.numel() // size,)](
        x, weight.contiguous(), output, size, eps, triton.next_power_of_2(size)
    )
    return output


@triton.jit(do_not_specialize=["batch", "rows", "table_stride", "table_width"])
def _prepare_attention_tokens(
    query_offsets,
    prefixes_or_lengths,
    block_table,
    layout,
    rows,
    batch,
    table_stride,
    table_width,
    page_size: tl.constexpr,
    prefill: tl.constexpr,
    paged: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0) * block + tl.arange(0, block)
    if prefill:
        lo = tl.full((block,), 0, tl.int32)
        hi = tl.full((block,), batch, tl.int32)
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            end = tl.load(query_offsets + mid + 1, lo < hi, other=0)
            advance = (lo < hi) & (end <= token)
            lo = tl.where(advance, mid + 1, lo)
            hi = tl.where((lo < hi) & ~advance, mid, hi)
        sequence = lo
        active = (token < rows) & (sequence < batch)
        start = tl.load(query_offsets + sequence, active, other=0)
        end = tl.load(query_offsets + sequence + 1, active, other=0)
        prefix = tl.load(prefixes_or_lengths + sequence, active, other=0)
        position = prefix + token - start
    else:
        sequence = token
        active = (token < rows) & (sequence < batch)
        start, end = token, token + 1
        position = tl.load(prefixes_or_lengths + sequence, active, other=0) - 1

    page = tl.full((block,), 0, tl.int32)
    if paged:
        page = tl.load(
            block_table + sequence.to(tl.int64) * table_stride + position // page_size,
            active & (position >= 0) & (position // page_size < table_width),
            other=0,
        )
        # RTP reserves page zero for padding and fake streams.
        active = active & (page > 0)
    tl.store(layout + token * 5, sequence, token < rows)
    tl.store(layout + token * 5 + 1, start, token < rows)
    tl.store(layout + token * 5 + 2, end, token < rows)
    tl.store(layout + token * 5 + 3, tl.where(active, position, -1), token < rows)
    tl.store(layout + token * 5 + 4, page, token < rows)


def attention_token_layout(
    query_offsets: torch.Tensor | None,
    prefixes_or_lengths: torch.Tensor,
    block_table: torch.Tensor | None,
    rows: int,
    page_size: int,
) -> torch.Tensor:
    """Map packed queries to their request, query range, position and cache page."""
    layout = prefixes_or_lengths.new_empty((rows, 5))
    if rows:
        _prepare_attention_tokens[(triton.cdiv(rows, 128),)](
            query_offsets,
            prefixes_or_lengths,
            block_table,
            layout,
            rows,
            prefixes_or_lengths.numel(),
            block_table.stride(0) if block_table is not None else 0,
            block_table.size(1) if block_table is not None else 0,
            page_size,
            query_offsets is not None,
            block_table is not None,
            128,
        )
    return layout


@triton.jit
def _rope(
    tensor,
    layout,
    cos_sin,
    position_ids,
    row_stride: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    position_id_stride: tl.constexpr,
    mrope_time: tl.constexpr,
    mrope_height: tl.constexpr,
    mrope_width: tl.constexpr,
    mrope_interleaved: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    default_position = tl.load(layout + token * 5 + 3)
    if default_position >= 0:
        index = tl.arange(0, block)
        valid = index < half_dim
        if position_id_stride == 3:
            if mrope_interleaved:
                axis = tl.where((index % 3 == 1) & (index < 3 * mrope_height), 1, 0)
                axis = tl.where((index % 3 == 2) & (index < 3 * mrope_width), 2, axis)
            else:
                axis = tl.where(
                    index < mrope_time,
                    0,
                    tl.where(index < mrope_time + mrope_height, 1, 2),
                )
            position = tl.load(position_ids + 3 * token + axis, valid, other=0)
        elif position_id_stride == 1:
            position = tl.load(position_ids + token)
        else:
            position = default_position
        cache_offset = position.to(tl.int64) * (2 * half_dim) + index
        cos = tl.load(cos_sin + cache_offset, valid, other=0)
        sin = tl.load(cos_sin + cache_offset + half_dim, valid, other=0)
        offset = token.to(tl.int64) * row_stride + tl.program_id(1) * head_dim + index
        first = tl.load(tensor + offset, valid, other=0).to(tl.float32)
        second = tl.load(tensor + offset + half_dim, valid, other=0).to(tl.float32)
        tl.store(tensor + offset, first * cos - second * sin, valid)
        tl.store(tensor + offset + half_dim, second * cos + first * sin, valid)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: torch.Tensor,
    cos_sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    *,
    mrope_sections: tuple[int, int, int] | None = None,
    mrope_interleaved: bool = False,
) -> None:
    rows = query.size(0)
    axes = 3 if mrope_sections is not None else 1
    if position_ids is not None and position_ids.numel() == 0:
        position_ids = None
    if position_ids is not None:
        if position_ids.numel() != rows * axes:
            raise ValueError(
                f"batch-invariant RoPE requires {axes} position IDs per token"
            )
        position_ids = position_ids.contiguous()
    if rows:
        half_dim = cos_sin.size(1) // 2
        for tensor in (query, key):
            _rope[(rows, tensor.size(1))](
                tensor,
                layout,
                cos_sin,
                position_ids,
                tensor.stride(0),
                tensor.size(2),
                half_dim,
                axes if position_ids is not None else 0,
                *(mrope_sections or (0, 0, 0)),
                mrope_interleaved,
                triton.next_power_of_2(half_dim),
            )


@triton.jit
def _append_kv(
    key,
    value,
    cache,
    scales,
    layout,
    key_row_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    page_stride: tl.constexpr,
    kv_stride: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    kv_heads: tl.constexpr,
    fp8: tl.constexpr,
    store_scale: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    position = tl.load(layout + token * 5 + 3)
    if position >= 0:
        page = tl.load(layout + token * 5 + 4).to(tl.int64)
        dim = tl.arange(0, block)
        cache_offset = (
            page * page_stride + (head * page_size + position % page_size) * head_dim
        )
        k = tl.load(
            key + token.to(tl.int64) * key_row_stride + head * head_dim + dim,
            dim < head_dim,
            other=0,
        )
        v = tl.load(
            value + token.to(tl.int64) * value_row_stride + head * head_dim + dim,
            dim < head_dim,
            other=0,
        )
        if store_scale:
            k = k.to(tl.float32)
            v = v.to(tl.float32)
            k_scale = tl.maximum(tl.max(tl.abs(k), axis=0), 1e-6) / 448.0
            v_scale = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-6) / 448.0
            scale_offset = (
                page * 2 * kv_heads + head
            ) * page_size + position % page_size
            tl.store(scales + scale_offset, k_scale)
            tl.store(scales + scale_offset + kv_heads * page_size, v_scale)
            k = k / k_scale
            v = v / v_scale
        if fp8:
            k = tl.clamp(k.to(tl.float32), -448.0, 448.0)
            v = tl.clamp(v.to(tl.float32), -448.0, 448.0)
        tl.store(cache + cache_offset + dim, k, dim < head_dim)
        tl.store(cache + cache_offset + kv_stride + dim, v, dim < head_dim)


def append_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    layout: torch.Tensor,
    scales: torch.Tensor | None = None,
) -> None:
    rows, heads, head_dim = key.shape
    if rows:
        _append_kv[(rows, heads)](
            key,
            value,
            cache,
            scales,
            layout,
            key.stride(0),
            value.stride(0),
            cache.stride(0),
            cache.stride(1),
            head_dim,
            cache.size(3),
            heads,
            cache.dtype == torch.float8_e4m3fn,
            scales is not None,
            triton.next_power_of_2(head_dim),
        )


@triton.jit(do_not_specialize=["table_stride"])
def _attention(
    query,
    key,
    value,
    cache,
    kv_scales,
    alibi_slopes,
    block_table,
    layout,
    output,
    table_stride,
    query_row_stride: tl.constexpr,
    key_row_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    page_stride: tl.constexpr,
    kv_stride: tl.constexpr,
    query_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    scale: tl.constexpr,
    causal: tl.constexpr,
    sliding_window: tl.constexpr,
    paged: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    group_head = tl.program_id(2) * block_h + tl.arange(0, block_h)
    q_head = kv_head * group_size + group_head
    dim = tl.arange(0, block_d)
    valid_output = (group_head[:, None] < group_size) & (dim[None, :] < head_dim)
    out = (
        output
        + (token.to(tl.int64) * query_heads + q_head[:, None]) * head_dim
        + dim[None, :]
    )
    position = tl.load(layout + token * 5 + 3)
    if position < 0:
        tl.store(out, 0, valid_output)
    else:
        sequence = tl.load(layout + token * 5)
        start = tl.load(layout + token * 5 + 1)
        if causal:
            length = position + 1
        else:
            length = tl.load(layout + token * 5 + 2) - start
        first_key = 0
        if sliding_window > 0:
            first_key = tl.maximum(0, position - sliding_window + 1)
            if not causal:
                length = tl.minimum(length, position + sliding_window)
        q = tl.load(
            query
            + token.to(tl.int64) * query_row_stride
            + q_head[:, None] * head_dim
            + dim[None, :],
            valid_output,
            other=0,
        )
        maximum = tl.full((block_h,), float("-inf"), tl.float32)
        total = tl.full((block_h,), 1.0, tl.float32)
        acc = tl.zeros((block_h, block_d), tl.float32)
        if alibi_slopes is not None:
            slopes = tl.load(alibi_slopes + q_head, group_head < group_size, other=0)
        # Sequence-aligned KV tiles with one online reduction per query.
        for tile in range(first_key // block_n, tl.cdiv(length, block_n)):
            index = tile * block_n + tl.arange(0, block_n)
            valid_key = (index >= first_key) & (index < length)
            valid = valid_key[:, None] & (dim[None, :] < head_dim)
            if paged:
                page = tl.load(
                    block_table
                    + sequence.to(tl.int64) * table_stride
                    + index // page_size,
                    valid_key,
                    other=0,
                ).to(tl.int64)
                offset = (
                    page * page_stride
                    + (kv_head * page_size + index % page_size) * head_dim
                )
                k = tl.load(cache + offset[:, None] + dim[None, :], valid, other=0.0)
                v = tl.load(
                    cache + offset[:, None] + kv_stride + dim[None, :], valid, other=0.0
                )
                if kv_scales is not None:
                    scale_offset = (
                        page * 2 * kv_heads + kv_head
                    ) * page_size + index % page_size
                    k_scale = tl.load(kv_scales + scale_offset, valid_key, other=0.0)
                    v_scale = tl.load(
                        kv_scales + scale_offset + kv_heads * page_size,
                        valid_key,
                        other=0.0,
                    )
                    k = k.to(tl.float32) * k_scale[:, None]
                    v = v.to(tl.float32) * v_scale[:, None]
            else:
                row = (start + index).to(tl.int64)
                k = tl.load(
                    key
                    + row[:, None] * key_row_stride
                    + kv_head * head_dim
                    + dim[None, :],
                    valid,
                    other=0,
                )
                v = tl.load(
                    value
                    + row[:, None] * value_row_stride
                    + kv_head * head_dim
                    + dim[None, :],
                    valid,
                    other=0,
                )
            k = k.to(q.dtype)
            v = v.to(q.dtype)
            score = tl.dot(q, tl.trans(k)) * scale
            if alibi_slopes is not None:
                score -= slopes[:, None] * tl.abs(index - position)[None, :]
            score = tl.where(valid_key[None, :], score, float("-inf"))
            maximum, total, probability, correction = _softmax_step(
                score, maximum, total
            )
            acc = acc * correction[:, None]
            acc += tl.dot(probability.to(v.dtype), v)
        tl.store(out, acc / total[:, None], valid_output)


@triton.jit
def _softmax_step(score, maximum, total):
    """Online softmax update from vLLM's unified attention."""
    next_maximum = tl.maximum(maximum, tl.max(score, axis=1))
    next_maximum = tl.where(next_maximum > float("-inf"), next_maximum, 0.0)
    probability = tl.exp(score - next_maximum[:, None])
    correction = tl.exp(maximum - next_maximum)
    total = total * correction + tl.sum(probability, axis=1)
    return next_maximum, total, probability, correction


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: torch.Tensor,
    cache: torch.Tensor | None,
    block_table: torch.Tensor | None,
    *,
    causal: bool,
    scale: float,
    sliding_window: int = 0,
    kv_scales: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    rows, query_heads, head_dim = query.shape
    kv_heads = key.size(1)
    group_size = query_heads // kv_heads
    output = torch.empty(
        (rows, query_heads * head_dim), device=query.device, dtype=query.dtype
    )
    if rows:
        block_h = 16
        _attention[(rows, kv_heads, triton.cdiv(group_size, block_h))](
            query,
            key,
            value,
            cache,
            kv_scales,
            alibi_slopes,
            block_table,
            layout,
            output,
            block_table.stride(0) if block_table is not None else 0,
            query.stride(0),
            key.stride(0),
            value.stride(0),
            cache.stride(0) if cache is not None else 0,
            cache.stride(1) if cache is not None else 0,
            query_heads,
            kv_heads,
            group_size,
            head_dim,
            cache.size(3) if cache is not None else 0,
            scale,
            causal,
            sliding_window,
            cache is not None,
            block_h,
            32,
            max(32, triton.next_power_of_2(head_dim)),
            num_warps=4,
        )
    return output
