"""Device kernels; model state and execution policy belong to the caller."""

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_key(
    history,
    cosine,
    sine,
    gamma,
    D: tl.constexpr,
    ROTARY: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
):
    d = tl.arange(0, D)
    pooled = (tl.sum(history, axis=0) * (1.0 / RATIO)).to(tl.bfloat16).to(tl.float32)
    gain = 1.0 + gamma.to(tl.float32)
    rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / D + EPS)
    normalized = (pooled * rms * gain).to(tl.bfloat16)

    # Upstream RoPE rounds each multiplication and the addition to BF16.
    partner = tl.where(
        d < ROTARY, tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2), d
    )
    rotated_half = tl.gather(normalized, partner, axis=0)
    left = (normalized.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
    right = (
        rotated_half.to(tl.float32)
        * sine.to(tl.float32)
        * tl.where(d < ROTARY // 2, -1.0, 1.0)
    ).to(tl.bfloat16)
    return tl.where(
        d < ROTARY,
        (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
        normalized,
    )


@triton.jit
def _compress_prefill_keys(
    raw,
    cos,
    sin,
    gamma,
    out,
    D: tl.constexpr,
    ROTARY: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_RATIO: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    group = tl.arange(0, BLOCK_RATIO)
    offsets = (row * RATIO + group[:, None]) * D + d[None, :]
    history = tl.load(raw + offsets, mask=group[:, None] < RATIO, other=0)
    c = tl.load(cos + row * ROTARY + d, mask=d < ROTARY, other=1)
    s = tl.load(sin + row * ROTARY + d, mask=d < ROTARY, other=0)
    value = _compress_key(history, c, s, tl.load(gamma + d), D, ROTARY, RATIO, EPS)
    tl.store(out + row * D + d, value)


def compress_prefill_keys(history, cos, sin, gamma, eps):
    """Apply the same compression and BF16 boundaries as the decode writer."""
    rows, ratio, width = history.shape
    out = torch.empty((rows, width), device=history.device, dtype=torch.bfloat16)
    _compress_prefill_keys[(rows,)](
        history.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        gamma.contiguous(),
        out,
        width,
        cos.shape[-1],
        ratio,
        eps,
        triton.next_power_of_2(ratio),
        num_warps=4,
    )
    return out


@triton.jit
def _write_decode_key(
    raw_ptr,
    positions_ptr,
    valid_ptr,
    cos_ptr,
    sin_ptr,
    gamma_ptr,
    kv_pool_ptr,
    kv_table_ptr,
    state_pool_ptr,
    state_table_ptr,
    KV_TABLE_WIDTH: tl.constexpr,
    STATE_TABLE_WIDTH: tl.constexpr,
    KV_POOL_BLOCKS: tl.constexpr,
    STATE_POOL_BLOCKS: tl.constexpr,
    KV_TOKENS_PER_BLOCK: tl.constexpr,
    STATE_TOKENS_PER_BLOCK: tl.constexpr,
    KV_ENTRIES_PER_BLOCK: tl.constexpr,
    D: tl.constexpr,
    ROTARY: tl.constexpr,
    EPS: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    request = tl.program_id(0)
    d = tl.arange(0, D)
    active = tl.load(valid_ptr + request) != 0 if HAS_VALID else True
    position = tl.load(positions_ptr + request)
    state_column = position // STATE_TOKENS_PER_BLOCK
    state_id = tl.load(
        state_table_ptr + request * STATE_TABLE_WIDTH + state_column,
        mask=active & (position >= 0) & (state_column < STATE_TABLE_WIDTH),
        other=0,
    ).to(tl.int64)
    state_valid = (position >= 0) & (state_id > 0) & (state_id < STATE_POOL_BLOCKS)
    raw = tl.load(raw_ptr + request * D + d, mask=active, other=0).to(tl.float32)
    tl.store(
        state_pool_ptr + (state_id * 8 + position % 8) * D + d,
        raw,
        mask=active & state_valid,
    )

    if active and (position + 1) % 4 == 0:
        group = tl.arange(0, 4)
        source_positions = position - 3 + group
        source_columns = source_positions // STATE_TOKENS_PER_BLOCK
        source_ids = tl.load(
            state_table_ptr + request * STATE_TABLE_WIDTH + source_columns,
            mask=(source_positions >= 0) & (source_columns < STATE_TABLE_WIDTH),
            other=0,
        ).to(tl.int64)
        source_valid = (source_ids > 0) & (source_ids < STATE_POOL_BLOCKS)
        source_slots = source_ids * 8 + source_positions % 8
        history = tl.load(
            state_pool_ptr + source_slots[:, None] * D + d[None, :],
            mask=source_valid[:, None],
            other=0.0,
        )
        # The current key has just been written by this program. Read the
        # input directly so no global-memory fence is needed for that row.
        history = tl.where(group[:, None] == 3, raw[None, :], history)
        cosine = tl.load(cos_ptr + request * ROTARY + d, mask=d < ROTARY, other=1)
        sine = tl.load(sin_ptr + request * ROTARY + d, mask=d < ROTARY, other=0)
        value = _compress_key(
            history, cosine, sine, tl.load(gamma_ptr + d), D, ROTARY, 4, EPS
        )

        kv_column = position // KV_TOKENS_PER_BLOCK
        kv_id = tl.load(
            kv_table_ptr + request * KV_TABLE_WIDTH + kv_column,
            mask=(position >= 0) & (kv_column < KV_TABLE_WIDTH),
            other=0,
        ).to(tl.int64)
        kv_valid = (kv_id > 0) & (kv_id < KV_POOL_BLOCKS)
        entry = (position // 4) % KV_ENTRIES_PER_BLOCK
        tl.store(
            kv_pool_ptr + (kv_id * KV_ENTRIES_PER_BLOCK + entry) * D + d,
            value,
            mask=kv_valid,
        )
