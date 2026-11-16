import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import gated_delta_step
from rtp_llm.models_py.triton_kernels.fla.op import exp, softplus


@triton.jit
def _page_id(pages, logical_page, PAGE_COUNT: tl.constexpr, POOL_SIZE: tl.constexpr):
    page = tl.load(
        pages + logical_page,
        mask=(logical_page >= 0) & (logical_page < PAGE_COUNT),
        other=0,
    ).to(tl.int64)
    return tl.where((page > 0) & (page < POOL_SIZE), page, 0)


@triton.jit
def _checkpoint_index(prefix, token, PAGE_SIZE: tl.constexpr, IS_VERIFY: tl.constexpr):
    if IS_VERIFY:
        # Target verification reserves a separate slot for every candidate
        # and the executor publishes the accepted checkpoint after sampling.
        return prefix // PAGE_SIZE + token
    return (prefix + token) // PAGE_SIZE


@triton.jit
def _request_layout(
    cu_seqlens,
    lengths,
    pages,
    row,
    num_tokens,
    PAGE_COUNT: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    IS_VERIFY: tl.constexpr,
):
    start = tl.load(cu_seqlens + row).to(tl.int64)
    end = tl.minimum(tl.load(cu_seqlens + row + 1), num_tokens).to(tl.int64)
    prefix = tl.load(lengths + row)
    if IS_DECODE or IS_VERIFY:
        prefix -= 1
    read_page = tl.full((), 0, tl.int64)
    active = (end > start) & (prefix >= 0)
    if POOL_SIZE:
        read_page = _page_id(pages, (prefix - 1) // PAGE_SIZE, PAGE_COUNT, POOL_SIZE)
        write_page = _page_id(
            pages,
            _checkpoint_index(prefix, end - start - 1, PAGE_SIZE, IS_VERIFY),
            PAGE_COUNT,
            POOL_SIZE,
        )
        # The allocator may omit old pages. The starting checkpoint (when
        # continuing a request) and the tail must be materialized for a live row.
        active &= ((prefix == 0) | (read_page > 0)) & (write_page > 0)
        read_page = tl.where(prefix > 0, read_page, 0)
    return start, end, prefix, read_page, active


@triton.jit(do_not_specialize=["num_tokens", "page_stride"])
def _conv1d(
    x,
    weight,
    out,
    conv_state,
    cu_seqlens,
    lengths,
    block_map,
    num_tokens,
    page_stride,
    D: tl.constexpr,
    X_STRIDE: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_T: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_TYPE: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE_COUNT: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    IS_VERIFY: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    pages = block_map + row.to(tl.int64) * page_stride if POOL_SIZE else None
    start, end, prefix, read_page, active = _request_layout(
        cu_seqlens,
        lengths,
        pages,
        row,
        num_tokens,
        PAGE_COUNT,
        POOL_SIZE,
        PAGE_SIZE,
        IS_DECODE,
        IS_VERIFY,
    )
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    if not active:
        for token in range(start, end):
            tl.store(out + token * D + channels, 0, channels < D)
        return

    weights = ()
    history = ()
    for i in tl.static_range(WIDTH):
        w = tl.load(
            weight + channels * W_STRIDE_C + i * W_STRIDE_T, channels < D, other=0
        )
        weights += (w.to(tl.float32),)
        if i < WIDTH - 1:
            if POOL_SIZE:
                item = tl.load(
                    conv_state + read_page * STATE_STRIDE + i * D + channels,
                    (read_page > 0) & (channels < D),
                    other=0,
                )
            else:
                item = tl.full((BLOCK_C,), 0, STATE_TYPE)
            history += (item,)

    for token in range(start, end):
        current = tl.load(x + token * X_STRIDE + channels, channels < D, other=0)
        window = history + (current.to(STATE_TYPE),)
        acc = tl.full((BLOCK_C,), 0, tl.float32)
        for i in tl.static_range(WIDTH):
            acc += window[i].to(tl.float32) * weights[i]
        tl.store(out + token * D + channels, acc / (1 + tl.exp(-acc)), channels < D)
        history = window[1:]

        if POOL_SIZE:
            step = token - start
            if IS_VERIFY or (prefix + step + 1) % PAGE_SIZE == 0 or token + 1 == end:
                page = _page_id(
                    pages,
                    _checkpoint_index(prefix, step, PAGE_SIZE, IS_VERIFY),
                    PAGE_COUNT,
                    POOL_SIZE,
                )
                for i in tl.static_range(WIDTH - 1):
                    tl.store(
                        conv_state + page * STATE_STRIDE + i * D + channels,
                        history[i],
                        (page > 0) & (channels < D),
                    )


@triton.jit(do_not_specialize=["num_tokens", "page_stride"])
def _recurrent(
    qkv,
    g,
    beta,
    decay_log,
    decay_bias,
    out,
    ssm_state,
    cu_seqlens,
    lengths,
    block_map,
    num_tokens,
    page_stride,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_TYPE: tl.constexpr,
    PAGE_COUNT: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    IS_VERIFY: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    head_v = tl.program_id(1)
    head_k = head_v // (HV // H)
    pages = block_map + row.to(tl.int64) * page_stride if POOL_SIZE else None
    start, end, prefix, read_page, active = _request_layout(
        cu_seqlens,
        lengths,
        pages,
        row,
        num_tokens,
        PAGE_COUNT,
        POOL_SIZE,
        PAGE_SIZE,
        IS_DECODE,
        IS_VERIFY,
    )
    k_cols = tl.arange(0, BLOCK_K)
    v_cols = tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)
    if not active:
        for token in range(start, end):
            tl.store(out + (token * HV + head_v) * V + v_cols, 0, v_cols < V)
        return

    if decay_log is not None:
        # KDA stores [K, V] per head; GDN stores [V, K].
        state_offsets = head_v * V * K + k_cols[None, :] * V + v_cols[:, None]
        head_decay = -exp(tl.load(decay_log + head_v).to(tl.float32))
        bias = tl.load(decay_bias + head_v * K + k_cols, k_cols < K, other=0.0).to(
            tl.float32
        )
    else:
        state_offsets = (head_v * V + v_cols[:, None]) * K + k_cols[None, :]
    mask = (v_cols[:, None] < V) & (k_cols[None, :] < K)
    if POOL_SIZE:
        state = tl.load(
            ssm_state + read_page * STATE_STRIDE + state_offsets,
            (read_page > 0) & mask,
            other=0,
        ).to(tl.float32)
    else:
        state = tl.full((BLOCK_V, BLOCK_K), 0, tl.float32)

    for token in range(start, end):
        qkv_row = qkv + token * (2 * H * K + HV * V)
        q = tl.load(qkv_row + head_k * K + k_cols, k_cols < K, other=0).to(tl.float32)
        k = tl.load(qkv_row + (H + head_k) * K + k_cols, k_cols < K, other=0).to(
            tl.float32
        )
        v = tl.load(qkv_row + 2 * H * K + head_v * V + v_cols, v_cols < V, other=0).to(
            tl.float32
        )
        if decay_log is not None:
            forget = tl.load(
                g + (token * HV + head_v) * K + k_cols, k_cols < K, other=0.0
            ).to(tl.float32)
            decay = head_decay * softplus(forget + bias)
        else:
            decay = tl.load(g + token * HV + head_v).to(tl.float32)
        step = tl.load(beta + token * HV + head_v).to(tl.float32)
        state, result = gated_delta_step(
            state, q, k, v, decay, step, K**-0.5, True, decay_log is not None
        )
        tl.store(out + (token * HV + head_v) * V + v_cols, result, v_cols < V)

        # Each transition starts from the same cache precision regardless of
        # how the scheduler splits the sequence across forwards.
        stored_state = state.to(STATE_TYPE)
        state = stored_state.to(tl.float32)
        if POOL_SIZE:
            step = token - start
            if IS_VERIFY or (prefix + step + 1) % PAGE_SIZE == 0 or token + 1 == end:
                page = _page_id(
                    pages,
                    _checkpoint_index(prefix, step, PAGE_SIZE, IS_VERIFY),
                    PAGE_COUNT,
                    POOL_SIZE,
                )
                tl.store(
                    ssm_state + page * STATE_STRIDE + state_offsets,
                    stored_state,
                    (page > 0) & mask,
                )


def batch_invariant_gated_delta_rule(
    qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    lengths: torch.Tensor,
    block_map: torch.Tensor | None,
    conv_state: torch.Tensor | None,
    ssm_state: torch.Tensor | None,
    conv_dtype: torch.dtype,
    ssm_dtype: torch.dtype,
    seq_size_per_block: int,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    is_decode: bool,
    is_target_verify: bool = False,
    decay_log: torch.Tensor | None = None,
    decay_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance packed GDN or KDA requests using their RTP cache layouts."""
    dtypes = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    tokens, channels = qkv.shape
    output = qkv.new_empty((tokens, value_heads, value_dim))
    requests = cu_seqlens.numel() - 1
    if not tokens or not requests:
        return output
    if block_map is not None and block_map.stride(1) != 1:
        block_map = block_map.contiguous()
    if ssm_state is not None and (block_map is None or block_map.numel() == 0):
        raise ValueError("batch-invariant Gated DeltaNet requires a cache block table")
    page_stride = block_map.stride(0) if ssm_state is not None else 0
    layout = dict(
        PAGE_COUNT=block_map.shape[1] if ssm_state is not None else 0,
        POOL_SIZE=ssm_state.shape[0] if ssm_state is not None else 0,
        PAGE_SIZE=seq_size_per_block,
        IS_DECODE=is_decode,
        IS_VERIFY=is_target_verify,
        num_warps=1,
    )
    conv = torch.empty(qkv.shape, device=qkv.device, dtype=qkv.dtype)
    _conv1d[(requests, triton.cdiv(channels, 32))](
        qkv,
        conv_weight,
        conv,
        conv_state,
        cu_seqlens,
        lengths,
        block_map,
        tokens,
        page_stride,
        channels,
        qkv.stride(0),
        *conv_weight.stride(),
        conv_state.stride(0) if conv_state is not None else 0,
        dtypes[conv_dtype],
        conv_weight.shape[1],
        BLOCK_C=32,
        **layout,
    )
    _recurrent[(requests, value_heads, triton.cdiv(value_dim, 8))](
        conv,
        g,
        beta,
        decay_log,
        decay_bias,
        output,
        ssm_state,
        cu_seqlens,
        lengths,
        block_map,
        tokens,
        page_stride,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        ssm_state.stride(0) if ssm_state is not None else 0,
        dtypes[ssm_dtype],
        BLOCK_K=triton.next_power_of_2(key_dim),
        BLOCK_V=8,
        **layout,
    )
    return output
