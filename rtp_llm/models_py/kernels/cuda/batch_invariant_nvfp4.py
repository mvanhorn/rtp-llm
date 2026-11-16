"""E2M1 operands with ModelOpt's per-group E4M3 and outer FP32 scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def nvfp4_activation(x, global_scale, rows: tl.constexpr, width: tl.constexpr):
    grouped = x.to(tl.float32).reshape((rows, width // 16, 16))
    peak = tl.max(tl.abs(grouped), axis=2)
    scale = tl.minimum(peak / (6.0 * global_scale), 448.0)
    scale = scale.to(tl.float8e4nv).to(tl.float32)
    magnitude = (
        tl.abs(grouped) / tl.where(scale > 0, scale * global_scale, 1.0)[:, :, None]
    )
    # E2M1 round-to-nearest-even at each of the seven representable midpoints.
    q = tl.where(magnitude > 0.25, 0.5, 0.0)
    q = tl.where(magnitude >= 0.75, 1.0, q)
    q = tl.where(magnitude > 1.25, 1.5, q)
    q = tl.where(magnitude >= 1.75, 2.0, q)
    q = tl.where(magnitude > 2.5, 3.0, q)
    q = tl.where(magnitude >= 3.5, 4.0, q)
    q = tl.where(magnitude > 5.0, 6.0, q)
    return (
        (tl.where(grouped < 0, -q, q) * scale[:, :, None])
        .reshape((rows, width))
        .to(tl.float16)
    )


@triton.jit
def e2m1_weight(packed, k, n, stride_wn, stride_wk, valid):
    byte = tl.load(
        packed + k[:, None] // 2 * stride_wk + n[None, :] * stride_wn,
        valid,
        other=0,
    ).to(tl.int32)
    code = (byte >> (4 * (k[:, None] % 2))) & 15
    magnitude = code & 7
    q = tl.where(
        magnitude < 4,
        magnitude * 0.5,
        (2 + (magnitude & 1)) << (magnitude >= 6).to(tl.int32),
    )
    return tl.where(code < 8, q, -q)


@triton.jit
def nvfp4_weight(
    packed, scales, k, n, stride_wn, stride_wk, stride_sn, stride_sk, valid
):
    q = e2m1_weight(packed, k, n, stride_wn, stride_wk, valid)
    scale = tl.load(
        scales + k[:, None] // 16 * stride_sk + n[None, :] * stride_sn,
        valid,
        other=0.0,
    ).to(tl.float32)
    return (q * scale).to(tl.float16)


def validate_nvfp4_weight(weight, scales, weight_global_scale, input_global_scale):
    if (
        weight.dtype != torch.uint8
        or weight.size(-1) % 8
        or scales.dtype != torch.float8_e4m3fn
        or scales.shape != (*weight.shape[:-1], weight.size(-1) // 8)
        or weight_global_scale.numel()
        not in (1, weight.size(0) if weight.ndim == 3 else 1)
        or input_global_scale.numel()
        not in (1, weight.size(0) if weight.ndim == 3 else 1)
        or not bool(torch.isfinite(weight_global_scale).all())
        or not bool((weight_global_scale >= 0).all())
        or not bool(torch.isfinite(input_global_scale).all())
        or not bool((input_global_scale > 0).all())
    ):
        raise ValueError(
            "batch-invariant NVFP4 requires row-major ModelOpt weights and scales"
        )
