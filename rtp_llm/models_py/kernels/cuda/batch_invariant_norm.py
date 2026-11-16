"""Row-wise normalization with fixed reduction geometry on NVIDIA."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalize(
    x,
    weight,
    bias,
    output,
    size: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, block)
    valid = col < size
    offset = row * size + col
    values = tl.load(x + offset, valid, other=0).to(tl.float32)
    if weight is None:
        l2 = tl.sqrt(tl.sum(values * values, 0))
        result = values / tl.maximum(l2, eps)
    else:
        mean = tl.sum(values, 0) / size
        centered = tl.where(valid, values - mean, 0.0)
        scale = tl.rsqrt(tl.sum(centered * centered, 0) / size + eps)
        gamma = tl.load(weight + col, valid, other=0).to(tl.float32)
        result = (centered * scale) * gamma
        if bias is not None:
            result += tl.load(bias + col, valid, other=0).to(tl.float32)
    tl.store(output + offset, result, valid)


def _launch(
    x: torch.Tensor,
    eps: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x.contiguous()
    if x.numel() == 0:
        return x
    size = x.size(-1)
    output = torch.empty_like(x)
    _normalize[(x.numel() // size,)](
        x, weight, bias, output, size, eps, triton.next_power_of_2(size)
    )
    return output


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return _launch(x, eps)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return _launch(x, eps, weight, bias)
