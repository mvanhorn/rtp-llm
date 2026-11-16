"""Fixed-shape reductions for Qwen's batched NVIDIA vision towers."""

import torch
from torch import nn

from rtp_llm.models_py.kernels.cuda import batch_invariant
from rtp_llm.models_py.kernels.cuda.batch_invariant_norm import layer_norm


class _Linear(nn.Module):
    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.weight = layer.weight
        self.bias = layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return batch_invariant.linear(x, self.weight, self.bias)


class _LayerNorm(nn.Module):
    def __init__(self, layer: nn.LayerNorm):
        super().__init__()
        self.weight = layer.weight
        self.bias = layer.bias
        self.eps = layer.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.weight, self.bias, self.eps)


class _PatchProjection(nn.Module):
    """The vision tower feeds one complete, non-overlapping patch per row."""

    def __init__(self, layer: nn.Conv3d):
        super().__init__()
        if (
            layer.groups != 1
            or layer.stride != layer.kernel_size
            or layer.padding != (0, 0, 0)
            or layer.dilation != (1, 1, 1)
        ):
            raise ValueError("unsupported batch-invariant vision patch projection")
        self.weight = layer.weight
        self.bias = layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != self.weight.shape[1:]:
            raise ValueError(
                "batch-invariant patch projection expects complete patches"
            )
        result = batch_invariant.linear(x.flatten(1), self.weight.flatten(1), self.bias)
        return result[:, :, None, None, None]


def _vision_attention(
    _module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seq_lens_q: torch.Tensor,
    scaling: float,
    **_kwargs,
) -> tuple[torch.Tensor, None]:
    # Transformers supplies [1, heads, packed_tokens, head_dim] and
    # cumulative frame offsets after applying its own vision RoPE.
    query, key, value = (x[0].transpose(0, 1) for x in (query, key, value))
    rows, heads, head_dim = query.shape
    layout = batch_invariant.attention_token_layout(
        cu_seq_lens_q,
        torch.zeros_like(cu_seq_lens_q[:-1]),
        None,
        rows,
        1,
    )
    result = batch_invariant.attention(
        query, key, value, layout, None, None, causal=False, scale=scaling
    )
    return result.view(1, rows, heads, head_dim), None


def enable_qwen_vision_batch_invariant(vision: nn.Module) -> None:
    from transformers import AttentionInterface

    # Transformers selects its packed vision branch for flash implementations.
    implementation = "rtp_batch_invariant_flash"
    AttentionInterface.register(implementation, _vision_attention)
    vision.config._attn_implementation = implementation
    replacements = {
        nn.Conv3d: _PatchProjection,
        nn.Linear: _Linear,
        nn.LayerNorm: _LayerNorm,
    }
    for module in vision.modules():
        for name, child in list(module.named_children()):
            replacement = replacements.get(type(child))
            if replacement is not None:
                setattr(module, name, replacement(child))
