import torch
from torch import nn


class BatchInvariantProjection(nn.Module):
    """Fixed GEMM for Qwen4's [out,in] dense and per-tensor-FP8 weights."""

    def __init__(self, weight, scale=None, bias=None):
        super().__init__()
        self.weight = weight
        self.scale = (
            None if scale is None else scale.to(weight.device, torch.float32).reshape(1)
        )
        self.bias = bias
        self.compute_weight = None
        self.use_scaled_mm = False
        if self.scale is not None and any(size % 16 for size in weight.shape):
            self.compute_weight = (weight.float() * self.scale).to(torch.bfloat16)

    def maybe_cache_quant_scale(self, max_len):
        pass

    def forward_with_bias_gelu(self, x):
        return torch.nn.functional.gelu(self(x))

    def forward(self, x):
        from rtp_llm.models_py.kernels.cuda.batch_invariant import (
            fp8_per_tensor_linear,
            linear,
        )

        if self.compute_weight is not None:
            return linear(x, self.compute_weight, self.bias)
        if self.scale is not None:
            return fp8_per_tensor_linear(x, self.weight, self.scale, bias=self.bias)
        return linear(x, self.weight, self.bias)
