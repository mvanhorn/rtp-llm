"""Qwen4 BF16 HC consumer with the existing residual/norm rounding order."""

import torch

from rtp_llm.models_py.modules.base.common.fp8_linear import dense_projection
from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual


class Qwen4GatedResidual(GatedResidual):
    def __init__(self, *args, batch_invariant=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._hc_role = str(kwargs.get("role", "PDFUSION")).rsplit(".", 1)[-1]
        self.batch_invariant = batch_invariant
        if batch_invariant:
            from .batch_invariant_projection import BatchInvariantProjection

            scales = kwargs.get("weight_scales") or (None, None, None)
            self.down_projection = BatchInvariantProjection(self.mix_down, scales[0])
            self.up_projection = BatchInvariantProjection(self.mix_up, scales[1])
            self.inject_projection = (
                BatchInvariantProjection(self.inject, scales[2])
                if self.inject is not None
                else None
            )
            self.merged_down_inject = None

    def _norm(self, hyper_input):
        if self.batch_invariant:
            from rtp_llm.models_py.triton_kernels.gated_residual.norm import (
                grouped_rms_norm_triton,
            )

            return grouped_rms_norm_triton(
                hyper_input.contiguous(),
                self.norm_gamma.contiguous(),
                self.hidden_size,
                self.norm_eps,
            )
        return super()._norm(hyper_input)

    def _forward_normed(self, hyper_input, normed, *, down_input_scale=None):
        if self.batch_invariant:
            from rtp_llm.models_py.triton_kernels.gated_residual.mix import (
                fused_mix_reduce,
            )

            down = torch.nn.functional.silu(self.down_projection(normed) / self.hc_mult)
            mixed = fused_mix_reduce(
                self.up_projection(down), normed, self.hc_mult, batch_invariant=True
            )
            gates = (
                2 * torch.sigmoid(self.inject_projection(normed) / self.hc_mult)
                if self.inject_projection is not None
                else None
            )
            return mixed, hyper_input, gates
        up = self.up_projection
        weight = self.mix_up if up is None else up.weight
        scale = None if up is None else up.scale
        if self._hc_role in (
            "PREFILL",
            "PDFUSION",
        ):
            from .hc_prefill_mix import hc_up_mix, supports_hc_prefill

            # Only replace the large Prefill shape. The native small-batch
            # projection/graph path and its independent scales remain active.
            if (
                self.hc_mult == 4
                and self.mix_down.shape == (320, 10240)
                and (up is None or up.bias is None)
                and weight.dtype == torch.float8_e4m3fn
                and normed.ndim == 2
                and normed.shape[0] >= 8192
                and supports_hc_prefill(normed, weight, scale)
            ):
                from rtp_llm.models_py.triton_kernels.gated_residual.down import (
                    separate_down_inject_epilogue,
                )

                down = (
                    dense_projection(normed, self.mix_down, self.down_projection)
                    if down_input_scale is None
                    else self.down_projection(normed, input_scale=down_input_scale)
                )
                injection = (
                    dense_projection(normed, self.inject, self.inject_projection)
                    if self.inject is not None
                    else None
                )
                mix, gates = separate_down_inject_epilogue(down, injection)
                return hc_up_mix(mix, weight, normed, scale), hyper_input, gates
        return super()._forward_normed(
            hyper_input, normed, down_input_scale=down_input_scale
        )

    def forward_with_injection(self, hyper_input, sublayer_out, inject_weights):
        # A BF16 descriptor has no FP8 projection/scale consumer. Fuse the
        # producer independently of the projection dtype, then ignore the
        # optional FP8 scale. Other layouts keep the shared implementation.
        if (
            not self.batch_invariant
            and not self._has_forward_hooks()
            and self.down_projection is None
            and self.hc_mult == 4
        ):
            from rtp_llm.models_py.triton_kernels.gated_residual.prefill_inject_norm import (
                try_inject_grouped_norm,
            )

            result = try_inject_grouped_norm(
                hyper_input,
                sublayer_out,
                inject_weights,
                self.norm_gamma,
                self.norm_eps,
            )
            if result is not None:
                residual, normed, _ = result
                return self._forward_normed(residual, normed)
        return super().forward_with_injection(hyper_input, sublayer_out, inject_weights)
