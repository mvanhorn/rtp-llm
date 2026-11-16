from typing import Any, Dict, Optional

import torch
import triton.language as tl

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    get_default_config,
    invoke_fused_moe_kernel,
    moe_align_block_size_torch,
)
from rtp_llm.utils.model_weight import W

from .decode_capabilities import decode_capabilities


class TritonFusedMoeExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
        *,
        batch_invariant: bool = False,
    ):
        super().__init__(config, quant_config, weights)
        self.batch_invariant = batch_invariant

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
        self.top_k = config.moe_k

        # w1: [E, 2*inter_size, hidden_size], w2: [E, hidden_size, inter_size]
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.nvfp4 = batch_invariant and quant_config.quant_dtype == torch.uint8
        self.mxfp4 = batch_invariant and quant_config.quant_dtype == torch.int8
        self.mxfp4_activation_group_size = (
            32 if getattr(config, "model_type", "") == "deepseek_v41" else 128
        )
        self.fp8_per_tensor = (
            batch_invariant
            and quant_config.quant_dtype == torch.float8_e4m3fn
            and not quant_config.is_block_quantized
        )
        self.w1_scale = (
            weights[W.moe_s1]
            if batch_invariant
            and (quant_config.is_block_quantized or self.fp8_per_tensor)
            else None
        )
        self.w2_scale = weights[W.moe_s2] if self.w1_scale is not None else None
        if batch_invariant and (
            self.w1.dtype
            not in (
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.uint8,
                torch.int8,
            )
            or self.w2.dtype != self.w1.dtype
            or (self.w1.dtype == torch.uint8) != self.nvfp4
            or (self.w1.dtype == torch.int8) != self.mxfp4
            or (self.w1.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8))
            != (self.w1_scale is not None)
        ):
            raise ValueError(
                "RTP_BATCH_INVARIANT requires FP16, BF16, FP8, NVFP4 or MXFP4 experts"
            )
        self.nvfp4_scales = ()
        self.fp8_input_scales = ()
        if self.fp8_per_tensor:
            self.w1_scale = self.w1_scale.reshape(self.w1.size(0), 1, 1)
            self.w2_scale = self.w2_scale.reshape(self.w2.size(0), 1, 1)
            self.fp8_input_scales = (
                weights.get(W.moe_w1_input_sr),
                weights.get(W.moe_w2_input_sr),
            )
            if any(
                scale is not None and scale.numel() != 1
                for scale in self.fp8_input_scales
            ):
                raise ValueError(
                    "batch-invariant per-tensor FP8 requires scalar input scales"
                )
        if self.nvfp4:
            from rtp_llm.models_py.kernels.cuda.batch_invariant_nvfp4 import (
                validate_nvfp4_weight,
            )

            self.nvfp4_scales = (
                (weights[W.moe_w1_i_s].contiguous(), weights[W.moe_w1_s2].contiguous()),
                (weights[W.moe_w2_i_s].contiguous(), weights[W.moe_w2_s2].contiguous()),
            )
            for weight, scale, (input_scale, weight_scale) in zip(
                (self.w1, self.w2), (self.w1_scale, self.w2_scale), self.nvfp4_scales
            ):
                validate_nvfp4_weight(weight, scale, weight_scale, input_scale)
        if self.mxfp4:
            self.w1_scale = self._mxfp4_weight_scales(self.w1, self.w1_scale)
            self.w2_scale = self._mxfp4_weight_scales(self.w2, self.w2_scale)
        self.E = self.w1.size(0)
        self.N = self.w1.size(1)  # 2 * inter_size
        self.K = self.w1.size(2) * (2 if self.nvfp4 or self.mxfp4 else 1)  # hidden_size
        self.inter_size = self.N // 2
        if (
            batch_invariant
            and quant_config.is_block_quantized
            and not (self.nvfp4 or self.mxfp4)
            and (self.K % 128 or self.inter_size % 128)
        ):
            raise ValueError(
                "batch-invariant block FP8 experts require K divisible by 128"
            )
        capabilities = decode_capabilities(
            self.w1,
            self.top_k,
            self.ep_size,
            getattr(getattr(config, "model_config", None), "model_type", ""),
        )
        self._decode_alignment = capabilities.alignment
        self._decode_tiles = capabilities.tuned_tiles
        self._small_moe_gemv = capabilities.small_gemv

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    @staticmethod
    def _mxfp4_weight_scales(weight, scales):
        if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError("batch-invariant MXFP4 requires UE8M0 expert scales")
        if scales.shape != (*weight.shape[:-1], weight.size(-1) // 16):
            raise ValueError("batch-invariant MXFP4 requires row-major group-32 scales")
        return scales.view(torch.uint8)

    def _quantize_expert_input(
        self, x: torch.Tensor, weight_scales: torch.Tensor | None, projection: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if weight_scales is None:
            return x, {}
        if self.nvfp4:
            input_scale, weight_scale = self.nvfp4_scales[projection]
            return x, {
                "B_scale": weight_scales,
                "nvfp4_input_scale": input_scale,
                "nvfp4_weight_scale": weight_scale,
            }
        if self.fp8_per_tensor:
            from rtp_llm.models_py.kernels.cuda.batch_invariant import quantize_fp8_rows

            quantized, scales = quantize_fp8_rows(x, self.fp8_input_scales[projection])
            return quantized, {
                "A_scale": scales,
                "B_scale": weight_scales,
                "per_tensor_fp8": True,
            }
        from rtp_llm.models_py.kernels.cuda.batch_invariant import quantize_fp8_blocks

        if self.mxfp4:
            group_size = self.mxfp4_activation_group_size
            eps = torch.finfo(torch.float32).tiny if group_size == 32 else 1e-4
            quantized, scales = quantize_fp8_blocks(x, True, group_size, eps=eps)
            return quantized, {
                "A_scale": scales,
                "B_scale": weight_scales,
                "mxfp4": True,
                "mxfp4_activation_group_size": group_size,
            }
        packed = weight_scales.dtype == torch.int32
        quantized, scales = quantize_fp8_blocks(x, packed)
        return quantized, {
            "A_scale": scales,
            "B_scale": weight_scales,
            "block_shape": self.quant_config.block_shape,
            "scale_ue8m0": packed,
        }

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        hidden_states = payload.expert_x  # [M, K]
        topk_ids = payload.expert_topk_ids  # [M, top_k]
        topk_weights = payload.expert_topk_weights  # [M, top_k]

        assert topk_ids is not None
        assert topk_weights is not None

        if (
            not self.batch_invariant
            and self._small_moe_gemv
            and activation in ("silu", "SiGLU")
            and expert_map is None
            and a2_scale is None
            and not apply_router_weight_on_input
            and not extra_expert_args
        ):
            from rtp_llm.models_py.triton_kernels.moe.small_moe_gemv import (
                small_moe_gemv,
            )

            output = small_moe_gemv(
                hidden_states, self.w1, self.w2, topk_ids, topk_weights
            )
            if output is not None:
                return CombineForwardPayload(fused_expert_output=output)

        if self.ep_size > 1:
            local_ids = topk_ids - self.start_expert_id
            local_mask = (local_ids >= 0) & (local_ids < self.num_experts_per_partition)
            topk_ids = local_ids.clamp(min=0, max=self.num_experts_per_partition - 1)
            topk_weights = topk_weights * local_mask

        M, K = hidden_states.shape
        top_k = topk_ids.size(1)
        device = hidden_states.device
        dtype = hidden_states.dtype
        compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

        # Config selection for both GEMMs
        if self.batch_invariant:
            config1 = config2 = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": (
                    128
                    if self.quant_config.is_block_quantized
                    and not (self.nvfp4 or self.mxfp4)
                    else 32
                ),
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
            }
        else:
            config1 = get_default_config(M, self.E, self.N, K, top_k)
            config2 = get_default_config(M, self.E, K, self.inter_size, top_k)
        if self._decode_tiles and not self.batch_invariant:
            from rtp_llm.models_py.triton_kernels.moe.decode_alignment import (
                sm120_decode_gemm_configs,
            )

            config1, config2 = sm120_decode_gemm_configs(M, config1, config2)

        # Flat routing tensors for the kernel
        flat_topk_weights = topk_weights.view(-1)  # [M * top_k]
        flat_topk_ids = topk_ids.view(-1)  # [M * top_k]

        # Token-expert alignment: compute once if block sizes match, else twice
        block_m1 = config1["BLOCK_SIZE_M"]
        block_m2 = config2["BLOCK_SIZE_M"]
        align = moe_align_block_size_torch
        if (
            not self.batch_invariant
            and self._decode_alignment
            and 0 < M <= 32
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
        ):
            from rtp_llm.models_py.triton_kernels.moe.decode_alignment import (
                moe_align_decode,
            )

            align = moe_align_decode
        sorted_token_ids, expert_ids, num_tokens_post_padded = align(
            topk_ids, block_m1, self.E
        )
        if block_m1 == block_m2:
            sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
            )
        else:
            sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = align(
                topk_ids, block_m2, self.E
            )

        # GEMM1: hidden_states @ w1.T → intermediate1 [M*top_k, 2*inter]
        intermediate1 = torch.empty(M * top_k, self.N, device=device, dtype=dtype)
        expert_input, scale_options = self._quantize_expert_input(
            hidden_states, self.w1_scale, 0
        )
        invoke_fused_moe_kernel(
            expert_input,
            self.w1,
            intermediate1,
            flat_topk_weights,
            flat_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=config1,
            compute_type=compute_type,
            **scale_options,
        )

        # Activation: silu_and_mul(intermediate1) → intermediate2 [M*top_k, inter]
        intermediate2 = torch.empty(
            M * top_k, self.inter_size, device=device, dtype=dtype
        )
        if self.batch_invariant:
            silu_and_mul(
                intermediate2,
                intermediate1,
                gate_first=getattr(self.config, "moe_w1_layout", "") == "gate_up",
                clamp_limit=getattr(self.config, "swiglu_limit", 0.0),
            )
        else:
            silu_and_mul(intermediate2, intermediate1)

        # GEMM2: intermediate2 @ w2.T → out [M*top_k, K]
        out = torch.empty(M * top_k, K, device=device, dtype=dtype)
        expert_input, scale_options = self._quantize_expert_input(
            intermediate2, self.w2_scale, 1
        )
        invoke_fused_moe_kernel(
            expert_input,
            self.w2,
            out,
            flat_topk_weights,
            flat_topk_ids,
            sorted_token_ids2,
            expert_ids2,
            num_tokens_post_padded2,
            mul_routed_weight=True,
            top_k=1,
            config=config2,
            compute_type=compute_type,
            **scale_options,
        )

        # Sum reduce over top_k: out is [M*top_k, K], reshape and sum
        routed = out.view(M, top_k, K)
        if self.batch_invariant:
            output = routed[:, 0].float()
            for expert_slot in range(1, top_k):
                output = output + routed[:, expert_slot]
            output = output.to(dtype)
        else:
            output = routed.sum(dim=1)

        return CombineForwardPayload(fused_expert_output=output)
