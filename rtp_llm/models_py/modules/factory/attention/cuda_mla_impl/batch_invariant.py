"""Shared NVIDIA absorbed MLA path for prefill and decode."""

import torch

from rtp_llm.models_py.kernels.cuda import batch_invariant, batch_invariant_mla
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W

from .mla_kv_cache_write_op import MlaKVCacheWriteOp
from .rope_emb_new import NewMlaRotaryEmbeddingOp


class BatchInvariantMlaAttention(MlaImplBase):
    cuda_graph_device_metadata_only = True

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: list[dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor | None,
        *,
        is_cuda_graph: bool = False,
        parallelism_config: ParallelismConfig | None = None,
    ) -> None:
        super().__init__(
            attn_configs,
            attn_inputs,
            weights,
            cos_sin_cache,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
        )
        self.page_size = attn_configs.kernel_tokens_per_block
        self.scale = (
            attn_configs.nope_head_dim + attn_configs.rope_head_dim
        ) ** -0.5 * attn_configs.softmax_extra_scale
        self.fp8_cache = attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        self.append_cache = MlaKVCacheWriteOp(attn_configs.kv_cache_dtype)
        self.cache_store = common.create_write_cache_store_impl(attn_inputs)
        self.rope = (
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache, is_neox_style=attn_configs.rope_config.is_neox_style
            )
            if cos_sin_cache is not None
            else None
        )

    @staticmethod
    def supports_config(config: AttentionConfigs) -> bool:
        return (
            config.use_mla
            and not config.use_logn_attn
            and config.sliding_window == 0
            and config.is_causal
            and config.kernel_tokens_per_block > 0
            and config.dtype in (torch.float16, torch.bfloat16)
            and config.head_num > 0
            and config.nope_head_dim > 0
            and config.v_head_dim > 0
            and 0 < config.kv_lora_rank <= 512
            and 0 <= config.rope_head_dim <= 128
            and config.rope_head_dim % 2 == 0
            and (
                not config.is_sparse
                or (
                    config.dtype == torch.bfloat16
                    and config.q_lora_rank > 0
                    and config.indexer_head_dim == 128
                    and config.indexer_topk > 0
                    and config.indexer_head_num > 0
                )
            )
            and (
                config.kv_cache_dtype == KvCacheDataType.BASE
                or (
                    config.kv_cache_dtype == KvCacheDataType.FP8
                    and config.kv_lora_rank == 512
                    and config.rope_head_dim == 64
                )
            )
        )

    @staticmethod
    def support(config: AttentionConfigs, inputs: PyAttentionInputs) -> bool:
        return (
            BatchInvariantMlaAttention.supports_config(config)
            and not inputs.headwise_config
            and inputs.context_parallel_info is None
        )

    def supports_generation_prefill_cuda_graph(self) -> bool:
        return True

    def prepare_cuda_graph(self, _inputs: PyAttentionInputs) -> None:
        # Lengths and page IDs are read from the graph-owned device inputs.
        pass

    def prepare_metadata(
        self, rows: int, kv_cache: LayerKVCache | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        inputs = self.attn_inputs
        pages = None
        if kv_cache is not None:
            pages = inputs.kv_cache_kernel_block_id_device
            if pages is None or pages.numel() == 0:
                raise ValueError("batch-invariant MLA requires a paged KV block table")
            if pages.stride(1) != 1:
                pages = pages.contiguous()
        layout = batch_invariant.attention_token_layout(
            inputs.cu_seqlens_device if inputs.is_prefill else None,
            (
                inputs.prefix_lengths_device
                if inputs.is_prefill
                else inputs.sequence_lengths_plus_1_device
            ),
            pages,
            rows,
            self.page_size,
        )
        return layout, pages

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: LayerKVCache | None,
        layer_id: int,
        topk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config, inputs = self.attn_configs, self.attn_inputs
        rows, heads, _ = q.shape
        if not rows:
            return q.new_empty((rows, heads, config.v_head_dim))
        layout, pages = self.prepare_metadata(rows, kv_cache)
        q_nope, q_rope = q.split((config.nope_head_dim, config.rope_head_dim), dim=-1)
        if self.rope is not None:
            positions = layout[:, 3].clamp_min(0)
            self.rope.forward(q_rope, k_pe, None, precomputed_pos_ids=positions)

        if kv_cache is None:
            # Cacheless model warmup supplies only the rows of the current input.
            cache = torch.cat((compressed_kv, k_pe), dim=-1).unsqueeze(0)
        else:
            cache = kv_cache.kv_cache_base
            cache_width = (
                656 // cache.element_size()
                if self.fp8_cache
                else (config.kv_lora_rank + config.rope_head_dim)
            )
            if (
                cache.shape[1:] != (self.page_size, cache_width)
                or cache.stride(2) != 1
                or (not self.fp8_cache and cache.dtype != q.dtype)
            ):
                raise ValueError("unsupported batch-invariant MLA KV cache layout")
            slots = batch_invariant_mla.cache_slots(layout, self.page_size)
            self.append_cache.append(compressed_kv, k_pe, kv_cache, slots)
        common.apply_write_cache_store(self.cache_store, inputs, kv_cache)

        weights = self.weights[layer_id]
        q_latent = batch_invariant_mla.head_matmul(q_nope, weights[W.mla_kc])
        latent_output = batch_invariant_mla.attention(
            q_latent,
            q_rope,
            cache,
            layout,
            pages,
            scale=self.scale,
            fp8_cache=self.fp8_cache and kv_cache is not None,
            topk_indices=topk_indices,
        )
        return batch_invariant_mla.head_matmul(latent_output, weights[W.mla_vc])
