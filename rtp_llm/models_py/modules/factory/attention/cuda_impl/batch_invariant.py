"""Shared NVIDIA attention path for batch-invariant prefill and decode."""

import torch

from rtp_llm.models_py.kernels.cuda import batch_invariant
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import (
    AttentionConfigs,
    KvCacheDataType,
    RopeConfig,
    RopeStyle,
    get_rope_cache,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


def _mrope_sections(rope: RopeConfig) -> tuple[int, int, int]:
    return rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3


def _supports_rope(rope: RopeConfig, head_dim: int) -> bool:
    if rope.offset != 0:
        return False
    if rope.style == RopeStyle.No:
        return rope.index_factor == 1
    if (
        not 0 < rope.dim <= head_dim
        or rope.dim % 2
        or not rope.is_neox_style
        or rope.scale < 1
    ):
        return False
    if rope.style == RopeStyle.Base:
        return rope.index_factor == 1
    if rope.style != RopeStyle.Mrope or rope.index_factor != 3:
        return False
    pairs = rope.dim // 2
    sections = _mrope_sections(rope)
    return (
        min(sections) >= 0
        and sum(sections) == pairs
        and (
            not rope.mrope_interleaved
            or (sections[1] <= (pairs + 1) // 3 and sections[2] <= pairs // 3)
        )
    )


class BatchInvariantAttention(FMHAImplBase):
    cuda_graph_device_metadata_only = True

    def __init__(
        self,
        config: AttentionConfigs,
        inputs: PyAttentionInputs,
        cos_sin: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self.inputs = inputs
        self.scale = (
            config.softmax_extra_scale / config.q_scaling * config.size_per_head**-0.5
        )
        self.cache_store = common.create_write_cache_store_impl(inputs)
        self.cos_sin = cos_sin
        self.alibi_slopes = alibi_slopes
        rope = config.rope_config
        self.mrope_sections = (
            _mrope_sections(rope) if rope.style == RopeStyle.Mrope else None
        )
        self.mrope_interleaved = rope.mrope_interleaved

    @staticmethod
    def create_rope_cache(
        config: AttentionConfigs, device: torch.device
    ) -> torch.Tensor:
        rope = config.rope_config
        if rope.style == RopeStyle.Mrope:
            # MRoPE selects a position per rotary pair from the same base
            # frequency table. The layout of the axes lives in the kernel.
            base = RopeConfig()
            base.style = RopeStyle.Base
            base.dim = rope.dim
            base.base = rope.base
            base.scale = rope.scale
            rope = base
        with torch.cuda.device(device):
            return get_rope_cache(rope, config.max_seq_len, interleave=False)

    @staticmethod
    def supports_config(
        config: AttentionConfigs, *, allow_sparse: bool = False
    ) -> bool:
        return (
            not config.use_mla
            and (allow_sparse or not config.is_sparse)
            and not config.use_logn_attn
            and config.sliding_window >= 0
            and config.kernel_tokens_per_block > 0
            and config.kv_cache_dtype in (KvCacheDataType.BASE, KvCacheDataType.FP8)
            and config.dtype in (torch.float16, torch.bfloat16)
            and 0 < config.size_per_head <= 256
            and _supports_rope(config.rope_config, config.size_per_head)
            and config.head_num > 0
            and 0 < config.kv_head_num
            and config.head_num % config.kv_head_num == 0
        )

    @staticmethod
    def support(config: AttentionConfigs, inputs: PyAttentionInputs) -> bool:
        return (
            BatchInvariantAttention.supports_config(config)
            and not inputs.headwise_config
            and inputs.context_parallel_info is None
        )

    def supports_generation_prefill_cuda_graph(self) -> bool:
        return True

    def prepare_cuda_graph(self, _inputs: PyAttentionInputs) -> None:
        # The runner updates the graph-owned length and page-table tensors in place.
        pass

    def prepare_qkv(self, qkv: torch.Tensor, kv_cache: LayerKVCache | None):
        config, inputs = self.config, self.inputs
        qkv = qkv.contiguous()
        rows = qkv.shape[0]
        query, key, value = qkv.view(
            rows, config.head_num + 2 * config.kv_head_num, config.size_per_head
        ).split((config.head_num, config.kv_head_num, config.kv_head_num), dim=1)
        pages = None
        cache = None
        scales = None
        if kv_cache is not None:
            if not config.is_causal:
                raise ValueError(
                    "batch-invariant noncausal attention requires packed inputs"
                )
            pages = inputs.kv_cache_kernel_block_id_device
            if pages is None or pages.numel() == 0:
                raise ValueError(
                    "batch-invariant paged attention requires a block table"
                )
            cache = common.reshape_paged_kv_cache(
                kv_cache.kv_cache_base,
                config.kv_head_num,
                config.kernel_tokens_per_block,
                config.size_per_head,
            )
            expected_dtype = (
                torch.float8_e4m3fn
                if config.kv_cache_dtype == KvCacheDataType.FP8
                else qkv.dtype
            )
            if cache.dtype != expected_dtype or cache.shape[1:] != (
                2,
                config.kv_head_num,
                config.kernel_tokens_per_block,
                config.size_per_head,
            ):
                raise ValueError("unsupported batch-invariant KV cache layout")
            if pages.stride(1) != 1:
                pages = pages.contiguous()
        prefixes_or_lengths = (
            inputs.prefix_lengths_device
            if inputs.is_prefill
            else inputs.sequence_lengths_plus_1_device
        )
        layout = batch_invariant.attention_token_layout(
            inputs.cu_seqlens_device if inputs.is_prefill else None,
            prefixes_or_lengths,
            pages,
            rows,
            config.kernel_tokens_per_block,
        )
        if self.cos_sin is not None:
            batch_invariant.apply_rope(
                query,
                key,
                layout,
                self.cos_sin,
                inputs.combo_position_ids,
                mrope_sections=self.mrope_sections,
                mrope_interleaved=self.mrope_interleaved,
            )
        if cache is not None:
            if cache.dtype == torch.float8_e4m3fn:
                scales = kv_cache.kv_scale_base
                if scales is not None and not scales.numel():
                    scales = None
                if scales is not None and (
                    scales.dtype != torch.float32
                    or not scales.is_contiguous()
                    or scales.numel()
                    != cache.size(0)
                    * 2
                    * config.kv_head_num
                    * config.kernel_tokens_per_block
                ):
                    raise ValueError("unsupported batch-invariant FP8 KV scale layout")
            batch_invariant.append_kv(key, value, cache, layout, scales)
        common.apply_write_cache_store(self.cache_store, inputs, kv_cache)
        return query, key, value, layout, cache, pages, scales

    def forward(
        self, qkv: torch.Tensor, kv_cache: LayerKVCache | None, layer_idx: int = 0
    ) -> torch.Tensor:
        query, key, value, layout, cache, pages, scales = self.prepare_qkv(
            qkv, kv_cache
        )
        return batch_invariant.attention(
            query,
            key,
            value,
            layout,
            cache,
            pages,
            causal=self.config.is_causal,
            scale=self.scale,
            sliding_window=self.config.sliding_window,
            kv_scales=scales,
            alibi_slopes=self.alibi_slopes,
        )
