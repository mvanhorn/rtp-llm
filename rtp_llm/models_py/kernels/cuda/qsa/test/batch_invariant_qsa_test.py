import unittest

import torch

from rtp_llm.models_py.kernels.cuda.qsa.indexer_decode import (
    _write_decode_key,
    compress_prefill_keys,
)
from rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score import (
    qsa_paged_indexer_score,
)
from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import (
    batch_invariant_sparse_paged_gqa,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "requires CUDA")
class BatchInvariantQsaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def ints(self, values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    def bf16(self, *shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    def assert_bits(self, left, right):
        self.assertTrue(
            torch.equal(
                left.contiguous().view(torch.uint8),
                right.contiguous().view(torch.uint8),
            )
        )

    def test_prefill_and_decode_compression(self):
        history = self.bf16(3, 16, 128)
        gamma = self.bf16(128)
        angles = self.bf16(3, 4, 64)
        cos, sin = angles.cos(), angles.sin()
        prefill = compress_prefill_keys(
            history.view(-1, 4, 128).float(),
            cos.flatten(0, 1),
            sin.flatten(0, 1),
            gamma,
            1e-6,
        ).view(3, 4, 128)
        kv = self.bf16(5, 32, 128).zero_()
        state = self.bf16(5, 8, 128).float().zero_()
        pages = self.ints([[1], [3], [2]])
        for pos in range(16):
            raw = history[:, pos].contiguous()
            _write_decode_key[(3,)](
                raw,
                self.ints([pos] * 3),
                None,
                cos[:, pos // 4].contiguous(),
                sin[:, pos // 4].contiguous(),
                gamma,
                kv,
                pages,
                state,
                pages,
                KV_TABLE_WIDTH=1,
                STATE_TABLE_WIDTH=1,
                KV_POOL_BLOCKS=5,
                STATE_POOL_BLOCKS=5,
                KV_TOKENS_PER_BLOCK=128,
                STATE_TOKENS_PER_BLOCK=128,
                KV_ENTRIES_PER_BLOCK=32,
                D=128,
                ROTARY=64,
                EPS=1e-6,
                HAS_VALID=False,
                num_warps=4,
            )
        self.assert_bits(prefill, kv[[1, 3, 2], :4])
        self.assert_bits(history[:, -8:].float(), state[[1, 3, 2]])

    def test_fixed_score_and_selection_with_different_column_counts(self):
        q = self.bf16(1, 19, 4, 128)
        kv = self.bf16(96, 128)
        w = torch.full((19, 4), 128**-0.5, dtype=torch.float32, device="cuda")
        lengths = self.ints([[5] * 19])
        single = qsa_paged_indexer_score(
            q[:, 18:],
            w[18:],
            kv,
            self.ints([[1]]),
            lengths[:, 18:],
            block_size=32,
            max_ctx_len=5,
            batch_invariant=True,
        )
        full = qsa_paged_indexer_score(
            q,
            w,
            kv,
            self.ints([[1, 2]]),
            lengths,
            block_size=32,
            max_ctx_len=63,
            batch_invariant=True,
        )
        self.assert_bits(single[0], full[-1, :5])
        for width in (5, 63):
            logits = full.new_zeros((1, width))
            selected = select_qsa_paged_tokens(
                logits,
                self.ints([21]),
                compress_ratio=4,
                token_budget=8,
                batch_invariant=True,
            )
            torch.testing.assert_close(
                selected, self.ints([[0, 1, 2, 3, 4, 5, 6, 7, 20, -1, -1]])
            )

    def test_sparse_paged_attention_with_reordered_graph_queries(self):
        query = self.bf16(3, 2, 256)
        cache = self.bf16(4, 2, 1, 128, 256)
        selected = self.ints([list(range(73)), list(range(73)), list(range(73))])
        layout = self.ints([[0, 0, 1, 74, 3], [0, 0, 1, 74, 3], [1, 1, 3, 74, 1]])
        pages = self.ints([[3], [1]])
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(cache=dtype):
                storage = cache.to(dtype)
                scales = (
                    None
                    if dtype == torch.bfloat16
                    else torch.full(
                        (4, 2, 1, 128), 0.75, dtype=torch.float32, device="cuda"
                    )
                )
                args = dict(scale=256**-0.5, kv_scale=scales)
                ref = batch_invariant_sparse_paged_gqa(
                    query[:1], storage, pages[:1], layout[:1], selected[:1], **args
                )
                graph = torch.cuda.CUDAGraph()
                batch_invariant_sparse_paged_gqa(
                    query, storage, pages, layout, selected, **args
                )
                with torch.cuda.graph(graph):
                    output = batch_invariant_sparse_paged_gqa(
                        query, storage, pages, layout, selected, **args
                    )
                order = self.ints([2, 1, 0]).long()
                query.copy_(query[order])
                layout.copy_(layout[order])
                graph.replay()
                self.assert_bits(ref, output[-1:])
                query.copy_(query[order])
                layout.copy_(layout[order])


if __name__ == "__main__":
    unittest.main()
