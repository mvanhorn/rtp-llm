import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
    roundtrip_swa_kv,
)
from rtp_llm.models_py.modules.dsv4.fp8.batch_invariant_attention import (
    indexed_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_fp4_triton, _v41_swa_triton
from rtp_llm.models_py.modules.dsv41.fp8._v41_batch_invariant_indexer import score_paged

_CUDA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (8, 9)
)


@unittest.skipUnless(_CUDA, "requires NVIDIA SM89+")
class BatchInvariantIndexedAttentionTest(unittest.TestCase):
    def test_prefill_and_dual_pool_decode(self):
        self._run_dual_pool(v41=False)

    def test_v41_prefill_and_dual_pool_decode(self):
        self._run_dual_pool(v41=True)

    def _run_dual_pool(self, *, v41):
        torch.manual_seed(314159)
        device, dtype = "cuda", torch.bfloat16
        heads = 18
        requests = 3
        dims = 512
        swa = torch.randn((64, dims), device=device, dtype=dtype) * 0.25
        cmp = torch.randn((64, dims), device=device, dtype=dtype) * 0.25
        query = torch.randn((requests, heads, dims), device=device, dtype=dtype)
        sink = torch.randn(heads, device=device)

        slots = torch.arange(swa.size(0), device=device, dtype=torch.int64)
        if v41:
            swa_pool = torch.empty((2, 32, 528), dtype=torch.uint8, device=device)
            cmp_pool = torch.empty((4, 16, 288), dtype=torch.uint8, device=device)
            _v41_swa_triton.quantize_and_insert_swa_k_cache(swa, swa_pool, slots)
            _v41_fp4_triton.quantize_and_insert_k_cache_fp4(cmp, cmp_pool, slots)
            swa = _v41_swa_triton.roundtrip_swa_kv(swa)
            cmp = _v41_fp4_triton.dequantize_k_cache_slots_fp4(cmp_pool, slots)
        else:
            swa_pool = torch.empty((2, 32, 584), dtype=torch.uint8, device=device)
            cmp_pool = torch.empty((4, 16, 584), dtype=torch.uint8, device=device)
            quantize_and_insert_k_cache(swa, swa_pool, slots)
            quantize_and_insert_k_cache(cmp, cmp_pool, slots)
            swa, cmp = roundtrip_swa_kv(swa), roundtrip_swa_kv(cmp)

        swa_indices = torch.full((requests, 40), -1, device=device, dtype=torch.int32)
        cmp_indices = torch.full((requests, 36), -1, device=device, dtype=torch.int32)
        swa_counts = [39, 5, 0]
        cmp_counts = [11, 34, 0]
        prefill_indices = torch.full(
            (requests, 76), -1, device=device, dtype=torch.int32
        )
        for row, (swa_count, cmp_count) in enumerate(zip(swa_counts, cmp_counts)):
            swa_indices[row, :swa_count] = slots[row : row + swa_count]
            cmp_indices[row, :cmp_count] = slots[row : row + cmp_count]
            prefill_indices[row, :cmp_count] = cmp_indices[row, :cmp_count] + swa.size(
                0
            )
            prefill_indices[row, cmp_count : cmp_count + swa_count] = swa_indices[
                row, :swa_count
            ]
        swa_sizes = torch.tensor(swa_counts, device=device, dtype=torch.int32)
        cmp_sizes = torch.tensor(cmp_counts, device=device, dtype=torch.int32)
        prefill_kv = torch.cat((swa, cmp))
        scale = dims**-0.5
        prefill = indexed_attention(
            query,
            prefill_kv[:, None],
            prefill_indices,
            swa_sizes + cmp_sizes,
            sink,
            scale,
        )

        op = SparseAttnV4DecodeFp8Op(heads, dims, scale, batch_invariant=True)

        def decode(q, sw, cp, sw_len, cp_len):
            return op.forward(
                q[:, None],
                swa_pool,
                sink,
                sw[:, None],
                None,
                topk_length=sw_len,
                extra_k_cache=cmp_pool,
                extra_topk_idxs=cp[:, None],
                extra_topk_length=cp_len,
            )[:, 0]

        mixed = decode(query, swa_indices, cmp_indices, swa_sizes, cmp_sizes)
        self.assertTrue(torch.equal(prefill, mixed))
        self.assertEqual(torch.count_nonzero(mixed[-1]).item(), 0)
        one = decode(
            *[x[:1] for x in (query, swa_indices, cmp_indices, swa_sizes, cmp_sizes)]
        )
        self.assertTrue(torch.equal(prefill[:1], one))

        first_keys = prefill_kv[prefill_indices[0, : swa_counts[0] + cmp_counts[0]]]
        scores = query[0].float() @ first_keys.float().T * scale
        prob = torch.softmax(torch.cat((scores, sink[:, None]), dim=1), dim=1)[:, :-1]
        expected = (prob @ first_keys.float()).to(dtype)
        torch.testing.assert_close(prefill[0], expected, atol=0.002, rtol=0.02)

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = decode(query, swa_indices, cmp_indices, swa_sizes, cmp_sizes)
        for x in query, swa_indices, cmp_indices, swa_sizes, cmp_sizes:
            x.copy_(x.flip(0))
        graph.replay()
        self.assertTrue(torch.equal(captured, prefill.flip(0)))

    def test_v41_fp4_indexer_half_pages(self):
        torch.manual_seed(13)
        device, dtype = "cuda", torch.bfloat16
        heads, rows, logical, physical, head_dim = 32, 9, 16, 32, 128
        key0 = torch.randn((logical * 2, head_dim), dtype=dtype, device=device)
        key1 = torch.randn_like(key0)
        query = torch.randn((rows, heads, head_dim), dtype=dtype, device=device)
        weights = torch.randn((rows, heads), device=device)
        pages = torch.tensor([[4, 1], [3, 2]], dtype=torch.int32, device=device)
        pool = torch.empty((5, physical, 68), dtype=torch.uint8, device=device)
        offsets = torch.arange(logical, device=device)
        for keys, table in zip((key0, key1), pages):
            slots = (table[:, None] * physical + offsets).reshape(-1)
            _v41_fp4_triton.quantize_indexer_k_fp4(keys, slots, pool)
        q, q_scale = _v41_fp4_triton.quantize_rows_fp4(query)
        lengths = torch.tensor([31, 17, 2, 6, 5, 16, 28, 9, 8], device=device)
        requests = torch.arange(rows, device=device) % 2
        score = score_paged(
            q, q_scale, weights, pool, pages, lengths, requests, logical
        )
        order = torch.arange(rows - 1, -1, -1, device=device)
        mixed = score_paged(
            q[order],
            q_scale[order],
            weights[order],
            pool,
            pages.flip(0),
            lengths[order],
            1 - requests[order],
            logical,
        )
        self.assertTrue(torch.equal(score[order], mixed))
        for row in range(rows):
            single = score_paged(
                q[row : row + 1],
                q_scale[row : row + 1],
                weights[row : row + 1],
                pool,
                pages,
                lengths[row : row + 1],
                requests[row : row + 1],
                logical,
            )
            self.assertTrue(torch.equal(score[row : row + 1], single))
            self.assertTrue(torch.isneginf(score[row, lengths[row] :]).all().item())


if __name__ == "__main__":
    unittest.main()
