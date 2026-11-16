import unittest

import torch

from rtp_llm.models_py.kernels.cuda import batch_invariant, batch_invariant_mla
from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
    packed_fp8_paged_mqa_topk,
)

_CUDA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (8, 0)
)


@unittest.skipUnless(_CUDA, "requires NVIDIA SM80+")
class BatchInvariantMlaTest(unittest.TestCase):
    def _run_case(self, dtype, fp8_cache, sparse=False):
        torch.manual_seed(10493)
        page_size, rows, heads = 32, 42, 20
        page_ids = (4, 1)
        rank, nope_dim, rope_dim, v_dim = 512, 128, 64, 128

        def rand(shape):
            return torch.randn(shape, device="cuda", dtype=dtype) * 0.25

        def meta(values):
            return torch.tensor(values, device="cuda", dtype=torch.int32)

        q = rand((rows, heads, nope_dim + rope_dim))
        latent = rand((rows, rank))
        rope_key = rand((rows, rope_dim))
        kc = rand((heads, nope_dim, rank))
        vc = rand((heads, rank, v_dim))
        if fp8_cache:
            scales = latent.float().reshape(-1, 4, 128).abs().amax(-1) / 448
            fp8 = (
                (latent.float().reshape(-1, 4, 128) / scales.unsqueeze(-1))
                .clamp(-448, 448)
                .reshape(rows, rank)
                .to(torch.float8_e4m3fn)
            )
            packed = torch.cat(
                (
                    fp8.view(torch.uint8),
                    scales.view(torch.uint8),
                    rope_key.view(torch.uint8),
                ),
                dim=-1,
            )
            latent = (
                (fp8.float().reshape(-1, 4, 128) * scales.unsqueeze(-1))
                .reshape(rows, rank)
                .to(dtype)
            )
        else:
            packed = torch.cat((latent, rope_key), dim=-1)
        cache = packed.new_zeros((6, page_size, packed.size(-1)))
        cache[page_ids[0]] = packed[:page_size]
        cache[page_ids[1], : rows - page_size] = packed[page_size:]
        cache[5] = packed[:page_size].flip(0)
        pages = meta([page_ids])

        if sparse:
            index_keys = rand((rows, 128)).to(torch.float8_e4m3fn)
            index_scales = (torch.arange(rows, device=q.device) + 1).float() / rows
            index_weights = (
                torch.linspace(-1, 1, heads, device=q.device)
                .expand(rows, -1)
                .contiguous()
            )
            index_cache = torch.zeros(
                (6, page_size, 132), dtype=torch.uint8, device=q.device
            )
            blocks = index_cache.flatten(1)
            for page, start in zip((*page_ids, 5), (0, page_size, 0)):
                size = min(page_size, rows - start)
                blocks[page, : size * 128] = (
                    index_keys[start : start + size].view(torch.uint8).flatten()
                )
                blocks[page, page_size * 128 : page_size * 128 + size * 4] = (
                    index_scales[start : start + size].view(torch.uint8)
                )
            index_topk = 9

            def select(q_rows, layout, block_table):
                return packed_fp8_paged_mqa_topk(
                    q_rows[..., :128].to(torch.float8_e4m3fn),
                    index_cache,
                    index_weights[:1].expand(q_rows.size(0), -1),
                    layout,
                    block_table,
                    index_topk,
                )

        def evaluate(q_rows, lengths, block_table, offsets=None):
            q_nope, q_rope = q_rows.split((nope_dim, rope_dim), dim=-1)
            layout = batch_invariant.attention_token_layout(
                offsets, lengths, block_table, q_rows.size(0), page_size
            )
            q_latent = batch_invariant_mla.head_matmul(q_nope, kc)
            value = batch_invariant_mla.attention(
                q_latent,
                q_rope,
                cache,
                layout,
                block_table,
                scale=(nope_dim + rope_dim) ** -0.5,
                fp8_cache=fp8_cache,
                topk_indices=select(q_rows, layout, block_table) if sparse else None,
            )
            return batch_invariant_mla.head_matmul(value, vc)

        prefill = evaluate(q, meta([0]), pages, meta([0, rows]))
        split = 17
        for start, end in ((0, split), (split, rows)):
            chunk = evaluate(q[start:end], meta([start]), pages, meta([0, end - start]))
            self.assertTrue(torch.equal(prefill[start:end], chunk))
        mixed = evaluate(
            torch.cat((q[:7], q)),
            meta([0, 0]),
            meta([(5, 0), page_ids]),
            meta([0, 7, 7 + rows]),
        )
        self.assertTrue(torch.equal(prefill, mixed[7:]))

        # Decode reorders live requests and includes a graph padding row.
        token_order = (38, 0, 21)
        decode_q = torch.cat(
            (q[list(token_order)], rand((1, heads, nope_dim + rope_dim)))
        )
        seq_lens = meta([token + 1 for token in token_order] + [1])
        decode_pages = meta([page_ids, page_ids, page_ids, (0, 0)])
        decode = evaluate(decode_q, seq_lens, decode_pages)
        self.assertTrue(torch.equal(decode[:3], prefill[list(token_order)]))
        self.assertEqual(torch.count_nonzero(decode[3]).item(), 0)

        # Capture an inactive slot and turn it into a live request on replay.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = evaluate(decode_q, seq_lens, decode_pages)
        token = 31
        decode_q[3].copy_(q[token])
        seq_lens[3] = token + 1
        decode_pages[3].copy_(pages[0])
        graph.replay()
        self.assertTrue(torch.equal(captured[3], prefill[token]))
        self.assertTrue(torch.equal(captured[:3], decode[:3]))

        token = token_order[0]
        q1, pe1 = q[token].float().split((nope_dim, rope_dim), dim=-1)
        q1 = torch.bmm(q1[:, None], kc.float())[:, 0].to(dtype).float()
        context = slice(token + 1)
        if sparse:
            q_fp8 = q[token, :, :128].to(torch.float8_e4m3fn).float()
            logits = (q_fp8 @ index_keys[: token + 1].float().T).relu()
            logits = (logits * index_weights[0, :, None]).sum(0) * index_scales[
                : token + 1
            ]
            context = (
                logits.argsort(descending=True, stable=True)[:index_topk].sort().values
            )
            layout = batch_invariant.attention_token_layout(
                None, meta([token + 1]), pages, 1, page_size
            )
            actual = select(q[token : token + 1], layout, pages)[0]
            self.assertTrue(torch.equal(actual, context))
            tied = select(torch.zeros_like(q[:1]), layout, pages)[0]
            self.assertTrue(torch.equal(tied, meta(list(range(index_topk)))))
        scores = q1 @ latent[context].float().T
        scores += pe1 @ rope_key[context].float().T
        probs = torch.softmax(scores * (nope_dim + rope_dim) ** -0.5, dim=-1)
        expected = (probs @ latent[context].float()).to(dtype)
        expected = torch.bmm(expected.float()[:, None], vc.float())[:, 0].to(dtype)
        torch.testing.assert_close(prefill[token], expected, atol=0.003, rtol=0.03)

    def test_bfloat16(self):
        self._run_case(torch.bfloat16, fp8_cache=False)

    def test_float16(self):
        self._run_case(torch.float16, fp8_cache=False)

    @unittest.skipUnless(
        _CUDA and torch.cuda.get_device_capability() >= (8, 9), "requires SM89+"
    )
    def test_fp8_ds_mla(self):
        self._run_case(torch.bfloat16, fp8_cache=True)

    @unittest.skipUnless(
        _CUDA and torch.cuda.get_device_capability() >= (8, 9), "requires SM89+"
    )
    def test_sparse_mla(self):
        self._run_case(torch.bfloat16, fp8_cache=True, sparse=True)


if __name__ == "__main__":
    unittest.main()
