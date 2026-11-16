import unittest

import torch

from rtp_llm.models_py.kernels.cuda import batch_invariant as ops
from rtp_llm.models_py.kernels.cuda.batch_invariant_norm import layer_norm, l2_normalize


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8,
    "requires NVIDIA SM80+",
)
class BatchInvariantEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(64)

    @staticmethod
    def metadata(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    def packed_layout(self, lengths):
        prefixes = self.metadata([0] * len(lengths))
        offsets = self.metadata([0] + list(torch.tensor(lengths).cumsum(0).tolist()))
        return ops.attention_token_layout(offsets, prefixes, None, sum(lengths), 16)

    def test_noncausal_packed_attention(self):
        target_rows, other_rows, heads, width = 41, 7, 4, 64
        qkv = torch.randn(
            (3, target_rows + other_rows, heads, width),
            device="cuda",
            dtype=torch.bfloat16,
        )

        def run(qkv, lengths, slopes):
            return ops.attention(
                *qkv,
                self.packed_layout(lengths),
                None,
                None,
                causal=False,
                scale=width**-0.5,
                alibi_slopes=slopes,
            )

        q, k, v = qkv[:, :target_rows].float().transpose(1, 2)
        positions = torch.arange(target_rows, device=q.device)
        distance = (positions[:, None] - positions).abs()
        alibi_slopes = torch.tensor([0.25, 0.125, 0.0625, 0.03125], device=q.device)
        for slopes in (None, alibi_slopes):
            with self.subTest(alibi=slopes is not None):
                baseline = run(qkv[:, :target_rows], [target_rows], slopes)
                mixed = run(qkv, [target_rows, other_rows], slopes)
                reversed_batch = run(
                    qkv.roll(other_rows, dims=1), [other_rows, target_rows], slopes
                )
                self.assertTrue(torch.equal(baseline, mixed[:target_rows]))
                self.assertTrue(torch.equal(baseline, reversed_batch[other_rows:]))
                mask = -slopes[:, None, None] * distance if slopes is not None else None
                reference = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask
                    )
                    .transpose(0, 1)
                    .reshape(target_rows, -1)
                )
                torch.testing.assert_close(
                    baseline.float(), reference, atol=1e-2, rtol=1e-2
                )

    def test_mrope_uses_packed_axes_and_preserves_nonrotary_dimensions(self):
        positions = self.metadata([[0, 2, 5], [1, 4, 2], [2, 6, 3], [3, 1, 8]])
        sections = (11, 11, 10)
        pairs, width = sum(sections), 80
        query = torch.randn((4, 3, width), device="cuda", dtype=torch.float16)
        key = query[:, :2].contiguous()
        angle = torch.randn((16, pairs), device="cuda", dtype=torch.float32)
        cosine, sine = angle.cos(), angle.sin()
        table = torch.cat((cosine, sine), dim=-1)

        def rotated(query, key, positions, lengths, interleaved):
            query, key = query.clone(), key.clone()
            ops.apply_rope(
                query,
                key,
                self.packed_layout(lengths),
                table,
                positions,
                mrope_sections=sections,
                mrope_interleaved=interleaved,
            )
            return query, key

        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                axis = torch.zeros(pairs, device="cuda", dtype=torch.long)
                if interleaved:
                    axis[1 : 3 * sections[1] : 3] = 1
                    axis[2 : 3 * sections[2] : 3] = 2
                else:
                    axis[sections[0] : sections[0] + sections[1]] = 1
                    axis[sections[0] + sections[1] :] = 2
                index = positions[:, axis]
                pair_ids = torch.arange(pairs, device="cuda")
                cos = cosine[index, pair_ids].unsqueeze(1)
                sin = sine[index, pair_ids].unsqueeze(1)
                q0, q1 = query[..., : 2 * pairs].float().chunk(2, dim=-1)
                expected = torch.cat((q0 * cos - q1 * sin, q1 * cos + q0 * sin), dim=-1)
                q, k = rotated(query, key, positions, [1, 3], interleaved)
                torch.testing.assert_close(q[..., : 2 * pairs], expected.half())
                self.assertTrue(
                    torch.equal(q[..., 2 * pairs :], query[..., 2 * pairs :])
                )
                self.assertTrue(torch.equal(q[:, :2], k))
                q_solo, _ = rotated(query[1:], key[1:], positions[1:], [3], interleaved)
                self.assertTrue(torch.equal(q[1:], q_solo))
                implicit, _ = rotated(query, key, None, [4], interleaved)
                text_positions = self.metadata([[i] * 3 for i in range(4)])
                explicit, _ = rotated(query, key, text_positions, [4], interleaved)
                self.assertTrue(torch.equal(implicit, explicit))

    def test_mrope_cuda_graph_reads_updated_positions(self):
        rows, pairs = 3, 32
        x = torch.randn((rows, 1, 2 * pairs), device="cuda", dtype=torch.bfloat16)
        layout = self.packed_layout([rows])
        theta = torch.randn((12, pairs), device="cuda")
        table = torch.cat((theta.cos(), theta.sin()), dim=-1)
        positions = self.metadata([[0, 0, 0], [1, 2, 3], [2, 3, 4]])

        def run():
            q, k = x.clone(), x.clone()
            ops.apply_rope(
                q,
                k,
                layout,
                table,
                positions,
                mrope_sections=(11, 11, 10),
                mrope_interleaved=True,
            )
            return q

        initial = run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        positions.copy_(self.metadata([[2, 1, 0], [2, 5, 2], [7, 3, 6]]))
        graph.replay()
        self.assertFalse(torch.equal(initial, captured))
        self.assertTrue(torch.equal(run(), captured))

    def test_output_normalizations_are_fixed_per_row(self):
        x = torch.randn((3, 11, 763), device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn(763, device="cuda", dtype=x.dtype)
        beta = torch.randn_like(gamma)
        x[0, 0].zero_()
        for norm, reference in (
            (l2_normalize, torch.nn.functional.normalize(x.float(), dim=-1)),
            (
                lambda x: layer_norm(x, gamma, beta, 1e-6),
                torch.nn.functional.layer_norm(x, (763,), gamma, beta, 1e-6),
            ),
        ):
            mixed = norm(x)
            solo = norm(x[1, 0])
            self.assertTrue(torch.equal(mixed[1, 0], solo))
            self.assertTrue(torch.equal(norm(x.flip(0))[1, 0], solo))
            torch.testing.assert_close(
                mixed, reference.to(x.dtype), atol=2e-2, rtol=1e-2
            )


if __name__ == "__main__":
    unittest.main()
