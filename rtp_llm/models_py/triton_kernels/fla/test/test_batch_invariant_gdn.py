import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.batch_invariant_gdn import (
    batch_invariant_gated_delta_rule,
)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "requires CUDA")
class BatchInvariantGdnTest(unittest.TestCase):
    page_size = 4
    key_heads = 1
    value_heads = 2
    head_dim = 128
    channels = (2 * key_heads + value_heads) * head_dim
    conv_dtype = torch.bfloat16

    def setUp(self):
        torch.manual_seed(42)
        self.weight = self.random((self.channels, 4))
        # Retain different row and channel strides, as with fused qkvz+ba.
        self.qkv = self.random((20, self.channels + 7))[:, : self.channels]
        self.g = -torch.rand((20, self.value_heads), device="cuda")
        self.beta = torch.rand((20, self.value_heads), device="cuda").to(
            self.conv_dtype
        )

    def random(self, shape):
        return torch.randn(shape, device="cuda", dtype=self.conv_dtype) * 0.25

    def ints(self, values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    def cache(self, ssm_dtype):
        # Emulate the packed layer pool: a gap follows each typed page.
        conv = torch.zeros((13, 4, self.channels), device="cuda", dtype=self.conv_dtype)
        ssm = torch.zeros(
            (13, 3, self.head_dim, self.head_dim), device="cuda", dtype=ssm_dtype
        )
        return conv[:, :3], ssm[:, : self.value_heads]

    def run_gdn(
        self,
        token_ids,
        cu_seqlens,
        lengths,
        blocks,
        cache,
        *,
        decode=False,
        verify=False
    ):
        conv, ssm = cache
        return batch_invariant_gated_delta_rule(
            self.qkv[token_ids],
            self.weight,
            self.g[token_ids],
            self.beta[token_ids],
            cu_seqlens=cu_seqlens,
            lengths=lengths,
            block_map=blocks,
            conv_state=conv,
            ssm_state=ssm,
            conv_dtype=self.conv_dtype,
            ssm_dtype=ssm.dtype,
            seq_size_per_block=self.page_size,
            key_heads=self.key_heads,
            value_heads=self.value_heads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
            is_decode=decode,
            is_target_verify=verify,
        )

    def reference(self, length, ssm_dtype):
        conv = F.conv1d(
            F.pad(self.qkv[:length].T[None].float(), (3, 0)),
            self.weight[:, None].float(),
            groups=self.channels,
        )
        q, k, v = (
            F.silu(conv[0].T)
            .to(self.conv_dtype)
            .float()
            .split(
                (self.head_dim, self.head_dim, self.value_heads * self.head_dim), dim=1
            )
        )
        state = torch.zeros(
            (self.value_heads, self.head_dim, self.head_dim), device="cuda"
        )
        outputs = []
        for t in range(length):
            q_t = q[t] * (q[t].square().sum() + 1e-6).rsqrt() * self.head_dim**-0.5
            k_t = k[t] * (k[t].square().sum() + 1e-6).rsqrt()
            state = state * self.g[t, :, None, None].exp()
            residual = v[t].view(self.value_heads, self.head_dim) - (state * k_t).sum(
                -1
            )
            state = state + (residual * self.beta[t, :, None].float())[..., None] * k_t
            outputs.append((state * q_t).sum(-1).to(self.conv_dtype))
            state = state.to(ssm_dtype).float()
        return torch.stack(outputs)

    def assert_bitwise_equal(self, left, right):
        self.assertTrue(
            torch.equal(
                left.contiguous().view(torch.uint8),
                right.contiguous().view(torch.uint8),
            )
        )

    def test_keywise_decay_uses_kda_state_layout(self):
        tokens = self.ints([0, 1, 2, 3, 4])
        width = self.head_dim
        gate = self.random((5, self.value_heads, width))
        log_decay = torch.randn(self.value_heads, device="cuda")
        bias = self.random((self.value_heads, width))
        conv, ssm = self.cache(torch.bfloat16)
        blocks = self.ints([[3, 6]])

        def run(begin, end, decode=False):
            return batch_invariant_gated_delta_rule(
                self.qkv[tokens[begin:end]],
                self.weight,
                gate[begin:end],
                self.beta[begin:end],
                cu_seqlens=self.ints([0, end - begin]),
                lengths=self.ints([end if decode else begin]),
                block_map=blocks,
                conv_state=conv,
                ssm_state=ssm,
                conv_dtype=self.conv_dtype,
                ssm_dtype=ssm.dtype,
                seq_size_per_block=self.page_size,
                key_heads=self.key_heads,
                value_heads=self.value_heads,
                key_dim=width,
                value_dim=width,
                is_decode=decode,
                decay_log=log_decay,
                decay_bias=bias,
            )

        full = run(0, 5)
        state = ssm[6].clone()
        conv.zero_()
        ssm.zero_()
        self.assert_bitwise_equal(full, torch.cat((run(0, 4), run(4, 5, True))))
        self.assert_bitwise_equal(state, ssm[6])

        convolved = (
            F.silu(
                F.conv1d(
                    F.pad(self.qkv[:5].T[None].float(), (3, 0)),
                    self.weight[:, None].float(),
                    groups=self.channels,
                )[0].T
            )
            .to(self.conv_dtype)
            .float()
        )
        q, k, v = convolved.split((width, width, self.value_heads * width), dim=1)
        decay = (
            -log_decay.exp()[:, None] * F.softplus(gate.float() + bias.float())
        ).exp()
        ref = torch.zeros_like(state, dtype=torch.float32)
        outputs = []
        for row in range(5):
            qr = q[row] * (q[row].square().sum() + 1e-6).rsqrt() * width**-0.5
            kr = k[row] * (k[row].square().sum() + 1e-6).rsqrt()
            ref = ref * decay[row, :, :, None]
            resid = v[row].reshape(self.value_heads, width) - (
                ref * kr[None, :, None]
            ).sum(1)
            ref = ref + kr[None, :, None] * (resid * self.beta[row, :, None]).unsqueeze(
                1
            )
            outputs.append((ref * qr[None, :, None]).sum(1))
            ref = ref.to(ssm.dtype).float()
        torch.testing.assert_close(
            full.float(), torch.stack(outputs), atol=7e-4, rtol=2e-2
        )
        torch.testing.assert_close(state.float(), ref, atol=7e-4, rtol=2e-2)

    def test_batching_prefix_chunking_and_cuda_graph(self):
        a_pages = [3, 6, 2, 11]
        b_pages = [5, 8, 0, 0]
        blocks = self.ints([a_pages])
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(ssm_dtype=dtype):
                full_cache = self.cache(dtype)
                ref = self.run_gdn(
                    slice(0, 13), self.ints([0, 13]), self.ints([0]), blocks, full_cache
                )
                torch.testing.assert_close(
                    ref, self.reference(13, dtype), atol=4e-5, rtol=0.02
                )

                mixed_cache = self.cache(dtype)
                reordered = self.run_gdn(
                    self.ints(list(range(13, 20)) + list(range(13))),
                    self.ints([0, 7, 20]),
                    self.ints([0, 0]),
                    self.ints([b_pages, a_pages]),
                    mixed_cache,
                )
                self.assert_bitwise_equal(ref, reordered[7:])

                hit_cache = self.cache(dtype)
                for dst, src in zip(hit_cache, full_cache):
                    dst[a_pages[1]].copy_(src[a_pages[1]])
                hit = self.run_gdn(
                    slice(8, 13), self.ints([0, 5]), self.ints([8]), blocks, hit_cache
                )
                self.assert_bitwise_equal(ref[8:], hit)

                chunk_cache = self.cache(dtype)
                first = self.run_gdn(
                    slice(0, 5), self.ints([0, 5]), self.ints([0]), blocks, chunk_cache
                )
                second = self.run_gdn(
                    slice(5, 12), self.ints([0, 7]), self.ints([5]), blocks, chunk_cache
                )
                self.assert_bitwise_equal(ref[:12], torch.cat([first, second]))

                graph_tokens = self.ints([12, 19]).long()
                graph_offsets = self.ints([0, 1, 2])
                graph_lengths = self.ints([1, 1])
                graph_pages = self.ints([[0] * 4, [0] * 4])
                self.run_gdn(
                    graph_tokens,
                    graph_offsets,
                    graph_lengths,
                    graph_pages,
                    chunk_cache,
                    decode=True,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay = self.run_gdn(
                        graph_tokens,
                        graph_offsets,
                        graph_lengths,
                        graph_pages,
                        chunk_cache,
                        decode=True,
                    )
                graph_lengths[0] = 13
                graph_pages[0].copy_(blocks[0])
                graph.replay()
                self.assert_bitwise_equal(ref[12:], replay[:1])
                self.assert_bitwise_equal(replay[1:], torch.zeros_like(replay[1:]))
                for chunk, full, mixed in zip(chunk_cache, full_cache, mixed_cache):
                    self.assert_bitwise_equal(chunk[a_pages], full[a_pages])
                    self.assert_bitwise_equal(mixed[a_pages], full[a_pages])
                    self.assert_bitwise_equal(chunk[0], torch.zeros_like(chunk[0]))

    def test_target_verify_keeps_each_candidate_checkpoint(self):
        dtype = torch.bfloat16
        pages = [3, 6, 2, 11]
        blocks = self.ints([pages])
        verify_cache = self.cache(dtype)
        self.run_gdn(
            slice(0, 6), self.ints([0, 6]), self.ints([0]), blocks, verify_cache
        )
        verified = self.run_gdn(
            slice(6, 9),
            self.ints([0, 3]),
            self.ints([7]),
            blocks,
            verify_cache,
            verify=True,
        )
        for accepted in range(1, 4):
            regular_cache = self.cache(dtype)
            end = 6 + accepted
            regular = self.run_gdn(
                slice(0, end),
                self.ints([0, end]),
                self.ints([0]),
                blocks,
                regular_cache,
            )
            self.assert_bitwise_equal(verified[accepted - 1], regular[-1])
            scratch_page = pages[6 // self.page_size + accepted - 1]
            regular_page = pages[(end - 1) // self.page_size]
            for scratch, regular in zip(verify_cache, regular_cache):
                self.assert_bitwise_equal(scratch[scratch_page], regular[regular_page])


if __name__ == "__main__":
    unittest.main()
