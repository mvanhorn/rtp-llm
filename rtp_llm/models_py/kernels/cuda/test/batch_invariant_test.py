import os
import unittest

import torch

from rtp_llm.models_py.kernels.cuda import batch_invariant as ops

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
_CUDA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8
)


@unittest.skipUnless(_CUDA or _INTERPRET, "requires NVIDIA SM80+ or Triton interpreter")
class BatchInvariantTest(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if _CUDA else "cpu"
        torch.manual_seed(12)

    def metadata(self, values):
        return torch.tensor(values, device=self.device, dtype=torch.int32)

    def test_linear_with_padding_and_reordering(self):
        dtype = torch.float16
        x = torch.randn((35, 67), device=self.device, dtype=dtype)
        weight = torch.randn((67, 89), device=self.device, dtype=dtype).T
        bias = torch.randn(178, device=self.device, dtype=dtype)[::2]

        def linear(rows):
            if _CUDA:
                return ops.linear(rows, weight, bias, out_dtype=torch.float32)
            return ops._launch_linear(rows, weight, bias, torch.float32)

        baseline = linear(x[8:9])
        for batch in (x, x.flip(0), x[8:].contiguous()):
            row = (batch == x[8]).all(dim=1).nonzero().item()
            self.assertTrue(torch.equal(linear(batch)[row : row + 1], baseline))
        expected = x[8:9].float() @ weight.float().T + bias.float()
        torch.testing.assert_close(baseline, expected, atol=3e-5, rtol=3e-5)

        if _CUDA:
            # Multiple work tiles per SM and a partial M group.
            wide = torch.randn((9217, 67), device=self.device, dtype=x.dtype)
            full = ops.lm_head(x.flip(0), wide, None)
            self.assertEqual(full.dtype, torch.float32)
            self.assertTrue(torch.equal(full[26:27], ops.lm_head(x[8:9], wide, None)))

    def test_rms_norm_reorders_rows(self):
        x = torch.randn((19, 127), device=self.device, dtype=torch.float16)
        gamma = torch.randn(127, device=self.device, dtype=x.dtype)
        baseline = ops.rms_norm(x[8:9], gamma, 1e-6)
        reordered = ops.rms_norm(x.flip(0), gamma, 1e-6)
        self.assertTrue(torch.equal(reordered[10:11], baseline))
        row = x[8:9].float()
        expected = row * (row.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        torch.testing.assert_close(baseline, (expected * gamma).half())

    def test_block_fp8_scales_follow_their_token_and_k_group(self):
        rows, cols, groups = 19, 133, 5
        inner = groups * 128
        x = torch.randint(-2, 3, (rows, inner), device=self.device).half()
        weight = torch.randint(-2, 3, (cols, inner), device=self.device).half()
        x_scales = 2.0 ** torch.randint(-5, 0, (rows, groups), device=self.device)
        weight_scales = 2.0 ** torch.randint(-5, 0, (2, groups), device=self.device)
        expanded_scales = weight_scales.repeat_interleave(128, dim=0)[:cols]
        expected = (x.float() * x_scales.repeat_interleave(128, dim=1)) @ (
            weight.float() * expanded_scales.repeat_interleave(128, dim=1)
        ).T

        def pack_scales(scales):
            exponents = scales.view(torch.int32) >> 23
            padded = torch.nn.functional.pad(exponents, (0, 3)).view(-1, 2, 4)
            shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=self.device)
            return (padded << shifts).sum(dim=-1, dtype=torch.int32)

        if _CUDA and torch.cuda.get_device_capability() >= (8, 9):
            x, weight = x.to(torch.float8_e4m3fn), weight.to(torch.float8_e4m3fn)
        for packed in (False, True):
            with self.subTest(packed=packed):
                activations = pack_scales(x_scales) if packed else x_scales
                weights = pack_scales(expanded_scales) if packed else weight_scales
                baseline = ops._launch_linear(
                    x[8:9], weight, None, torch.float32, activations[8:9], weights
                )
                mixed = ops._launch_linear(
                    x, weight, None, torch.float32, activations, weights
                )
                torch.testing.assert_close(mixed, expected, atol=0, rtol=0)
                self.assertTrue(torch.equal(mixed[8:9], baseline))

    @unittest.skipUnless(_CUDA, "requires NVIDIA GPU")
    def test_fp8_scaling_is_local_to_each_token(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_per_tensor_linear import (
            CudaFp8PerTensorLinear,
        )

        x = torch.randn((19, 96), device=self.device, dtype=torch.bfloat16)
        x[0] *= 50
        weight = torch.randn((113, 96), device=self.device).to(torch.float8_e4m3fn)
        scale = torch.tensor(0.01, device=self.device)
        for static_scale in (None, scale):
            baseline = ops.fp8_per_tensor_linear(
                x[8:9], weight, scale, static_scale, out_dtype=torch.float32
            )
            mixed = ops.fp8_per_tensor_linear(
                x, weight, scale, static_scale, out_dtype=torch.float32
            )
            self.assertTrue(torch.equal(mixed[8:9], baseline))
            linear = CudaFp8PerTensorLinear(
                weight, scale, static_scale, batch_invariant=True
            )
            self.assertTrue(torch.equal(linear(x[8:9]), baseline.to(x.dtype)))

    @unittest.skipUnless(
        _CUDA and torch.cuda.get_device_capability() >= (8, 9), "requires NVIDIA SM89+"
    )
    def test_block_fp8_linear_quantizes_activations_independently(self):
        x = torch.randn((19, 640), device=self.device, dtype=torch.bfloat16)
        x[0] *= 50
        weight = torch.randn((133, 640), device=self.device).to(torch.float8_e4m3fn)
        for scales in (
            torch.full((2, 5), 0.125, device=self.device),
            torch.full((133, 2), 0x7C7C7C7C, device=self.device, dtype=torch.int32),
        ):
            baseline = ops.fp8_per_block_linear(x[8:9], weight, scales)
            mixed = ops.fp8_per_block_linear(x, weight, scales)
            self.assertTrue(torch.equal(mixed[8:9], baseline))

    @unittest.skipUnless(
        _CUDA and torch.cuda.get_device_capability() >= (8, 9), "requires NVIDIA SM89+"
    )
    def test_nvfp4_dense_and_routed_gemm(self):
        from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
            invoke_fused_moe_kernel,
            moe_align_block_size_torch,
        )
        import triton.language as tl

        m, n, k, experts = 37, 71, 48, 3
        e2m1 = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=self.device,
        )
        codes = torch.randint(16, (experts, n, k), device=self.device)
        weights = (codes[:, :, ::2] | (codes[:, :, 1::2] << 4)).to(torch.uint8)
        scales = torch.randint(1, 4, (experts, n, k // 16), device=self.device)
        input_scale = torch.tensor(0.5, device=self.device)
        weight_scale = torch.tensor([0.125, 0.25, 0.5], device=self.device)
        # Each group has a peak of 3, so its exactly representable scale is 1.
        x = e2m1[torch.randint(16, (m, k), device=self.device)] * input_scale
        x[:, ::16] = 3
        x = x.bfloat16()
        scale_fp8 = scales.to(torch.float8_e4m3fn)
        dequant = (
            e2m1[codes]
            * scales.repeat_interleave(16, dim=-1)
            * weight_scale[:, None, None]
        )

        def dense(x):
            return ops._launch_linear(
                x,
                weights[1],
                None,
                torch.float32,
                weight_scales=scale_fp8[1],
                nvfp4_input_scale=input_scale,
                nvfp4_weight_scale=weight_scale[1],
            )

        torch.testing.assert_close(dense(x), x.float() @ dequant[1].T, atol=0, rtol=0)
        self.assertTrue(torch.equal(dense(x.flip(0))[21:22], dense(x[15:16])))

        def routed(x, ids):
            sorted_ids, expert_ids, count = moe_align_block_size_torch(ids, 16, experts)
            output = torch.empty((x.size(0), n), device=self.device, dtype=x.dtype)
            invoke_fused_moe_kernel(
                x,
                weights,
                output,
                None,
                ids,
                sorted_ids,
                expert_ids,
                count,
                False,
                1,
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                },
                tl.bfloat16,
                B_scale=scale_fp8,
                nvfp4_input_scale=input_scale,
                nvfp4_weight_scale=weight_scale,
            )
            return output

        ids = torch.randint(experts, (m, 1), dtype=torch.int32, device=self.device)
        expected = torch.einsum("mk,mnk->mn", x.float(), dequant[ids[:, 0]]).bfloat16()
        torch.testing.assert_close(routed(x, ids), expected, atol=0, rtol=0)
        self.assertTrue(
            torch.equal(
                routed(x[15:16], ids[15:16]),
                routed(x.flip(0), ids.flip(0))[21:22],
            )
        )

    def test_attention_across_prefill_chunk_decode_and_cache_pages(self):
        length, page_size, q_heads, kv_heads, head_dim = 39, 16, 4, 2, 32
        width = (q_heads + 2 * kv_heads) * head_dim
        target = torch.randn((length, width), device=self.device, dtype=torch.float16)
        other = torch.randn((5, width), device=self.device, dtype=torch.float16)
        angles = torch.arange(length, device=self.device)[
            :, None
        ].float() * torch.linspace(0.01, 0.07, head_dim // 2, device=self.device)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
        scale = head_dim**-0.5

        def new_cache():
            return torch.zeros(
                (8, 2, kv_heads, page_size, head_dim),
                device=self.device,
                dtype=target.dtype,
            )

        def run(qkv, cache, pages, offsets, prefixes_or_lengths):
            offsets = self.metadata(offsets) if offsets is not None else None
            layout = ops.attention_token_layout(
                offsets,
                self.metadata(prefixes_or_lengths),
                pages,
                qkv.size(0),
                page_size,
            )
            q, k, v = (
                qkv.clone()
                .view(qkv.size(0), -1, head_dim)
                .split((q_heads, kv_heads, kv_heads), dim=1)
            )
            ops.apply_rope(q, k, layout, cos_sin, None)
            ops.append_kv(k, v, cache, layout)
            result = ops.attention(
                q, k, v, layout, cache, pages, causal=True, scale=scale
            )
            return result, (q, k, v)

        pages = self.metadata([[1, 2, 3]])
        baseline, (q, k, v) = run(target, new_cache(), pages, [0, length], [0])
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.float().transpose(0, 1),
                k.float().repeat_interleave(q_heads // kv_heads, dim=1).transpose(0, 1),
                v.float().repeat_interleave(q_heads // kv_heads, dim=1).transpose(0, 1),
                is_causal=True,
            )
            .transpose(0, 1)
            .reshape(length, -1)
        )
        torch.testing.assert_close(baseline.float(), expected, atol=2e-3, rtol=2e-3)

        mixed_pages = self.metadata([[4, 0, 0], [7, 5, 6]])
        mixed, _ = run(
            torch.cat((other, target)),
            new_cache(),
            mixed_pages,
            [0, 5, 5 + length],
            [0, 0],
        )
        self.assertTrue(torch.equal(mixed[5:], baseline))

        cache = new_cache()
        start, _ = run(target[:19], cache, pages, [0, 19], [0])
        suffix, _ = run(target[19:-1], cache, pages, [0, length - 20], [19])
        decoded, _ = run(target[-1:], cache, pages, None, [length])
        self.assertTrue(torch.equal(torch.cat((start, suffix, decoded)), baseline))

        # A rounded graph batch can have zero-length rows and page zero.
        padded, _ = run(
            torch.cat((target, other)),
            new_cache(),
            self.metadata([[0, 0, 0], [1, 2, 3], [0, 0, 0]]),
            [0, 0, length, length + 5],
            [0, 0, 0],
        )
        self.assertTrue(torch.equal(padded[:length], baseline))
        self.assertEqual(torch.count_nonzero(padded[length:]).item(), 0)

    @unittest.skipUnless(
        _CUDA and torch.cuda.get_device_capability() >= (8, 9), "requires NVIDIA SM89+"
    )
    def test_fp8_paged_cache_keeps_per_token_head_scales(self):
        rows, page_size, heads, dim = 21, 16, 2, 32
        x = torch.randn((rows, heads, dim), device=self.device, dtype=torch.bfloat16)
        value = x * 0.125
        cache = torch.empty(
            (3, 2, heads, page_size, dim), device=self.device, dtype=torch.float8_e4m3fn
        )
        scales = torch.empty(cache.shape[:-1], device=self.device, dtype=torch.float32)
        pages = self.metadata([[2, 1]])

        def attend(x, value, offsets, lengths):
            layout = ops.attention_token_layout(
                self.metadata(offsets) if offsets is not None else None,
                self.metadata(lengths),
                pages,
                x.size(0),
                page_size,
            )
            ops.append_kv(x, value, cache, layout, scales)
            return ops.attention(
                x,
                x,
                value,
                layout,
                cache,
                pages,
                causal=True,
                scale=dim**-0.5,
                kv_scales=scales,
            )

        full = attend(x, value, [0, rows], [0])
        split = attend(x[:20], value[:20], [0, 20], [0])
        last = attend(x[-1:], value[-1:], None, [rows])
        self.assertTrue(torch.equal(torch.cat((split, last)), full))
        expected_scale = x[-1].float().abs().amax(dim=-1) / 448
        torch.testing.assert_close(scales[1, 0, :, 4], expected_scale)
        torch.testing.assert_close(scales[1, 1, :, 4], expected_scale * 0.125)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                x.float().transpose(0, 1),
                x.float().transpose(0, 1),
                value.float().transpose(0, 1),
                is_causal=True,
            )
            .transpose(0, 1)
            .reshape_as(full)
        )
        torch.testing.assert_close(full.float(), expected, atol=0.015, rtol=0.05)

    @unittest.skipUnless(
        _CUDA and torch.cuda.get_device_capability() >= (8, 9), "requires NVIDIA SM89+"
    )
    def test_fp8_and_mxfp4_expert_scale_layouts(self):
        import triton.language as tl

        from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
            invoke_fused_moe_kernel,
            moe_align_block_size_torch,
        )

        m, n, k = 23, 71, 128
        rows = torch.randint(-3, 4, (m, k), device=self.device).bfloat16()
        fp8 = torch.float8_e4m3fn
        ids = torch.arange(m, dtype=torch.int32, device=self.device).remainder(2)[
            :, None
        ]
        weight = torch.randint(-3, 4, (2, n, k), device=self.device).to(fp8)
        a_scale = torch.tensor([[0.125], [0.25]], device=self.device)
        b_scale = torch.tensor([0.5, 0.25], device=self.device).view(2, 1, 1)

        def route(x, w, ids, sa, sb, **mode):
            sorted_ids, expert_ids, count = moe_align_block_size_torch(ids, 16, 2)
            output = torch.empty(
                (x.size(0), n), device=self.device, dtype=torch.bfloat16
            )
            invoke_fused_moe_kernel(
                x,
                w,
                output,
                None,
                ids,
                sorted_ids,
                expert_ids,
                count,
                False,
                1,
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                },
                tl.bfloat16,
                A_scale=sa,
                B_scale=sb,
                **mode,
            )
            return output

        values = rows.to(fp8)
        sa = a_scale[ids[:, 0]]
        expected = torch.einsum(
            "mk,mnk->mn",
            rows.float() * sa,
            weight.float()[ids[:, 0]] * b_scale[ids[:, 0]],
        ).bfloat16()
        actual = route(values, weight, ids, sa, b_scale, per_tensor_fp8=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        solo = route(
            values[5:6], weight, ids[5:6], sa[5:6], b_scale, per_tensor_fp8=True
        )
        self.assertTrue(torch.equal(solo, actual[5:6]))

        # 0x22 is 1 in both E2M1 nibbles. UE8M0 uses exponent bias 127.
        packed = torch.full((2, n, k // 2), 0x22, device=self.device, dtype=torch.int8)
        sb = torch.arange(123, 127, device=self.device, dtype=torch.uint8)
        sb = sb.view(1, 1, 4).expand(2, n, 4).contiguous()
        for group_size in (32, 128):
            group_scales = torch.tensor(
                [126, 127, 124, 125] if group_size == 32 else [126],
                device=self.device,
                dtype=torch.int32,
            )
            shifts = torch.arange(
                0, group_scales.numel() * 8, 8, device=self.device, dtype=torch.int32
            )
            encoded = torch.sum(group_scales << shifts, dtype=torch.int32)
            sa = encoded.reshape(1, 1).expand(m, 1)
            true_sa = (2.0 ** (group_scales - 127)).repeat_interleave(group_size)
            true_sb = (2.0 ** (sb.int() - 127)).repeat_interleave(32, dim=-1)
            expected = torch.einsum(
                "mk,mnk->mn", rows.float() * true_sa, true_sb[ids[:, 0]]
            ).bfloat16()
            mode = {"mxfp4": True, "mxfp4_activation_group_size": group_size}
            actual = route(values, packed, ids, sa, sb, **mode)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            solo = route(values[5:6], packed, ids[5:6], sa[5:6], sb, **mode)
            self.assertTrue(torch.equal(solo, actual[5:6]))

    @unittest.skipUnless(_CUDA, "requires NVIDIA GPU")
    def test_graph_replay_uses_updated_metadata(self):
        dtype = torch.bfloat16
        page_size = 16
        cache = torch.randn((4, 2, 1, page_size, 32), device=self.device, dtype=dtype)
        q = torch.randn((3, 2, 32), device=self.device, dtype=dtype)
        k = torch.empty((3, 1, 32), device=self.device, dtype=dtype)
        pages = self.metadata([[1, 2], [0, 0], [0, 0]])
        lengths = self.metadata([7, 1, 1])

        def run(query, key, blocks, sizes):
            layout = ops.attention_token_layout(
                None, sizes, blocks, query.size(0), page_size
            )
            return ops.attention(
                query, key, key, layout, cache, blocks, causal=True, scale=32**-0.5
            )

        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            run(q, k, pages, lengths)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(q, k, pages, lengths)

        pages.copy_(self.metadata([[3, 2], [0, 0], [0, 0]]))
        lengths.copy_(self.metadata([21, 1, 1]))
        graph.replay()
        eager = run(q[:1], k[:1], pages[:1], lengths[:1])
        self.assertTrue(torch.equal(output[:1], eager))
        self.assertEqual(torch.count_nonzero(output[1:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
