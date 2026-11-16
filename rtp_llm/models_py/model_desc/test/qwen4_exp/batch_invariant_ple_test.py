import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.qwen4_exp.gated_residual import Qwen4GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "requires CUDA")
class BatchInvariantPleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(46)

    def weights(self, *shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1

    def assert_bits(self, left, right):
        self.assertTrue(
            torch.equal(
                left.contiguous().view(torch.uint8),
                right.contiguous().view(torch.uint8),
            )
        )

    def test_ple_page_checkpoints_and_mixed_decode_graph(self):
        hidden = self.weights(2, 11, 256)
        embeds = self.weights(2, 11, 128)
        tokens = torch.arange(26, device="cuda").view(2, 13)
        epsilon = 1e-6
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(weight=dtype):
                scale = (
                    torch.ones(1, device="cuda")
                    if dtype == torch.float8_e4m3fn
                    else None
                )
                layer = Qwen4ExpPLELayer(
                    SimpleNamespace(ngram_size=3, context_len=2),
                    self.weights(256, 128).to(dtype),
                    self.weights(64, 128).to(dtype),
                    self.weights(256, 1, 4),
                    self.weights(256),
                    self.weights(256),
                    self.weights(256),
                    hc_mult=4,
                    hidden_size=64,
                    conv_kernel_size=4,
                    norm_eps=epsilon,
                    projection_scales=(scale, scale),
                    batch_invariant=True,
                )
                ref, snapshots = layer.prefill_page_snapshots(
                    hidden, tokens, 4, embeddings=embeds
                )
                mixed, mixed_snapshots = layer.prefill_page_snapshots(
                    hidden.flip(0), tokens.flip(0), 4, embeddings=embeds.flip(0)
                )
                self.assert_bits(ref, mixed.flip(0))
                self.assert_bits(snapshots[-1], mixed_snapshots[-1].flip(0))
                prefix, prefix_state = layer.prefill(
                    hidden[:, :4], tokens[:, :6], embeddings=embeds[:, :4]
                )
                self.assert_bits(ref[:, :4], prefix)
                self.assert_bits(prefix_state, snapshots[0])
                remainder, tail = layer.prefill_with_state(
                    hidden[:, 4:], tokens[:, 4:], prefix_state, embeddings=embeds[:, 4:]
                )
                self.assert_bits(ref[:, 4:], remainder)
                self.assert_bits(snapshots[-1], tail)

                # Start the graph at the preceding complete page. Retain the
                # live convolution history and update the packed query lanes.
                state = snapshots[1].clone()
                x = hidden[:, 8:9].contiguous()
                e = embeds[:, 8:9].contiguous()
                t = tokens[:, 8:11].contiguous()
                layer.decode_step(x, t, state, embeddings=e)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output, state_after = layer.decode_step(x, t, state, embeddings=e)
                x.copy_(x.flip(0))
                e.copy_(e.flip(0))
                t.copy_(t.flip(0))
                state.copy_(state.flip(0))
                graph.replay()
                self.assert_bits(ref[:, 8:9], output.flip(0))
                _, ref_state = layer.prefill(
                    hidden[:, :9], tokens[:, :11], embeddings=embeds[:, :9]
                )
                self.assert_bits(ref_state, state_after.flip(0))

    def test_hyper_connection_bitwise_across_batch_sizes(self):
        scale = torch.ones(1, device="cuda", dtype=torch.float32)
        layer = Qwen4GatedResidual(
            self.weights(256),
            self.weights(32, 256).to(torch.float8_e4m3fn),
            self.weights(256, 32).to(torch.float8_e4m3fn),
            self.weights(4, 256).to(torch.float8_e4m3fn),
            hc_mult=4,
            norm_eps=1e-6,
            weight_scales=(scale, scale, scale),
            batch_invariant=True,
        )
        x = self.weights(17, 256)
        result, _, gates = layer(x)
        one, _, one_gate = layer(x[8:9])
        self.assert_bits(result[8:9], one)
        self.assert_bits(gates[8:9], one_gate)


if __name__ == "__main__":
    unittest.main()
