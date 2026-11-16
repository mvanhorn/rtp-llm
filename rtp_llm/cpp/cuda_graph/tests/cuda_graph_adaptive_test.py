import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
    rtp_llm_ops,
)


class Attention:
    def __init__(self, inputs):
        self.params = rtp_llm_ops.SparseMlaParams()
        self.params.fill_params(inputs.attention_inputs, 64, False)

    def prepare_cuda_graph(self, inputs):
        lengths = inputs.input_lengths.tolist()
        offsets = inputs.cu_seqlens.tolist()
        assert offsets == [0] + list(torch.tensor(lengths).cumsum(0).tolist())
        assert inputs.cu_seqlens_host.tolist() == offsets
        prefixes = inputs.prefix_lengths.tolist()
        expected_kv = [0] + list(
            torch.tensor([q + k for q, k in zip(lengths, prefixes)]).cumsum(0).tolist()
        )
        # Legacy padded metadata behavior is unchanged; exact-row graphs have
        # no padded tokens, and require fully consistent empty-request slots.
        if sum(lengths) == self.params.multi_token_decode_total_tokens:
            assert inputs.cu_kv_seqlens.tolist() == expected_kv
        for i, length in enumerate(lengths):
            if length == 0:
                assert inputs.prefix_lengths[i].item() == 0
                assert (
                    inputs.kv_cache_kernel_block_id_device[i].count_nonzero().item()
                    == 0
                )
        self.params.fill_multi_token_decode_cuda_graph_params(
            inputs.input_lengths,
            inputs.prefix_lengths,
            inputs.kv_cache_kernel_block_id_device,
            64,
        )


class ShapeRecordingModel:
    def __init__(self):
        self.captured_rows = []

    def prepare_fmha_impl(self, inputs, *args):
        return Attention(inputs)

    def forward(self, inputs, attn=None):
        rows = inputs.input_hiddens.shape[0]
        if torch.cuda.is_current_stream_capturing():
            self.captured_rows.append(rows)
        result = PyModelOutputs()
        # Real SparseMLA metadata kernel must retain exact token-to-request,
        # causal position and physical KV write-slot mappings across replays.
        params = attn.params
        bias = params.batch_indice_d + params.positions_d + params.slot_mapping
        result.hidden_states = inputs.input_hiddens + rows + bias[:, None]
        return result


def make_inputs(lengths, value):
    rows = sum(lengths)
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    inputs.input_hiddens = torch.full(
        (rows, 8), value, device="cuda", dtype=torch.float16
    )
    attn = PyAttentionInputs()
    attn.is_prefill = True
    attn.is_target_verify = True
    attn.input_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    attn.prefix_lengths = torch.full_like(attn.input_lengths, 8)
    attn.sequence_lengths = torch.empty(0, device="cuda", dtype=torch.int32)
    attn.cu_seqlens_host = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32
    )
    attn.cu_seqlens = attn.cu_seqlens_host.cuda()
    attn.cu_kv_seqlens = torch.cat(
        (
            torch.zeros(1, device="cuda", dtype=torch.int32),
            (attn.input_lengths + 8).cumsum(0).int(),
        )
    )
    attn.kv_cache_kernel_block_id_host = torch.ones(
        (len(lengths), 1), dtype=torch.int32
    )
    attn.kv_cache_kernel_block_id_device = attn.kv_cache_kernel_block_id_host.cuda()
    attn.kv_cache_block_id_host = attn.kv_cache_kernel_block_id_host
    attn.kv_cache_block_id_device = attn.kv_cache_kernel_block_id_device
    attn.padding_offset = torch.zeros(rows, device="cuda", dtype=torch.int32)
    attn.total_tokens = rows
    attn.context_total_kv_length = rows + 8 * len(lengths)
    attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.float16))
    inputs.attention_inputs = attn
    return inputs


class AdaptiveGraphTest(unittest.TestCase):
    def test_short_graph_and_ragged_replay(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = ShapeRecordingModel()
                runner = CudaGraphRunner()
                runner.init_decode(model, 8, 64, 64, 64, [1, 4], 6, enabled, 4)
                self.assertEqual(len(model.captured_rows), 24 if enabled else 2)
                for value, lengths in enumerate(
                    ([4], [1, 3], [6, 1, 2], [1], [6], [4]), 1
                ):
                    inputs = make_inputs(lengths, value)
                    self.assertTrue(runner.canRun(inputs))
                    result = runner.forward(inputs).hidden_states.clone()
                    batch = 1 if len(lengths) == 1 else 4
                    capacity = sum(lengths) if enabled else batch * 6
                    expected_bias = torch.tensor(
                        [
                            i + 80 + 2 * j
                            for i, length in enumerate(lengths)
                            for j in range(length)
                        ],
                        device="cuda",
                        dtype=torch.float16,
                    )
                    torch.testing.assert_close(
                        result,
                        inputs.input_hiddens + capacity + expected_bias[:, None],
                        rtol=0,
                        atol=0,
                    )
                del runner

    def test_every_row_and_same_graph_different_request_boundaries(self):
        model = ShapeRecordingModel()
        runner = CudaGraphRunner()
        # Sparse user BS list must not leave holes in adaptive row coverage.
        runner.init_decode(model, 8, 64, 64, 64, [1, 4], 6, True, 4)
        self.assertEqual(sorted(model.captured_rows), list(range(1, 25)))
        for rows in list(range(1, 25)) + list(range(24, 0, -1)):
            for batch in range((rows + 5) // 6, min(rows, 4) + 1):
                lengths = [rows // batch + (i < rows % batch) for i in range(batch)]
                for lengths in (lengths, lengths[::-1]):
                    with self.subTest(rows=rows, lengths=lengths):
                        inputs = make_inputs(lengths, 3)
                        attn = inputs.attention_inputs
                        prefixes = (
                            torch.arange(batch, device="cuda", dtype=torch.int32) + 8
                        )
                        blocks = (
                            torch.arange(batch, device="cuda", dtype=torch.int32) + 1
                        )
                        attn.prefix_lengths = prefixes
                        attn.cu_kv_seqlens = torch.cat(
                            (
                                torch.zeros(1, device="cuda", dtype=torch.int32),
                                (attn.input_lengths + prefixes).cumsum(0).int(),
                            )
                        )
                        attn.kv_cache_kernel_block_id_device = blocks[
                            :, None
                        ].contiguous()
                        attn.kv_cache_kernel_block_id_host = (
                            attn.kv_cache_kernel_block_id_device.cpu()
                        )
                        self.assertTrue(runner.canRun(inputs))
                        result = runner.forward(inputs).hidden_states.clone()
                        expected_bias = torch.tensor(
                            [
                                i + 2 * (8 + i + j) + (i + 1) * 64
                                for i, length in enumerate(lengths)
                                for j in range(length)
                            ],
                            device="cuda",
                            dtype=torch.float16,
                        )
                        expected = inputs.input_hiddens + rows + expected_bias[:, None]
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        for lengths in ([1] * 5, [7], []):
            with self.subTest(invalid=lengths), self.assertRaises(Exception):
                runner.canRun(make_inputs(lengths, 0))

    def test_default_dense_rows_single_request(self):
        model = ShapeRecordingModel()
        runner = CudaGraphRunner()
        runner.init_decode(model, 8, 64, 64, 64, [], 3, True, 1)
        self.assertEqual(sorted(model.captured_rows), [1, 2, 3])
        for rows in (3, 1, 2, 1):
            inputs = make_inputs([rows], 2)
            self.assertTrue(runner.canRun(inputs))
            actual = runner.forward(inputs).hidden_states.clone()
            bias = torch.arange(rows, device="cuda", dtype=torch.float16) * 2 + 80
            torch.testing.assert_close(
                actual, inputs.input_hiddens + rows + bias[:, None], rtol=0, atol=0
            )

    def test_non_adaptive_single_token_decode_keeps_bs_buckets(self):
        class PlainAttention:
            def prepare_cuda_graph(self, inputs):
                pass

        class PlainModel(ShapeRecordingModel):
            def prepare_fmha_impl(self, inputs, *args):
                return PlainAttention()

            def forward(self, inputs, attn=None):
                rows = inputs.input_hiddens.shape[0]
                if torch.cuda.is_current_stream_capturing():
                    self.captured_rows.append(rows)
                result = PyModelOutputs()
                result.hidden_states = inputs.input_hiddens + rows
                return result

        model = PlainModel()
        runner = CudaGraphRunner()
        runner.init_decode(model, 8, 64, 64, 64, [1, 4], 1, False, 4)
        self.assertEqual(sorted(model.captured_rows), [1, 4])
        for batch in (4, 2, 1, 3):
            inputs = make_inputs([1] * batch, 2)
            inputs.attention_inputs.is_prefill = False
            inputs.attention_inputs.is_target_verify = False
            inputs.attention_inputs.sequence_lengths = torch.full(
                (batch,), 8, dtype=torch.int32
            ).pin_memory()
            self.assertTrue(runner.canRun(inputs))
            actual = runner.forward(inputs).hidden_states.clone()
            torch.testing.assert_close(
                actual,
                torch.full_like(inputs.input_hiddens, 3 if batch == 1 else 6),
                rtol=0,
                atol=0,
            )

    def test_concurrency_48_captures_and_replays_all_288_rows(self):
        model = ShapeRecordingModel()
        runner = CudaGraphRunner()
        runner.init_decode(model, 8, 64, 64, 64, [], 6, True, 48)
        self.assertEqual(sorted(model.captured_rows), list(range(1, 289)))
        for rows in range(1, 289):
            # Exercise both minimal and maximal request counts at the same key,
            # including row counts across CUDA block/warp boundaries.
            for batch in sorted({(rows + 5) // 6, min(rows, 48)}, reverse=True):
                lengths = [rows // batch + (i < rows % batch) for i in range(batch)]
                with self.subTest(rows=rows, batch=batch):
                    inputs = make_inputs(lengths, 2)
                    self.assertTrue(runner.canRun(inputs))
                    actual = runner.forward(inputs).hidden_states.clone()
                    bias = torch.tensor(
                        [
                            i + 80 + 2 * j
                            for i, length in enumerate(lengths)
                            for j in range(length)
                        ],
                        device="cuda",
                        dtype=torch.float16,
                    )
                    torch.testing.assert_close(
                        actual,
                        inputs.input_hiddens + rows + bias[:, None],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
