"""Compare actual embedding and rerank service outputs across request batches."""

import json
import os
import unittest
from pathlib import Path

import requests
import torch

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class EmbeddingBatchInvariantTest(unittest.TestCase):
    def test_service_preserves_output_under_rebatching(self):
        mode = os.environ["RTP_TEST_BATCH_INVARIANT_TASK"]
        fixture_name = {"embedding": "q_r.json", "rerank": "reranker_q_r.json"}[mode]
        fixture = json.loads(
            (Path(__file__).parent / "data/model/bert" / fixture_name).read_text()
        )
        server = MagaServerManager(
            env_args={"LOAD_PYTHON_MODEL": "1", "FT_SERVER_TEST": "1"},
            role_name=mode,
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        self.addCleanup(server.stop_server)
        self.assertTrue(
            server.start_server(
                model_type=fixture["model_type"],
                model_path=os.environ.get("RTP_TEST_MODEL_PATH", fixture["model_path"]),
            )
        )
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)
        url = f"http://127.0.0.1:{server.port}{fixture['endpoint']}"

        def score(items, target_index):
            if mode == "embedding":
                query = {"input": items, "model": "embedding_batch_invariant"}
            else:
                query = {
                    "query": "What habitat do giant pandas need?",
                    "documents": items,
                    "sorted": False,
                }
            with session.post(url, json=query, timeout=120) as response:
                response.raise_for_status()
                result = response.json()
            key = "data" if mode == "embedding" else "results"
            self.assertEqual(len(result[key]), len(items))
            item = result[key][target_index]
            self.assertEqual(item["index"], target_index)
            value = (
                item["embedding"] if mode == "embedding" else item["relevance_score"]
            )
            assert isinstance(value, (list, float, int))
            tensor = torch.tensor(value, dtype=torch.float32)
            self.assertGreater(tensor.numel(), 0)
            self.assertTrue(torch.isfinite(tensor).all().item())
            return tensor.reshape(-1).view(torch.int32)

        target = "Giant pandas live in temperate mountain forests with bamboo."
        other = "Other forests in the region have very different rainfall. " * 11
        baseline = score([target], 0)
        self.assertTrue(torch.equal(score(["Hi", target, other], 1), baseline))
        self.assertTrue(torch.equal(score([other, "Hi", target], 2), baseline))


if __name__ == "__main__":
    unittest.main()
