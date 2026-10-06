import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from gridpfn.core.model import (
    _embedding_cache,
    configure_embedding_device,
    embedding_rows,
    encoder_context_identity,
    precompute_embeddings,
    validate_encoder_context,
)
from gridpfn.core.utils.feature_cache import FrozenFeatureCache


class FrozenFeatureTests(unittest.TestCase):
    def test_context_binds_training_data_and_clears_features_when_labels_change(self):
        rows = np.arange(24, dtype=np.float32).reshape(3, 8)
        client = SimpleNamespace(
            home_id=27, train_data=rows[None], scaler={"train_dates": ["2019-06-01"]}
        )
        strategy, market = {"ac_energy_quota": True}, {"enabled": True, "price": 0.1}
        metadata = dict(
            split="train",
            home_ids=[27],
            dates=["2019-06-01"],
            training_sha256=hashlib.sha256(client.train_data.tobytes()).hexdigest(),
            strategy=strategy,
            market=market,
        )

        class Backbone(torch.nn.Module):
            def forward(self, x, y, **kwargs):
                return {"test_embeddings": x[len(y) :, :, :4] + y.mean()}

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("gridpfn.core.model._tabpfn_backbone", return_value=Backbone()),
        ):
            archive = Path(tmp) / "context.npz"
            try:
                for label in (1, 2):
                    np.savez(
                        archive,
                        features=rows,
                        labels=np.full(3, label),
                        metadata=json.dumps(metadata),
                    )
                    configure_embedding_device("cpu", encoder_context=archive)
                    self.assertEqual(len(_embedding_cache("cpu")), 0)
                    validate_encoder_context([client], strategy, market)
                    actual = embedding_rows(rows, "cpu")
                    torch.testing.assert_close(actual, torch.from_numpy(rows[:, :4]) + label)
                    self.assertIsNotNone(encoder_context_identity())
                client.train_data = client.train_data + 1
                with self.assertRaisesRegex(ValueError, "training cohort"):
                    validate_encoder_context([client], strategy, market)
                client.train_data = rows[None]
                client.scaler["train_dates"] = ["2019-07-18"]
                with self.assertRaisesRegex(ValueError, "training cohort"):
                    validate_encoder_context([client], strategy, market)
                metadata["split"] = "validation"
                np.savez(archive, features=rows, labels=np.ones(3), metadata=json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "training-only"):
                    configure_embedding_device("cpu", encoder_context=archive)
            finally:
                configure_embedding_device()

    def test_exact_rows_and_encoder_identity_isolate_entries(self):
        rows = np.arange(24, dtype=np.float32).reshape(3, 8)
        matrix = np.ones((3, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            cache = FrozenFeatureCache(tmp, {"weights": "first"})
            cache.put(rows, matrix)
            np.testing.assert_array_equal(cache.get(rows, 4), matrix)
            self.assertIsNone(cache.get(rows[::-1], 4))
            changed = rows.copy()
            changed[0, 3] += 0.01
            self.assertIsNone(cache.get(changed, 4))
            self.assertIsNone(FrozenFeatureCache(tmp, {"weights": "second"}).get(rows, 4))
            with self.assertRaisesRegex(ValueError, "Invalid"):
                cache.get(rows, 5)

    def test_concurrent_writes_are_complete_and_nonfinite_values_fail(self):
        rows = np.ones((3, 8), dtype=np.float32)
        matrix = np.ones((3, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            cache = FrozenFeatureCache(tmp, {"weights": "same"})
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: cache.put(rows, matrix), range(12)))
            np.testing.assert_array_equal(cache.get(rows, 4), matrix)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
            matrix[0, 0] = np.nan
            cache.put(rows, matrix)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                cache.get(rows, 4)

    def test_warm_features_skip_backbone_and_remain_trainable_inputs(self):
        class Backbone(torch.nn.Module):
            embedding_dim = 4

            def forward(self, x, *args, **kwargs):
                return {"test_embeddings": x[2:, :, :4] * 2}

        rows = torch.arange(27, dtype=torch.float32).reshape(3, 9)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("gridpfn.core.model._tabpfn_backbone", return_value=Backbone()),
            patch("gridpfn.core.utils.run_io.backbone_identity", return_value={"weights": "test"}),
        ):
            configure_embedding_device("cpu", tmp)
            table = _embedding_cache("cpu")
            try:
                table.clear()
                with torch.inference_mode():
                    self.assertEqual(precompute_embeddings(rows, "cpu"), 3)
                expected = table.matrix.clone()
                table.clear()
                with patch.object(Backbone, "forward", side_effect=AssertionError("Missed")):
                    with torch.inference_mode():
                        self.assertEqual(precompute_embeddings(rows, "cpu"), 3)
                torch.testing.assert_close(table.matrix, expected, atol=0, rtol=0)
                layer = torch.nn.Linear(4, 1)
                layer(table.matrix).sum().backward()
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertFalse(table.matrix.is_inference())
                self.assertEqual(precompute_embeddings(rows[:0], "cpu"), 0)
                with patch("gridpfn.core.model._tabpfn_backbone", side_effect=AssertionError("Reloaded")):
                    self.assertEqual(precompute_embeddings(rows, "cpu"), 0)
            finally:
                table.clear()
                configure_embedding_device()
