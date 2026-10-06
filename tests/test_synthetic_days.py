"""Synthetic scenarios preserve daily markets and cannot use held-out donors."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gridpfn.core.agents.onpolicy import OnPolicyAgent, OnPolicyLearner
from gridpfn.core.model import Actor, Critic
from gridpfn.core.synthetic_days import ScenarioConfig, attach_synthetic_days, generate


class SyntheticDaysTests(unittest.TestCase):
    def bundles(self):
        result = []
        for home in range(2):
            data = np.zeros((4, 24, 8))
            data[..., 2] = np.arange(24)
            data[..., [0, 5, 6, 7]] = 0.2 + 0.05 * home
            data[..., 1] = np.maximum(0, np.sin(np.arange(24) * np.pi / 12 - np.pi / 2))
            data[..., 3] = np.arange(4)[:, None] / 4
            data[..., 4] = 0.3 + np.arange(4)[:, None] / 10
            scaler = {
                "min": [0, 0, 0, 20, 0, 0, 0],
                "max": [2, 2, 0.5, 35, 2.5, 6, 1],
                "cols": ["load", "pv", "price", "temp", "ac", "ev", "wm"],
                "col_to_scaler_idx": {0: 0, 1: 1, 2: None, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},
                "delta_t": 1,
                "train_dates": [f"2019-06-0{i}" for i in range(1, 5)],
                "train_end_exclusive": "2019-07-18",
            }
            result.append((data, np.ones((1, 24, 8)), ["2019-07-18"], scaler))
        return result

    def test_generation_is_deterministic_and_ignores_all_heldout_values(self):
        bundles = self.bundles()
        original, manifest = generate(bundles, [1, 2], ScenarioConfig())
        modified = copy.deepcopy(bundles)
        for bundle in modified:
            bundle[1][:] = 1e9
        actual, actual_manifest = generate(modified, [1, 2], ScenarioConfig())
        np.testing.assert_array_equal(original, actual)
        self.assertEqual(manifest, actual_manifest)

    def test_whole_day_mixes_share_weather_prices_clock_and_keep_night_pv_zero(self):
        rows, manifest = generate(self.bundles(), [1, 2], ScenarioConfig())
        np.testing.assert_allclose(rows[0, ..., [3, 4]], rows[1, ..., [3, 4]])
        np.testing.assert_array_equal(
            rows[..., 2], np.broadcast_to(np.arange(24), rows[..., 2].shape)
        )
        np.testing.assert_array_equal(rows[..., 1][:, :, :6], 0)
        self.assertEqual(len(manifest["provenance"]), 64)
        self.assertTrue(np.isfinite(rows).all())

    def test_perturbations_preserve_shared_weather_and_feasible_service_demands(self):
        rows, _ = generate(self.bundles(), [1, 2], ScenarioConfig(method="mixed_jitter"))
        np.testing.assert_allclose(rows[0, ..., [3, 4]], rows[1, ..., [3, 4]])
        self.assertTrue((rows[..., [0, 1, 3, 5, 6, 7]] >= 0).all())
        self.assertLessEqual(rows[..., 5].max(), 1)
        self.assertLessEqual(rows[..., 6].max(), 1)

    def test_validation_donors_and_mismatched_shared_weather_are_rejected(self):
        bundles = self.bundles()
        bundles[0][3]["train_dates"][0] = "2019-08-01"
        with self.assertRaisesRegex(ValueError, "cutoff"):
            generate(bundles, [1, 2], ScenarioConfig())
        bundles = self.bundles()
        bundles[0][0][..., 4] += 0.5
        with self.assertRaisesRegex(ValueError, "weather"):
            generate(bundles, [1, 2], ScenarioConfig())

    def test_synthetic_days_cannot_change_the_simulator_time_grid(self):
        bundles = self.bundles()
        bundles[0][0][..., 2] = np.arange(24)[::-1]
        with self.assertRaisesRegex(ValueError, "ordered hours"):
            generate(bundles, [1, 2], ScenarioConfig())
        bundles = self.bundles()
        bundles[0][3]["delta_t"] = 0.5
        with self.assertRaisesRegex(ValueError, "hourly"):
            generate(bundles, [1, 2], ScenarioConfig())

    def test_mixing_cannot_select_a_previous_tariff_slot_through_clock_roundoff(self):
        from gridpfn.core.environment import HOME_ENERGY_MGNT

        rows, _ = generate(self.bundles(), [1, 2], ScenarioConfig(days=64))
        env = HOME_ENERGY_MGNT(np.zeros((24, 8)))
        env.tou_prices_by_hour = np.arange(24)
        prices = np.array([env._resolve_price(hour, 0) for hour in rows[..., 2].ravel()])
        np.testing.assert_array_equal(prices.reshape(rows.shape[:3]), rows[..., 2])

    def test_archive_matches_training_and_leaves_real_dates_and_scaling_unchanged(self):
        bundles = self.bundles()
        rows, manifest = generate(bundles, [1, 2], ScenarioConfig(days=3))
        clients = [
            SimpleNamespace(home_id=h, train_data=b[0].copy(), scaler=copy.deepcopy(b[3]))
            for h, b in zip([1, 2], bundles, strict=True)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            np.savez_compressed(path / "days.npz", days=rows)
            manifest["data_sha256"] = hashlib.sha256((path / "days.npz").read_bytes()).hexdigest()
            (path / "manifest.json").write_text(json.dumps(manifest))
            attach_synthetic_days(clients, path)
            self.assertEqual([c.real_train_count for c in clients], [4, 4])
            for client, bundle in zip(clients, bundles, strict=True):
                self.assertEqual(client.scaler, bundle[3])
                np.testing.assert_array_equal(client.train_data[:4], bundle[0])
            (path / "days.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                attach_synthetic_days(clients, path)

    def test_loading_rejects_inexact_clocks_and_checks_all_homes_before_mutating(self):
        bundles = self.bundles()
        rows, manifest = generate(bundles, [1, 2], ScenarioConfig(days=3))
        clients = [
            SimpleNamespace(home_id=h, train_data=b[0].copy(), scaler=copy.deepcopy(b[3]))
            for h, b in zip([1, 2], bundles, strict=True)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)

            def save(data):
                np.savez_compressed(path / "days.npz", days=data)
                manifest["data_sha256"] = hashlib.sha256(
                    (path / "days.npz").read_bytes()
                ).hexdigest()
                (path / "manifest.json").write_text(json.dumps(manifest))

            invalid = rows.copy()
            invalid[0, 0, 19, 2] = np.nextafter(19.0, 0)
            save(invalid)
            with self.assertRaisesRegex(ValueError, "exact ordered hours"):
                attach_synthetic_days(clients, path)
            save(rows)
            clients[1].train_data[0, 0, 0] += 0.1
            with self.assertRaisesRegex(ValueError, "donors or preprocessing"):
                attach_synthetic_days(clients, path)
            self.assertFalse(hasattr(clients[0], "real_train_count"))
            np.testing.assert_array_equal(clients[0].train_data, bundles[0][0])

    def test_joint_sampler_switches_to_real_days_after_pretraining(self):
        torch.set_num_threads(1)
        agent = OnPolicyAgent(
            Actor(
                17, 3, feature_mode="raw", hidden_dim=16, learned_discrete=True, stochastic_std=0.1
            ),
            Critic(17, 3, 2, feature_mode="raw", hidden_dim=16, value_head=True),
            2,
            3,
            17,
            hyperparams=dict(ppo_shuffle_days=True, synthetic_fraction=1, synthetic_until=5),
        )
        learner = OnPolicyLearner([agent], 1)
        client = SimpleNamespace(
            train_data=np.zeros((7, 24, 8)), real_train_count=4, scaler=self.bundles()[0][3]
        )
        rng_before = torch.get_rng_state().clone()
        self.assertGreaterEqual(learner.training_indices([client], 0)[0], 4)
        self.assertTrue(learner.last_synthetic)
        self.assertLess(learner.training_indices([client], 5)[0], 4)
        self.assertFalse(learner.last_synthetic)
        torch.testing.assert_close(torch.get_rng_state(), rng_before)

    def test_synthetic_days_cannot_recalibrate_the_original_tariff(self):
        from gridpfn.core.em_strategy import ToUTariff

        bundle = self.bundles()[0]
        original = SimpleNamespace(train_data=bundle[0].copy(), scaler=bundle[3])
        augmented = copy.deepcopy(original)
        synthetic = bundle[0].copy()
        synthetic[..., 3] = 10
        augmented.real_train_count = len(augmented.train_data)
        augmented.train_data = np.concatenate((augmented.train_data, synthetic))
        np.testing.assert_array_equal(
            ToUTariff().compute_global_tou_prices([original]),
            ToUTariff().compute_global_tou_prices([augmented]),
        )
