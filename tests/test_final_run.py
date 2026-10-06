import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from gridpfn.core.training_config import parse_args
from gridpfn.core.training_metrics import PeriodicEvaluator
from gridpfn.core.utils.convergence import ValidationStopping, feasible
from gridpfn.core.utils.run_io import atomic_json


class FinalRunTests(unittest.TestCase):
    def test_flat_tariff_expands_exactly_and_cannot_add_an_implicit_tou_charge(self):
        args = parse_args(["--preset", "ppo", "--no-tou", "--flat_grid_price", ".3"])
        self.assertEqual(args.grid_prices, [0.3] * 24)
        with self.assertRaises(SystemExit):
            parse_args(["--preset", "ppo", "--flat_grid_price", ".3"])

    def test_concurrent_manifests_remain_complete_and_failed_writes_preserve_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda i: atomic_json(path, {"worker": i}), range(32)))
            previous = json.loads(path.read_text())
            self.assertIn(previous["worker"], range(32))
            with self.assertRaises(ValueError):
                atomic_json(path, {"reward": float("nan")})
            self.assertEqual(json.loads(path.read_text()), previous)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_default_preserves_original_constraints_and_optimized_training(self):
        args = parse_args([])
        self.assertEqual(args.ac_service, "energy_quota")
        self.assertEqual(args.scaler_mode, "local")
        self.assertEqual(args.run_models, ["fedavg"])
        self.assertEqual((args.state_dim, args.validation_days, args.eval_step), (17, 14, 50))
        self.assertEqual((args.bc_rounds, args.bc_weight, args.actor_q_weight), (20, 1, 0.1))
        self.assertTrue(args.batched_updates)
        self.assertFalse(args.warmup_learning)
        self.assertEqual((args.min_episodes, args.patience, args.episode), (3000, 20, 6000))
        self.assertEqual(parse_args(["--episode", "4000"]).episode, 4000)
        legacy = parse_args(["--preset", "legacy"])
        self.assertFalse(legacy.batched_updates)
        self.assertEqual(legacy.bc_rounds, 0)
        self.assertEqual(len(legacy.run_models), 5)

    def test_ppo_default_uses_confirmed_day_mixing_and_reference_is_reproducible(self):
        final = parse_args(["--preset", "ppo"])
        original = parse_args(["--preset", "ppo_reference"])
        self.assertTrue(final.ppo_shuffle_days)
        self.assertTrue(final.ppo_compile_mapping)
        self.assertEqual(final.embedding_cache, "results/frozen_features")
        self.assertFalse(original.ppo_shuffle_days)
        self.assertFalse(original.ppo_compile_mapping)
        self.assertIsNone(original.embedding_cache)
        for key in ("gamma", "ppo_gae_lambda", "head_width", "value_width"):
            self.assertEqual(getattr(final, key), getattr(original, key))
        self.assertEqual(final.quota_correction, 6.0)
        self.assertEqual(final.target_temperature_bounds, (-5, 26))
        self.assertAlmostEqual(final.ppo_ac_std, 1 / 30)
        self.assertEqual((final.bc_rounds, final.bc_weight), (60, 0))
        self.assertEqual((original.bc_rounds, original.bc_weight), (20, 0.1))
        self.assertIsNone(final.synthetic_data)
        self.assertEqual(final.ppo_value_warmup_days, 0)
        self.assertEqual(original.quota_correction, 2.0)
        self.assertEqual(original.target_temperature_bounds, (-5, 21.8))
        self.assertIsNone(original.ppo_ac_std)
        self.assertFalse(parse_args(["--preset", "ppo", "--no-ppo_shuffle_days"]).ppo_shuffle_days)

    def test_stopping_uses_feasible_validation_improvement_and_minimum_duration(self):
        reference = dict(reward=0, comfort_pct=90, elec_cost=5)
        stopping = ValidationStopping(min_episodes=200, patience=2, min_delta=0.01)
        self.assertFalse(stopping.update(0, reference, reference))
        self.assertFalse(stopping.update(50, reference | {"reward": 0.005}, reference))
        self.assertFalse(stopping.update(100, reference, reference))
        self.assertFalse(stopping.update(150, reference | {"reward": 1}, reference))
        # Large reward cannot reset patience by violating service or cost.
        self.assertFalse(
            stopping.update(200, reference | {"reward": 10, "comfort_pct": 88}, reference)
        )
        self.assertTrue(stopping.update(250, reference | {"reward": 10, "elec_cost": 6}, reference))
        self.assertEqual(stopping.best, 1)
        self.assertFalse(feasible({"reward": 100}, reference))

    def test_test_data_cannot_control_stopping_and_zero_patience_disables_it(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            PeriodicEvaluator(SimpleNamespace(), 50, split="test", patience=2)
        with self.assertRaises(ValueError):
            ValidationStopping(patience=-1)
        stopping = ValidationStopping(min_episodes=0, patience=0)
        ref = dict(reward=0, comfort_pct=90, elec_cost=5)
        for episode in range(10):
            self.assertFalse(stopping.update(episode, ref, ref))


if __name__ == "__main__":
    unittest.main()
