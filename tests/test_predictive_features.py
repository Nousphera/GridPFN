"""Observation boundaries, immutable contexts and causal feature preparation."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gridpfn.core.batched_learning import _Stack
from gridpfn.core.forecasting import DirectForecaster
from gridpfn.core.model import Actor, Critic, heads_from_state
from gridpfn.core.predictive_features import PredictiveContext, observe
from gridpfn.core.tabpfn_adaptation import PromptRegressor
from gridpfn.experiments.build_predictive_features import numeric_features


class PredictiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_privileged_future_features_cannot_enter_the_causal_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            np.savez(path, privileged=True)
            with self.assertRaisesRegex(ValueError, "Privileged future features"):
                PredictiveContext(path, np.zeros((1, 24, 5)), {})

    def test_auxiliary_checkpoint_preserves_physical_state_and_actions(self):
        actor = Actor(
            17,
            3,
            feature_mode="raw",
            auxiliary_dim=17,
            hidden_dim=16,
            learned_discrete=True,
            stochastic_std=0.1,
            feasible_dt=1,
        )
        critic = Critic(
            17,
            3,
            2,
            feature_mode="raw",
            auxiliary_dim=17,
            hidden_dim=16,
            value_head=True,
            value_hidden_dim=32,
        )
        states = torch.randn(5, 34)
        states[:, 8], states[:, 11] = 0.2, 0.1
        features = actor.prepare_features(states)
        torch.testing.assert_close(features[:, -17:], states[:, :17])
        a, c = heads_from_state(actor.state_dict(), critic.state_dict(), "cpu")
        torch.testing.assert_close(a(states), actor(states))
        torch.testing.assert_close(c(states, a(states)), critic(states, actor(states)))
        torch.testing.assert_close(
            a._feasible_bounds(features)[0], actor._feasible_bounds(features)[0]
        )
        self.assertEqual(a.fc1.in_features, 34)
        self.assertEqual(c.value_fc1.out_features, 32)

    def test_fast_heads_preserve_forward_and_parameter_gradients(self):
        for kind in ("distribution", "value"):
            heads = [
                Actor(17, 3, feature_mode="raw", learned_discrete=True, stochastic_std=0.1)
                if kind == "distribution"
                else Critic(17, 3, 2, feature_mode="raw", value_head=True, value_normalization=True)
                for _ in range(3)
            ]
            stack = _Stack(heads, "actor" if kind == "distribution" else "critic")
            x = torch.randn(3, 8, 17)
            fast, reference = stack(x, kind=kind), stack.functional(x, kind=kind)
            fast = fast if isinstance(fast, tuple) else (fast,)
            reference = reference if isinstance(reference, tuple) else (reference,)
            for a, b in zip(fast, reference, strict=True):
                torch.testing.assert_close(a, b)
            parameters = stack.trainable()
            ga = torch.autograd.grad(
                sum(a.square().sum() for a in fast), parameters, allow_unused=True
            )
            gb = torch.autograd.grad(
                sum(a.square().sum() for a in reference), parameters, allow_unused=True
            )
            for a, b in zip(ga, gb, strict=True):
                if a is None:
                    self.assertIsNone(b)
                else:
                    torch.testing.assert_close(a, b)

    def test_context_rejects_training_drift_and_leaves_simulator_state_intact(self):
        train = np.zeros((2, 24, 5), dtype=np.float32)
        dates = ["2026-06-01", "2026-06-02"]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "features.npz"
            np.savez(
                p,
                dates=dates,
                values=np.ones((2, 25, 3)),
                train_dates=dates,
                train_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
            )
            context = PredictiveContext(p, train, {"train_dates": dates})
            state = np.arange(17, dtype=np.float32)
            result = observe(
                SimpleNamespace(predictive_context=context, scaler={"train_dates": dates}),
                state,
                0,
                0,
            )
            np.testing.assert_array_equal(result[:17], state)
            self.assertEqual(len(result), 20)
            with self.assertRaisesRegex(ValueError, "differs"):
                PredictiveContext(p, train + 1, {"train_dates": dates})

    def test_future_measurements_do_not_enter_history_or_forecast_inputs(self):
        rng = np.random.default_rng(5)
        train = rng.normal(size=(5, 24, 4))
        train[:, :, :2] = abs(train[:, :, :2])
        query = train[:1].copy()
        changed = query.copy()
        changed[:, 10:, :] += 100
        model = DirectForecaster("seasonal").fit(train[1:])
        tables = [model.predict_table(x) for x in (query, changed)]
        features = [
            numeric_features(x, ["2026-07-20"], f)
            for x, f in zip((query, changed), tables, strict=True)
        ]
        np.testing.assert_array_equal(features[0][:, :10], features[1][:, :10])
        self.assertFalse(np.array_equal(features[0][:, 10], features[1][:, 10]))

    def test_constant_prompt_context_uses_exact_constant_without_loading_a_backbone(self):
        x = np.zeros((24, 16))
        y = np.ones(24)
        groups = np.repeat(np.arange(6), 4)
        model = PromptRegressor("cpu", steps=12).fit(x, y, groups)
        np.testing.assert_array_equal(model.predict(x), y)
        self.assertEqual(model.metrics["parameters"], 0)
        self.assertFalse(hasattr(model, "model"))

    def test_short_forecast_queries_preserve_consumed_features(self):
        rng = np.random.default_rng(7)
        train = rng.uniform(size=(4, 24, 4))
        query = train[:1]
        model = DirectForecaster("seasonal").fit(train[1:])
        full = model.predict_table(query)
        short = model.predict_table(query, max_lead=6)
        np.testing.assert_array_equal(
            numeric_features(query, ["2026-07-20"], full),
            numeric_features(query, ["2026-07-20"], short),
        )


if __name__ == "__main__":
    unittest.main()
