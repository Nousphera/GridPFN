import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.client import Client
from gridpfn.core.dataset import _construct_dataset, feature_columns
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.evaluate import ModelEvaluator
from gridpfn.core.model import (
    Actor,
    Critic,
    checkpoint_feature_mode,
    checkpoint_state_dim,
    precompute_embeddings,
)
from gridpfn.core.server import Server
from gridpfn.core.training_metrics import MetricsLogger, PeriodicEvaluator
from gridpfn.core.utils.agent_utils import soft_update_target_network
from gridpfn.experiments.live_plot import IncrementalRecords, PlotDashboard


class EfficientPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_observation_adapter_preserves_live_and_terminal_coordinates(self):
        from gridpfn.experiments.audit_constraints import (
            REFERENCE,
            reference_module,
            reference_observation,
        )

        original, _ = reference_module("environment", REFERENCE)
        day = np.tile([3, 4, 0, 0.3, 30, 1.4, 2, 0.12], (24, 1)).astype(float)
        day[:, 2] = np.arange(24)
        scaler = {
            "delta_t": 1,
            "col_to_scaler_idx": {0: 0, 1: 1, 2: None, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},
            "min": [1, 0, 0.1, 20, 0, 0, 0],
            "max": [6, 5, 0.5, 40, 2, 3, 1],
        }
        normalized = day.copy()
        for column, index in scaler["col_to_scaler_idx"].items():
            if index is not None:
                normalized[:, column] = (day[:, column] - scaler["min"][index]) / (
                    scaler["max"][index] - scaler["min"][index]
                )
        old = original.HOME_ENERGY_MGNT(day, scaler={"delta_t": 1}, fixed_cost=5)
        new = HOME_ENERGY_MGNT(normalized, scaler=scaler, fixed_cost=5, state_dim=17)
        old.reset()
        new.reset()
        for step in range(25):
            np.testing.assert_allclose(
                reference_observation(old, scaler),
                new._state_for_step(step).astype(np.float32),
                atol=1e-7,
                rtol=0,
            )
            if step < 24:
                action = (int(step >= 10), [0.5, 1.0, 0.1])
                old.step(action)
                new.step(action)

    def test_peer_volume_and_bill_match_analytical_grid_first_settlement(self):
        class IdleActor(torch.nn.Module):
            feature_mode = "raw"

            def forward(self, states):
                return torch.zeros(len(states), 3)

        clients = []
        for home in (27, 950):
            day = np.zeros((24, 8))
            day[:, 2], day[:, 3], day[:, 4] = np.arange(24), 0.2, 20
            day[12, 1 if home == 27 else 0] = 2
            clients.append(
                SimpleNamespace(
                    home_id=home,
                    test_dates=["test"],
                    test_data=[day],
                    scaler={},
                    fixed_cost=0,
                    state_dim=17,
                    device=torch.device("cpu"),
                    fedavg_agent=SimpleNamespace(
                        actor_net=IdleActor(),
                        device=torch.device("cpu"),
                        critic_net=lambda state, control: torch.zeros(len(state), 2),
                    ),
                )
            )
        server = SimpleNamespace(
            clients=clients,
            p2p_config={"enabled": True, "price": 0.1},
            em_strategy={"pv_curtail": 2.5, "export_price": 0.025},
        )
        evaluator = PeriodicEvaluator(SimpleNamespace(home_ids=[27, 950]), 50)
        record = evaluator.evaluate(server, include_losses=False, include_actions=True)
        self.assertEqual(np.shape(record["action_records"]), (2, 1, 24, 4))
        np.testing.assert_array_equal(record["action_records"], np.zeros((2, 1, 24, 4)))
        self.assertAlmostEqual(record["p2p_kwh"], 0.25)
        self.assertAlmostEqual(record["peer_export_kwh"], 0.25)
        self.assertAlmostEqual(record["homes"][1]["p2p_kwh"], 0.5)
        self.assertAlmostEqual(record["energy_bill_without_dr"], (0.2 * 1.5 - 0.025 * 1.5) / 2)

    def test_chronological_split_fits_only_earlier_training_and_reuses_test_scaler(self):
        frames = []
        for date, temp in [
            ("2019-06-01", 20),
            ("2019-06-02", 25),
            ("2019-07-20", 100),
            ("2019-08-01", 70),
        ]:
            f = pd.DataFrame(
                {"datetime": pd.date_range(date, periods=96, freq="15min"), "t": np.arange(96)}
            )
            for col in feature_columns:
                f[col] = 0.0
            f["temp (C)"] = temp
            frames.append(f)
        with patch("gridpfn.core.dataset.merge_temp_price", side_effect=lambda data, **_: data):
            train, val, dates, scaler = _construct_dataset(
                pd.concat(frames), validation_days=14, split="validation"
            )
            train2, test, test_dates, scaler2 = _construct_dataset(
                pd.concat(frames), validation_days=14
            )
        self.assertEqual(dates, ["2019-07-20"])
        self.assertEqual(test_dates, ["2019-08-01"])
        np.testing.assert_array_equal(train, train2)
        self.assertEqual(scaler["max"], scaler2["max"])
        self.assertEqual(val[0, 0, 4], 16)
        self.assertEqual(test[0, 0, 4], 10)

    def test_incremental_reader_partial_tail_no_reread_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_bytes(b'{"a":1}\n{"a":')
            reader = IncrementalRecords(path)
            self.assertEqual(reader.poll(), [{"a": 1}])
            initial = reader.bytes_read
            reader.poll()
            self.assertEqual(reader.bytes_read, initial)
            with path.open("ab") as f:
                f.write(b"2}\n")
            self.assertEqual(reader.poll(), [{"a": 1}, {"a": 2}])
            self.assertEqual(reader.bytes_read, path.stat().st_size)
            generation = reader.generation
            path.write_text('{"a":3}\n')
            self.assertEqual(reader.poll(), [{"a": 3}])
            self.assertNotEqual(reader.generation, generation)

    def test_monitor_discovery_prefers_active_study_over_recent_completion(self):
        import json
        import os

        from gridpfn.experiments.monitor_run import newest_run

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, state, created in [("active", "running", 10), ("complete", "completed", 20)]:
                folder = root / name
                folder.mkdir()
                (folder / "study.json").write_text(
                    json.dumps({"state": state, "pid": os.getpid(), "created_at": created})
                )
            self.assertEqual(newest_run(root), root / "active")
            (root / "active/study.json").write_text(
                json.dumps({"state": "completed", "created_at": 10})
            )
            self.assertEqual(newest_run(root), root / "complete")

    def test_dashboard_rejects_run_outside_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dashboard = PlotDashboard(root, 1, 10)
            with self.assertRaises(ValueError):
                dashboard.select_run("../outside")

    def test_raw_client_constructs_only_selected_agent_without_backbone(self):
        day = np.zeros((24, 8))
        day[:, 2] = np.arange(24)
        day[:, 4] = 20
        with patch(
            "gridpfn.core.model._tabpfn_backbone", side_effect=AssertionError("Raw policy loaded backbone")
        ):
            client = Client(
                (np.array([day]), np.array([day]), ["2019-07-20"], {}),
                17,
                3,
                2,
                10,
                0.3,
                active_model="fedavg",
                feature_mode="raw",
                device="cpu",
            )
            self.assertFalse(hasattr(client, "agent"))
            self.assertFalse(hasattr(client, "fedfit_agent"))
            self.assertEqual(checkpoint_state_dim(client.fedavg_actor_net.state_dict()), 17)
            self.assertEqual(checkpoint_feature_mode(client.fedavg_actor_net.state_dict()), "raw")

    def test_optimized_update_matches_reference_update(self):
        # Compare parameter updates, not just loss decrease. Preserve the same replay draw.
        for mode in ["raw", "frozen", "normalized"]:
            with self.subTest(mode=mode):
                torch.manual_seed(5)
                a = P_DQN(
                    Actor(17, 3, feature_mode=mode),
                    Critic(17, 3, 2, feature_mode=mode),
                    2,
                    3,
                    17,
                    {"batch_size": 4},
                )
                states = np.random.default_rng(6).random((6, 17)).astype("float32")
                if mode != "raw":
                    precompute_embeddings(states, "cpu")
                for i in range(6):
                    a.store_transition(
                        states[i],
                        (i % 2, np.array([1.0, 2.0, 0.0])),
                        float(-i),
                        states[(i + 1) % 6],
                        i == 5,
                    )
                b = copy.deepcopy(a)
                random.seed(42)
                actual = a.learn()
                random.seed(42)
                s, actions, r, ns, done = b.memory.sample(4)
                s, ns, r, done = [torch.as_tensor(v, dtype=torch.float32) for v in (s, ns, r, done)]
                act = torch.tensor(np.array([v[1] for v in actions]), dtype=torch.float32)
                disc = torch.tensor([v[0] for v in actions])[:, None]
                with torch.no_grad():
                    target = (
                        r
                        + b.gamma
                        * (1 - done)
                        * b.critic_target_net(ns, b.actor_target_net(ns)).max(1).values
                    )
                lc = (b.critic_net(s, act).gather(1, disc).squeeze(1) - target).square().mean()
                b.critic_optimizer.zero_grad()
                lc.backward()
                b.critic_optimizer.step()
                la = -b.critic_net(s, b.actor_net(s)).max(1).values.mean()
                b.actor_optimizer.zero_grad()
                b.critic_optimizer.zero_grad()
                la.backward()
                b.actor_optimizer.step()
                soft_update_target_network(b.actor_net, b.actor_target_net, b.actor_tau)
                soft_update_target_network(b.critic_net, b.critic_target_net, b.critic_tau)
                np.testing.assert_allclose(actual, [la.item(), lc.item()], rtol=1e-6, atol=1e-6)
                for name in ["actor_net", "critic_net", "actor_target_net", "critic_target_net"]:
                    for x, y in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
                        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)
                self.assertTrue(all(p.grad is None for p in a.critic_net.parameters()))

    def test_validation_checkpoint_retains_best_separate_from_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = MetricsLogger(Path(directory) / "metrics.jsonl", [27])
            evaluator = PeriodicEvaluator(logger, 50, split="validation")
            actor = Actor(17, 3, feature_mode="raw")
            critic = Critic(17, 3, 2, feature_mode="raw")
            server = SimpleNamespace(
                clients=[
                    SimpleNamespace(
                        fedavg_agent=SimpleNamespace(actor_net=actor, critic_net=critic)
                    )
                ]
            )
            evaluator.save_checkpoint(server, 50, {"reward": -10, "comfort_pct": 80})
            evaluator.save_checkpoint(server, 100, {"reward": -20, "comfort_pct": 60})
            base = Path(directory) / "checkpoints"
            self.assertEqual(torch.load(base / "best/heads.pt", weights_only=True)["episode"], 50)
            self.assertEqual(
                torch.load(base / "latest/heads.pt", weights_only=True)["episode"], 100
            )

    def test_feasible_checkpoint_rejects_cost_or_comfort_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = PeriodicEvaluator(
                MetricsLogger(Path(directory) / "metrics.jsonl", [27]), 50, split="validation"
            )
            actor = Actor(17, 3, feature_mode="raw")
            critic = Critic(17, 3, 2, feature_mode="raw")
            server = SimpleNamespace(
                clients=[
                    SimpleNamespace(
                        fedavg_agent=SimpleNamespace(actor_net=actor, critic_net=critic)
                    )
                ]
            )
            for episode, reward, comfort, cost in [
                (0, -30, 89, 3.5),
                (50, -25, 90, 3.3),
                (100, -20, 90, 4),
                (150, -21, 85, 3),
            ]:
                evaluator.save_checkpoint(
                    server, episode, dict(reward=reward, comfort_pct=comfort, elec_cost=cost)
                )
            base = Path(directory) / "checkpoints"
            self.assertEqual(torch.load(base / "initial/heads.pt", weights_only=True)["episode"], 0)
            self.assertEqual(torch.load(base / "best/heads.pt", weights_only=True)["episode"], 100)
            self.assertEqual(
                torch.load(base / "best_feasible/heads.pt", weights_only=True)["episode"], 50
            )

    def test_fixed_service_reference_rejects_ineligible_initial_policy_and_wrong_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            contract = dict(
                home_ids=[27], split="validation", dates=["2019-07-18"], comfort_pct=90, elec_cost=2
            )
            reference.write_text(json.dumps(contract))
            evaluator = PeriodicEvaluator(
                MetricsLogger(root / "metrics.jsonl", [27]),
                50,
                split="validation",
                selection_reference=reference,
            )
            server = SimpleNamespace(
                clients=[
                    SimpleNamespace(
                        fedavg_agent=SimpleNamespace(
                            actor_net=Actor(17, 3, feature_mode="raw"),
                            critic_net=Critic(17, 3, 2, feature_mode="raw"),
                        )
                    )
                ]
            )
            evaluator.save_checkpoint(
                server, 0, dict(reward=100, comfort_pct=50, elec_cost=1, dates=contract["dates"])
            )
            self.assertFalse((root / "checkpoints/best_feasible").exists())
            evaluator.save_checkpoint(
                server, 50, dict(reward=10, comfort_pct=90, elec_cost=2, dates=contract["dates"])
            )
            selected = torch.load(root / "checkpoints/best_feasible/heads.pt", weights_only=True)
            self.assertEqual(selected["episode"], 50)
            self.assertEqual(selected["reference_metrics"], {"comfort_pct": 90, "elec_cost": 2})
            with self.assertRaisesRegex(ValueError, "Service reference cohort"):
                evaluator.save_checkpoint(
                    server, 100, dict(reward=20, comfort_pct=100, elec_cost=0, dates=["2019-08-01"])
                )
            with self.assertRaisesRegex(ValueError, "never test"):
                PeriodicEvaluator(
                    MetricsLogger(root / "metrics.jsonl", [27]),
                    50,
                    split="test",
                    selection_reference=reference,
                )

    def test_federation_preserves_shared_feature_coordinates(self):
        state = {
            "fc1.weight": torch.ones(2),
            "feature_mean": torch.tensor([0.1, 0.2]),
            "feature_scale": torch.tensor([0.3, 0.7]),
        }
        result = Server._average_parameters([state] * 10)
        torch.testing.assert_close(result["feature_scale"], state["feature_scale"], rtol=0, atol=0)
        other = copy.deepcopy(state)
        other["feature_mean"][0] += 1
        with self.assertRaisesRegex(ValueError, "shared feature normalization"):
            Server._average_parameters([state, other])

    def test_batched_raw_final_evaluation_matches_sequential_without_tabpfn(self):
        with patch(
            "gridpfn.core.model._tabpfn_backbone", side_effect=AssertionError("Raw evaluation loaded TabPFN")
        ):
            torch.manual_seed(6)
            actor, critic = Actor(17, 3, feature_mode="raw"), Critic(17, 3, 2, feature_mode="raw")
            days = []
            for temperature in (20, 30):
                day = np.zeros((24, 8))
                day[:, 2] = np.arange(24)
                day[:, 4] = temperature
                days.append(day)
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.device, evaluator.fixed_cost = torch.device("cpu"), 0
            evaluator.homes = {
                1: SimpleNamespace(
                    test_data=days,
                    test_dates=["2019-08-01", "2019-08-02"],
                    scaler={},
                    actual_home_id=27,
                )
            }
            evaluator._model_for_home = lambda *_: (actor, critic)
            batched = evaluator.evaluate_model("fedavg", 1)["mean"]
            sequential = [
                evaluator.calculate_metrics(
                    evaluator.evaluate_episode(actor, critic, HOME_ENERGY_MGNT(day, state_dim=17))
                )
                for day in days
            ]
            for key in sequential[0]:
                if key == "squared_violation":
                    np.testing.assert_allclose(
                        batched[key], np.mean([r[key] for r in sequential]), atol=1e-4, rtol=1e-7
                    )
                    continue
                self.assertAlmostEqual(
                    batched[key], np.mean([r[key] for r in sequential]), places=4, msg=key
                )


if __name__ == "__main__":
    unittest.main()
