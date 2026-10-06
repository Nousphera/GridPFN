import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from gridpfn.core.evaluate import ModelEvaluator
from gridpfn.core.training_metrics import MetricsLogger, PeriodicEvaluator
from gridpfn.core.utils.plots import convergence_series
from gridpfn.experiments.live_plot import read_records


class ConstantActor(torch.nn.Module):
    def forward(self, state):
        return state.new_zeros((len(state), 3))


class ConstantCritic(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, state, control):
        return state.new_full((len(state), 2), self.value)


def make_client(solar=0):
    day = np.zeros((24, 8))
    day[:, 0], day[:, 1], day[:, 2], day[:, 3], day[:, 4] = 1, solar, np.arange(24), 0.5, 20
    agent = SimpleNamespace(
        actor_net=ConstantActor(),
        critic_net=ConstantCritic(1),
        actor_target_net=ConstantActor(),
        critic_target_net=ConstantCritic(4),
        gamma=0.5,
        device=torch.device("cpu"),
        frame_idx=7,
        memory=["sentinel"],
    )
    return SimpleNamespace(
        test_data=[day],
        test_dates=["2019-08-01"],
        scaler={},
        fixed_cost=5,
        device=agent.device,
        fedavg_agent=agent,
    )


class LiveMonitoringTests(unittest.TestCase):
    def test_td_loss_includes_terminal_mask_and_requested_discrete_action(self):
        agent = make_client().fedavg_agent
        transitions = [
            (np.zeros(9), 0, np.zeros(3), np.ones(9), False, 2),
            (np.ones(9), 1, np.zeros(3), np.zeros(9), True, 3),
        ]
        actor, critic = PeriodicEvaluator.losses(agent, transitions)
        self.assertEqual(actor, -1)
        self.assertEqual(critic, ((1 - 4) ** 2 + (1 - 3) ** 2) / 2)

    def test_interval_final_episode_and_rng_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = MetricsLogger(Path(directory) / "metrics.jsonl", [27])
            evaluator = PeriodicEvaluator(logger, 50)
            server = SimpleNamespace(episode=103, clients=[make_client()])
            expected_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

            def consume_rng(_):
                random.random(), np.random.random(), torch.rand(1)
                return {
                    "days": 1,
                    "actor_loss": -1,
                    "critic_loss": 2,
                    "reward": 3,
                    "task_success_pct": 100,
                }

            with patch.object(evaluator, "evaluate", side_effect=consume_rng) as evaluate:
                for episode in (0, 1, 49, 50, 51, 100, 103):
                    evaluator(server, episode)
                self.assertEqual(evaluate.call_count, 4)
            records = read_records(logger.path)
            self.assertEqual(
                [r["episode"] for r in records if r["kind"] == "eval"], [0, 50, 100, 103]
            )
            self.assertEqual(random.getstate(), expected_rng[0])
            np.testing.assert_equal(np.random.get_state(), expected_rng[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng[2]))

    def test_p2p_metrics_match_existing_evaluation_without_agent_mutation(self):
        clients = [make_client(3), make_client()]
        for client in clients:
            original = client.test_data[0]
            client.test_data = [original.copy() for _ in range(3)]
            client.test_dates = [f"2019-08-0{i}" for i in (1, 2, 3)]
            client.test_data[1][:, 4] = 25
            client.test_data[2][:, 4] = 30
        strategy = {"tou": {"enabled": False}, "export_price": 0.025, "pv_curtail": None}
        server = SimpleNamespace(
            clients=clients, em_strategy=strategy, p2p_config={"enabled": True, "price": 0.1}
        )
        with tempfile.TemporaryDirectory() as directory:
            monitor = PeriodicEvaluator(
                MetricsLogger(Path(directory) / "metrics.jsonl", [27, 950]), 50
            )
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.device, evaluator.fixed_cost = torch.device("cpu"), 5
            evaluator.homes = {
                i: SimpleNamespace(
                    test_data=c.test_data,
                    test_dates=c.test_dates,
                    scaler=c.scaler,
                    actual_home_id=i,
                )
                for i, c in enumerate(clients, 1)
            }
            evaluator._model_for_home = lambda name, i: (
                clients[i - 1].fedavg_agent.actor_net,
                clients[i - 1].fedavg_agent.critic_net,
            )
            with (
                patch("gridpfn.core.training_metrics.precompute_embeddings"),
                patch("gridpfn.core.evaluate.precompute_embeddings"),
            ):
                actual = monitor.evaluate(server)
                expected = evaluator._evaluate_p2p_model("fedavg", strategy, 0.1)
            for i, home in enumerate(actual["homes"], 1):
                for key, value in expected[i]["mean"].items():
                    self.assertAlmostEqual(home[key], value, places=6, msg=key)
                self.assertEqual(home["task_success_pct"], 100)
                agent = clients[i - 1].fedavg_agent
                self.assertEqual(agent.memory, ["sentinel"])
                self.assertEqual(agent.frame_idx, 7)
                self.assertTrue(agent.actor_net.training)
                self.assertTrue(agent.critic_net.training)

    def test_fixed_dates_use_intersection_not_mismatched_day_indices(self):
        clients = [make_client(), make_client()]
        clients[1].test_dates = ["2019-08-02"]
        with tempfile.TemporaryDirectory() as directory:
            evaluator = PeriodicEvaluator(MetricsLogger(Path(directory) / "m.jsonl", [1, 2]), 50)
            with self.assertRaisesRegex(ValueError, "shared held-out"):
                evaluator.evaluate(SimpleNamespace(clients=clients))

    def test_variance_is_across_homes_after_per_home_smoothing(self):
        records = [
            {
                "kind": "train",
                "episode": i,
                "homes": [{"home_id": 27, "reward": a}, {"home_id": 950, "reward": b}],
            }
            for i, a, b in ((1, 0, 2), (2, 4, 8))
        ]
        x, mean, sd = convergence_series(records, "reward", "train", window=2)
        np.testing.assert_allclose(x, [1, 2])
        np.testing.assert_allclose(mean, [1, 3.5])
        np.testing.assert_allclose(sd, [1, 1.5])
        _, mean, sd = convergence_series(records, "reward", "train", window=1, home_id=950)
        np.testing.assert_allclose(mean, [2, 8])
        np.testing.assert_allclose(sd, 0)
        for record in records:
            record["kind"] = "eval"
        _, mean, sd = convergence_series(records, "reward", "eval", window=2)
        np.testing.assert_allclose(mean, [1, 6])
        np.testing.assert_allclose(sd, [1, 2])

    def test_stream_reader_ignores_partial_record_and_logger_keeps_homes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.jsonl"
            logger = MetricsLogger(path, [27, 950])
            logger.training("fedavg", 1, [(1, None, None), (3, 2, 4)])
            with path.open("a") as handle:
                handle.write('{"kind":')
            records = read_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["reward"], 2)
            self.assertEqual(records[0]["actor_loss"], 2)
            self.assertEqual(records[0]["homes"][0]["home_id"], 27)
            json.dumps(records, allow_nan=False)
            with self.assertRaisesRegex(ValueError, "fresh run"):
                MetricsLogger(path, [27, 950])


if __name__ == "__main__":
    unittest.main()
