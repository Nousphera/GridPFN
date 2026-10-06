"""Regression checks for the training audit, without requiring a backbone download."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from gridpfn.core.client import Client
from gridpfn.core.dataset import _construct_dataset, dataset_columns, feature_columns
from gridpfn.core.em_strategy import P2P_TRADING, _denorm_client_feature
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.server import Server, TrainLogs


class RecordingAgent:
    def __init__(self):
        self.transitions = []

    def choose_action(self, state):
        return 0, np.zeros(3)

    def store_transition(self, *transition):
        self.transitions.append(transition)

    def learn(self):
        return None, None


class AveragingClient:
    def __init__(self, initial, increment):
        self.train_data = [None]
        self.increment = increment
        self.fedavg_actor_net = torch.nn.Linear(1, 1, bias=False)
        self.fedavg_critic_net = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.fedavg_actor_net.weight.fill_(initial)
            self.fedavg_critic_net.weight.fill_(initial)
        self.fedavg_agent = SimpleNamespace(
            actor_net=self.fedavg_actor_net,
            critic_net=self.fedavg_critic_net,
            actor_target_net=copy.deepcopy(self.fedavg_actor_net),
            critic_target_net=copy.deepcopy(self.fedavg_critic_net),
        )
        self.warmup_weights = None

    def set_em_strategy(self, strategy):
        pass

    def get_fedavg_model_params(self):
        return tuple(
            copy.deepcopy(n.state_dict()) for n in (self.fedavg_actor_net, self.fedavg_critic_net)
        )

    def set_fedavg_model_params(self, actor, critic):
        self.fedavg_actor_net.load_state_dict(actor)
        self.fedavg_critic_net.load_state_dict(critic)

    def warmup_episode(self, agent, index):
        self.warmup_weights = [
            n.weight.item()
            for n in (
                agent.actor_net,
                agent.critic_net,
                agent.actor_target_net,
                agent.critic_target_net,
            )
        ]

    def fedavg_train(self, index):
        with torch.no_grad():
            self.fedavg_actor_net.weight.add_(self.increment)
            self.fedavg_critic_net.weight.add_(self.increment)
        return 0, None, None


class TrainingPipelineTests(unittest.TestCase):
    def test_control_observation_distinguishes_hidden_device_state(self):
        day = np.zeros((24, 8))
        day[:, 2], day[:, 4] = np.arange(24), 25
        day[0, 6], day[10:13, 7] = 6, 1
        env = HOME_ENERGY_MGNT(day, state_dim=17)
        before = env.reset()
        env.indoor_temp += 5
        env.SoE_EV += 0.1
        env.ev_energy_delivered += 1
        env.wm_pending = False
        env.wm_running = True
        env.wm_cycle_index = 1
        env.exported_kwh += 1
        after = env._state_for_step(0)
        np.testing.assert_array_equal(before[:9], after[:9])
        self.assertTrue(
            np.all(before[[9, 10, 11, 12, 13, 14, 16]] != after[[9, 10, 11, 12, 13, 14, 16]])
        )
        self.assertTrue(np.isfinite(env._state_for_step(24)).all())
        with self.assertRaises(ValueError):
            HOME_ENERGY_MGNT(day, state_dim=10)

    def test_p2p_replay_uses_state_after_export_budget_adjustment(self):
        day = np.zeros((24, 8))
        day[:, 2], day[:, 4] = np.arange(24), 20
        agent = RecordingAgent()
        client = SimpleNamespace(
            train_data=[day],
            scaler={},
            fixed_cost=0,
            state_dim=17,
            em_strategy={"pv_curtail": 100},
            fedavg_agent=agent,
        )
        with patch.object(P2P_TRADING, "compute_adjustments", return_value=([0], [0], [1])):
            P2P_TRADING.run_episode(
                [client], 0, use_fed=True, p2p_config={"enabled": True, "price": 0.1}
            )
        self.assertAlmostEqual(agent.transitions[0][3][16], 0.99)
        np.testing.assert_array_equal(agent.transitions[0][3], agent.transitions[1][0])

    def test_fedavg_shared_initialization_round_boundary_and_fresh_final_average(self):
        clients = [AveragingClient(1, 1), AveragingClient(9, 2)]
        seen = {}

        def observe(server, episode):
            seen[episode] = [c.fedavg_actor_net.weight.item() for c in server.clients]

        server = Server(
            clients,
            episode=3,
            actor_sparsity=0.25,
            critic_sparsity=0.25,
            update=1,
            aggregate=2,
            em_strategy={"tou": {"enabled": False}},
            warmup_rounds=1,
            eval_callback=observe,
        )
        with (
            patch.object(TrainLogs, "save_model_logs"),
            patch.object(TrainLogs, "save_best_models"),
            patch.object(TrainLogs, "save_global_models") as save,
        ):
            server.fedavg_train()
        for client in clients:
            self.assertEqual(client.warmup_weights, [1, 1, 1, 1])
        self.assertEqual(seen[2], [4, 4])
        self.assertEqual(seen[3], [5, 6])
        self.assertEqual(save.call_args.args[1]["weight"].item(), 5.5)
        self.assertEqual(save.call_args.args[2]["weight"].item(), 5.5)

    def test_local_critics_keep_home_values_while_actor_communication_is_shared(self):
        clients = [AveragingClient(1, 1), AveragingClient(9, 2)]
        server = Server(
            clients,
            episode=2,
            actor_sparsity=0.25,
            critic_sparsity=0.25,
            update=1,
            aggregate=2,
            em_strategy={"tou": {"enabled": False}},
            warmup_rounds=0,
            local_critics=True,
        )
        with (
            patch.object(TrainLogs, "save_model_logs"),
            patch.object(TrainLogs, "save_best_models"),
            patch.object(TrainLogs, "save_global_models"),
        ):
            server.fedavg_train()
        self.assertEqual([c.fedavg_actor_net.weight.item() for c in clients], [4, 4])
        self.assertEqual([c.fedavg_critic_net.weight.item() for c in clients], [3, 5])
        with patch.object(clients[0], "get_fedavg_model_params", side_effect=AssertionError):
            _, global_critic = server.fedavg_model_average()
        self.assertIsNone(global_critic)

    def test_validation_stop_saves_last_heads_without_running_extra_episodes(self):
        clients = [AveragingClient(1, 1), AveragingClient(9, 2)]
        seen = []

        class StopAfterTwo:
            stopping = SimpleNamespace(stopped=False)

            def __call__(self, server, episode):
                seen.append(episode)
                self.stopping.stopped = episode == 2

        server = Server(
            clients,
            episode=10,
            actor_sparsity=0.25,
            critic_sparsity=0.25,
            update=1,
            aggregate=5,
            em_strategy={"tou": {"enabled": False}},
            warmup_rounds=0,
            eval_callback=StopAfterTwo(),
        )
        with (
            patch.object(TrainLogs, "save_model_logs"),
            patch.object(TrainLogs, "save_best_models") as local_save,
            patch.object(TrainLogs, "save_global_models") as global_save,
        ):
            logs = server.fedavg_train()
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(len(logs["rewards"][1]), 2)
        self.assertEqual(local_save.call_count, 2)
        self.assertEqual(global_save.call_args.args[1]["weight"].item(), 4)

    def test_replay_keeps_requested_action_when_deadlines_force_devices(self):
        day = np.zeros((24, 8))
        day[:, 2], day[:, 3], day[:, 4] = np.arange(24), 0.2, 25
        day[0, 6], day[10:13, 7] = 6, 1
        # Ensure the fixture really invokes the automatic interventions.
        env = HOME_ENERGY_MGNT(day)
        applied = []
        for _ in range(24):
            env.step((0, np.zeros(3)))
            applied.append(env.action)
        self.assertTrue(any(a[0] == 1 for a in applied))
        self.assertTrue(any(a[1][1] > 0 for a in applied))
        for mode in ("local", "warmup", "p2p"):
            with self.subTest(mode=mode):
                agent = RecordingAgent()
                client = SimpleNamespace(
                    train_data=[day], scaler={}, fixed_cost=0, em_strategy=None, fedavg_agent=agent
                )
                if mode == "local":
                    Client._run_episode(client, agent, 0)
                elif mode == "warmup":
                    Client.warmup_episode(client, agent, 0)
                else:
                    P2P_TRADING.run_episode(
                        [client], 0, use_fed=True, p2p_config={"enabled": True, "price": 0.1}
                    )
                self.assertEqual(len(agent.transitions), 24)
                for _, action, _, _, _ in agent.transitions:
                    self.assertEqual(action[0], 0)
                    np.testing.assert_array_equal(action[1], 0)

    def test_unseen_physical_values_survive_scaling_including_constant_features(self):
        frames = []
        for date, temp, load in [
            ("2019-06-01", 20, 1),
            ("2019-06-02", 25, 1),
            ("2019-08-01", 35, 2),
        ]:
            frame = pd.DataFrame(
                {"datetime": pd.date_range(date, periods=96, freq="15min"), "t": np.arange(96)}
            )
            for col in feature_columns:
                frame[col] = 0.0
            frame["temp (C)"], frame["fixed_load (kWh)"] = temp, load
            frames.append(frame)
        with patch("gridpfn.core.dataset.merge_temp_price", side_effect=lambda data, **_: data):
            _, test, _, scaler = _construct_dataset(pd.concat(frames, ignore_index=True))
        env = HOME_ENERGY_MGNT(test[0], scaler=scaler)
        client = SimpleNamespace(scaler=scaler)
        for column, physical in [("temp (C)", 35), ("fixed_load (kWh)", 8)]:
            idx = dataset_columns.index(column)
            normalized = test[0, 0, idx]
            self.assertGreater(normalized, 1)
            self.assertEqual(env._denorm_feature(idx, normalized), physical)
            self.assertEqual(_denorm_client_feature(client, idx, normalized), physical)
            self.assertEqual(env._norm_feature(idx, physical), normalized)


if __name__ == "__main__":
    unittest.main()
