import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.model import Actor, Critic
from gridpfn.core.utils.agent_utils import soft_update_target_network


class StrongerPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)

    def agent(self, **hp):
        return P_DQN(
            Actor(17, 3, feature_mode="raw"),
            Critic(17, 3, 2, feature_mode="raw"),
            2,
            3,
            17,
            {"batch_size": 2, **hp},
        )

    def test_residual_zero_start_projection_and_frozen_reference(self):
        actor = Actor(17, 3, feature_mode="raw")
        x = torch.randn(16, 17)
        before = actor(x).detach()
        actor.enable_residual([0.04, 0.25, 0.5])
        torch.testing.assert_close(actor(x), before, rtol=0, atol=0)
        radius = actor.residual_scale * (actor.action_max - actor.action_min)
        with torch.no_grad():
            actor.fc4.bias.fill_(100)
        self.assertTrue(((actor(x) - before).abs() <= radius + 1e-6).all())
        projected = actor.project_action(x, torch.ones_like(before) * 1000)
        self.assertTrue(((projected - before).abs() <= radius + 1e-6).all())
        target = copy.deepcopy(actor)
        with torch.no_grad():
            target.base_fc1.weight.add_(1)
        expected = target.base_fc1.weight.clone()
        soft_update_target_network(actor, target, 0.1)
        torch.testing.assert_close(target.base_fc1.weight, expected, rtol=0, atol=0)

    def test_episode_returns_end_at_each_boundary(self):
        agent = self.agent(gamma=0.5, return_mode="episode")
        state = np.ones(17, dtype=np.float32)
        agent.store_transition(state, (0, np.zeros(3)), 1.0, state, False)
        self.assertEqual(len(agent.memory), 0)
        agent.store_transition(state, (0, np.zeros(3)), 2.0, state, True)
        agent.store_transition(state, (0, np.zeros(3)), 10.0, state, True)
        self.assertEqual([v[2] for v in agent.memory.buffer], [2.0, 2.0, 10.0])
        self.assertTrue(all(v[4] for v in agent.memory.buffer))
        with patch.object(
            agent.actor_target_net,
            "forward_features",
            side_effect=AssertionError("MC should not bootstrap"),
        ):
            loss = agent.learn()
        self.assertTrue(np.isfinite(loss).all())

    def test_checkpoint_evaluation_rejects_service_mode_mismatch(self):
        from gridpfn.experiments.evaluate_checkpoint import evaluate_saved

        with (
            patch(
                "gridpfn.experiments.evaluate_checkpoint.read_logged_settings",
                return_value={
                    "home_ids": [27],
                    "em_strategy": {"ac_energy_quota": False},
                },
            ),
            patch(
                "gridpfn.experiments.evaluate_checkpoint.safe_torch_load",
                return_value={"home_ids": [27], "ac_service": "energy_quota"},
            ),
            self.assertRaisesRegex(ValueError, "service mode"),
        ):
            evaluate_saved(Path("unused"), Path("unused.pt"))

    def test_thermal_ac_service_does_not_force_historical_consumption(self):
        from gridpfn.core.em_strategy import apply_em_strategy
        from gridpfn.core.environment import HOME_ENERGY_MGNT

        day = np.zeros((24, 8))
        day[:, 2] = np.arange(24)
        day[:, 4] = 20
        day[:, 5] = 2.5
        legacy = HOME_ENERGY_MGNT(day, state_dim=17)
        thermal = HOME_ENERGY_MGNT(day, state_dim=17)
        apply_em_strategy(thermal, {"ac_energy_quota": False})
        for env in [legacy, thermal]:
            env.current_step = 23
            env.indoor_temp = 20
        _, _, _, _, done = thermal.step((0, np.zeros(3)))
        self.assertTrue(done)
        self.assertEqual(thermal.power_AC, 0)
        self.assertAlmostEqual(thermal.indoor_temp, 20)
        self.assertEqual(thermal._state_for_step(24)[15], 0)
        legacy.step((0, np.zeros(3)))
        self.assertEqual(legacy.power_AC, 2.5)
        self.assertLess(legacy.indoor_temp, 18)

    def test_zero_local_noise_never_replaces_continuous_policy(self):
        agent = self.agent(exploration_noise=0, epsilon_start=1, epsilon_end=1)
        state = np.ones(17, dtype=np.float32)
        expected = agent.actor_net(torch.from_numpy(state[None])).detach().numpy()[0]
        for _ in range(10):
            _, action = agent.choose_action(state)
            np.testing.assert_array_equal(action, expected)

    def test_residual_checkpoint_roundtrip(self):
        from gridpfn.core.evaluate import ModelEvaluator

        agent = self.agent()
        agent.actor_net.enable_residual([0.04, 0.25, 0.5])
        with tempfile.TemporaryDirectory() as d:
            a, c = Path(d) / "a.pt", Path(d) / "c.pt"
            torch.save(agent.actor_net.state_dict(), a)
            torch.save(agent.critic_net.state_dict(), c)
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.device = "cpu"
            loaded, _ = evaluator.load_model(a, c)
            x = torch.randn(8, 17)
            torch.testing.assert_close(loaded(x), agent.actor_net(x), rtol=0, atol=0)
            self.assertFalse(loaded.base_fc1.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
