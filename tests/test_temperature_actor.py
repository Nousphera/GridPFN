import copy
import random
import unittest

import numpy as np
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.batched_learning import BatchedLearner
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.model import Actor, Critic, heads_from_state


class TemperatureActorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def actor(self, feasible=False):
        return Actor(
            17, 3, feature_mode="raw", thermal_bounds=[0, 40], feasible_dt=1 if feasible else None
        )

    def test_feasible_controls_match_original_ev_battery_and_quota_constraints(self):
        actor = self.actor(feasible=True)
        for hour, soe in [(0, 0), (7, 0.2), (8, 1), (23, 0.5)]:
            day = np.zeros((24, 8))
            day[:, 0], day[:, 2], day[:, 4], day[:, 5] = 10, np.arange(24), 0.75, 2.5
            day[0, 6] = 12
            env = HOME_ENERGY_MGNT(
                day, scaler={"min": [0], "max": [40], "col_to_scaler_idx": {4: 0}}, state_dim=17
            )
            env.current_step, env.SoE_BESS = hour, soe
            env.ac_energy_delivered = 2.5 * hour
            state = torch.as_tensor(env._state_for_step(hour)[None], dtype=torch.float32)
            action = actor(state)[0].detach().numpy()
            env.step((0, action))
            np.testing.assert_allclose(
                action, [env.power_AC, env.power_EV, env.power_BESS], atol=2e-6
            )
        loaded, _ = heads_from_state(
            actor.state_dict(), Critic(17, 3, 2, feature_mode="raw").state_dict(), "cpu"
        )
        torch.testing.assert_close(loaded(state), actor(state), atol=0, rtol=0)

    def test_exact_thermal_equation_preserves_feasible_comfort_and_action_contract(self):
        actor = self.actor()
        rng = np.random.default_rng(5)
        for indoor, outdoor in rng.uniform([15, 10], [35, 40], (30, 2)):
            day = np.zeros((24, 8))
            day[:, 2], day[:, 4] = np.arange(24), outdoor / 40
            env = HOME_ENERGY_MGNT(
                day, scaler={"min": [0], "max": [40], "col_to_scaler_idx": {4: 0}}, state_dim=17
            )
            env.indoor_temp = indoor
            state = torch.as_tensor(env._state_for_step(0)[None], dtype=torch.float32)
            control = actor(state)[0].detach().numpy()
            self.assertEqual(control.shape, (3,))
            self.assertTrue(np.all(control >= [0, 0, -2.4]))
            self.assertTrue(np.all(control <= [2.5, 6, 2.4]))
            natural = 0.7 * indoor + 0.3 * outdoor
            env.step((0, control))
            if natural >= 18.2 and natural - 7.5 <= 21.8:
                self.assertGreaterEqual(env.indoor_temp + 1e-5, 18.2)
                self.assertLessEqual(env.indoor_temp - 1e-5, 21.8)

    def test_quota_still_overrides_policy_when_comfort_conflicts(self):
        day = np.zeros((24, 8))
        day[:, 2], day[:, 4], day[:, 5] = np.arange(24), 0.5, 2.5
        env = HOME_ENERGY_MGNT(
            day, scaler={"min": [0], "max": [40], "col_to_scaler_idx": {4: 0}}, state_dim=17
        )
        env.current_step, env.indoor_temp = 23, 20
        state = torch.as_tensor(env._state_for_step(23)[None], dtype=torch.float32)
        control = self.actor()(state)[0].detach().numpy()
        self.assertLess(control[0], 1)
        env.step((0, control))
        self.assertEqual(env.power_AC, 2.5)
        self.assertLess(env.indoor_temp, 18)

    def test_gradients_projection_and_checkpoint_reload(self):
        actor, critic = self.actor(), Critic(17, 3, 2, feature_mode="raw")
        states = torch.zeros(4, 17)
        states[:, 7], states[:, 9] = 0.75, 0.3  # outdoor30, indoor23
        controls = actor(states)
        controls[:, 0].sum().backward()
        self.assertGreater(actor.fc4.bias.grad[0].abs().item(), 0)
        projected = actor.project_action(states, torch.ones_like(controls) * 100)
        predicted = 0.7 * 23 + 0.3 * 30 - 3 * projected[:, 0]
        self.assertTrue(((predicted >= 18.2 - 1e-5) & (predicted <= 21.8 + 1e-5)).all())
        loaded, _ = heads_from_state(actor.state_dict(), critic.state_dict(), "cpu")
        torch.testing.assert_close(loaded(states), controls, atol=0, rtol=0)

    def test_batched_updates_and_exploration_match_sequential_temperature_actors(self):
        agents = []
        rng = np.random.default_rng(10)
        for _ in range(3):
            agent = P_DQN(
                self.actor(feasible=True),
                Critic(17, 3, 2, feature_mode="raw", twin=True),
                2,
                3,
                17,
                {
                    "batch_size": 4,
                    "critic_huber_delta": 1,
                    "exploration_noise": 0.05,
                    "policy_delay": 2,
                },
            )
            for i in range(10):
                state = rng.random(17).astype("float32")
                agent.store_transition(
                    state, (i % 2, np.ones(3)), -i, rng.random(17).astype("float32"), i == 9
                )
            agents.append(agent)
        refs = copy.deepcopy(agents)
        batch = BatchedLearner(agents)
        states = rng.random((3, 17)).astype("float32")
        random.seed(5)
        np.random.seed(5)
        expected = [a.choose_action(s) for a, s in zip(refs, states)]
        random.seed(5)
        np.random.seed(5)
        actual = batch.choose_actions(states)
        for (d, p), (e, q) in zip(expected, actual):
            self.assertEqual(d, e)
            np.testing.assert_allclose(p, q, atol=2e-6)
        for step in range(3):
            random.seed(step)
            expected = [a.learn() for a in refs]
            random.seed(step)
            actual = batch.learn()
            for e, a in zip(expected, actual):
                for x, y in zip(e, a):
                    if x is None:
                        self.assertIsNone(y)
                    else:
                        self.assertAlmostEqual(x, y, places=4)
            for ref, agent in zip(refs, agents):
                for net in ("actor_net", "critic_net", "actor_target_net", "critic_target_net"):
                    for p, q in zip(
                        getattr(ref, net).parameters(), getattr(agent, net).parameters()
                    ):
                        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
