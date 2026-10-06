import copy
import random
import unittest

import numpy as np
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.batched_learning import BatchedLearner
from gridpfn.core.control_guidance import FeedbackTeacher
from gridpfn.core.model import Actor, Critic
from gridpfn.core.server import Server


class BatchedLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def make_agents(self, residual=False, twin=False, **hp):
        agents = []
        rng = np.random.default_rng(5)
        for _ in range(3):
            actor = Actor(17, 3, feature_mode="raw")
            if residual:
                actor.enable_residual([0.04, 0.25, 0.5])
            agent = P_DQN(
                actor,
                Critic(17, 3, 2, feature_mode="raw", twin=twin),
                2,
                3,
                17,
                {"batch_size": 4, **hp},
            )
            agent.feedback_teacher = FeedbackTeacher(
                {"col_to_scaler_idx": {4: 0}, "min": [10], "max": [40]}
            )
            for j in range(24):
                state = rng.random(17).astype("float32")
                nxt = rng.random(17).astype("float32")
                agent.store_transition(
                    state,
                    (j % 2, np.array([1.0, 2.0, 0.1])),
                    float(-j % 4),
                    nxt,
                    j % 8 == 7,
                )
            agents.append(agent)
        return agents

    def test_sequential_parameter_and_loss_parity(self):
        for settings in [
            {},
            {"bc_weight": 1, "actor_q_weight": 0.1},
            {"bc_weight": 1, "actor_q_weight": 0.1, "guidance_weights": (1, 0, 0)},
            {"bc_weight": 1, "actor_q_weight": 0},
            {"critic_huber_delta": 1, "twin": True, "policy_delay": 2},
            {
                "residual": True,
                "twin": True,
                "policy_delay": 2,
                "bc_weight": 1,
                "actor_q_weight": 0.1,
            },
            {"return_mode": "episode", "residual": True},
        ]:
            with self.subTest(settings=settings):
                references = self.make_agents(**settings)
                agents = copy.deepcopy(references)
                batch = BatchedLearner(agents)
                for step in range(4):
                    random.seed(step)
                    expected = [a.learn() for a in references]
                    random.seed(step)
                    actual = batch.learn()
                    for e, a in zip(expected, actual):
                        for x, y in zip(e, a):
                            if x is None:
                                self.assertIsNone(y)
                            else:
                                self.assertAlmostEqual(x, y, places=4)
                    for ref, agent in zip(references, agents):
                        for net in [
                            "actor_net",
                            "critic_net",
                            "actor_target_net",
                            "critic_target_net",
                        ]:
                            for p, q in zip(
                                getattr(ref, net).parameters(),
                                getattr(agent, net).parameters(),
                            ):
                                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)

    def test_batched_exploration_preserves_actions_and_rng(self):
        for residual in (False, True):
            refs = self.make_agents(residual=residual, exploration_noise=0.05, epsilon_start=0.4)
            agents = copy.deepcopy(refs)
            batch = BatchedLearner(agents)
            states = np.random.default_rng(2).random((3, 17)).astype("float32")
            for step in range(4):
                random.seed(step)
                np.random.seed(step)
                expected = [a.choose_action(s) for a, s in zip(refs, states)]
                py_rng = random.getstate()
                np_rng = np.random.get_state()
                random.seed(step)
                np.random.seed(step)
                actual = batch.choose_actions(states)
                self.assertEqual(random.getstate(), py_rng)
                np.testing.assert_array_equal(np.random.get_state()[1], np_rng[1])
                for (d, a), (e, b) in zip(expected, actual):
                    self.assertEqual(d, e)
                    np.testing.assert_allclose(a, b, atol=1e-6)

    def test_fedavg_copies_update_canonical_stacked_parameters(self):
        agents = self.make_agents()
        batch = BatchedLearner(agents)
        actor = Server._average_parameters([a.get_model_params()[0] for a in agents])
        critic = Server._average_parameters([a.get_model_params()[1] for a in agents])
        for a in agents:
            a.set_model_params(actor, critic)
        for name, value in actor.items():
            stacked = batch.actor.params.get("head." + name)
            if stacked is not None:
                for row in stacked:
                    torch.testing.assert_close(row, value, rtol=0, atol=0)
        random.seed(10)
        batch.learn()
        self.assertFalse(torch.equal(batch.actor.params["head.fc4.weight"][0], actor["fc4.weight"]))


if __name__ == "__main__":
    unittest.main()
