import unittest
from types import SimpleNamespace

import numpy as np
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.control_guidance import FeedbackTeacher, initialize_guided_actors
from gridpfn.core.model import Actor, Critic
from gridpfn.core.server import Server
from gridpfn.core.training_metrics import PeriodicEvaluator


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def agent(self, **hp):
        agent = P_DQN(
            Actor(17, 3, feature_mode="raw"),
            Critic(17, 3, 2, feature_mode="raw"),
            2,
            3,
            17,
            {"batch_size": 4, **hp},
        )
        rng = np.random.default_rng(1)
        for _ in range(8):
            state = rng.random(17).astype(np.float32)
            agent.store_transition(state, (0, np.array([1.0, 2.0, 0.0])), -1.0, state, False)
        return agent

    def test_teacher_current_state_bounds_and_deadline(self):
        teacher = FeedbackTeacher({"col_to_scaler_idx": {4: 0}, "min": [10], "max": [40]})
        states = np.zeros((3, 17), dtype=np.float32)
        states[:, 7] = 2 / 3  # 30 degrees
        states[:, 9] = 0.3  # 23 degrees
        states[:, 11] = 0.5  # 12 kWh remaining
        states[:, 0] = [0, 7 / 24, 8 / 24]
        result = teacher(states)
        np.testing.assert_allclose(result[:, 0], 1.7, atol=1e-6)
        np.testing.assert_allclose(result[:, 1], [1.5, 6.0, 0.0])
        self.assertTrue((result[:, 2] == 0).all())
        # Constant training temperature still preserves unseen physical values.
        teacher = FeedbackTeacher({"col_to_scaler_idx": {4: 0}, "min": [30], "max": [30]})
        states[:, 7] = 0
        np.testing.assert_allclose(teacher(states)[:, 0], 1.7, atol=1e-6)

    def test_delayed_updates_leave_actor_and_targets_until_due(self):
        agent = self.agent(policy_delay=2)
        actor = [p.detach().clone() for p in agent.actor_net.parameters()]
        target = [p.detach().clone() for p in agent.critic_target_net.parameters()]
        loss = agent.learn()
        self.assertIsNone(loss[0])
        self.assertTrue(np.isfinite(loss[1]))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(actor, agent.actor_net.parameters())))
        self.assertTrue(
            all(torch.equal(x, y) for x, y in zip(target, agent.critic_target_net.parameters()))
        )
        self.assertIsNotNone(agent.learn()[0])
        self.assertTrue(
            any(not torch.equal(x, y) for x, y in zip(actor, agent.actor_net.parameters()))
        )

    def test_imitation_updates_actor_without_critic_gradients(self):
        agent = self.agent(bc_weight=1, actor_q_weight=0)
        agent.feedback_teacher = lambda state: np.zeros((len(state), 3), dtype=np.float32)
        states = np.ones((4, 17), dtype=np.float32)
        before = agent.actor_net.prepare_features(states)
        initial = agent.actor_net.forward_features(before).detach().clone()
        from unittest.mock import patch

        with patch.object(
            agent.critic_net, "forward_features", wraps=agent.critic_net.forward_features
        ) as forward:
            losses = agent.learn()
            self.assertEqual(forward.call_count, 1)  # no unused actor-through-critic pass
        self.assertTrue(all(np.isfinite(losses)))
        self.assertFalse(torch.equal(initial, agent.actor_net.forward_features(before)))
        self.assertTrue(
            all(p.grad is None or not p.grad.any() for p in agent.critic_net.parameters())
        )

    def test_federated_imitation_uses_only_training_and_syncs_targets(self):
        clients = []
        for _ in range(2):
            agent = self.agent(bc_rounds=2, bc_steps=2)
            day = np.zeros((24, 8), dtype=np.float32)
            day[:, 2] = np.arange(24)
            day[:, 4] = 0.5
            client = SimpleNamespace(
                fedavg_agent=agent,
                scaler={"col_to_scaler_idx": {4: 0}, "min": [10], "max": [40], "delta_t": 1},
                train_data=[day],
                fixed_cost=5,
                device=torch.device("cpu"),
                em_strategy={},
            )
            # No test_data field: initialization cannot depend on validation.
            clients.append(client)
        initial = clients[0].fedavg_agent.actor_net.state_dict()
        clients[1].fedavg_agent.actor_net.load_state_dict(initial)
        server = SimpleNamespace(clients=clients, _average_parameters=Server._average_parameters)
        initialize_guided_actors(server)
        for key, value in clients[0].fedavg_agent.actor_net.state_dict().items():
            self.assertTrue(torch.equal(value, clients[1].fedavg_agent.actor_net.state_dict()[key]))
            self.assertTrue(
                torch.equal(value, clients[0].fedavg_agent.actor_target_net.state_dict()[key])
            )

    def test_demonstration_cache_reuses_labels_and_invalidates_on_target_change(self):
        import tempfile

        from gridpfn.core.control_guidance import prepare_demonstrations

        agent = self.agent()
        day = np.zeros((24, 8), dtype=np.float32)
        day[:, 2], day[:, 4] = np.arange(24), 0.5
        scaler = {"col_to_scaler_idx": {4: 0}, "min": [10], "max": [40], "delta_t": 1}
        agent.feedback_teacher = FeedbackTeacher(scaler)
        client = SimpleNamespace(
            fedavg_agent=agent,
            scaler=scaler,
            train_data=[day],
            fixed_cost=5,
            device=torch.device("cpu"),
            em_strategy={},
        )
        with tempfile.TemporaryDirectory() as directory:
            agent.demonstration_cache = directory
            expected = prepare_demonstrations(client)
            agent.feedback_teacher = lambda _: (_ for _ in ()).throw(AssertionError("cache miss"))
            actual = prepare_demonstrations(client)
            for x, y in zip(expected, actual):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            agent.feedback_target = 18.2
            with self.assertRaisesRegex(AssertionError, "cache miss"):
                prepare_demonstrations(client)

    def test_twin_minimum_is_per_action_and_both_heads_learn(self):
        critic = Critic(17, 3, 2, feature_mode="raw", twin=True)
        with torch.no_grad():
            for p in critic.parameters():
                p.zero_()
            critic.fc4.bias.copy_(torch.tensor([10.0, 0.0]))
            critic.twin_fc4.bias.copy_(torch.tensor([0.0, 10.0]))
        state, action = torch.zeros((4, 17)), torch.zeros((4, 3))
        minimum = critic.minimum_features(state, action)
        self.assertTrue(torch.equal(minimum, torch.zeros((4, 2))))
        agent = self.agent(policy_delay=2, target_noise=0.1)
        agent.critic_net = critic
        import copy

        agent.critic_target_net = copy.deepcopy(critic)
        agent.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
        before = [critic.fc4.bias.clone(), critic.twin_fc4.bias.clone()]
        agent.learn()
        self.assertFalse(torch.equal(before[0], critic.fc4.bias))
        self.assertFalse(torch.equal(before[1], critic.twin_fc4.bias))
        self.assertEqual(critic(state, action).shape, (4, 2))

    def test_twin_checkpoint_loaders_preserve_predictions(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import gridpfn.core.schedule as schedule
        from gridpfn.core.evaluate import ModelEvaluator

        actor, critic = (
            Actor(17, 3, feature_mode="raw"),
            Critic(17, 3, 2, feature_mode="raw", twin=True),
        )
        with tempfile.TemporaryDirectory() as directory:
            ap, cp = Path(directory) / "actor.pt", Path(directory) / "critic.pt"
            torch.save(actor.state_dict(), ap)
            torch.save(critic.state_dict(), cp)
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.device = torch.device("cpu")
            a, c = evaluator.load_model(ap, cp)
            with patch.object(schedule, "DEVICE", torch.device("cpu")):
                sa, sc = schedule.load_actor_critic(ap, cp)
            for loaded in (c, sc):
                for name, value in critic.state_dict().items():
                    self.assertTrue(torch.equal(value, loaded.state_dict()[name]))
                self.assertTrue(loaded.twin)

    def test_return_diagnostics_do_not_mix_dates(self):
        class ConstantCritic(torch.nn.Module):
            def forward(self, states, controls):
                return torch.zeros((len(states), 2))

        agent = SimpleNamespace(critic_net=ConstantCritic(), device="cpu", gamma=0.5)
        # Day A rewards [1, 2], day B [10, 20], time-major order.
        transitions = [
            (np.zeros(17), 0, np.zeros(3), np.zeros(17), done, r)
            for r, done in [(1, False), (10, False), (2, True), (20, True)]
        ]
        metrics = PeriodicEvaluator.value_diagnostics(agent, transitions, 2)
        self.assertAlmostEqual(metrics["q_return_bias"], -11)
        self.assertAlmostEqual(metrics["q_return_rmse"], np.sqrt(202))


if __name__ == "__main__":
    unittest.main()
