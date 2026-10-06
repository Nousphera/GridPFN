"""No held-out demonstrations; expert mixing keeps independent home updates."""

import copy
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.batched_learning import BatchedLearner
from gridpfn.core.model import Actor, Critic
from gridpfn.core.utils.agent_utils import ReplayBuffer
from oracle import OracleConfig
from oracle.demonstrations import validate_protocol


class ExpertLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def agents(self, implicit=False):
        result = []
        for home in range(2):
            agent = P_DQN(
                Actor(17, 3, feature_mode="raw", learned_discrete=implicit),
                Critic(17, 3, 2, feature_mode="raw", twin=True, value_head=implicit),
                2,
                3,
                17,
                {
                    "batch_size": 8,
                    "expert_fraction": 0.25,
                    "bc_weight": 0 if implicit else 0.3,
                    "actor_update": "implicit" if implicit else "q_gradient",
                    "actor_q_weight": 0.1,
                    "policy_delay": 2,
                    "critic_huber_delta": 1,
                },
            )
            agent.expert_memory = ReplayBuffer(24)
            for row in range(24):
                state = np.full(17, row / 24, dtype=np.float32)
                action = (row % 2, np.array([0.5 + home, 1.0, -0.2], dtype=np.float32))
                for buffer, reward in [(agent.memory, 0.1), (agent.expert_memory, 2.0)]:
                    buffer.store_transition(state, action, reward, state, row == 23)
            result.append(agent)
        return result

    def test_sampling_preserves_expert_labels_and_online_rewards(self):
        agent = self.agents()[0]
        batch, mask = agent.sample_batch()
        self.assertEqual(mask.tolist(), [True, True, False, False, False, False, False, False])
        np.testing.assert_array_equal(batch[2][:2], [2, 2])
        np.testing.assert_array_equal(batch[2][2:], [0.1] * 6)
        self.assertEqual(len(agent.expert_memory), 24)

    def test_expert_batched_updates_match_independent_optimizers(self):
        self.parity(False)
        self.parity(True)

    def parity(self, implicit):
        refs = self.agents(implicit)
        actual = copy.deepcopy(refs)
        learner = BatchedLearner(actual)
        for step in range(4):
            random.seed(step)
            expected_losses = [a.learn() for a in refs]
            random.seed(step)
            actual_losses = learner.learn()
            for expected, got in zip(expected_losses, actual_losses, strict=True):
                for a, b in zip(expected, got, strict=True):
                    if a is None:
                        self.assertIsNone(b)
                    else:
                        self.assertAlmostEqual(a, b, places=5)
            for ref, agent in zip(refs, actual, strict=True):
                for key in ("actor_net", "critic_net", "actor_target_net", "critic_target_net"):
                    for a, b in zip(
                        getattr(ref, key).parameters(),
                        getattr(agent, key).parameters(),
                        strict=True,
                    ):
                        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_implicit_target_never_queries_an_unseen_action(self):
        agent = self.agents(implicit=True)[0]
        states = torch.randn(3, 17)
        rewards, done = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 1.0, 0.0])
        expected = rewards + agent.gamma * agent.critic_net.value_features(states) * (1 - done)
        with patch.object(
            agent.actor_target_net, "forward_features", side_effect=AssertionError("Unseen action")
        ):
            actual = agent.bellman_target(states, rewards, done)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual[1].item(), 2)

    def test_implicit_checkpoint_restores_categorical_inference_and_contracts(self):
        from gridpfn.core.model import heads_from_state
        from gridpfn.core.training_metrics import greedy_actions

        agent = self.agents(implicit=True)[0]
        with torch.no_grad():
            agent.actor_net.discrete_fc4.weight.zero_()
            agent.actor_net.discrete_fc4.bias.copy_(torch.tensor([0.0, 10.0]))
            agent.critic_net.fc4.weight.zero_()
            agent.critic_net.fc4.bias.copy_(torch.tensor([100.0, 0.0]))
        actor, critic = heads_from_state(
            agent.actor_net.state_dict(), agent.critic_net.state_dict(), "cpu"
        )
        loaded = SimpleNamespace(actor_net=actor, critic_net=critic)
        states = np.ones((3, 17), dtype=np.float32)
        output = greedy_actions(loaded, states)
        np.testing.assert_array_equal(output[:, 0], [1, 1, 1])
        self.assertEqual(actor(torch.as_tensor(states)).shape, (3, 3))
        self.assertEqual(
            critic(torch.as_tensor(states), actor(torch.as_tensor(states))).shape, (3, 2)
        )

    def test_expectile_and_actor_weights_have_no_critic_gradients(self):
        from gridpfn.core.agents.implicit import actor_regression, expectile_loss

        value = torch.tensor([0.0, 0.0], requires_grad=True)
        target = torch.tensor([2.0, -2.0], requires_grad=True)
        torch.testing.assert_close(expectile_loss(value, target, 0.7), torch.tensor([2.8, 1.2]))
        control = torch.zeros(2, 3, requires_grad=True)
        loss = actor_regression(
            control,
            torch.ones(2, 3),
            torch.zeros(2, 2, requires_grad=True),
            torch.tensor([0, 1]),
            target,
            torch.ones(3),
            3,
        ).mean()
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(control.grad.abs().sum().item(), 0)

    def test_balanced_embeddings_preserve_raw_state_and_checkpoint_units(self):
        from gridpfn.core.model import heads_from_state

        backbone = SimpleNamespace(embedding_dim=4)
        states = np.ones((3, 17), dtype=np.float32)
        with (
            patch("gridpfn.core.model._tabpfn_backbone", return_value=backbone),
            patch("gridpfn.core.model.embedding_rows", return_value=torch.ones(3, 4)),
        ):
            actor = Actor(17, 3, feature_mode="hybrid", embedding_weight=0.1)
            critic = Critic(17, 3, 2, feature_mode="hybrid", embedding_weight=0.1)
            features = actor.prepare_features(states)
            torch.testing.assert_close(features[:, :4], torch.ones(3, 4) * 0.1)
            torch.testing.assert_close(features[:, 4:], torch.as_tensor(states))
            loaded, _ = heads_from_state(actor.state_dict(), critic.state_dict(), "cpu")
            torch.testing.assert_close(
                loaded.forward_features(features), actor.forward_features(features)
            )

    def test_dense_embedding_gather_preserves_duplicates_order_and_growth(self):
        from gridpfn.core.model import _EmbeddingTable

        table = _EmbeddingTable()
        table.extend(["a", "b"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        table.extend(["c"], torch.tensor([[5.0, 6.0]]))
        torch.testing.assert_close(
            table.gather(["c", "a", "c", "b"]),
            torch.tensor([[5.0, 6.0], [1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]),
        )
        table.clear()
        self.assertEqual(len(table), 0)

    def test_quota_actor_zero_correction_matches_causal_feedback(self):
        from gridpfn.core.control_guidance import FeedbackTeacher

        scaler = {"col_to_scaler_idx": {4: 0}, "min": [0], "max": [40]}
        actor = Actor(
            17,
            3,
            feature_mode="raw",
            thermal_bounds=[0, 40],
            target_temperature_bounds=[-5, 21.8],
            feasible_dt=1,
            quota_actor=True,
        )
        state = np.zeros((2, 17), dtype=np.float32)
        state[:, 0], state[:, 7], state[:, 9], state[:, 8] = 0.5, 0.75, 0.0, 0.2
        state[:, 15] = [0.0, 0.4]
        teacher = FeedbackTeacher(scaler, quota_aware=True)
        np.testing.assert_allclose(
            actor(torch.as_tensor(state))[:, 0].detach().numpy(), teacher(state)[:, 0], atol=1e-6
        )

    def test_storage_guide_uses_only_observed_load_and_limits_state_of_charge(self):
        from gridpfn.core.control_guidance import FeedbackTeacher

        scaler = {
            "col_to_scaler_idx": {0: 0, 1: 1, 4: 2, 7: 3},
            "min": [0, 0, 0, 0],
            "max": [10, 10, 40, 5],
        }
        guide = FeedbackTeacher(scaler, quota_aware=True, storage_aware=True)
        state = np.zeros((3, 17), dtype=np.float32)
        state[:, 0], state[:, 7], state[:, 9] = 0.5, 0.5, 0
        state[:, 8] = [0.2, 1, 0]
        state[:, 2] = [1, 1, 0]
        state[:, 3] = [0, 0, 1]
        action = guide(state)
        np.testing.assert_allclose(action[:, 2], [2.4, 0, 0], atol=1e-6)

    def test_physics_critic_known_thermal_reward_and_gradient_match_simulator(self):
        from gridpfn.core.environment import HOME_ENERGY_MGNT
        from gridpfn.core.model import heads_from_state

        critic = Critic(17, 3, 2, feature_mode="raw", twin=True, thermal_bounds=[0, 40])
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.zero_()
        state = torch.zeros(1, 17)
        state[:, 7], state[:, 9] = 0.75, 0.0
        control = torch.tensor([[2.5, 0.0, 0.0]], requires_grad=True)
        q1, q2 = critic.both_features(state, control)
        torch.testing.assert_close(q1, torch.full((1, 2), -0.5))
        torch.testing.assert_close(q2, q1)
        q1[:, 0].sum().backward()
        self.assertAlmostEqual(control.grad[0, 0].item(), -1.2, places=5)
        day = np.zeros((24, 8))
        day[:, 2], day[:, 4] = np.arange(24), 30
        env = HOME_ENERGY_MGNT(day, state_dim=17)
        env.indoor_temp = 20
        _, _, _, comfort, _ = env.step((0, control.detach().numpy()[0]))
        self.assertAlmostEqual(q1[0, 0].item(), comfort, places=6)
        actor = Actor(17, 3, feature_mode="raw")
        _, loaded = heads_from_state(actor.state_dict(), critic.state_dict(), "cpu")
        torch.testing.assert_close(loaded(state, control), q1)

    def test_quota_aware_temperature_actor_starts_without_saturated_gradient(self):
        from gridpfn.core.model import heads_from_state

        actor = Actor(
            17,
            3,
            feature_mode="raw",
            thermal_bounds=[0, 40],
            target_temperature_bounds=[-5, 21.8],
            feasible_dt=1,
        )
        state = torch.zeros(2, 17)
        state[:, 7], state[:, 9], state[:, 8] = 0.75, 0.3, 0.2
        action = actor(state)
        self.assertGreater(action[0, 0].item(), 0)
        self.assertLess(action[0, 0].item(), 2.5)
        action[:, 0].sum().backward()
        self.assertGreater(actor.fc4.bias.grad[0].abs().item(), 0)
        loaded, _ = heads_from_state(
            actor.state_dict(), Critic(17, 3, 2, feature_mode="raw").state_dict(), "cpu"
        )
        torch.testing.assert_close(loaded(state), actor(state), atol=0, rtol=0)

    def test_matched_oracle_regret_catches_impossible_improvements(self):
        from oracle.benchmark import OracleComparison

        comparison = OracleComparison.__new__(OracleComparison)
        comparison.home_ids = [27]
        comparison.records = {
            "day": {
                "frontiers": [{"minimum_squared_violation": 2}],
                "oracles": {
                    "paper_reward": {
                        "audit": {"homes": [{"reward": -3}]},
                        "solution": {"lower_bound": 3},
                    }
                },
            }
        }
        record = {
            "dates": ["day"],
            "reward": -5,
            "homes": [{"home_id": 27, "reward": -5, "squared_violation": 4}],
        }
        comparison.annotate(record)
        self.assertEqual(record["oracle_regret"], 2)
        self.assertEqual(record["excess_squared_violation"], 2)
        invalid = copy.deepcopy(record)
        invalid["homes"][0]["home_id"] = 950
        with self.assertRaisesRegex(ValueError, "cohort"):
            comparison.annotate(invalid)
        invalid = copy.deepcopy(record)
        invalid["homes"][0]["squared_violation"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            comparison.annotate(invalid)
        record["reward"] = record["homes"][0]["reward"] = 0
        with self.assertRaisesRegex(AssertionError, "certified bound"):
            comparison.annotate(record)

    def protocol(self):
        date = "2019-06-01"
        client = SimpleNamespace(
            home_id=27, state_dim=17, fixed_cost=5, scaler={"train_dates": [date]}
        )
        manifest = dict(
            split="train",
            training_labels=True,
            dates=[date],
            home_ids=[27],
            scenario=OracleConfig(home_ids=(27,), split="train").settings(),
            strategy={"dr_limit": 5},
            input_file_sha256={},
            price_weather_sha256="weather",
        )
        return manifest, [client], {"dr_limit": 5}, {"enabled": True, "price": 0.1}

    def test_training_protocol_rejects_heldout_dates_and_different_physics(self):
        with patch("oracle.demonstrations.file_sha256", return_value="weather"):
            validate_protocol(*self.protocol())
            for split in ("validation", "test"):
                manifest, *rest = self.protocol()
                manifest["split"] = split
                with self.assertRaisesRegex(ValueError, "held-out"):
                    validate_protocol(manifest, *rest)
            manifest, *rest = self.protocol()
            manifest["dates"] = ["2019-08-01"]
            with self.assertRaisesRegex(ValueError, "training split"):
                validate_protocol(manifest, *rest)
            manifest, *rest = self.protocol()
            manifest["scenario"]["ac_energy_quota"] = False
            with self.assertRaisesRegex(ValueError, "physical scenario"):
                validate_protocol(manifest, *rest)

    def test_evaluation_allows_matched_peer_tariff_but_training_labels_do_not(self):
        with patch("oracle.demonstrations.file_sha256", return_value="weather"):
            manifest, clients, strategy, market = self.protocol()
            manifest["scenario"]["peer_price"] = 0.04
            market["price"] = 0.04
            with self.assertRaisesRegex(ValueError, "physical scenario"):
                validate_protocol(manifest, clients, strategy, market)
            manifest["split"] = "validation"
            clients[0].test_dates = manifest["dates"]
            validate_protocol(manifest, clients, strategy, market, training=False)
            market["price"] = 0.10
            with self.assertRaisesRegex(ValueError, "peer market"):
                validate_protocol(manifest, clients, strategy, market, training=False)

    def test_evaluation_requires_identical_declared_grid_tariff(self):
        with patch("oracle.demonstrations.file_sha256", return_value="weather"):
            manifest, clients, strategy, market = self.protocol()
            manifest["split"] = "validation"
            manifest["scenario"].update(flat_price_per_kwh=0.3, tou_enabled=False)
            manifest["strategy"]["tou"] = {"enabled": False, "n_blocks": 5}
            strategy["tou"] = {"enabled": False, "n_blocks": 5}
            clients[0].test_dates = manifest["dates"]
            with self.assertRaisesRegex(ValueError, "explicit grid tariff differs"):
                validate_protocol(manifest, clients, strategy, market, training=False)
            clients[0].scaler["grid_prices"] = [0.3] * 24
            validate_protocol(manifest, clients, strategy, market, training=False)
            clients[0].scaler["grid_prices"][12] = 0.2
            with self.assertRaisesRegex(ValueError, "explicit grid tariff differs"):
                validate_protocol(manifest, clients, strategy, market, training=False)


if __name__ == "__main__":
    unittest.main()
