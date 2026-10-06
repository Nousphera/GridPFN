"""Integration contracts using the real TabPFN 3.5 checkpoint.

Run: python -m unittest discover -s tests -v
Requires the dependencies and normal TabPFN checkpoint access described in README.
"""

import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.agents.agent_feddmpq import P_DQN_FedDMPQ
from gridpfn.core.agents.agent_fedfit import P_DQN_FedFit
from gridpfn.core.agents.agent_pffdst import P_DQN_PFFDST
from gridpfn.core.agents.agent_sparse import P_DQN_sparse
from gridpfn.core.environment import CONTINUOUS_ACTION_MAX, CONTINUOUS_ACTION_MIN, HOME_ENERGY_MGNT
from gridpfn.core.model import (
    Actor,
    Critic,
    _embedding_cache,
    _tabpfn_backbone,
    checkpoint_state_dim,
    heads_from_state,
    precompute_embeddings,
)
from gridpfn.core.server import Server
from gridpfn.core.utils.agent_utils import hard_update_target_network, soft_update_target_network


class TabPFNModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        # Exercise the first-load inference-mode path before training uses it.
        with torch.inference_mode():
            cls.backbone = _tabpfn_backbone("cpu")

    def setUp(self):
        torch.manual_seed(7)
        self.actor = Actor(9, 3)
        self.critic = Critic(9, 3, 2)
        self.states = torch.rand(3, 9)

    def test_shapes_bounds_and_batch_independence(self):
        with torch.inference_mode():
            actions = self.actor(self.states)
            values = self.critic(self.states, actions)
            self.assertEqual(actions.shape, (3, 3))
            self.assertEqual(values.shape, (3, 2))
            self.assertTrue(torch.isfinite(actions).all())
            self.assertTrue(torch.isfinite(values).all())
            self.assertTrue((actions >= torch.tensor(CONTINUOUS_ACTION_MIN)).all())
            self.assertTrue((actions <= torch.tensor(CONTINUOUS_ACTION_MAX)).all())
            # Recompute the singleton so this checks TabPFN batch independence,
            # rather than simply retrieving the earlier cached batch output.
            _embedding_cache("cpu").clear()
            single_action = self.actor(self.states[:1])
            single_value = self.critic(self.states[:1], actions[:1])
            torch.testing.assert_close(single_action, actions[:1], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(single_value, values[:1], atol=2e-5, rtol=2e-5)
            self.assertEqual(self.actor(self.states[:0]).shape, (0, 3))
            self.assertEqual(self.critic(self.states[:0], actions[:0]).shape, (0, 2))

    def test_actor_custom_bounds_and_unrestricted_q_values(self):
        self.actor.set_action_bounds((-1, -2, -3), (1, 2, 3))
        with torch.no_grad():
            actions = self.actor(self.states)
            self.assertTrue((actions >= self.actor.action_min).all())
            self.assertTrue((actions <= self.actor.action_max).all())
            self.critic.fc4.weight.zero_()
            self.critic.fc4.bias.copy_(torch.tensor([-4.0, 7.0]))
            torch.testing.assert_close(
                self.critic(self.states, actions), torch.tensor([[-4.0, 7.0]]).expand(3, -1)
            )

    def test_hybrid_preserves_existing_observation_and_checkpoint_contract(self):
        actor, critic = Actor(17, 3, feature_mode="hybrid"), Critic(17, 3, 2, feature_mode="hybrid")
        states = torch.rand(3, 17)
        features = actor.prepare_features(states)
        torch.testing.assert_close(features[:, -17:], states, atol=0, rtol=0)
        self.assertEqual(checkpoint_state_dim(actor.state_dict()), 17)
        loaded_actor, loaded_critic = heads_from_state(
            actor.state_dict(), critic.state_dict(), "cpu"
        )
        actions = actor(states)
        torch.testing.assert_close(loaded_actor(states), actions, atol=0, rtol=0)
        torch.testing.assert_close(
            loaded_critic(states, actions), critic(states, actions), atol=0, rtol=0
        )
        self.assertEqual(critic(states[:0], actions[:0]).shape, (0, 2))

    def test_policy_gradient_through_frozen_critic(self):
        self.actor.train()
        self.critic.train()
        actions = self.actor(self.states)
        actions.retain_grad()
        loss = -self.critic(self.states, actions).max(1).values.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(actions.grad.abs().sum().item(), 0)
        for parameter in self.actor.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.actor.fc4.weight.grad.abs().sum().item(), 0)
        self.assertFalse(self.backbone.training)
        self.assertIs(self.backbone, _tabpfn_backbone("cpu"))
        self.assertTrue(
            all(not p.requires_grad and p.grad is None for p in self.backbone.parameters())
        )

    def test_head_only_checkpoint_reload_and_target_updates(self):
        for source, target, args in (
            (self.actor, Actor(9, 3), (self.states,)),
            (self.critic, Critic(9, 3, 2), (self.states, torch.rand(3, 3))),
        ):
            self.assertEqual(
                set(source.state_dict()), {"fc1.weight", "fc1.bias", "fc4.weight", "fc4.bias"}
            )
            self.assertEqual(set(dict(source.named_parameters())), set(source.state_dict()))
            buffer = io.BytesIO()
            torch.save(source.state_dict(), buffer)
            buffer.seek(0)
            target.load_state_dict(torch.load(buffer, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(source(*args), target(*args), atol=0, rtol=0)
                target.fc4.weight.zero_()
            soft_update_target_network(source, target, 0.25)
            torch.testing.assert_close(target.fc4.weight, 0.25 * source.fc4.weight)
            hard_update_target_network(source, target)
            torch.testing.assert_close(target.fc4.weight, source.fc4.weight)

    def test_all_agent_learning_and_topology_contracts(self):
        classes = (P_DQN, P_DQN_sparse, P_DQN_FedFit, P_DQN_PFFDST, P_DQN_FedDMPQ)
        for agent_cls in classes:
            with self.subTest(agent=agent_cls.__name__):
                agent = agent_cls(
                    actor_net=Actor(9, 3),
                    critic_net=Critic(9, 3, 2),
                    state_dim=9,
                    continuous_action_dim=3,
                    discrete_action_dim=2,
                    hyperparams={"batch_size": 2, "epsilon_start": 0, "epsilon_end": 0, "K": 1},
                )
                if isinstance(agent, P_DQN_PFFDST):
                    agent.initialize(0.5, 0.5)
                elif isinstance(agent, P_DQN_sparse):
                    params = agent.get_trainable_params()
                    masks = [
                        agent.init_masks(p, agent.calculate_sparsities(p, sparse=0.5))
                        for p in params
                    ]
                    agent.set_model_masks(*masks)
                if isinstance(agent, P_DQN_FedDMPQ):
                    agent.init_soft_prune_state()
                for i in range(3):
                    state = self.states[i].numpy()
                    action = agent.choose_action(state)
                    self.assertIn(action[0], (0, 1))
                    self.assertEqual(action[1].shape, (3,))
                    agent.store_transition(state, action, float(i), state * 0.9, i == 2)
                losses = agent.learn()
                self.assertEqual(len(losses), 2)
                self.assertTrue(np.isfinite(losses).all())
                if isinstance(agent, P_DQN_FedFit):
                    self.assertTrue(agent.adjust_topology(1))
                elif isinstance(agent, P_DQN_PFFDST):
                    agent.server_readjust("stage1", grow=False)
                    agent.freeze_subnetwork()
                    agent.server_readjust("stage2")
                    frozen = agent.get_frozen_state()
                    self.assertTrue(np.isfinite(agent.learn()).all())
                    for net, masks, values in (
                        (agent.actor_net, frozen[0], frozen[2]),
                        (agent.critic_net, frozen[1], frozen[3]),
                    ):
                        for name, param in net.named_parameters():
                            torch.testing.assert_close(
                                param[masks[name]], values[name][masks[name]]
                            )
                elif isinstance(agent, P_DQN_FedDMPQ):
                    agent.soft_prune_step(0.5, 0.5)
                    self.assertTrue(
                        all(
                            torch.isfinite(t).all()
                            for state in agent.get_mixed_precision_params()
                            for t in state.values()
                        )
                    )
                elif isinstance(agent, P_DQN_sparse):
                    gradients = agent.screen_gradients()
                    self.assertTrue(
                        all(torch.isfinite(t).all() for group in gradients for t in group.values())
                    )
                actor_state, critic_state = agent.get_model_params()
                agent.set_model_params(actor_state, critic_state)
                average = Server._average_parameters([actor_state, actor_state])
                for name in actor_state:
                    torch.testing.assert_close(average[name], actor_state[name])

    def test_dtype_and_input_shape(self):
        self.actor.double()
        self.critic.double()
        states = self.states.double()
        actions = self.actor(states)
        self.assertEqual(actions.dtype, torch.float64)
        self.assertEqual(self.critic(states, actions).dtype, torch.float64)
        with self.assertRaisesRegex(ValueError, "Expected"):
            self.actor(torch.ones(9))
        with self.assertRaisesRegex(ValueError, "Expected"):
            self.actor(torch.ones(1, 8))

    def test_cached_updates_do_not_call_or_backpropagate_through_tabpfn(self):
        _embedding_cache("cpu").clear()
        with torch.inference_mode():
            self.assertEqual(precompute_embeddings(self.states, "cpu"), 3)
        with patch.object(
            self.backbone, "forward", side_effect=AssertionError("Unexpected TabPFN call")
        ):
            changed = self.states.clone()
            changed[:, -1] += 0.4
            actions = self.actor(changed)
            actions.retain_grad()
            loss = -self.critic(changed, actions).max(1).values.mean()
            loss.backward()
            self.assertGreater(actions.grad.abs().sum().item(), 0)
            self.assertFalse(torch.allclose(actions, self.actor(self.states)))
            self.assertEqual(precompute_embeddings(changed, "cpu"), 0)
        self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))

    def test_precomputed_day_covers_rollout_and_terminal_state(self):
        day = np.ones((24, 8))
        day[:, 2] = np.arange(24)
        day[:, 3] = 0.1
        day[:, 4] = 25
        env = HOME_ENERGY_MGNT(day)
        states = np.asarray([env._state_for_step(step) for step in range(25)])
        precompute_embeddings(states, "cpu")
        state = env.reset()
        with patch.object(
            self.backbone, "forward", side_effect=AssertionError("Cache missed rollout state")
        ):
            for _ in range(24):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = self.actor(state_tensor)
                value = self.critic(state_tensor, action)
                state, _, _, _, done = env.step(
                    (value.argmax(1).item(), action[0].detach().numpy())
                )
            self.assertTrue(done)
            self.actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))

    def test_control_state_bypasses_backbone_and_preserves_checkpoint_contract(self):
        actor, critic = Actor(17, 3), Critic(17, 3, 2)
        states = torch.rand(3, 17)
        precompute_embeddings(states, "cpu")
        self.assertEqual(checkpoint_state_dim(actor.state_dict()), 17)
        self.assertEqual(checkpoint_state_dim(self.actor.state_dict()), 9)
        with patch.object(
            self.backbone, "forward", side_effect=AssertionError("Live state entered TabPFN")
        ):
            changed = states.clone()
            changed[:, 8:] += 0.2
            changed.requires_grad_(True)
            features = actor._state_features(changed)
            torch.testing.assert_close(features[:, -9:], changed[:, 8:])
            actions = actor(changed)
            self.assertEqual(actions.shape, (3, 3))
            q = critic(changed, actions)
            self.assertEqual(q.shape, (3, 2))
            q.sum().backward()
            self.assertGreater(changed.grad[:, 9].abs().sum().item(), 0)
            self.assertEqual(precompute_embeddings(changed, "cpu"), 0)
            self.assertEqual(actor(changed[:0]).shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
