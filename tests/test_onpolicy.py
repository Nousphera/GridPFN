"""Actual latent likelihoods, daily returns and private local PPO updates."""

import copy
import unittest

import numpy as np
import torch

from gridpfn.core.agents.onpolicy import (
    OnPolicyAgent,
    OnPolicyLearner,
    advantages,
    log_probability,
    policy_kl,
)
from gridpfn.core.model import Actor, Critic, heads_from_state


class OnPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)

    def make_agent(self, **options):
        return OnPolicyAgent(
            Actor(
                17,
                3,
                feature_mode="raw",
                hidden_dim=16,
                learned_discrete=True,
                stochastic_std=0.1,
                feasible_dt=1,
            ),
            Critic(17, 3, 2, feature_mode="raw", hidden_dim=16, value_head=True),
            2,
            3,
            17,
            hyperparams={
                "gamma": 1,
                "bc_weight": 0,
                "lr_actor": 0.001,
                "lr_critic": 0.001,
                **options,
            },
        )

    def test_zero_value_updates_are_rejected_instead_of_silently_using_actor_epochs(self):
        with self.assertRaisesRegex(ValueError, "Invalid PPO"):
            self.make_agent(ppo_value_epochs=0)
        self.assertEqual(self.make_agent(ppo_value_epochs=None).ppo_value_epochs, 4)

    def test_actor_depth_activation_preserve_stacked_outputs_gradients_and_checkpoints(self):
        from gridpfn.core.batched_learning import _Stack

        features = torch.randn(2, 4, 17)
        critic = Critic(17, 3, 2, feature_mode="raw", hidden_dim=16, value_head=True)
        for depth, activation in ((1, "relu"), (1, "tanh"), (2, "relu"), (2, "tanh")):
            actor = Actor(
                17,
                3,
                feature_mode="raw",
                hidden_dim=16,
                hidden_layers=depth,
                activation=activation,
                learned_discrete=True,
                stochastic_std=0.1,
            )
            stack = _Stack([actor, copy.deepcopy(actor)], "actor")
            actual = stack(features, kind="distribution")
            expected = stack.functional(features, kind="distribution")
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b)
            params = stack.trainable()
            a_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), params)
            b_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), params)
            for a, b in zip(a_grad, b_grad, strict=True):
                torch.testing.assert_close(a, b)
            restored, _ = heads_from_state(actor.state_dict(), critic.state_dict(), "cpu")
            for a, b in zip(
                restored.distribution_features(features[0]),
                actor.distribution_features(features[0]),
                strict=True,
            ):
                torch.testing.assert_close(a, b)

    def test_lr_decay_uses_completed_days_and_rejects_invalid_schedules(self):
        for option in ("ppo_lr_decay_days", "ppo_bc_decay_days"):
            with self.assertRaisesRegex(ValueError, "Invalid PPO"):
                self.make_agent(**{option: -1})
        agent = self.make_agent(ppo_lr_decay_days=10)
        learner = OnPolicyLearner([agent], 1)
        learner.completed_days = 9
        state = np.zeros((1, 17), dtype=np.float32)
        state[:, 8] = 0.2
        action = learner.choose_actions(state)[0]
        agent.store_transition(state[0], action, 1, state[0], True)
        learner.learn()
        self.assertAlmostEqual(learner.actor_optimizer.param_groups[0]["lr"], 0.0001)

    def test_cooperative_targets_do_not_change_stored_private_rewards(self):
        agents = [self.make_agent(ppo_team_reward=True, ppo_entropy=0) for _ in range(2)]
        for agent in agents:
            for layer in (agent.critic_net.value_fc1, agent.critic_net.value_fc4):
                for p in layer.parameters():
                    p.data.zero_()
        learner = OnPolicyLearner(agents, 1)
        state = np.zeros((2, 17), dtype=np.float32)
        state[:, 8] = 0.2
        actions = learner.choose_actions(state)
        for agent, row, action, reward in zip(agents, state, actions, (2, -2), strict=True):
            agent.store_transition(row, action, reward, row, True)
        self.assertEqual([a.rollout[0][6] for a in agents], [2, -2])
        learner.learn()
        self.assertEqual(learner.last_diagnostics["value_rmse"], [0, 0])

    def test_gradient_federation_keeps_shared_actor_and_private_values(self):
        first = self.make_agent(
            ppo_federation="gradient",
            ppo_entropy=0,
            ppo_shared_gradient_clip=True,
            ppo_advantage_scale="cohort",
        )
        second = self.make_agent(
            ppo_federation="gradient",
            ppo_entropy=0,
            ppo_shared_gradient_clip=True,
            ppo_advantage_scale="cohort",
        )
        second.actor_net.load_state_dict(first.actor_net.state_dict())
        original = copy.deepcopy(first.actor_net.state_dict())
        learner = OnPolicyLearner([first, second], 1)
        for _ in range(2):
            for hour in range(4):
                state = np.zeros((2, 17), dtype=np.float32)
                state[:, 0], state[:, 8], state[:, 9] = hour / 24, 0.2, [0.1, 0.5]
                actions = learner.choose_actions(state)
                for agent, row, action, reward in zip(
                    (first, second), state, actions, (hour + 1, -4), strict=True
                ):
                    agent.store_transition(row, action, reward, row, hour == 3)
                learner.learn()
        self.assertTrue(
            any(
                not torch.equal(first.actor_net.state_dict()[key], value)
                for key, value in original.items()
            )
        )
        for key, value in first.actor_net.state_dict().items():
            torch.testing.assert_close(value, second.actor_net.state_dict()[key], atol=0, rtol=0)
        self.assertTrue(
            any(
                not torch.equal(value, second.critic_net.state_dict()[key])
                for key, value in first.critic_net.state_dict().items()
            )
        )

    def test_shared_gradient_clipping_requires_gradient_federation(self):
        with self.assertRaisesRegex(ValueError, "Invalid PPO"):
            self.make_agent(ppo_shared_gradient_clip=True)

    def test_value_warmup_changes_only_private_values_and_preserves_training_streams(self):
        from types import SimpleNamespace

        agent = self.make_agent(ppo_shuffle_days=True, training_seed=913)
        learner = OnPolicyLearner([agent], 5)
        day = np.zeros((24, 8))
        day[:, 2], day[:, 4], day[:, 3], day[:, 0] = np.arange(24), 30, 0.1, 0.5
        client = SimpleNamespace(
            train_data=np.stack([day, day]),
            state_dim=17,
            scaler={"train_dates": ["2019-06-01", "2019-06-02"], "delta_t": 1},
            fedavg_agent=agent,
            device=torch.device("cpu"),
        )
        actor = copy.deepcopy(agent.actor_net.state_dict())
        critic = copy.deepcopy(agent.critic_net.state_dict())
        rng = torch.get_rng_state().clone()
        calendar_rng = copy.deepcopy(learner.day_rng.bit_generator.state)
        result = learner.warmup_values([client], {"enabled": True, "price": 0.1}, 2, 16)
        self.assertTrue(np.isfinite(result["rmse"]).all())
        for key, value in actor.items():
            torch.testing.assert_close(value, agent.actor_net.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(
            any(
                not torch.equal(value, agent.critic_net.state_dict()[key])
                for key, value in critic.items()
            )
        )
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(learner.day_rng.bit_generator.state, calendar_rng)
        self.assertEqual(learner.interval, 5)
        self.assertEqual(learner.completed_days, 0)
        self.assertEqual(agent.frame_idx, 0)
        self.assertFalse(agent.rollout)

    def test_ac_exploration_can_be_matched_without_changing_other_actuators(self):
        actor = Actor(
            17,
            3,
            feature_mode="raw",
            learned_discrete=True,
            stochastic_std=0.1,
            stochastic_ac_std=1 / 30,
        )
        _, std, _ = actor.distribution_features(torch.zeros(2, 17))
        torch.testing.assert_close(std, torch.tensor([[1 / 30, 0.1, 0.1]]).expand(2, -1))
        with self.assertRaisesRegex(ValueError, "AC exploration"):
            Actor(
                17,
                3,
                feature_mode="raw",
                learned_discrete=True,
                stochastic_std=0.1,
                stochastic_ac_std=0.001,
            )

    def test_thermal_conditioning_preserves_local_power_sensitivity_and_checkpoint(self):
        from gridpfn.core.batched_learning import _Stack

        actors = []
        for radius, conditioned in ((2, False), (6, True)):
            torch.manual_seed(71)
            actor = Actor(
                17,
                3,
                feature_mode="raw",
                hidden_dim=16,
                learned_discrete=True,
                stochastic_std=0.1,
                thermal_bounds=(20, 35),
                target_temperature_bounds=(-5, 26),
                quota_actor=True,
                quota_correction=radius,
                thermal_conditioning=conditioned,
                feasible_dt=1,
            )
            actors.append(actor)
        state = torch.zeros(3, 17)
        state[:, 7], state[:, 9], state[:, 8] = 2 / 3, 0.3, 0.2
        sensitivities = []
        for actor in actors:
            actor.zero_grad()
            actor.forward_features(state)[:, 0].mean().backward()
            sensitivities.append(actor.fc4.bias.grad[0].item())
        np.testing.assert_allclose(sensitivities, [-2 / 3, -2 / 3], rtol=1e-5)
        actor = actors[1]
        stack = _Stack([actor, copy.deepcopy(actor)], "actor")
        features = state[None].expand(2, -1, -1)
        for actual, expected in zip(
            stack(features, kind="distribution"),
            stack.functional(features, kind="distribution"),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(stack(features), stack.functional(features))
        critic = Critic(17, 3, 2, feature_mode="raw", value_head=True, hidden_dim=16)
        restored, _ = heads_from_state(actor.state_dict(), critic.state_dict(), "cpu")
        torch.testing.assert_close(restored(state), actor(state))
        _, std, _ = restored.distribution_features(state)
        torch.testing.assert_close(std[:, 0], torch.full((3,), 1 / 30))

    def test_terminal_advantages_never_cross_day_boundaries(self):
        rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        values = torch.ones_like(rewards)
        done = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
        advantage, returns = advantages(rewards, values, done, 1, 1)
        torch.testing.assert_close(
            returns, torch.tensor([[3.0, 30.0], [2.0, 20.0], [7.0, 70.0], [4.0, 40.0]])
        )
        torch.testing.assert_close(advantage, returns - values)

    def test_latent_likelihood_matches_gaussian_and_categorical_distributions(self):
        mean, std, logits = torch.randn(4, 3), torch.full((4, 3), 0.2), torch.randn(4, 2)
        latent, choice = torch.randn(4, 3), torch.tensor([0, 1, 0, 1])
        expected = torch.distributions.Normal(mean, std).log_prob(latent).sum(
            -1
        ) + torch.distributions.Categorical(logits=logits).log_prob(choice)
        torch.testing.assert_close(log_probability(mean, std, logits, latent, choice), expected)

    def test_clipped_ac_score_has_zero_expectation_and_finite_constant_action_gradients(self):
        from scipy.special import ndtr

        # Integrate the mixed distribution: two atoms and its interior density.
        nodes, weights = np.polynomial.legendre.leggauss(128)
        lower, upper, center, sigma = -0.2, 0.3, 0.1, 0.4
        interior = lower + (nodes + 1) * (upper - lower) / 2
        mass = (
            weights
            * (upper - lower)
            / 2
            * np.exp(-0.5 * ((interior - center) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi))
        )
        mass = np.r_[ndtr((lower - center) / sigma), mass, ndtr((center - upper) / sigma)]
        samples = torch.zeros(130, 3, dtype=torch.float64)
        samples[:, 0] = torch.from_numpy(np.r_[lower - 1, interior, upper + 1])
        mean = torch.tensor([center, 0, 0], dtype=torch.float64, requires_grad=True)
        std = torch.tensor([sigma, 0.4, 0.4], dtype=torch.float64, requires_grad=True)
        logits = torch.zeros(130, 2, dtype=torch.float64)
        mask = torch.tensor([1, 0, 0, 0], dtype=torch.float64)
        bounds = (torch.full((130,), lower), torch.full((130,), upper))
        score = log_probability(
            mean.expand_as(samples),
            std.expand_as(samples),
            logits,
            samples,
            torch.zeros(130, dtype=torch.long),
            mask,
            bounds,
        )
        gradients = torch.autograd.grad((score * torch.from_numpy(mass)).sum(), (mean, std))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)
        bounds = (torch.full((130,), float("inf")), torch.full((130,), float("inf")))
        constant = log_probability(
            mean.expand_as(samples),
            std.expand_as(samples),
            logits,
            samples,
            torch.zeros(130, dtype=torch.long),
            mask,
            bounds,
        )
        self.assertTrue(torch.equal(constant, torch.zeros_like(constant)))
        for gradient in torch.autograd.grad(constant.sum(), (mean, std)):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_clipped_ac_boundaries_match_original_executed_powers(self):
        from gridpfn.core.environment import HOME_ENERGY_MGNT

        day = np.zeros((24, 8))
        day[:, 2], day[:, 4], day[:, 5] = np.arange(24), 30, 0.3
        original = HOME_ENERGY_MGNT(day, state_dim=17)
        actor = Actor(
            17,
            3,
            feature_mode="raw",
            hidden_dim=16,
            thermal_bounds=(0, 1),
            target_temperature_bounds=(-5, 26),
            quota_actor=True,
            quota_correction=6,
            feasible_dt=1,
            learned_discrete=True,
            stochastic_std=0.1,
        )
        agent = OnPolicyAgent(
            actor,
            Critic(17, 3, 2, feature_mode="raw", value_head=True),
            2,
            3,
            17,
            hyperparams={"ppo_clip_ac_likelihood": True},
        )
        learner = OnPolicyLearner([agent], 1)
        for step in (0, 15, 23):
            env = copy.deepcopy(original)
            env.current_step = step
            if step == 23:
                env.ac_energy_delivered = env.ac_required_energy - 0.15
            state = torch.as_tensor(env._state_for_step(step), dtype=torch.float32)[None, None]
            lower, upper = learner._ac_bounds(state)

            def executed(z):
                trial = copy.deepcopy(env)
                latent = torch.tensor([[float(z), 0, 0]])
                control = actor.latent_features(state[0], latent).detach().numpy()[0]
                trial.step((0, control))
                return trial.power_AC

            if torch.isfinite(lower).item():
                self.assertAlmostEqual(
                    executed(lower.item() - 0.1), executed(lower.item() - 1), places=5
                )
            if torch.isfinite(upper).item():
                self.assertAlmostEqual(
                    executed(upper.item() + 0.1), executed(upper.item() + 1), places=5
                )

    def test_sampling_records_latents_before_saturation_and_rejects_stale_samples(self):
        agent = self.make_agent()
        learner = OnPolicyLearner([agent], 1)
        state = np.zeros((1, 17), dtype=np.float32)
        state[:, 8] = 0.2
        action = learner.choose_actions(state)[0]
        latent = agent.pending_policy[1].clone()
        features = agent.pending_policy[0].clone()
        state[:] = 10  # Caller mutation cannot alter the recorded observation.
        torch.testing.assert_close(agent.pending_policy[0], features)
        self.assertEqual(action[1][1], 0)  # No EV demand, many latents map to zero.
        with self.assertRaisesRegex(RuntimeError, "not been settled"):
            learner.choose_actions(state)
        agent.store_transition(state[0], action, 1, state[0], True)
        torch.testing.assert_close(agent.rollout[0][1], latent)
        with self.assertRaisesRegex(RuntimeError, "freshly sampled"):
            agent.store_transition(state[0], action, 1, state[0], True)
        learner.learn()
        self.assertEqual(len(agent.rollout), 0)

    def test_vectorized_local_updates_match_independent_home_updates(self):
        agents = [self.make_agent(), self.make_agent()]
        reference = copy.deepcopy(agents)
        batched = OnPolicyLearner(agents, 1)
        separate = [OnPolicyLearner([a], 1) for a in reference]
        for step in range(4):
            state = np.zeros((2, 17), dtype=np.float32)
            state[:, 0], state[:, 8], state[:, 11] = step / 24, 0.2, 0.2
            actions = batched.choose_actions(state)
            for i, (agent, ref) in enumerate(zip(agents, reference, strict=True)):
                ref.pending_policy = copy.deepcopy(agent.pending_policy)
                for a in (agent, ref):
                    a.store_transition(state[i], actions[i], (1 + i) * step, state[i], step == 3)
        actual = batched.learn()
        expected = [learner.learn()[0] for learner in separate]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        for agent, ref in zip(agents, reference, strict=True):
            for key in ("actor_net", "critic_net"):
                for a, b in zip(
                    getattr(agent, key).parameters(), getattr(ref, key).parameters(), strict=True
                ):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_stochastic_checkpoint_keeps_deterministic_deployment_contract(self):
        agent = self.make_agent()
        actor, critic = heads_from_state(
            agent.actor_net.state_dict(), agent.critic_net.state_dict(), "cpu"
        )
        state = torch.zeros(5, 17)
        state[:, 8] = 0.2
        torch.testing.assert_close(actor(state), agent.actor_net(state))
        self.assertEqual(actor(state).shape, (5, 3))
        self.assertEqual(critic(state, actor(state)).shape, (5, 2))
        torch.testing.assert_close(
            critic.value_features(state), agent.critic_net.value_features(state)
        )

    def test_preset_is_explicit_and_preserves_original_physical_constraints(self):
        from gridpfn.core.training_config import parse_args
        from gridpfn.experiments.audit_constraints import CONSTRAINT_OPTIONS

        legacy, ppo = parse_args([]), parse_args(["--preset", "ppo", "--episode", "5000"])
        self.assertEqual(
            {k: getattr(legacy, k) for k in CONSTRAINT_OPTIONS},
            {k: getattr(ppo, k) for k in CONSTRAINT_OPTIONS},
        )
        self.assertEqual(ppo.actor_update, "ppo")
        self.assertTrue(ppo.strict_convergence)
        self.assertEqual(ppo.episode, 5000)
        self.assertEqual(legacy.actor_update, "q_gradient")

    def test_popart_preserves_values_and_checkpoint_with_independent_value_width(self):
        agent = self.make_agent()
        agent.critic_net = Critic(
            17,
            3,
            2,
            feature_mode="raw",
            hidden_dim=16,
            value_head=True,
            value_hidden_dim=32,
            value_normalization=True,
        )
        learner = OnPolicyLearner([agent], 1)
        features = torch.randn(1, 6, 17)
        before = learner.critic(features, kind="value").detach().clone()
        learner._normalize_values(torch.tensor([[-500.0, -200.0, -100.0, 20.0, 200.0, 600.0]]))
        torch.testing.assert_close(
            learner.critic(features, kind="value"), before, atol=2e-6, rtol=2e-5
        )
        _, restored = heads_from_state(
            agent.actor_net.state_dict(), agent.critic_net.state_dict(), "cpu"
        )
        torch.testing.assert_close(
            restored.value_features(features[0]), before[0], atol=2e-6, rtol=2e-5
        )
        self.assertEqual(restored.value_fc1.out_features, 32)

    def test_masked_likelihood_ignores_only_inactive_factors(self):
        mean = torch.zeros(2, 3)
        std, logits = torch.ones_like(mean), torch.zeros(2, 2)
        latent, choice = torch.zeros_like(mean), torch.zeros(2, dtype=torch.long)
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        before = log_probability(mean, std, logits, latent, choice, mask)
        mean[0, 1], logits[0, 0] = 100, 100
        torch.testing.assert_close(log_probability(mean, std, logits, latent, choice, mask), before)
        reference = (torch.zeros_like(mean), std.clone(), torch.zeros_like(logits))
        self.assertEqual(float(policy_kl(reference, (mean, std, logits), mask)[0]), 0)
        agent = self.make_agent()
        agent.ppo_mask_inactive = True
        learner = OnPolicyLearner([agent], 1)
        state = torch.zeros(1, 1, 17)
        state[..., 0], state[..., 8], state[..., 12] = 9 / 24, 0.2, 1
        self.assertEqual(learner._mask(state)[0, 0, 3], 1)  # Early WM start incurs a penalty.
        state[..., 0] = 17 / 24
        self.assertEqual(learner._mask(state)[0, 0, 3], 0)  # Deadline forces a start.

    def test_latent_kl_agrees_with_distribution_kl(self):
        before = (torch.randn(5, 3), torch.rand(5, 3) + 0.1, torch.randn(5, 2))
        after = (torch.randn(5, 3), torch.rand(5, 3) + 0.1, torch.randn(5, 2))
        kl = torch.distributions.kl_divergence
        expected = kl(
            torch.distributions.Normal(*before[:2]), torch.distributions.Normal(*after[:2])
        ).sum(-1)
        expected += kl(
            torch.distributions.Categorical(logits=before[2]),
            torch.distributions.Categorical(logits=after[2]),
        )
        torch.testing.assert_close(policy_kl(before, after), expected)

    def test_federation_preserves_integer_observation_metadata(self):
        from gridpfn.core.server import Server

        rows = [
            {"auxiliary_width": torch.tensor(17), "weight": torch.tensor([1.0])},
            {"auxiliary_width": torch.tensor(17), "weight": torch.tensor([3.0])},
        ]
        merged = Server._average_parameters(rows)
        self.assertEqual(merged["auxiliary_width"].dtype, torch.int64)
        self.assertEqual(int(merged["auxiliary_width"]), 17)
        torch.testing.assert_close(merged["weight"], torch.tensor([2.0]))
        rows[1]["auxiliary_width"] = torch.tensor(8)
        with self.assertRaisesRegex(ValueError, "metadata"):
            Server._average_parameters(rows)

    def test_shuffled_daily_markets_share_dates_and_keep_action_rng_unchanged(self):
        from types import SimpleNamespace

        agent = self.make_agent()
        agent.ppo_shuffle_days = True
        agent.training_seed = 791
        learner = OnPolicyLearner([agent], 1)
        clients = [
            SimpleNamespace(scaler={"train_dates": ["a", "b", "c"]}),
            SimpleNamespace(scaler={"train_dates": ["b", "c", "d"]}),
        ]
        before = torch.random.get_rng_state().clone()
        dates = []
        for episode in range(4):
            indices = learner.training_indices(clients, episode)
            selected = [c.scaler["train_dates"][i] for c, i in zip(clients, indices, strict=True)]
            self.assertEqual(selected[0], selected[1])
            dates.append(selected[0])
        self.assertEqual(set(dates[:2]), {"b", "c"})
        self.assertEqual(set(dates[2:]), {"b", "c"})
        torch.testing.assert_close(torch.random.get_rng_state(), before)


if __name__ == "__main__":
    unittest.main()
