import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gridpfn.core.environment import CONTINUOUS_ACTION_MAX, CONTINUOUS_ACTION_MIN
from gridpfn.core.utils.agent_utils import ReplayBuffer, soft_update_target_network


# Dense P-DQN with a discrete WM decision and continuous AC/EV/BESS control.
class P_DQN:
    def __init__(
        self,
        actor_net: nn.Module,
        critic_net: nn.Module,
        discrete_action_dim,
        continuous_action_dim,
        state_dim,
        hyperparams=None,
    ):

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.device = next(actor_net.parameters()).device
        self.state_dim = state_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dim = discrete_action_dim

        hp = hyperparams or {}
        self.memory_capacity = int(hp.get("memory_capacity", 1000))
        default_action_min = CONTINUOUS_ACTION_MIN
        default_action_max = CONTINUOUS_ACTION_MAX
        self.continuous_action_min = hp.get("continuous_action_min", default_action_min)
        self.continuous_action_max = hp.get("continuous_action_max", default_action_max)
        self.actor_net.set_action_bounds(
            self.continuous_action_min,
            self.continuous_action_max,
        )
        self.actor_target_net = copy.deepcopy(self.actor_net)
        self.critic_target_net = copy.deepcopy(self.critic_net)

        self.gamma = float(hp.get("gamma", 0.99))
        self.batch_size = int(hp.get("batch_size", 32))
        self.lr_actor = float(hp.get("lr_actor", 0.00001))
        self.lr_critic = float(hp.get("lr_critic", 0.0001))
        self.epsilon_start = float(hp.get("epsilon_start", 1.0))
        self.epsilon_end = float(hp.get("epsilon_end", 0.005))
        self.epsilon_decay = int(hp.get("epsilon_decay", 10000))
        self.critic_tau = float(hp.get("critic_tau", 0.01))
        self.actor_tau = float(hp.get("actor_tau", 0.001))
        self.policy_delay = int(hp.get("policy_delay", 1))
        self.bc_rounds = int(hp.get("bc_rounds", 0))
        self.bc_steps = int(hp.get("bc_steps", 50))
        self.bc_lr = float(hp.get("bc_lr", 0.001))
        self.demonstration_cache = hp.get("demonstration_cache")
        self.oracle_demonstrations = hp.get("oracle_demonstrations")
        self.expert_objective = hp.get("expert_objective", "paper_reward")
        if self.expert_objective not in ("paper_reward", "comfort_first"):
            raise ValueError("Unknown expert objective")
        self.expert_fraction = float(hp.get("expert_fraction", 0.25))
        self.expert_critic_steps = int(hp.get("expert_critic_steps", 1000))
        self.expert_memory = None
        self.expert_returns = None
        if not 0 < self.expert_fraction < 1 or self.expert_critic_steps < 0:
            raise ValueError("expert_fraction must be in (0,1) and expert_critic_steps nonnegative")
        self.bc_weight = float(hp.get("bc_weight", 0))
        self.guidance_weights = tuple(float(v) for v in hp.get("guidance_weights", (1, 1, 1)))
        if len(self.guidance_weights) != 3 or any(
            not math.isfinite(v) or v < 0 for v in self.guidance_weights
        ):
            raise ValueError("guidance_weights requires three finite nonnegative coefficients")
        self.actor_q_weight = float(hp.get("actor_q_weight", 1))
        self.actor_update = hp.get("actor_update", "q_gradient")
        self.expectile = float(hp.get("expectile", 0.7))
        self.advantage_temperature = float(hp.get("advantage_temperature", 3))
        if (
            self.actor_update not in ("q_gradient", "implicit")
            or not 0.5 <= self.expectile < 1
            or self.advantage_temperature <= 0
        ):
            raise ValueError("Invalid actor update, expectile or advantage temperature")
        if self.actor_update == "implicit" and (
            self.bc_weight
            or not getattr(critic_net, "twin", False)
            or not hasattr(critic_net, "value_fc1")
            or not hasattr(actor_net, "discrete_fc4")
            or hp.get("return_mode", "td") != "td"
            or hp.get("target_noise", 0)
        ):
            raise ValueError(
                "Implicit learning requires twin Q/value and categorical actor heads, TD replay, bc_weight=0 and target_noise=0"
            )
        self.target_noise = float(hp.get("target_noise", 0))
        self.critic_huber_delta = float(hp.get("critic_huber_delta", 0))
        if self.critic_huber_delta < 0:
            raise ValueError("critic_huber_delta must be nonnegative")
        self.exploration_noise = hp.get("exploration_noise")
        if self.exploration_noise is not None and self.exploration_noise < 0:
            raise ValueError("exploration_noise must be nonnegative")
        if self.target_noise < 0:
            raise ValueError("target_noise must be nonnegative")
        if self.policy_delay < 1 or self.bc_rounds < 0 or self.bc_steps < 1:
            raise ValueError("policy_delay/bc_steps must be positive and bc_rounds nonnegative")
        if self.bc_weight < 0 or self.actor_q_weight < 0 or self.bc_lr <= 0:
            raise ValueError("Guidance weights must be nonnegative and bc_lr positive")
        if (self.bc_rounds or self.bc_weight) and state_dim != 17:
            raise ValueError("Feedback guidance requires state_dim=17")
        self.feedback_teacher = None
        self.feedback_target = float(hp.get("feedback_target", 20))
        self.quota_guidance = bool(hp.get("quota_guidance", False))
        self.storage_guidance = bool(hp.get("storage_guidance", False))
        if (self.quota_guidance or self.storage_guidance) and self.oracle_demonstrations:
            raise ValueError("Choose causal quota guidance or oracle demonstrations")
        if not 18 <= self.feedback_target <= 22:
            raise ValueError("feedback_target must be inside the original comfort band")
        self.residual_scale = hp.get("residual_scale")
        self.return_mode = hp.get("return_mode", "td")
        if self.return_mode not in ("td", "episode"):
            raise ValueError("return_mode must be td or episode")
        self.pending_episode = []
        self.learning_steps = 0

        self.memory = ReplayBuffer(self.memory_capacity)
        self.frame_idx = 0
        self.epsilon = lambda frame_idx: (
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * math.exp(-1.0 * frame_idx / self.epsilon_decay)
        )

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic)

    @torch.no_grad()
    def _policy_action(self, state, noisy=False):
        features = self.actor_net.prepare_features(np.asarray(state)[None])
        control = self.actor_net.forward_features(features)
        if noisy:
            span = self.actor_net.action_max - self.actor_net.action_min
            control = self.actor_net.project_action(
                features,
                control
                + torch.as_tensor(
                    np.random.normal(size=control.shape), dtype=control.dtype, device=self.device
                )
                * self.exploration_noise
                * span,
            )
        choice = (
            self.actor_net.discrete_features(features)
            if self.actor_update == "implicit"
            else self.critic_net.forward_features(features, control)
        ).argmax(1)
        return torch.cat((choice[:, None], control), dim=1)[0].cpu().numpy()

    def choose_action(self, state):
        """Preserve epsilon/continuous-exploration RNG order and requested actions."""
        self.frame_idx += 1
        if self.exploration_noise is not None:
            result = self._policy_action(state, noisy=True)
            if random.random() < self.epsilon(self.frame_idx):
                result[0] = random.randrange(self.discrete_action_dim)
            return int(result[0]), result[1:]
        if random.random() > self.epsilon(self.frame_idx):
            result = self._policy_action(state)
            return int(result[0]), result[1:]
        return random.randrange(self.discrete_action_dim), np.random.uniform(
            self.continuous_action_min, self.continuous_action_max, size=self.continuous_action_dim
        )

    # Proxy to store a transition into the replay buffer.
    def store_transition(self, state, action, reward, next_state, done):
        if self.return_mode == "episode":
            # Finite-episode behavior returns, an explicit experimental target.
            # The replay terminal flag here means no bootstrap, not env.done.
            self.pending_episode.append(
                (state.copy(), copy.deepcopy(action), reward, next_state.copy())
            )
            if done:
                total = 0.0
                completed = []
                for s, a, r, ns in reversed(self.pending_episode):
                    total = r + self.gamma * total
                    completed.append((s, a, total, ns, True))
                for transition in reversed(completed):
                    self.memory.store_transition(*transition)
                self.pending_episode.clear()
            return
        self.memory.store_transition(state, action, reward, next_state, done)

    def replay_action(self, env, requested):
        """Implicit regression fits executed powers, avoiding infeasible request aliases."""
        if self.actor_update != "implicit":
            return requested
        return int(env.action[0]), np.asarray(
            [env.power_AC, env.power_EV, env.power_BESS], dtype=np.float32
        )

    # Perform one learning update: update critic then actor, then soft-update targets.
    def learn(self):
        if len(self.memory) < self.batch_size:
            return None, None

        batch, expert_mask = self.sample_batch()
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        features = self.actor_net.prepare_features(state_batch)
        discrete_action_batch = [a[0] for a in action_batch]
        continuous_action_batch = [a[1] for a in action_batch]
        discrete_action_batch = torch.tensor(discrete_action_batch).unsqueeze(1).to(self.device)
        continuous_action_batch = np.array(continuous_action_batch)
        continuous_action_batch = torch.from_numpy(continuous_action_batch).float().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_features = (
            self.actor_target_net.prepare_features(next_state_batch)
            if self.return_mode == "td"
            else None
        )
        done_batch = torch.from_numpy(done_batch).float().to(self.device)

        # Update critic network using target networks to compute the TD target.
        with torch.no_grad():
            target = self.bellman_target(next_features, reward_batch, done_batch)

        if getattr(self.critic_net, "twin", False):
            q_values, second = self.critic_net.both_features(features, continuous_action_batch)
        else:
            q_values = self.critic_net.forward_features(features, continuous_action_batch)
        q_values = q_values.gather(1, index=discrete_action_batch)
        loss_critic_td = self.td_loss(q_values, target.unsqueeze(1))
        if getattr(self.critic_net, "twin", False):
            loss_critic_td = loss_critic_td + self.td_loss(
                second.gather(1, discrete_action_batch), target.unsqueeze(1)
            )
        if self.actor_update == "implicit":
            from gridpfn.core.agents.implicit import expectile_loss

            with torch.no_grad():
                observed_q = (
                    self.critic_target_net.minimum_features(features, continuous_action_batch)
                    .gather(1, discrete_action_batch)
                    .squeeze(1)
                )
            loss_critic_td = (
                loss_critic_td
                + expectile_loss(
                    self.critic_net.value_features(features), observed_q, self.expectile
                ).mean()
            )
        self.critic_net.train()
        self.critic_optimizer.zero_grad()
        loss_critic_td.backward()
        self.critic_optimizer.step()

        self.learning_steps += 1
        if self.learning_steps % self.policy_delay:
            # Report missing actor loss honestly; callers aggregate each loss
            # independently so critic-only steps remain in critic metrics.
            return None, float(loss_critic_td.detach())

        # Update actor by maximizing critic Q (minimizing -Q).
        update_continuous_action_batch = self.actor_net.forward_features(features)
        if self.actor_update == "implicit":
            from gridpfn.core.agents.implicit import actor_regression

            with torch.no_grad():
                advantage = self.critic_target_net.minimum_features(
                    features, continuous_action_batch
                ).gather(1, discrete_action_batch).squeeze(1) - self.critic_net.value_features(
                    features
                )
            loss_actor = actor_regression(
                update_continuous_action_batch,
                continuous_action_batch,
                self.actor_net.discrete_features(features),
                discrete_action_batch.squeeze(1),
                advantage,
                self.actor_net.action_max - self.actor_net.action_min,
                self.advantage_temperature,
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss_actor.backward()
            self.actor_optimizer.step()
            soft_update_target_network(self.actor_net, self.actor_target_net, self.actor_tau)
            soft_update_target_network(self.critic_net, self.critic_target_net, self.critic_tau)
            return float(loss_actor.detach()), float(loss_critic_td.detach())
        policy_q = None
        loss_actor = update_continuous_action_batch.sum() * 0
        if self.actor_q_weight:
            update_q_values = self.critic_net.forward_features(
                features, update_continuous_action_batch
            )
            policy_q = update_q_values.max(1)[0]
            loss_actor = -self.actor_q_weight * policy_q.mean()
        if self.bc_weight:
            if expert_mask is not None:
                teacher_actions = continuous_action_batch.detach()
            elif self.feedback_teacher is None:
                raise RuntimeError("Configure feedback guidance before learning")
            else:
                teacher_actions = self.guidance_actions(features, state_batch)
            action_range = self.actor_net.action_max - self.actor_net.action_min
            errors = (
                ((update_continuous_action_batch - teacher_actions) / action_range).square()
                * torch.as_tensor(self.guidance_weights, device=self.device)
            ).mean(1)
            imitation = (
                errors.mean()
                if expert_mask is None
                else errors[torch.as_tensor(expert_mask, device=self.device)].mean()
            )
            # Stop-gradient Q scaling follows the principle in TD3+BC. Here
            # labels are current-state feedback queries, not offline behavior.
            if policy_q is not None:
                loss_actor = loss_actor / policy_q.detach().abs().mean().clamp_min(1)
            loss_actor = loss_actor + self.bc_weight * imitation

        # Compute actor gradients and perform optimizer step.
        self.actor_net.train()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss_actor.backward(inputs=tuple(p for p in self.actor_net.parameters() if p.requires_grad))
        self.actor_optimizer.step()

        # Soft-update target networks with Polyak averaging.
        soft_update_target_network(self.actor_net, self.actor_target_net, self.actor_tau)
        soft_update_target_network(self.critic_net, self.critic_target_net, self.critic_tau)

        return tuple(torch.stack((loss_actor.detach(), loss_critic_td.detach())).cpu().tolist())

    def sample_batch(self):
        """Mix private expert and online replay; expert labels never come from evaluation."""
        if self.expert_memory is None:
            return self.memory.sample(self.batch_size), None
        count = max(1, min(self.batch_size - 1, round(self.batch_size * self.expert_fraction)))
        expert = self.expert_memory.sample(count)
        online = self.memory.sample(self.batch_size - count)
        batch = tuple(
            [*a, *b] if index == 1 else np.concatenate((a, b))
            for index, (a, b) in enumerate(zip(expert, online, strict=True))
        )
        return batch, np.arange(self.batch_size) < count

    def td_loss(self, predicted, target, reduction="mean"):
        if self.critic_huber_delta:
            return nn.functional.huber_loss(
                predicted, target, delta=self.critic_huber_delta, reduction=reduction
            )
        return nn.functional.mse_loss(predicted, target, reduction=reduction)

    def bellman_target(self, next_features, rewards, done):
        if self.return_mode == "episode":
            return rewards
        if self.actor_update == "implicit":
            return rewards + self.gamma * self.critic_net.value_features(next_features) * (1 - done)
        control = self.actor_target_net.forward_features(next_features)
        if self.target_noise:
            span = self.actor_net.action_max - self.actor_net.action_min
            noise = (torch.randn_like(control) * self.target_noise).clamp(
                -2.5 * self.target_noise, 2.5 * self.target_noise
            )
            control = self.actor_target_net.project_action(next_features, control + noise * span)
        forward = (
            self.critic_target_net.minimum_features
            if getattr(self.critic_target_net, "twin", False)
            else self.critic_target_net.forward_features
        )
        return rewards + self.gamma * forward(next_features, control).max(1).values * (1 - done)

    @torch.no_grad()
    def guidance_actions(self, features, states):
        actions = torch.as_tensor(self.feedback_teacher(states), device=self.device)
        return (
            self.actor_net.project_action(features, actions)
            if hasattr(self.actor_net, "control_dt")
            else actions
        )

    def set_model_params(self, actor_model, critic_model):
        self.actor_net.load_state_dict(actor_model)
        self.critic_net.load_state_dict(critic_model)

    def get_model_params(self):
        actor = {
            name: value.detach().cpu().clone()
            for name, value in self.actor_net.state_dict().items()
        }
        critic = {
            name: value.detach().cpu().clone()
            for name, value in self.critic_net.state_dict().items()
        }
        return actor, critic
