"""Federated hybrid PPO: private finite-day rollouts and independent local values.

Likelihoods refer to sampled Gaussian latents and categorical WM requests, before
the deterministic actuator/constraint map. Recording executed powers as Gaussian
samples would give incorrect ratios when a power saturates or a task is forced.
"""

import copy
import math
import time

import numpy as np
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.batched_learning import _Stack


def log_probability(mean, std, logits, latent, choice, mask=None, ac_bounds=None):
    continuous = -0.5 * ((latent - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
    if ac_bounds is not None:
        # Integrate AC tails that execute an identical power. Interior Jacobians
        # cancel in PPO ratios; other latent factors retain their original score.
        low, high = ac_bounds
        ac_mean, ac_std, sampled = mean[..., 0], std[..., 0], latent[..., 0]
        safe_low = torch.where(torch.isfinite(low), low, ac_mean.detach())
        safe_high = torch.where(torch.isfinite(high), high, ac_mean.detach())
        ac = torch.where(
            sampled <= low,
            torch.special.log_ndtr((safe_low - ac_mean) / ac_std),
            torch.where(
                sampled >= high,
                torch.special.log_ndtr((ac_mean - safe_high) / ac_std),
                continuous[..., 0],
            ),
        )
        active = (low < high) & (low < float("inf")) & (high > -float("inf"))
        continuous = torch.cat(
            (torch.where(active, ac, torch.zeros_like(ac))[..., None], continuous[..., 1:]), -1
        )
    discrete = logits.log_softmax(-1).gather(-1, choice[..., None]).squeeze(-1)
    parts = torch.cat((continuous, discrete[..., None]), -1)
    return (parts if mask is None else parts * mask).sum(-1)


def policy_kl(before, after, mask=None):
    """Exact latent Normal/Categorical KL, per observed state."""
    mean, std, logits = before
    next_mean, next_std, next_logits = after
    normal = (
        (next_std / std).log()
        + (std.square() + (mean - next_mean).square()) / (2 * next_std.square())
        - 0.5
    )
    discrete = (logits.softmax(-1) * (logits.log_softmax(-1) - next_logits.log_softmax(-1))).sum(-1)
    parts = torch.cat((normal, discrete[..., None]), -1)
    return (parts if mask is None else parts * mask).sum(-1)


def advantages(rewards, values, done, gamma, gae_lambda):
    """Time-major independent homes; terminal flags prevent crossing daily resets."""
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    following = torch.zeros_like(values[0])
    for t in range(len(rewards) - 1, -1, -1):
        continuation = 1 - done[t]
        delta = rewards[t] + gamma * following * continuation - values[t]
        running = delta + gamma * gae_lambda * continuation * running
        result[t] = running
        following = values[t]
    return result, result + values


class OnPolicyAgent(P_DQN):
    def __init__(self, *args, hyperparams=None, **kwargs):
        hp = dict(hyperparams or {})
        if hp.get("oracle_demonstrations") or hp.get("return_mode", "td") != "td":
            raise ValueError(
                "PPO uses fresh on-policy rollouts, without expert replay or MC replay"
            )
        if hp.get("target_noise", 0) or hp.get("policy_delay", 1) != 1:
            raise ValueError("PPO does not use target smoothing or delayed Q-gradient updates")
        hp["actor_update"] = "q_gradient"
        super().__init__(*args, hyperparams=hp, **kwargs)
        if not hasattr(self.actor_net, "log_std") or not hasattr(self.critic_net, "value_fc1"):
            raise ValueError("PPO requires stochastic/categorical actor and state-value heads")
        self.actor_update = "ppo"
        self.ppo_epochs = int(hp.get("ppo_epochs", 4))
        value_epochs = hp.get("ppo_value_epochs")
        self.ppo_value_epochs = int(self.ppo_epochs if value_epochs is None else value_epochs)
        self.ppo_clip = float(hp.get("ppo_clip", 0.2))
        self.ppo_gae_lambda = float(hp.get("ppo_gae_lambda", 0.95))
        self.ppo_entropy = float(hp.get("ppo_entropy", 0.001))
        self.ppo_target_kl = float(hp.get("ppo_target_kl", 0))
        self.ppo_mask_inactive = bool(hp.get("ppo_mask_inactive", False))
        self.ppo_reset_momentum = bool(hp.get("ppo_reset_momentum", False))
        self.ppo_compile_mapping = bool(hp.get("ppo_compile_mapping", False))
        self.ppo_shuffle_days = bool(hp.get("ppo_shuffle_days", False))
        self.ppo_lr_decay_days = int(hp.get("ppo_lr_decay_days", 0))
        self.ppo_bc_decay_days = int(hp.get("ppo_bc_decay_days", 0))
        self.ppo_federation = hp.get("ppo_federation", "actor")
        self.ppo_team_reward = bool(hp.get("ppo_team_reward", False))
        self.ppo_advantage_scale = hp.get("ppo_advantage_scale", "home")
        self.ppo_shared_gradient_clip = bool(hp.get("ppo_shared_gradient_clip", False))
        self.ppo_oracle_init = hp.get("ppo_oracle_init")
        self.ppo_clip_ac_likelihood = bool(hp.get("ppo_clip_ac_likelihood", False))
        self.ppo_rollout_days = int(hp.get("ppo_rollout_days", 0))
        if self.ppo_clip_ac_likelihood and not hasattr(self.actor_net, "quota_actor"):
            raise ValueError("Clipped AC likelihood requires a quota-aware thermal actor")
        self.training_seed = int(hp.get("training_seed", 0))
        self.synthetic_fraction = float(hp.get("synthetic_fraction", 0.5))
        self.synthetic_until = int(hp.get("synthetic_until", 0))
        if (
            self.ppo_epochs < 1
            or self.ppo_value_epochs < 1
            or not 0 < self.ppo_clip < 1
            or not 0 <= self.ppo_gae_lambda <= 1
            or not math.isfinite(self.ppo_entropy)
            or self.ppo_entropy < 0
            or not math.isfinite(self.ppo_target_kl)
            or self.ppo_target_kl < 0
            or not 0 <= self.synthetic_fraction <= 1
            or self.synthetic_until < 0
            or self.ppo_lr_decay_days < 0
            or self.ppo_bc_decay_days < 0
            or self.ppo_federation not in ("actor", "trunk", "gradient", "none")
            or self.ppo_rollout_days < 0
            or self.ppo_advantage_scale not in ("home", "cohort")
            or (self.ppo_shared_gradient_clip and self.ppo_federation != "gradient")
        ):
            raise ValueError("Invalid PPO epochs, clip, GAE lambda or entropy coefficient")
        self.rollout = []
        self.pending_policy = None

    def store_transition(self, state, action, reward, next_state, done):
        if self.pending_policy is None:
            raise RuntimeError("PPO transitions require a freshly sampled policy action")
        self.rollout.append((*self.pending_policy, float(reward), bool(done)))
        self.pending_policy = None

    def learn(self):
        raise RuntimeError("PPO requires the synchronized on-policy learner")


class OnPolicyLearner:
    """Collect one communication interval, optimize locally, discard all samples.

    The shared actor is averaged by Server only after these local updates. No
    rollout survives a policy broadcast. Full per-home rollout batches are small
    (24 * interval) and vectorized without pooling homes' examples or gradients.
    """

    def __init__(self, agents, interval):
        self.agents, self.first = agents, agents[0]
        self.interval, self.completed_days = interval, 0
        if interval < 1 or any(a.actor_update != "ppo" for a in agents):
            raise ValueError("PPO requires a positive synchronized communication interval")
        for agent in agents:
            for key in (
                "gamma",
                "lr_actor",
                "lr_critic",
                "bc_weight",
                "ppo_epochs",
                "ppo_value_epochs",
                "ppo_clip",
                "ppo_gae_lambda",
                "ppo_entropy",
                "ppo_target_kl",
                "ppo_mask_inactive",
                "ppo_reset_momentum",
                "ppo_compile_mapping",
                "ppo_shuffle_days",
                "ppo_lr_decay_days",
                "ppo_bc_decay_days",
                "ppo_federation",
                "ppo_team_reward",
                "ppo_advantage_scale",
                "ppo_shared_gradient_clip",
                "ppo_clip_ac_likelihood",
                "ppo_rollout_days",
                "training_seed",
                "synthetic_fraction",
                "synthetic_until",
                "guidance_weights",
            ):
                if getattr(agent, key) != getattr(self.first, key):
                    raise ValueError("PPO homes must share optimizer and objective settings")
            if agent.rollout or agent.pending_policy is not None:
                raise ValueError("PPO must start with empty on-policy rollouts")
        self.actor = _Stack([a.actor_net for a in agents], "actor")
        if self.first.ppo_compile_mapping:
            self.actor.compile_mapping()
        self.critic = _Stack([a.critic_net for a in agents], "critic")
        self.actor_optimizer = torch.optim.Adam(self.actor.trainable(), lr=self.first.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.trainable(), lr=self.first.lr_critic)
        self.last_diagnostics, self.reference_batch = None, None
        self.feature_seconds = 0.0
        self.day_rng = np.random.default_rng(self.first.training_seed)
        self.synthetic_rng = np.random.default_rng(self.first.training_seed + 7919)
        self.last_synthetic = False
        self.calendar, self.calendar_indices, self.day_cycle, self.day_order = (
            None,
            None,
            None,
            None,
        )

    def training_indices(self, clients, episode):
        """Shuffle complete shared dates; each daily market remains contemporaneous."""
        self.last_synthetic = False
        counts = [
            len(c.train_data) - c.real_train_count if hasattr(c, "real_train_count") else 0
            for c in clients
        ]
        if any(counts):
            if min(counts) < 1 or len(set(counts)) != 1:
                raise ValueError("Synthetic daily markets require matching cohorts")
            if (
                not self.first.synthetic_until or episode < self.first.synthetic_until
            ) and self.synthetic_rng.random() < self.first.synthetic_fraction:
                index = int(self.synthetic_rng.integers(counts[0]))
                self.last_synthetic = True
                return [c.real_train_count + index for c in clients]
        if not self.first.ppo_shuffle_days:
            return (
                [episode % getattr(c, "real_train_count", len(c.train_data)) for c in clients]
                if any(counts)
                else None
            )
        if self.calendar is None:
            calendars = [c.scaler["train_dates"] for c in clients]
            self.calendar = sorted(set.intersection(*(set(d) for d in calendars)))
            if not self.calendar:
                raise ValueError("PPO sampling needs shared training dates")
            self.calendar_indices = [{date: i for i, date in enumerate(d)} for d in calendars]
        cycle, offset = divmod(episode, len(self.calendar))
        if cycle != self.day_cycle:
            self.day_order = self.day_rng.permutation(len(self.calendar))
            self.day_cycle = cycle
        date = self.calendar[self.day_order[offset]]
        return [indices[date] for indices in self.calendar_indices]

    def _features(self, states):
        started = time.perf_counter()
        states = np.asarray(states, dtype=np.float32)
        homes, batch, width = states.shape
        features = self.first.actor_net.prepare_features(states.reshape(-1, width)).reshape(
            homes, batch, -1
        )
        self.feature_seconds += time.perf_counter() - started
        return features

    def _mask(self, features):
        if not self.first.ppo_mask_inactive:
            return None
        # Integrate out factors whose actions are entirely ignored. WM before
        # 10:00 stays active: starting early changes the original comfort reward.
        state = features[..., -17:]
        low, high = self.first.actor_net._feasible_bounds(features.reshape(-1, features.shape[-1]))
        active = (high - low > 1e-6).reshape(*state.shape[:-1], 3)
        wm = (state[..., 12] > 0) & (state[..., 0] * 24 < 17)
        return torch.cat((active, wm[..., None]), -1).to(features.dtype)

    def _ac_bounds(self, features):
        """Exact latent interval before target, power and original AC-quota clipping.

        All bounds depend on the observation and fixed controller, not policy
        weights. Collapsed intervals have zero score, as their power is constant.
        """
        if not self.first.ppo_clip_ac_likelihood:
            return None
        actor = self.first.actor_net
        flat = features.flatten(0, 1)
        state = flat[:, -17:]
        natural = actor._uncontrolled_temperature(flat)
        quota = (state[:, 15] * 60).clamp_min(0)
        power = (quota / (24 - state[:, 0] * 24).clamp_min(1)).clamp(0, 2.5)
        base = torch.minimum(torch.full_like(power, 20), natural - 3 * power)
        forced = actor._feasible_bounds(flat)[0][:, 0]
        low = torch.maximum(
            ((natural - actor.target_temperature_bounds[1]) / 3).clamp(0, 2.5), forced
        )
        high = torch.maximum(
            ((natural - actor.target_temperature_bounds[0]) / 3).clamp(0, 2.5), forced
        )
        radius = getattr(actor, "quota_correction", 2)

        def inverse(p):
            x = (natural - base - 3 * p) / radius
            value = torch.atanh(x.clamp(-1 + 1e-7, 1 - 1e-7))
            return torch.where(
                x <= -1, -float("inf"), torch.where(x >= 1, float("inf"), value)
            ).reshape(features.shape[:2])

        return inverse(high), inverse(low)

    @torch.no_grad()
    def choose_actions(self, states):
        if any(a.pending_policy is not None for a in self.agents):
            raise RuntimeError("Previous PPO action has not been settled")
        features = self._features(np.asarray(states)[:, None, :])
        mean, std, logits = self.actor(features, kind="distribution")
        latent = mean + torch.randn_like(mean) * std
        choice = torch.multinomial(logits[:, 0].softmax(-1), 1)[:, 0, None]
        logp = log_probability(
            mean, std, logits, latent, choice, self._mask(features), self._ac_bounds(features)
        )
        value = self.critic(features, kind="value")
        control = self.actor(features, latent, kind="latent")
        # These outputs are fresh immutable tensors. Retain row views until the
        # synchronized rollout is cleared; avoid fifty tiny copies every hour.
        columns = [column[:, 0].unbind(0) for column in (features, latent, choice, logp, value)]
        for agent, state, row in zip(self.agents, states, zip(*columns, strict=True), strict=True):
            agent.pending_policy = (
                *row,
                np.asarray(state, dtype=np.float32).copy(),
            )
            agent.frame_idx += 1
        rows = torch.cat((choice[..., None], control), -1)[:, 0].cpu().numpy()
        return [(int(row[0]), row[1:]) for row in rows]

    def learn(self):
        if not all(a.rollout and a.rollout[-1][-1] for a in self.agents):
            return [(None, None) for _ in self.agents]
        self.completed_days += 1
        if self.completed_days % self.interval:
            return [(None, None) for _ in self.agents]
        return self.update()

    def warmup_values(self, clients, p2p_config, days, steps):
        """Fit private values to fresh real-day returns before changing the actor.

        Restore exploration/calendar streams and discard all warmup rollouts.
        No held-out or oracle observations enter these value targets.
        """
        from gridpfn.core.em_strategy import P2P_TRADING

        if days < 1 or steps < 1 or any(a.rollout for a in self.agents):
            raise ValueError("Value warmup requires positive days/steps and empty rollouts")
        started = time.perf_counter()
        saved = copy.deepcopy(
            (
                self.day_rng.bit_generator.state,
                self.synthetic_rng.bit_generator.state,
                self.calendar,
                self.calendar_indices,
                self.day_cycle,
                self.day_order,
            )
        )
        interval, fraction = self.interval, self.first.synthetic_fraction
        frames = [a.frame_idx for a in self.agents]
        devices = sorted({a.device.index for a in self.agents if a.device.type == "cuda"})
        try:
            self.interval, self.first.synthetic_fraction = days + 1, 0
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.first.training_seed + 15401)
                for episode in range(days):
                    P2P_TRADING.run_episode(
                        clients, episode, use_fed=True, p2p_config=p2p_config, learner=self
                    )
                rows = [list(zip(*a.rollout, strict=True)) for a in self.agents]
                features = torch.stack([torch.stack(row[0]) for row in rows])
                rewards, done = [
                    torch.as_tensor(
                        np.asarray([row[col] for row in rows]),
                        dtype=torch.float32,
                        device=self.first.device,
                    )
                    for col in (6, 7)
                ]
                if self.first.ppo_team_reward:
                    rewards = rewards.mean(0, keepdim=True).expand_as(rewards)
                _, returns = advantages(
                    rewards.T, torch.zeros_like(rewards.T), done.T, self.first.gamma, 1
                )
                returns = returns.T
                scale = self._normalize_values(returns)
                for _ in range(steps):
                    idx = torch.randint(
                        features.shape[1], (min(128, features.shape[1]),), device=features.device
                    )
                    value = self.critic(features[:, idx], kind="value")
                    loss = 0.5 * ((value - returns[:, idx]) / scale).square().mean(1)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss.sum().backward()
                    self._clip_gradients(self.critic)
                    self.critic_optimizer.step()
                with torch.no_grad():
                    residual = returns - self.critic(features, kind="value")
                    return {
                        "days": days,
                        "steps": steps,
                        "seconds": time.perf_counter() - started,
                        "rmse": residual.square().mean(1).sqrt().tolist(),
                        "explained_variance": (
                            1
                            - residual.var(1, unbiased=False)
                            / returns.var(1, unbiased=False).clamp_min(1e-8)
                        ).tolist(),
                    }
        finally:
            self.interval, self.first.synthetic_fraction, self.completed_days = (
                interval,
                fraction,
                0,
            )
            (
                day_rng,
                synthetic_rng,
                self.calendar,
                self.calendar_indices,
                self.day_cycle,
                self.day_order,
            ) = saved
            self.day_rng.bit_generator.state = day_rng
            self.synthetic_rng.bit_generator.state = synthetic_rng
            self.last_synthetic = False
            for agent, frame in zip(self.agents, frames, strict=True):
                agent.rollout.clear()
                agent.pending_policy, agent.frame_idx = None, frame

    @staticmethod
    def _clip_gradients(stack, maximum=0.5):
        # Each home has an independent loss/optimizer: clip its own gradient.
        parameters = [p for p in stack.trainable() if p.grad is not None]
        norms = sum(p.grad.flatten(1).square().sum(1) for p in parameters).sqrt()
        scale = (maximum / norms.clamp_min(1e-8)).clamp_max(1)
        for parameter in parameters:
            parameter.grad.mul_(scale.reshape(-1, *([1] * (parameter.ndim - 1))))
        return norms.detach()

    @torch.no_grad()
    def _normalize_values(self, returns):
        """PopArt output rescaling preserves unnormalized V exactly."""
        buffers, params = self.critic.buffers, self.critic.params
        if "head.value_mean" not in buffers:
            return 1.0
        old_mean, old_scale = (
            buffers["head.value_mean"].clone(),
            buffers["head.value_scale"].clone(),
        )
        mean = 0.99 * old_mean + 0.01 * returns.mean(1, keepdim=True)
        second = 0.99 * buffers["head.value_second"] + 0.01 * returns.square().mean(1, keepdim=True)
        scale = (second - mean.square()).clamp_min(1e-4).sqrt()
        params["head.value_fc4.weight"].mul_((old_scale / scale)[..., None])
        bias = params["head.value_fc4.bias"]
        bias.copy_((old_scale * bias + old_mean - mean) / scale)
        buffers["head.value_mean"].copy_(mean)
        buffers["head.value_scale"].copy_(scale)
        buffers["head.value_second"].copy_(second)
        return scale

    @torch.no_grad()
    def after_broadcast(self):
        if self.reference_batch is None:
            return
        features, before, local = self.reference_batch
        global_policy = self.actor(features, kind="distribution")
        self.last_diagnostics["broadcast_kl"] = policy_kl(local, global_policy).mean(1).tolist()
        self.last_diagnostics["broadcast_active_kl"] = (
            policy_kl(local, global_policy, self._mask(features)).mean(1).tolist()
        )
        self.last_diagnostics["global_kl"] = policy_kl(before, global_policy).mean(1).tolist()
        if self.first.ppo_reset_momentum:
            self.actor_optimizer.state.clear()
        self.reference_batch = None

    def update(self):
        started = time.perf_counter()
        first = self.first
        if len({len(a.rollout) for a in self.agents}) != 1:
            raise ValueError("PPO daily trajectories must be aligned")
        data = [list(zip(*a.rollout, strict=True)) for a in self.agents]
        features, latent, choices, old_logp, old_values = [
            torch.stack([torch.stack(row[col]) for row in data]) for col in range(5)
        ]
        states = np.asarray([row[5] for row in data])
        rewards, done = [
            torch.as_tensor(
                np.asarray([row[col] for row in data]), dtype=torch.float32, device=first.device
            )
            for col in (6, 7)
        ]
        if first.ppo_team_reward:
            rewards = rewards.mean(0, keepdim=True).expand_as(rewards)
        lr_fraction = (
            max(0.1, 1 - 0.9 * self.completed_days / first.ppo_lr_decay_days)
            if first.ppo_lr_decay_days
            else 1.0
        )
        for group in self.actor_optimizer.param_groups:
            group["lr"] = first.lr_actor * lr_fraction
        bc_weight = first.bc_weight * (
            max(0, 1 - self.completed_days / first.ppo_bc_decay_days)
            if first.ppo_bc_decay_days
            else 1.0
        )
        advantage, returns = advantages(
            rewards.T, old_values.T, done.T, first.gamma, first.ppo_gae_lambda
        )
        advantage, returns = advantage.T, returns.T
        with torch.no_grad():
            before = tuple(t.detach().clone() for t in self.actor(features, kind="distribution"))
            _, mc = advantages(rewards.T, torch.zeros_like(old_values.T), done.T, first.gamma, 1)
            mc = mc.T
        scale = self._normalize_values(returns)
        mask = self._mask(features)
        ac_bounds = self._ac_bounds(features)
        centered = advantage - advantage.mean(1, keepdim=True)
        divisor = (
            advantage.std(1, keepdim=True, unbiased=False)
            if first.ppo_advantage_scale == "home"
            else centered.std(unbiased=False)
        )
        advantage = centered / divisor.clamp_min(1e-6)
        guidance = None
        if bc_weight:
            guidance = torch.as_tensor(
                np.stack(
                    [
                        a.feedback_teacher(s[:, :17])
                        for a, s in zip(self.agents, states, strict=True)
                    ]
                ),
                device=first.device,
            )
            guidance = self.actor(features, guidance, project=True).detach()
        epochs = 0
        for _ in range(first.ppo_epochs):
            mean, std, logits = self.actor(features, kind="distribution")
            if (
                first.ppo_target_kl
                and epochs
                and float(policy_kl(before, (mean, std, logits), mask).mean(1).max().detach())
                > 1.5 * first.ppo_target_kl
            ):
                break
            logp = log_probability(mean, std, logits, latent, choices, mask, ac_bounds)
            ratio = (logp - old_logp).exp()
            clipped = ratio.clamp(1 - first.ppo_clip, 1 + first.ppo_clip)
            actor_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean(1)
            normal_entropy = std.log() + 0.5 * math.log(2 * math.pi * math.e)
            categorical_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
            entropy_parts = torch.cat((normal_entropy, categorical_entropy[..., None]), -1)
            entropy = (entropy_parts if mask is None else entropy_parts * mask).sum(-1)
            actor_loss = actor_loss - first.ppo_entropy * entropy.mean(1)
            if guidance is not None:
                span = first.actor_net.action_max - first.actor_net.action_min
                weights = torch.as_tensor(first.guidance_weights, device=first.device)
                actor_loss = actor_loss + bc_weight * (
                    (((self.actor(features) - guidance) / span).square() * weights).mean((1, 2))
                )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.sum().backward()
            if not first.ppo_shared_gradient_clip:
                actor_norm = self._clip_gradients(self.actor)
            if first.ppo_federation == "gradient":
                # FedSGD shares clipped gradients, never private trajectories or
                # critics. Common initialization and Adam moments keep one actor.
                for parameter in self.actor.trainable():
                    if parameter.grad is not None:
                        parameter.grad.copy_(
                            parameter.grad.mean(0, keepdim=True).expand_as(parameter)
                        )
            if first.ppo_shared_gradient_clip:
                actor_norm = self._clip_gradients(self.actor)
            self.actor_optimizer.step()
            epochs += 1
        # A policy KL limit must not stop fitting the independent value heads.
        for _ in range(first.ppo_value_epochs):
            value = self.critic(features, kind="value")
            critic_loss = 0.5 * ((value - returns) / scale).square().mean(1)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.sum().backward()
            critic_norm = self._clip_gradients(self.critic)
            self.critic_optimizer.step()
        with torch.no_grad():
            after = tuple(t.detach() for t in self.actor(features, kind="distribution"))
            value = self.critic(features, kind="value")
            variance = mc.var(1, unbiased=False).clamp_min(1e-8)
            residual = mc - value
            self.last_diagnostics = {
                "local_kl": policy_kl(before, after).mean(1).tolist(),
                "local_active_kl": policy_kl(before, after, mask).mean(1).tolist(),
                "clip_fraction": ((ratio - 1).abs() > first.ppo_clip).float().mean(1).tolist(),
                "actor_lr": [first.lr_actor * lr_fraction] * len(self.agents),
                "guidance_weight": [bc_weight] * len(self.agents),
                "entropy": entropy.mean(1).tolist(),
                "value_rmse": residual.square().mean(1).sqrt().tolist(),
                "value_explained_variance": (
                    1 - residual.var(1, unbiased=False) / variance
                ).tolist(),
                "actor_gradient_norm": actor_norm.tolist(),
                "critic_gradient_norm": critic_norm.tolist(),
                "actor_gradient_clipped": (actor_norm > 0.5).float().tolist(),
                "critic_gradient_clipped": (critic_norm > 0.5).float().tolist(),
                "inactive_fraction": (1 - self._mask(features).mean((1, 2))).tolist()
                if mask is not None
                else [0.0] * len(self.agents),
                "epochs": [epochs] * len(self.agents),
                "value_epochs": [first.ppo_value_epochs] * len(self.agents),
                "update_seconds": [time.perf_counter() - started] * len(self.agents),
            }
            self.reference_batch = features.detach(), before, after
        for agent in self.agents:
            agent.rollout.clear()
            agent.learning_steps += epochs
        return list(
            zip(
                actor_loss.detach().cpu().tolist(), critic_loss.detach().cpu().tolist(), strict=True
            )
        )
