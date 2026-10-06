"""Vectorized independent per-home updates; FedAvg still happens at its interval."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state


class _FeatureCall(nn.Module):
    def __init__(self, head, kind):
        super().__init__()
        self.head = head
        self.kind = kind

    def forward(self, features, control=None):
        if self.kind == "critic":
            return self.head.both_features(features, control)
        if self.kind == "project":
            return self.head.project_action(features, control)
        if self.kind == "value":
            return self.head.value_features(features)
        if self.kind == "discrete":
            return self.head.discrete_features(features)
        if self.kind == "distribution":
            return self.head.distribution_features(features)
        if self.kind == "latent":
            return self.head.latent_features(features, control)
        return self.head.forward_features(features)


class _Stack:
    def __init__(self, heads, kind):
        wrappers = [_FeatureCall(head, kind) for head in heads]
        self.params, self.buffers = stack_module_state(wrappers)
        self.template = copy.deepcopy(wrappers[0]).to("meta")
        self.shared_mapping = None
        self.compiled_latent = None
        if kind == "actor":
            if not hasattr(heads[0], "residual_scale") and all(
                all(
                    torch.equal(value, dict(head.named_buffers())[name])
                    for name, value in heads[0].named_buffers()
                )
                for head in heads[1:]
            ):
                # Prepared features already use each home's feature coordinates.
                # Identical physical maps can act on flattened home/state rows.
                self.shared_mapping = heads[0]
            self.project_template = copy.deepcopy(self.template)
            self.project_template.kind = "project"
            if hasattr(heads[0], "discrete_fc4"):
                self.discrete_template = copy.deepcopy(self.template)
                self.discrete_template.kind = "discrete"
            if hasattr(heads[0], "log_std"):
                for kind in ("distribution", "latent"):
                    template = copy.deepcopy(self.template)
                    template.kind = kind
                    setattr(self, f"{kind}_template", template)
        elif hasattr(heads[0], "value_fc1"):
            self.value_template = copy.deepcopy(self.template)
            self.value_template.kind = "value"
        # Existing action/evaluation/checkpoint/FedAvg APIs access views of the
        # canonical stacked tensors. They must never run their own optimizers.
        for index, wrapper in enumerate(wrappers):
            for name, parameter in wrapper.named_parameters():
                parameter.data = self.params[name][index].detach()
            for name, buffer in wrapper.named_buffers():
                buffer.data = self.buffers[name][index].detach()

    def __call__(self, features, control=None, project=False, kind=None):
        if self.shared_mapping is not None and (project or kind == "latent"):
            method = (
                self.shared_mapping.project_action
                if project
                else self.shared_mapping.latent_features
            )
            if not project and self.compiled_latent is not None and not torch.is_grad_enabled():
                method = self.compiled_latent
            return method(features.flatten(0, 1), control.flatten(0, 1)).reshape(
                *features.shape[:2], -1
            )
        if kind is None and control is None and hasattr(self, "distribution_template"):
            mean, _, _ = self(features, kind="distribution")
            return self(features, mean, kind="latent")
        # These two small MLPs dominate cached PPO rollouts. Explicit batched
        # linear layers keep the same independent tensors and gradients while
        # avoiding repeated functional-call/module dispatch at every hour.
        if control is None and not project and kind in ("distribution", "value"):

            def linear(x, name):
                return (
                    torch.bmm(x, self.params[f"head.{name}.weight"].transpose(1, 2))
                    + self.params[f"head.{name}.bias"][:, None]
                )

            if kind == "distribution":
                head = self.template.head
                activate = torch.tanh if head.activation == "tanh" else torch.relu
                hidden = activate(linear(features, "fc1"))
                for index in range(len(head.hidden)):
                    hidden = activate(linear(hidden, f"hidden.{index}"))
                mean = linear(hidden, "fc4")
                std = self.params["head.log_std"].clamp(-4, 0).exp()[:, None].expand_as(mean)
                if "head.latent_scale" in self.buffers:
                    mean = mean * self.buffers["head.latent_scale"][:, None]
                    std = std * self.buffers["head.latent_scale"][:, None]
                return mean, std, linear(hidden, "discrete_fc4")
            value = linear(linear(features, "value_fc1").relu(), "value_fc4").squeeze(-1)
            if "head.value_scale" in self.buffers:
                value = value * self.buffers["head.value_scale"] + self.buffers["head.value_mean"]
            return value
        return self.functional(features, control, project, kind)

    def compile_mapping(self):
        if self.shared_mapping is None:
            raise ValueError("Compiled rollouts require identical physical mappings across homes")
        # Only inference is compiled. PPO/BC gradients retain the eager contract.
        self.compiled_latent = torch.compile(
            self.shared_mapping.latent_features, dynamic=True, fullgraph=True
        )

    def functional(self, features, control=None, project=False, kind=None):
        template = self.template
        if kind is not None:
            template = getattr(self, f"{kind}_template")
        if project:
            template = self.project_template
        if control is None:
            return torch.vmap(lambda p, b, x: functional_call(template, (p, b), (x,)))(
                self.params, self.buffers, features
            )
        return torch.vmap(lambda p, b, x, a: functional_call(template, (p, b), (x, a)))(
            self.params, self.buffers, features, control
        )

    def trainable(self):
        return [p for p in self.params.values() if p.requires_grad]

    @torch.no_grad()
    def update_from(self, source, tau):
        for name, parameter in source.params.items():
            if parameter.requires_grad:
                self.params[name].copy_(tau * parameter + (1 - tau) * self.params[name])


class BatchedLearner:
    """Same per-home losses/Adam states, summed across homes, not a pooled agent.

    Construct after replay-only warmup and BC. A single device, architecture,
    hyperparameter configuration, and synchronous update count are required.
    """

    def __init__(self, agents):
        self.agents = agents
        self.first = first = agents[0]
        keys = (
            "device",
            "batch_size",
            "gamma",
            "lr_actor",
            "lr_critic",
            "actor_tau",
            "critic_tau",
            "policy_delay",
            "bc_weight",
            "actor_q_weight",
            "target_noise",
            "return_mode",
            "critic_huber_delta",
            "learning_steps",
            "exploration_noise",
        )
        for agent in agents:
            for option in ("actor_update", "expectile", "advantage_temperature"):
                if getattr(agent, option, None) != getattr(first, option, None):
                    raise ValueError("Batched homes must share the actor update configuration")
            if getattr(agent, "guidance_weights", None) != getattr(first, "guidance_weights", None):
                raise ValueError("Batched homes must share guidance weights")
            if any(getattr(agent, key) != getattr(first, key) for key in keys):
                raise ValueError(
                    "Batched homes must share device, hyperparameters and update count"
                )
            if getattr(agent, "expert_fraction", None) != getattr(first, "expert_fraction", None):
                raise ValueError("Batched homes must share the expert sampling fraction")
            if (getattr(agent, "expert_memory", None) is None) != (
                getattr(first, "expert_memory", None) is None
            ):
                raise ValueError("Batched homes must share the expert replay mode")
            if agent.actor_net.feature_mode != first.actor_net.feature_mode:
                raise ValueError("Batched homes must share feature mode")
            if agent.actor_optimizer.state or agent.critic_optimizer.state:
                raise ValueError(
                    "Batched updates require replay-only warmup and fresh RL optimizers"
                )
            for key in (
                "feature_mean",
                "feature_scale",
                "action_min",
                "action_max",
                "thermal_bounds",
                "target_temperature_bounds",
                "quota_actor",
                "embedding_weight",
            ):
                if hasattr(first.actor_net, key) and not torch.equal(
                    getattr(first.actor_net, key), getattr(agent.actor_net, key)
                ):
                    raise ValueError("Batched homes must share feature coordinates and bounds")
        self.actor = _Stack([a.actor_net for a in agents], "actor")
        self.critic = _Stack([a.critic_net for a in agents], "critic")
        self.actor_target = _Stack([a.actor_target_net for a in agents], "actor")
        self.critic_target = _Stack([a.critic_target_net for a in agents], "critic")
        self.actor_optimizer = torch.optim.Adam(self.actor.trainable(), lr=first.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.trainable(), lr=first.lr_critic)

    def _features(self, states):
        homes, batch, width = states.shape
        return self.first.actor_net.prepare_features(states.reshape(-1, width)).reshape(
            homes, batch, -1
        )

    @torch.no_grad()
    def choose_actions(self, states):
        """One head call and host transfer for all homes; preserve RNG order."""
        if self.first.exploration_noise is None:
            return [a.choose_action(s) for a, s in zip(self.agents, states, strict=True)]
        features = self._features(np.asarray(states)[:, None, :])
        control = self.actor(features)
        noise = np.stack(
            [np.random.normal(size=(1, self.first.continuous_action_dim)) for _ in self.agents]
        )
        span = self.first.actor_net.action_max - self.first.actor_net.action_min
        control += (
            torch.as_tensor(noise, dtype=control.dtype, device=control.device)
            * self.first.exploration_noise
            * span
        )
        control = self.actor(features, control, project=True)
        q, _ = self.critic(features, control)
        if getattr(self.first, "actor_update", "q_gradient") == "implicit":
            q = self.actor(features, kind="discrete")
        values = torch.cat((q.argmax(2, keepdim=True), control), dim=2)[:, 0, :].cpu().numpy()
        actions = []
        for agent, row in zip(self.agents, values, strict=True):
            agent.frame_idx += 1
            choice = int(row[0])
            if random.random() < agent.epsilon(agent.frame_idx):
                choice = random.randrange(agent.discrete_action_dim)
            actions.append((choice, row[1:].copy()))
        return actions

    def learn(self):
        ready = [len(a.memory) >= a.batch_size for a in self.agents]
        if not all(ready):
            if any(ready):
                raise ValueError("Batched replay readiness must be synchronized")
            return [(None, None) for _ in self.agents]
        first = self.first
        sampled = [
            a.sample_batch()
            if hasattr(a, "sample_batch")
            else (a.memory.sample(a.batch_size), None)
            for a in self.agents
        ]
        batches = [row[0] for row in sampled]
        states = np.stack([b[0] for b in batches])
        features = self._features(states)

        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=first.device)

        discrete = tensor([[a[0] for a in b[1]] for b in batches], torch.long).unsqueeze(-1)
        controls = tensor([[a[1] for a in b[1]] for b in batches])
        reward = tensor([b[2] for b in batches])
        done = tensor([b[4] for b in batches])
        with torch.no_grad():
            target = reward
            if first.return_mode == "td":
                next_features = self._features(np.stack([b[3] for b in batches]))
                if getattr(first, "actor_update", "q_gradient") == "implicit":
                    target = reward + first.gamma * self.critic(next_features, kind="value") * (
                        1 - done
                    )
                else:
                    action = self.actor_target(next_features)
                if first.target_noise:
                    span = first.actor_net.action_max - first.actor_net.action_min
                    noise = (torch.randn_like(action) * first.target_noise).clamp(
                        -2.5 * first.target_noise, 2.5 * first.target_noise
                    )
                    action = action + noise * span
                    action = self.actor_target(next_features, action, project=True)
                if getattr(first, "actor_update", "q_gradient") != "implicit":
                    q1, q2 = self.critic_target(next_features, action)
                    target = reward + first.gamma * torch.minimum(q1, q2).max(2).values * (1 - done)
        q1, q2 = self.critic(features, controls)
        critic_loss = first.td_loss(
            q1.gather(2, discrete).squeeze(2), target, reduction="none"
        ).mean(1)
        if first.critic_net.twin:
            critic_loss = critic_loss + first.td_loss(
                q2.gather(2, discrete).squeeze(2), target, reduction="none"
            ).mean(1)
        self.critic_optimizer.zero_grad(set_to_none=True)
        implicit = getattr(first, "actor_update", "q_gradient") == "implicit"
        if implicit:
            from gridpfn.core.agents.implicit import expectile_loss

            with torch.no_grad():
                a, b = self.critic_target(features, controls)
                observed_q = torch.minimum(a, b).gather(2, discrete).squeeze(2)
            critic_loss = critic_loss + expectile_loss(
                self.critic(features, kind="value"), observed_q, first.expectile
            ).mean(1)
        critic_loss.sum().backward()
        self.critic_optimizer.step()
        for agent in self.agents:
            agent.learning_steps += 1
        if first.learning_steps % first.policy_delay:
            return [(None, float(x)) for x in critic_loss.detach().cpu().tolist()]
        control = self.actor(features)
        if implicit:
            from gridpfn.core.agents.implicit import actor_regression

            with torch.no_grad():
                advantage = observed_q - self.critic(features, kind="value")
            actor_loss = actor_regression(
                control,
                controls,
                self.actor(features, kind="discrete"),
                discrete.squeeze(2),
                advantage,
                first.actor_net.action_max - first.actor_net.action_min,
                first.advantage_temperature,
            ).mean(1)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.sum().backward()
            self.actor_optimizer.step()
            self.actor_target.update_from(self.actor, first.actor_tau)
            self.critic_target.update_from(self.critic, first.critic_tau)
            return list(
                zip(
                    actor_loss.detach().cpu().tolist(),
                    critic_loss.detach().cpu().tolist(),
                    strict=True,
                )
            )
        actor_loss = control.sum((1, 2)) * 0
        if first.actor_q_weight:
            q1, _ = self.critic(features, control)
            policy_q = q1.max(2).values
            actor_loss = -first.actor_q_weight * policy_q.mean(1)
            if first.bc_weight:
                actor_loss = actor_loss / policy_q.detach().abs().mean(1).clamp_min(1)
        if first.bc_weight:
            expert_mask = sampled[0][1]
            if expert_mask is not None:
                teacher = controls.detach()
            else:
                with torch.no_grad():
                    teacher = tensor(
                        [a.feedback_teacher(s) for a, s in zip(self.agents, states, strict=True)]
                    )
            span = first.actor_net.action_max - first.actor_net.action_min
            if hasattr(first.actor_net, "control_dt"):
                teacher = self.actor(features, teacher, project=True)
            errors = (
                ((control - teacher) / span).square()
                * tensor(getattr(first, "guidance_weights", (1, 1, 1)))
            ).mean(2)
            imitation = (
                errors.mean(1)
                if expert_mask is None
                else errors[:, torch.as_tensor(expert_mask, device=first.device)].mean(1)
            )
            actor_loss = actor_loss + first.bc_weight * imitation
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.sum().backward(inputs=tuple(self.actor.trainable()))
        self.actor_optimizer.step()
        self.actor_target.update_from(self.actor, first.actor_tau)
        self.critic_target.update_from(self.critic, first.critic_tau)
        return [
            tuple(row)
            for row in torch.stack((actor_loss.detach(), critic_loss.detach()), dim=1)
            .cpu()
            .tolist()
        ]
