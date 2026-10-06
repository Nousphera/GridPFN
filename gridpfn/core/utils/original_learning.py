"""Execution adapter for independent, unchanged original P-DQN updates."""

import random
from types import MethodType

import numpy as np
import torch

from gridpfn.core.batched_learning import BatchedLearner


def prepare_features(self, states):
    return torch.as_tensor(states, dtype=torch.float32, device=self.fc1.weight.device)


def both_features(self, state, control):
    value = self(state, control)
    return value, value


def td_loss(self, prediction, target, reduction="mean"):
    return torch.nn.functional.mse_loss(prediction, target, reduction=reduction)


@torch.no_grad()
def choose_actions(self, states):
    features = self._features(np.asarray(states)[:, None, :])
    controls = self.actor(features)
    q, _ = self.critic(features, controls)
    values = torch.cat((q.argmax(2, keepdim=True), controls), dim=2)[:, 0].cpu().numpy()
    actions = []
    for agent, row in zip(self.agents, values, strict=True):
        agent.frame_idx += 1
        if random.random() > agent.epsilon(agent.frame_idx):
            actions.append((int(row[0]), row[1:]))
        else:
            choice = random.randrange(agent.discrete_action_dim)
            control = np.random.uniform(
                agent.continuous_action_min,
                agent.continuous_action_max,
                size=agent.continuous_action_dim,
            )
            actions.append((choice, control))
    return actions


def batch_original_agents(agents):
    """Keep original networks/losses and carry warmup Adam moments into stacked updates."""
    for agent in agents:
        agent.td_loss = MethodType(td_loss, agent)
        actor_type, critic_type = type(agent.actor_net), type(agent.critic_net)
        actor_type.prepare_features = prepare_features
        actor_type.forward_features = actor_type.forward
        critic_type.forward_features = critic_type.forward
        critic_type.both_features = both_features
        agent.policy_delay, agent.bc_weight, agent.actor_q_weight = 1, 0, 1
        agent.target_noise, agent.critic_huber_delta = 0, 0
        agent.return_mode, agent.exploration_noise = "td", None
        agent.learning_steps = int(next(iter(agent.actor_optimizer.state.values()))["step"])
        for net in (
            agent.actor_net,
            agent.critic_net,
            agent.actor_target_net,
            agent.critic_target_net,
        ):
            net.feature_mode = "raw"
            net.twin = False
    saved = []
    for agent in agents:
        saved.append((agent.actor_optimizer.state, agent.critic_optimizer.state))
        # Construction rejects existing optimizers; restore them after transferring moments.
        agent.actor_optimizer.state = {}
        agent.critic_optimizer.state = {}
    try:
        batch = BatchedLearner(agents)
        batch.choose_actions = MethodType(choose_actions, batch)
        for index, (stack, optimizer, network) in enumerate(
            (
                (batch.actor, batch.actor_optimizer, "actor_net"),
                (batch.critic, batch.critic_optimizer, "critic_net"),
            )
        ):
            for name, parameter in stack.params.items():
                original_name = name.removeprefix("head.")
                states = [
                    saved[i][index][dict(getattr(agent, network).named_parameters())[original_name]]
                    for i, agent in enumerate(agents)
                ]
                if any(not torch.equal(state["step"], states[0]["step"]) for state in states[1:]):
                    raise ValueError("Original warmup optimizer counts differ across homes")
                optimizer.state[parameter] = {
                    "step": states[0]["step"].clone(),
                    "exp_avg": torch.stack(
                        [state["exp_avg"].to(parameter.device) for state in states]
                    ),
                    "exp_avg_sq": torch.stack(
                        [state["exp_avg_sq"].to(parameter.device) for state in states]
                    ),
                }
        return batch
    finally:
        for agent, (actor, critic) in zip(agents, saved, strict=True):
            agent.actor_optimizer.state, agent.critic_optimizer.state = actor, critic
