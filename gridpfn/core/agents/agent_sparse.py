import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gridpfn.core.environment import CONTINUOUS_ACTION_MAX, CONTINUOUS_ACTION_MIN
from gridpfn.core.model import Actor, Critic
from gridpfn.core.utils.agent_utils import (
    ReplayBuffer,
    hard_update_target_network,
    soft_update_target_network,
)


class P_DQN_sparse:
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
        self.actor_target_net = Actor(
            self.state_dim,
            self.continuous_action_dim,
            action_min=self.continuous_action_min,
            action_max=self.continuous_action_max,
        ).to(self.device)
        self.critic_target_net = Critic(
            self.state_dim,
            self.continuous_action_dim,
            self.discrete_action_dim,
        ).to(self.device)

        hard_update_target_network(self.actor_net, self.actor_target_net)
        hard_update_target_network(self.critic_net, self.critic_target_net)

        self.gamma = float(hp.get("gamma", 0.99))
        self.batch_size = int(hp.get("batch_size", 32))
        self.lr_actor = float(hp.get("lr_actor", 0.00001))
        self.lr_critic = float(hp.get("lr_critic", 0.0001))
        self.epsilon_start = float(hp.get("epsilon_start", 1.0))
        self.epsilon_end = float(hp.get("epsilon_end", 0.005))
        self.epsilon_decay = int(hp.get("epsilon_decay", 10000))
        self.critic_tau = float(hp.get("critic_tau", 0.01))
        self.actor_tau = float(hp.get("actor_tau", 0.001))

        self.memory = ReplayBuffer(self.memory_capacity)

        self.frame_idx = 0
        self.epsilon = lambda frame_idx: (
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * math.exp(-1.0 * frame_idx / self.epsilon_decay)
        )

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic)

    def init_masks(self, params, sparsities):
        masks = {}
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] = 1
        return masks

    def calculate_sparsities(self, params, tabu=(), distribution="ERK", sparse=0.5):
        sparsities = {}
        if distribution == "uniform":
            for name in params:
                if name not in tabu:
                    sparsities[name] = 1 - sparse
                else:
                    sparsities[name] = 0
        elif distribution == "ERK":
            is_epsilon_valid = False
            dense_layers = set()

            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if name in tabu:
                        dense_layers.add(name)
                    n_param = np.prod(params[name].shape)
                    n_zeros = n_param * sparse
                    n_ones = n_param * (1 - sparse)

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = np.sum(params[name].shape) / np.prod(
                            params[name].shape
                        )
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            for name in params:
                if name in dense_layers:
                    sparsities[name] = 0
                else:
                    sparsities[name] = 1 - epsilon * raw_probabilities[name]
        return sparsities

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                continuous_action = self.actor_net(state)
                q_values = self.critic_net(state, continuous_action)
                q_values = q_values.cpu().numpy()
                discrete_action = q_values.argmax().item()
            continuous_action = continuous_action.squeeze(0)
        else:
            discrete_action = random.randrange(self.discrete_action_dim)
            continuous_action = torch.tensor(
                np.random.uniform(
                    self.continuous_action_min,
                    self.continuous_action_max,
                    size=self.continuous_action_dim,
                )
            ).to(self.device)
        return discrete_action, continuous_action.cpu().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None, None

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size
        )
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        discrete_action_batch = [a[0] for a in action_batch]
        continuous_action_batch = [a[1] for a in action_batch]
        discrete_action_batch = torch.tensor(discrete_action_batch).unsqueeze(1).to(self.device)
        continuous_action_batch = np.array(continuous_action_batch)
        continuous_action_batch = torch.from_numpy(continuous_action_batch).float().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)

        # Update critic network.
        with torch.no_grad():
            next_continuous_action_batch = self.actor_target_net(next_state_batch)
            next_q_values = self.critic_target_net(next_state_batch, next_continuous_action_batch)
            next_q_values_max = next_q_values.max(1)[0].detach()
            target = reward_batch + self.gamma * next_q_values_max * (1 - done_batch)

        q_values = self.critic_net(state_batch, continuous_action_batch)
        q_values = q_values.gather(1, index=discrete_action_batch)
        loss_critic_td = nn.MSELoss()(q_values, target.unsqueeze(1))
        self.critic_net.train()
        self.critic_optimizer.zero_grad()
        loss_critic_td.backward()
        self.critic_optimizer.step()

        for name, param in self.critic_net.named_parameters():
            if name in self.critic_masks:
                param.data.mul_(self.critic_masks[name])

        # Update actor network.
        update_continuous_action_batch = self.actor_net(state_batch)
        update_q_values = self.critic_net(state_batch, update_continuous_action_batch)
        loss_actor = -update_q_values.max(1)[0].mean()

        self.actor_net.train()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        for name, param in self.actor_net.named_parameters():
            if name in self.actor_masks:
                param.data.mul_(self.actor_masks[name])

        soft_update_target_network(self.actor_net, self.actor_target_net, self.actor_tau)
        soft_update_target_network(self.critic_net, self.critic_target_net, self.critic_tau)

        for name, param in self.critic_target_net.named_parameters():
            if name in self.critic_masks:
                param.data.mul_(self.critic_masks[name])

        for name, param in self.actor_target_net.named_parameters():
            if name in self.actor_masks:
                param.data.mul_(self.actor_masks[name])

        return loss_actor.item(), loss_critic_td.item()

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

    def get_trainable_params(self):
        return dict(self.actor_net.named_parameters()), dict(self.critic_net.named_parameters())

    def set_model_params(self, actor_parameters, critic_parameters):
        self.actor_net.load_state_dict(actor_parameters)
        self.critic_net.load_state_dict(critic_parameters)

    def get_model_masks(self, device="cpu"):
        actor = {
            name: value.detach().to(device).clone() for name, value in self.actor_masks.items()
        }
        critic = {
            name: value.detach().to(device).clone() for name, value in self.critic_masks.items()
        }
        return actor, critic

    def set_model_masks(self, actor_masks, critic_masks):
        self.actor_masks = {
            name: value.detach().to(self.device).clone() for name, value in actor_masks.items()
        }
        self.critic_masks = {
            name: value.detach().to(self.device).clone() for name, value in critic_masks.items()
        }

    def screen_gradients(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size
        )
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        discrete_action_batch = [a[0] for a in action_batch]
        continuous_action_batch = [a[1] for a in action_batch]
        discrete_action_batch = torch.tensor(discrete_action_batch).unsqueeze(1).to(self.device)
        continuous_action_batch = np.array(continuous_action_batch)
        continuous_action_batch = torch.from_numpy(continuous_action_batch).float().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)

        self.actor_net.eval()
        self.critic_net.eval()

        next_continuous_action_batch = self.actor_target_net(next_state_batch)
        next_q_values = self.critic_target_net(next_state_batch, next_continuous_action_batch)
        next_q_values_max = next_q_values.max(1)[0].detach()
        target = reward_batch + self.gamma * next_q_values_max * (1 - done_batch)

        q_values = self.critic_net(state_batch, continuous_action_batch)
        q_values = q_values.gather(1, index=discrete_action_batch)
        loss_critic = nn.MSELoss()(q_values, target.unsqueeze(1))

        critic_gradient = {}
        self.critic_net.zero_grad()
        loss_critic.backward()
        for name, param in self.critic_net.named_parameters():
            critic_gradient[name] = param.grad.detach().clone()

        # Update actor network.
        update_continuous_action_batch = self.actor_net(state_batch)
        update_q_values = self.critic_net(state_batch, update_continuous_action_batch)
        loss_critic = -update_q_values.max(1)[0].mean()

        actor_gradient = {}
        self.critic_net.zero_grad()
        self.actor_net.zero_grad()
        loss_critic.backward()
        for name, param in self.actor_net.named_parameters():
            actor_gradient[name] = param.grad.detach().clone()

        return actor_gradient, critic_gradient
