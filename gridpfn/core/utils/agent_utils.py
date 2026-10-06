import random

import numpy as np
import torch


def validate_low_bit(low_bit):
    return int(low_bit)


def low_precision_param_column(low_bit):
    return f"params_int{validate_low_bit(low_bit)}"


def quantization_bounds(low_bit):
    low_bit = validate_low_bit(low_bit)
    return -(1 << (low_bit - 1)), (1 << (low_bit - 1)) - 1


# Soft-update target network parameters from a source network by factor tau (Polyak averaging).
def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        if not param.requires_grad:
            continue
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


# Replay buffer storing transitions and providing sampling utilities.
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = np.stack(state_batch)
        next_state_batch = np.stack(next_state_batch)
        reward_batch = np.array(reward_batch)
        done_batch = np.array(done_batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


def safe_torch_load(path, device):
    return torch.load(path, map_location=device, weights_only=True)
