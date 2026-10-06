import math

import numpy as np
import torch
import torch.nn as nn

from gridpfn.core.agents.agent_sparse import P_DQN_sparse


class P_DQN_FedFit(P_DQN_sparse):
    # Sparse P-DQN with FedFit's Fisher-guided topology adjustment.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = kwargs.get("hyperparams") or {}
        self.adjust_fraction = float(hp.get("adjust_fraction", 0.2))
        self.t_end = int(hp.get("t_end", 1000))
        self.fisher_eps = float(hp.get("fisher_eps", 1e-8))
        self.scores = None

    @staticmethod
    def _attach_kfac_hooks(net):
        factors, handles = {}, []
        for name, module in net.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            def forward_hook(layer, inputs, output, name=name):
                activation = inputs[0].detach().reshape(-1, layer.in_features)
                factors[name] = {"a": activation.square().sum(0) * activation.size(0)}

                def gradient_hook(gradient):
                    gradient = gradient.detach().reshape(-1, layer.out_features)
                    factors[name]["g"] = gradient.square().sum(0) * gradient.size(0)

                output.register_hook(gradient_hook)

            handles.append(module.register_forward_hook(forward_hook))
        return factors, handles

    def _score_loss(self, net, loss_fn):
        factors, handles = self._attach_kfac_hooks(net)
        net.zero_grad()
        loss_fn().backward()
        for handle in handles:
            handle.remove()

        prune, grow = {}, {}
        for name, param in net.named_parameters():
            if param.ndim < 2:
                continue
            gradient = param.grad.detach()
            factor = factors[name.rsplit(".", 1)[0]]
            fisher = factor["g"].unsqueeze(1) * factor["a"].unsqueeze(0)
            weight = param.detach()
            prune[name] = (-gradient * weight + 0.5 * weight.square() * fisher).detach().cpu()
            grow[name] = (0.5 * gradient.square() / (fisher + self.fisher_eps)).detach().cpu()
        return {"prune": prune, "grow": grow}

    def fisher_scores(self):
        if len(self.memory) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.from_numpy(state).float().to(self.device)
        discrete = torch.tensor([a[0] for a in action], device=self.device).unsqueeze(1)
        continuous = torch.from_numpy(np.array([a[1] for a in action])).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        with torch.no_grad():
            next_action = self.actor_target_net(next_state)
            target = reward + self.gamma * self.critic_target_net(next_state, next_action).max(1)[
                0
            ] * (1 - done)

        critic_scores = self._score_loss(
            self.critic_net,
            lambda: nn.functional.mse_loss(
                self.critic_net(state, continuous).gather(1, discrete), target.unsqueeze(1)
            ),
        )
        self.critic_net.zero_grad()
        actor_scores = self._score_loss(
            self.actor_net,
            lambda: -self.critic_net(state, self.actor_net(state)).max(1)[0].mean(),
        )
        self.critic_net.zero_grad()
        self.scores = actor_scores, critic_scores
        return self.scores

    @staticmethod
    def _adjust(masks, params, scores, fraction):
        adjusted = {name: mask.clone() for name, mask in masks.items()}
        for name, mask in adjusted.items():
            if params[name].ndim < 2:
                continue
            active = mask.flatten().bool().nonzero(as_tuple=False).squeeze(1)
            count = min(int(fraction * active.numel()), active.numel())
            if count == 0:
                continue
            prune = scores["prune"][name].flatten()[active]
            removed = active[torch.topk(prune, count, largest=False, sorted=False).indices]
            mask.flatten()[removed] = 0
            inactive = (~mask.flatten().bool()).nonzero(as_tuple=False).squeeze(1)
            grow = scores["grow"][name].flatten()[inactive]
            added = inactive[torch.topk(grow, count, largest=True, sorted=False).indices]
            mask.flatten()[added] = 1
        return adjusted

    def adjust_topology(self, round_idx):
        if round_idx > self.t_end or self.fisher_scores() is None:
            return False
        fraction = (
            self.adjust_fraction / 2 * (1 + math.cos(round_idx * math.pi / max(self.t_end, 1)))
        )
        actor_params, critic_params = self.get_trainable_params()
        actor_masks, critic_masks = self.get_model_masks()
        self.set_model_masks(
            self._adjust(actor_masks, actor_params, self.scores[0], fraction),
            self._adjust(critic_masks, critic_params, self.scores[1], fraction),
        )
        return True

    def apply_masks(self):
        for net, masks in (
            (self.actor_net, self.actor_masks),
            (self.critic_net, self.critic_masks),
            (self.actor_target_net, self.actor_masks),
            (self.critic_target_net, self.critic_masks),
        ):
            for name, param in net.named_parameters():
                param.data.mul_(masks[name].to(param.device))

    def set_model_masks(self, actor_masks, critic_masks):
        super().set_model_masks(actor_masks, critic_masks)
        self.apply_masks()

    def set_model_params(self, actor_parameters, critic_parameters):
        super().set_model_params(actor_parameters, critic_parameters)
        self.apply_masks()
