import torch

from gridpfn.core.agents.agent_sparse import P_DQN_sparse


class P_DQN_PFFDST(P_DQN_sparse):
    # P-DQN with PFFDST server rewiring and two-stage parameter freezing.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = kwargs.get("hyperparams") or {}
        self.differential_ratio = float(hp.get("differential_ratio", 0.5))
        self.readjust_interval = max(1, int(hp.get("readjust_interval", 1)))
        self.actor_counts = self.critic_counts = None
        self.frozen_actor_masks = self.frozen_critic_masks = None
        self.frozen_actor_params = self.frozen_critic_params = None
        self._optimizer_hooks = []

    def _counts(self, params, sparsity):
        weights = {name: param for name, param in params.items() if name.endswith("weight")}
        layer_sparsities = self.calculate_sparsities(weights, sparse=sparsity)
        return {
            name: int((1 - layer_sparsities.get(name, 0)) * param.numel())
            for name, param in params.items()
        }

    def _plan(self, params, final_sparsity):
        stage1 = (1 + final_sparsity) / 2
        extra = {
            name: int(self.differential_ratio * (1 - stage1) * param.numel())
            if name.endswith("weight")
            else 0
            for name, param in params.items()
        }
        counts = {
            "stage1": self._counts(params, stage1),
            "stage2": self._counts(params, final_sparsity),
        }
        for stage in (1, 2):
            counts[f"grown{stage}"] = {
                name: min(count + extra[name], params[name].numel())
                for name, count in counts[f"stage{stage}"].items()
            }
        return counts

    @staticmethod
    def _magnitude_masks(params, counts):
        masks = {}
        for name, param in params.items():
            mask = torch.zeros_like(param)
            kept = torch.topk(param.detach().abs().flatten(), counts[name], sorted=False).indices
            mask.flatten()[kept] = 1
            masks[name] = mask
        return masks

    def initialize(self, actor_sparsity, critic_sparsity, masks=None):
        actor_params, critic_params = self.get_trainable_params()
        self.actor_counts = self._plan(actor_params, actor_sparsity)
        self.critic_counts = self._plan(critic_params, critic_sparsity)
        if masks is None:
            actor_masks = self._magnitude_masks(actor_params, self.actor_counts["grown1"])
            critic_masks = self._magnitude_masks(critic_params, self.critic_counts["grown1"])
        else:
            actor_masks, critic_masks = masks
        self.set_model_masks(actor_masks, critic_masks)
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        return actor_masks, critic_masks

    def _apply_constraints(self):
        pairs = (
            (self.actor_net, self.actor_masks, self.frozen_actor_masks, self.frozen_actor_params),
            (
                self.actor_target_net,
                self.actor_masks,
                self.frozen_actor_masks,
                self.frozen_actor_params,
            ),
            (
                self.critic_net,
                self.critic_masks,
                self.frozen_critic_masks,
                self.frozen_critic_params,
            ),
            (
                self.critic_target_net,
                self.critic_masks,
                self.frozen_critic_masks,
                self.frozen_critic_params,
            ),
        )
        with torch.no_grad():
            for net, masks, frozen_masks, frozen_params in pairs:
                for name, param in net.named_parameters():
                    param.mul_(masks[name])
                    if frozen_masks is not None:
                        frozen = frozen_masks[name]
                        param[frozen] = frozen_params[name][frozen]

    def set_model_masks(self, actor_masks, critic_masks, survivor_masks=None):
        old_actor = getattr(self, "actor_masks", None)
        old_critic = getattr(self, "critic_masks", None)
        super().set_model_masks(actor_masks, critic_masks)
        with torch.no_grad():
            for index, (optimizer, net, target, masks, old_masks) in enumerate(
                (
                    (
                        self.actor_optimizer,
                        self.actor_net,
                        self.actor_target_net,
                        self.actor_masks,
                        old_actor,
                    ),
                    (
                        self.critic_optimizer,
                        self.critic_net,
                        self.critic_target_net,
                        self.critic_masks,
                        old_critic,
                    ),
                )
            ):
                for (name, param), target_param in zip(net.named_parameters(), target.parameters()):
                    survivor = masks[name].bool()
                    if survivor_masks is not None:
                        survivor = survivor_masks[index][name].to(self.device).bool()
                    elif old_masks is not None:
                        survivor = survivor & old_masks[name].bool()
                    target_param.mul_(survivor)
                    for value in optimizer.state.get(param, {}).values():
                        if torch.is_tensor(value) and value.shape == param.shape:
                            value.mul_(survivor)
        self._apply_constraints()

    def freeze_subnetwork(self, state=None):
        if state is None:
            actor_masks, critic_masks = self.get_model_masks(device=self.device)
            actor_params = {
                name: param.detach().clone() for name, param in self.actor_net.named_parameters()
            }
            critic_params = {
                name: param.detach().clone() for name, param in self.critic_net.named_parameters()
            }
        else:
            actor_masks, critic_masks, actor_params, critic_params = state
        self.frozen_actor_masks = {
            name: mask.bool().to(self.device) & name.endswith("weight")
            for name, mask in actor_masks.items()
        }
        self.frozen_critic_masks = {
            name: mask.bool().to(self.device) & name.endswith("weight")
            for name, mask in critic_masks.items()
        }
        self.frozen_actor_params = {
            name: value.to(self.device).clone() for name, value in actor_params.items()
        }
        self.frozen_critic_params = {
            name: value.to(self.device).clone() for name, value in critic_params.items()
        }
        for optimizer, net, masks in (
            (self.actor_optimizer, self.actor_net, self.frozen_actor_masks),
            (self.critic_optimizer, self.critic_net, self.frozen_critic_masks),
        ):
            frozen = [(param, masks[name]) for name, param in net.named_parameters()]

            def mask_gradients(_optimizer, _args, _kwargs, frozen=frozen):
                for param, mask in frozen:
                    param.grad.masked_fill_(mask, 0)

            self._optimizer_hooks.append(optimizer.register_step_pre_hook(mask_gradients))
            for name, param in net.named_parameters():
                for value in optimizer.state.get(param, {}).values():
                    if torch.is_tensor(value) and value.shape == param.shape:
                        value[masks[name]] = 0
        self._apply_constraints()

    def get_frozen_state(self):
        if self.frozen_actor_masks is None:
            return None
        maps = (
            self.frozen_actor_masks,
            self.frozen_critic_masks,
            self.frozen_actor_params,
            self.frozen_critic_params,
        )
        return tuple(
            {name: value.detach().cpu().clone() for name, value in mapping.items()}
            for mapping in maps
        )

    def get_communication_masks(self):
        actor_masks, critic_masks = self.get_model_masks()
        if self.frozen_actor_masks is None:
            return actor_masks, critic_masks
        return (
            {
                name: mask * (~self.frozen_actor_masks[name].cpu())
                for name, mask in actor_masks.items()
            },
            {
                name: mask * (~self.frozen_critic_masks[name].cpu())
                for name, mask in critic_masks.items()
            },
        )

    @staticmethod
    def _readjust(params, masks, target_counts, grown_counts, frozen_masks=None):
        updated_params = {name: value.clone() for name, value in params.items()}
        updated_masks = {}
        survivor_masks = {}
        for name, mask in masks.items():
            active = mask.bool().flatten()
            frozen = (
                torch.zeros_like(active)
                if frozen_masks is None
                else frozen_masks[name].bool().flatten().to(active.device)
            )
            mutable = (active & ~frozen).nonzero(as_tuple=False).squeeze(1)
            keep_count = max(0, min(target_counts[name] - int(frozen.sum()), mutable.numel()))
            kept = mutable
            if keep_count < mutable.numel():
                magnitude = params[name].flatten()[mutable].abs()
                kept = mutable[
                    torch.topk(magnitude, keep_count, largest=True, sorted=False).indices
                ]
            new_mask = frozen.clone()
            new_mask[kept] = True
            survivor_masks[name] = new_mask.reshape_as(mask).clone()
            updated_params[name].mul_(survivor_masks[name])
            grow_count = min(grown_counts[name], mask.numel()) - int(new_mask.sum())
            if grow_count > 0:
                candidates = (~new_mask).nonzero(as_tuple=False).squeeze(1)
                grown = candidates[
                    torch.randperm(candidates.numel(), device=mask.device)[:grow_count]
                ]
                new_mask[grown] = True
            updated_masks[name] = new_mask.reshape_as(mask).to(mask.dtype)
        return updated_params, updated_masks, survivor_masks

    def server_readjust(self, stage, grow=True):
        actor_params, critic_params = self.get_model_params()
        actor_masks, critic_masks = self.get_model_masks()
        suffix = stage if not grow else f"grown{stage[-1]}"
        actor_params, actor_masks, actor_survivors = self._readjust(
            actor_params,
            actor_masks,
            self.actor_counts[stage],
            self.actor_counts[suffix],
            self.frozen_actor_masks,
        )
        critic_params, critic_masks, critic_survivors = self._readjust(
            critic_params,
            critic_masks,
            self.critic_counts[stage],
            self.critic_counts[suffix],
            self.frozen_critic_masks,
        )
        self.set_model_params(actor_params, critic_params)
        survivors = actor_survivors, critic_survivors
        self.set_model_masks(actor_masks, critic_masks, survivors)
        return actor_params, critic_params, actor_masks, critic_masks, survivors
