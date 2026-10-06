import torch
import torch.nn as nn

from gridpfn.core.agents.agent_sparse import P_DQN_sparse
from gridpfn.core.utils.agent_utils import quantization_bounds, validate_low_bit


class P_DQN_FedDMPQ(P_DQN_sparse):
    # P-DQN agent for Federated Dynamic Mixed Precision Quantization.
    def __init__(
        self,
        actor_net: nn.Module,
        critic_net: nn.Module,
        discrete_action_dim: int,
        continuous_action_dim: int,
        state_dim: int,
        hyperparams=None,
    ):
        super().__init__(
            actor_net=actor_net,
            critic_net=critic_net,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            state_dim=state_dim,
            hyperparams=hyperparams,
        )

        hp = hyperparams or {}
        # Importance-score mixing coefficients: I_i = alpha_w*|W_i| + beta_g*|g_i| + gamma_e*|E_i|
        self.alpha_w = float(hp.get("alpha_w", 0.34))
        self.beta_g = float(hp.get("beta_g", 0.33))
        self.gamma_e = float(hp.get("gamma_e", 0.33))
        self.K = int(hp.get("K", 10))
        self.low_bit = validate_low_bit(hp.get("low_bit", 8))
        self.qmin, self.qmax = quantization_bounds(self.low_bit)

        # Per-weight precision state: True = low-bit, False = FP32.
        self.actor_precision: dict[str, torch.Tensor] = {}
        self.critic_precision: dict[str, torch.Tensor] = {}

        self.actor_low_precision_duration: dict[str, torch.Tensor] = {}
        self.critic_low_precision_duration: dict[str, torch.Tensor] = {}

    def init_soft_prune_state(self) -> None:
        # Initialize all weights to FP32 precision.
        for net, precision, duration in (
            (self.actor_net, self.actor_precision, self.actor_low_precision_duration),
            (self.critic_net, self.critic_precision, self.critic_low_precision_duration),
        ):
            for name, param in net.named_parameters():
                precision[name] = torch.zeros_like(param, dtype=torch.bool)
                duration[name] = torch.zeros_like(param, dtype=torch.int32)

    def _quantize_to_low_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        t = tensor.detach().float()
        max_abs = t.abs().amax().item()
        if max_abs < 1e-8:
            return tensor
        scale = max_abs / self.qmax
        quantized = torch.clamp(torch.round(t / scale), self.qmin, self.qmax)
        return (quantized * scale).to(orig_dtype)

    @staticmethod
    def _minmax_normalize(tensor: torch.Tensor) -> torch.Tensor:
        # Min-max normalize without leaving the tensor's current device
        min_value = tensor.amin()
        value_range = tensor.amax() - min_value
        return torch.where(
            value_range > 0,
            (tensor - min_value) / value_range.clamp_min(torch.finfo(tensor.dtype).eps),
            torch.zeros_like(tensor),
        )

    def _compute_importance(
        self,
        net: nn.Module,
        gradients: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        importance = {}
        for name, param in net.named_parameters():
            weight = param.detach().float()
            gradient = gradients.get(name)
            g_mag = (
                gradient.detach().to(weight.device, dtype=weight.dtype).abs()
                if gradient is not None
                else torch.zeros_like(weight)
            )
            importance[name] = (
                self.alpha_w * self._minmax_normalize(weight.abs())
                + self.beta_g * self._minmax_normalize(g_mag)
                + self.gamma_e
                * self._minmax_normalize((weight - self._quantize_to_low_precision(weight)).abs())
            )
        return importance

    @staticmethod
    def _derive_threshold(
        importance: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        sparsity: float,
    ) -> torch.Tensor:
        # Soft-pruning threshold is the *sparsity*-th percentile of importance scores among all currently active connections.
        active_scores = []
        for name, imp in importance.items():
            if name not in masks:
                continue
            mask_bool = masks[name].to(imp.device, dtype=torch.bool)
            scores = imp[mask_bool]
            if scores.numel() > 0:
                active_scores.append(scores)
        if not active_scores:
            return next(iter(importance.values())).new_zeros(())
        all_scores = torch.cat(active_scores)
        return torch.quantile(all_scores, sparsity)

    def _apply_soft_prune(
        self,
        net: nn.Module,
        masks: dict[str, torch.Tensor],
        importance: dict[str, torch.Tensor],
        precision: dict[str, torch.Tensor],
        low_precision_duration: dict[str, torch.Tensor],
        threshold: torch.Tensor,
        gradients: dict[str, torch.Tensor],
    ) -> None:

        for name, param in net.named_parameters():
            if name not in masks:
                continue

            param_device = param.device
            mask = masks[name].to(param_device, dtype=param.dtype)
            precision[name] = precision[name].to(param_device)
            low_precision_duration[name] = low_precision_duration[name].to(param_device)
            is_active = mask.bool()
            is_low_imp = importance[name] < threshold
            new_precision = torch.where(is_active, is_low_imp, precision[name])
            new_duration = torch.where(new_precision, low_precision_duration[name] + 1, 0)

            # Hard prune after K consecutive low-precision steps.
            hard_prune = new_precision & (new_duration >= self.K)
            num_hard_pruned = int(hard_prune.sum().item())
            new_mask = mask.clone()
            new_mask[hard_prune] = 0.0
            new_precision[hard_prune] = False
            new_duration[hard_prune] = 0

            # For every newly hard-pruned connection, regrow one dead connection at the position with the largest |gradient|.
            if num_hard_pruned > 0 and name in gradients and gradients[name] is not None:
                gradient = gradients[name].detach().to(param_device).abs().flatten()
                dead_indices = (~new_mask.bool()).flatten().nonzero(as_tuple=False).squeeze(1)
                regrow_count = min(num_hard_pruned, dead_indices.numel())
                if regrow_count > 0:
                    candidate_scores = gradient[dead_indices]
                    selected = torch.topk(candidate_scores, k=regrow_count, sorted=False).indices
                    new_mask.flatten()[dead_indices[selected]] = 1.0

            precision[name] = new_precision
            low_precision_duration[name] = new_duration
            masks[name] = new_mask

            with torch.no_grad():
                param.data *= new_mask

    def _apply_precision(self, net, precision):
        for name, param in net.named_parameters():
            idx = precision[name].to(param.device)
            if idx.any():
                param.data[idx] = self._quantize_to_low_precision(param.data[idx])

    def soft_prune_step(self, actor_sparsity: float, critic_sparsity: float) -> None:
        if len(self.memory) < self.batch_size:
            return

        actor_gradient, critic_gradient = self.screen_gradients()
        actor_masks, critic_masks = self.get_model_masks(device=self.device)

        for net, gradients, masks, precision, duration, sparsity in (
            (
                self.actor_net,
                actor_gradient,
                actor_masks,
                self.actor_precision,
                self.actor_low_precision_duration,
                actor_sparsity,
            ),
            (
                self.critic_net,
                critic_gradient,
                critic_masks,
                self.critic_precision,
                self.critic_low_precision_duration,
                critic_sparsity,
            ),
        ):
            importance = self._compute_importance(net, gradients)
            threshold = self._derive_threshold(importance, masks, sparsity)
            self._apply_soft_prune(
                net, masks, importance, precision, duration, threshold, gradients
            )

        self.set_model_masks(actor_masks, critic_masks)

        for net, precision, masks in (
            (self.actor_net, self.actor_precision, actor_masks),
            (self.critic_net, self.critic_precision, critic_masks),
            (self.actor_target_net, self.actor_precision, actor_masks),
            (self.critic_target_net, self.critic_precision, critic_masks),
        ):
            self._apply_precision(net, precision)
            for name, param in net.named_parameters():
                param.data *= masks[name].to(param.device, dtype=param.dtype)

    # Mixed-precision parameter export
    def get_mixed_precision_params(self):
        # Export CPU payloads without moving the live networks off their device.
        def export(net, precision):
            state = {
                name: tensor.detach().cpu().clone() for name, tensor in net.state_dict().items()
            }
            for name, tensor in state.items():
                if name in precision and precision[name].any():
                    idx = precision[name].cpu()
                    tensor[idx] = self._quantize_to_low_precision(tensor[idx])
            return state

        return export(self.actor_net, self.actor_precision), export(
            self.critic_net, self.critic_precision
        )
