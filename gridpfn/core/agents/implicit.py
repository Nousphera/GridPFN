"""IQL-style expectile value learning and advantage-weighted hybrid regression.

Continuous regression uses a fixed-variance Gaussian surrogate on normalized
executed powers; the categorical term learns the WM choice. Deployment uses the
feasible mean and modal WM choice. This is a hybrid adaptation, not vanilla IQL.
"""

import torch
import torch.nn.functional as F


def expectile_loss(value, target, expectile):
    residual = target.detach() - value
    return torch.where(residual > 0, expectile, 1 - expectile) * residual.square()


def actor_regression(control, behavior, logits, choices, advantage, span, temperature):
    weights = (advantage.detach() / temperature).clamp(-5, 3).exp()
    # Normalize within each home's batch to keep its optimizer step scale stable.
    weights = weights / weights.mean(-1, keepdim=True).clamp_min(1e-6)
    continuous = ((control - behavior) / span).square().mean(-1)
    discrete = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), choices.reshape(-1), reduction="none"
    ).reshape(choices.shape)
    return weights * (continuous + 0.05 * discrete)
