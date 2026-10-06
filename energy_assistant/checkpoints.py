"""Explicit opt-in loading of a locally trusted personalized policy checkpoint."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from utils.agent_utils import safe_torch_load
from utils.run_io import read_logged_settings

from model import configure_embedding_device, encoder_context_identity, heads_from_state
from predictive_features import PredictiveContext, observe
from training_metrics import greedy_actions


def checkpoint_policy(run, checkpoint, home, bundle):
    """Return a CPU callable and provenance; reject silently mismatched contexts."""
    run, checkpoint = Path(run).resolve(), Path(checkpoint).resolve()
    settings = read_logged_settings(run / "logs/fedavg/train_settings.txt")
    payload = safe_torch_load(checkpoint, "cpu")
    if payload["home_ids"] != settings["home_ids"] or home not in settings["home_ids"]:
        raise ValueError("Checkpoint home identities do not match the selected home")
    service = "energy_quota" if settings["em_strategy"].get("ac_energy_quota", True) else "thermal"
    if payload.get("ac_service") is not None and payload["ac_service"] != service:
        raise ValueError("Checkpoint appliance service differs from the training settings")
    device = torch.device("cpu")
    # No use of the shared training embedding cache: keep inference in this process.
    configure_embedding_device(device, None, settings.get("encoder_context"))
    if payload.get("encoder_context_sha256") != encoder_context_identity():
        raise ValueError("The checkpoint requires a different immutable encoder context")
    heads = payload["clients"][settings["home_ids"].index(home)]
    actor, critic = heads_from_state(heads["actor"], heads["critic"], device)
    train, heldout, dates, scaler = bundle
    context = None
    if settings.get("predictive_features"):
        context = PredictiveContext(
            Path(settings["predictive_features"]) / f"home_{home}.npz", train, scaler
        )
        if heads.get("predictive_context_sha256") != context.fingerprint:
            raise ValueError("Checkpoint forecast features do not match")
    client = SimpleNamespace(predictive_context=context)
    if actor.state_dim != 17 or getattr(actor, "auxiliary_dim", 0) != (
        context.width if context else 0
    ):
        raise ValueError(
            "The demo requires the 17-value control state with matching predictive features"
        )
    agent = SimpleNamespace(actor_net=actor.eval(), critic_net=critic.eval(), device=device)

    def policy(state, date, hour):
        features = observe(client, state, date, hour)
        action = greedy_actions(agent, np.asarray([features]))[0]
        return int(action[0]), action[1:]

    return policy, {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "episode": payload.get("episode"),
        "home": home,
        "feature_mode": getattr(actor, "feature_mode", "unknown"),
        "note": "Saved personalized policy, executed on CPU in a single-home replay without peer settlement. This is not the original cohort benchmark.",
    }
