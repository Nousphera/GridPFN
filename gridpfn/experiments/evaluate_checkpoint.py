"""Evaluate saved policy heads on a declared split, preserving training outputs."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from gridpfn.core.dataset import load_data, setup_seed
from gridpfn.core.em_strategy import compose_em_strategy
from gridpfn.core.model import (
    configure_embedding_device,
    encoder_context_identity,
    heads_from_state,
    validate_encoder_context,
)
from gridpfn.core.predictive_features import PredictiveContext
from gridpfn.core.training_metrics import PeriodicEvaluator
from gridpfn.core.utils.agent_utils import safe_torch_load
from gridpfn.core.utils.run_io import read_logged_settings


def evaluate_saved(run, checkpoint, split="test", gpu=1, oracle_reference=None):
    settings = read_logged_settings(run / "logs/fedavg/train_settings.txt")
    payload = safe_torch_load(checkpoint, "cpu")
    if payload["home_ids"] != settings["home_ids"]:
        raise ValueError("Checkpoint homes differ from training settings")
    service = "energy_quota" if settings["em_strategy"].get("ac_energy_quota", True) else "thermal"
    if payload.get("ac_service") is not None and payload["ac_service"] != service:
        raise ValueError("Checkpoint AC service mode differs from training settings")
    setup_seed(settings["fixed_seed"])
    embedding_device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") if settings.get("devices") == ["cpu"] else embedding_device
    if embedding_device.type == "cuda":
        torch.cuda.set_device(embedding_device)
        torch.cuda.set_per_process_memory_fraction(0.2, gpu)
    configure_embedding_device(
        embedding_device, settings.get("embedding_cache"), settings.get("encoder_context")
    )
    if payload.get("encoder_context_sha256") != encoder_context_identity():
        raise ValueError("Encoder context differs from the checkpoint's immutable context")
    data = load_data(
        settings["path_data"],
        "*.csv",
        choose=[f"home_{home}" for home in settings["home_ids"]],
        validation_days=settings["validation_days"],
        data_period=settings.get("data_period", "legacy"),
        split=split,
        scaler_mode=settings.get("scaler_mode", "local"),
        grid_prices=settings.get("grid_prices"),
    )
    clients = []
    for home_id, bundle, heads in zip(settings["home_ids"], data, payload["clients"], strict=True):
        actor_state, critic_state = heads["actor"], heads["critic"]
        actor, critic = heads_from_state(actor_state, critic_state, device)
        width = actor.state_dim
        train_data, test_data, test_dates, scaler = bundle
        clients.append(
            SimpleNamespace(
                train_data=train_data,
                home_id=home_id,
                test_data=test_data,
                test_dates=test_dates,
                scaler=scaler,
                fixed_cost=settings["fixed_cost"],
                state_dim=width,
                device=device,
                predictive_context=PredictiveContext(
                    Path(settings["predictive_features"]) / f"home_{home_id}.npz",
                    train_data,
                    scaler,
                )
                if settings.get("predictive_features")
                else None,
                fedavg_agent=SimpleNamespace(actor_net=actor, critic_net=critic, device=device),
            )
        )
        context = clients[-1].predictive_context
        if context is not None and heads.get("predictive_context_sha256") != context.fingerprint:
            raise ValueError("Predictive features differ from the checkpoint's immutable context")
    server = SimpleNamespace(
        clients=clients,
        em_strategy=compose_em_strategy(settings["em_strategy"], clients),
        p2p_config=settings["p2p_config"],
    )
    validate_encoder_context(clients, server.em_strategy, server.p2p_config)
    evaluator = PeriodicEvaluator(
        SimpleNamespace(home_ids=settings["home_ids"]),
        1,
        split=split,
        oracle_reference=oracle_reference,
    )
    started = time.monotonic()
    with torch.no_grad():
        result = evaluator.evaluate(
            server, include_days=True, include_losses=False, include_actions=True
        )
    result.update(
        split=split,
        ac_service=service,
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        episode=payload["episode"],
        validation_reward=payload["reward"],
        evaluation_seconds=time.monotonic() - started,
        head_device=str(device),
        embedding_device=str(embedding_device),
        selection_rule=payload.get("selection_rule", "max validation reward"),
        note="Policy rollout metrics; no reconstructed TD loss. August was used by earlier historical studies.",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--checkpoint",
        default="best_feasible",
        choices=("best_feasible", "best", "latest", "initial"),
    )
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument(
        "--oracle_reference", type=Path, help="Matched certified oracle for this split."
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    run = args.run.resolve()
    checkpoint = run / "checkpoints" / args.checkpoint / "heads.pt"
    output = run / "evaluation" / f"{args.split}_{args.checkpoint}.json"
    if output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {output}")
    record = evaluate_saved(run, checkpoint, args.split, args.gpu, args.oracle_reference)
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in {"homes", "day_records", "action_records", "dates"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
