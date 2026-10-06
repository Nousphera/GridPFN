"""Matched original-branch and current RL study with validation-only convergence."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gridpfn.core.dataset import home_data_dir, load_data, setup_seed
from gridpfn.core.em_strategy import P2P_TRADING, compose_em_strategy, make_em_strategy
from gridpfn.core.model import Actor, Critic, configure_embedding_device
from gridpfn.core.training_config import HOME_IDS, parse_args
from gridpfn.core.training_metrics import MetricsLogger, PeriodicEvaluator, greedy_actions
from gridpfn.core.utils.convergence import stability
from gridpfn.core.utils.original_learning import batch_original_agents
from gridpfn.core.utils.run_io import atomic_json, backbone_identity, file_sha256
from gridpfn.experiments.run_experiment import snapshot
from gridpfn.legacy.control_benchmark import load_clients, rollout_controllers
from gridpfn.paths import ROOT

ARMS = ("original", "current_mlp", "current_tabpfn")


class MatchedEvaluator(PeriodicEvaluator):
    def __init__(self, logger, minimum, checks):
        super().__init__(logger, 50, split="validation", min_episodes=minimum, patience=0)
        self.records = []
        self.minimum, self.checks = minimum, checks

    def save_checkpoint(self, server, episode, record):
        super().save_checkpoint(server, episode, record)
        self.records.append({"episode": episode, **record})

    def __call__(self, server, episode):
        before = len(self.records)
        super().__call__(server, episode)
        if len(self.records) == before:
            return
        evidence = stability(self.records, self.minimum, self.checks)
        self.stopping.stopped = evidence["stopped"]
        atomic_json(
            self.logger.path.parent / "convergence.json",
            {"episode": episode, "minimum_episodes": self.minimum, **evidence},
        )


def original_modules(source):
    """Load the pinned original learning code; use the common audited data/physics."""
    names = [
        "model",
        "gridpfn.core.utils.agent_utils",
        "gridpfn.core.agents.agent",
        "gridpfn.core.agents.agent_sparse",
        "gridpfn.core.agents.agent_fedfit",
        "gridpfn.core.agents.agent_pffdst",
        "gridpfn.core.agents.agent_feddmpq",
        "client",
        "server",
    ]
    for name in names:
        path = source / (name.replace(".", "/") + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["client"], sys.modules["server"]


class PlateauReached(Exception):
    pass


def original_server(module, evaluator, logger):
    class ObservedServer(module.Server):
        def _warmup_clients(self, get_agent_fn):
            super()._warmup_clients(get_agent_fn)
            self.batched_learner = batch_original_agents([c.fedavg_agent for c in self.clients])
            evaluator(self, 0)

        def _collect_episode_results(self, episode, *args, **kwargs):
            if episode:
                evaluator(self, episode)
                if evaluator.stopping.stopped:
                    raise PlateauReached
            return P2P_TRADING.run_episode(
                self.clients,
                episode,
                use_fed=True,
                p2p_config=self.p2p_config,
                learner=self.batched_learner,
            )

        def _log_episode_results(self, logs, results, episode, tag):
            super()._log_episode_results(logs, results, episode, tag)
            logger.training("fedavg", episode + 1, results)

    return ObservedServer


def worker(args):
    output = args.output.resolve()
    manifest = json.loads((output.parent / "study.json").read_text())
    output.mkdir(exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError(output / "run.json")
    started = time.monotonic()
    settings = parse_args(
        [
            "--seed",
            str(args.seed),
            "--episode",
            str(args.episodes),
            "--gpu",
            str(args.gpu),
            "--cpu_threads",
            "1",
            "--head_device",
            "cpu",
        ]
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configure_embedding_device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.set_per_process_memory_fraction(0.2, args.gpu)
    if args.arm == "original":
        client_module, server_module = original_modules(output.parent / "original_source")
        settings = parse_args(
            [
                "--preset",
                "legacy",
                "--run_models",
                "fedavg",
                "--eval_models",
                "fedavg",
                "--seed",
                str(args.seed),
                "--episode",
                str(args.episodes),
                "--state_dim",
                "9",
                "--validation_days",
                "14",
                "--eval_step",
                "50",
                "--actor_sparsity",
                "0.25",
                "--critic_sparsity",
                "0.25",
            ]
        )
    else:
        import gridpfn.core.client as client_module
        import gridpfn.core.server as server_module

        if args.arm == "current_mlp":
            settings.feature_mode = "raw"
    settings.path_train = str(output)
    settings.path_data = manifest["data_dir"]
    settings.gpu, settings.cpu_threads, settings.head_device = args.gpu, 1, "auto"
    head_device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    atomic_json(
        output / "run.json",
        {
            "created_at": time.time(),
            "models": ["fedavg"],
            "seed": args.seed,
            "gpu": args.gpu,
            "clients": len(HOME_IDS),
            "home_ids": list(HOME_IDS),
            "episodes": args.episodes,
            "eval_step": 50,
            "eval_split": "validation",
            "sparsity": 0,
            "arm": args.arm,
            "settings": vars(settings),
            "note": "FedAvg is dense; 0.25 sparsity only applies to sparse methods.",
        },
    )
    atomic_json(output / "status.json", {"state": "running", "pid": os.getpid()})
    try:
        setup_seed(settings.fixed_seed)
        bundles = load_data(
            settings.path_data,
            "*.csv",
            choose=[f"home_{h}" for h in HOME_IDS],
            validation_days=14,
            split="validation",
        )
        atomic_json(
            output / "training_inputs.json",
            [
                {
                    "home_id": home,
                    "train_sha256": hashlib.sha256(bundle[0].tobytes()).hexdigest(),
                    "validation_sha256": hashlib.sha256(bundle[1].tobytes()).hexdigest(),
                    "validation_dates": bundle[2],
                    "scaler": bundle[3],
                }
                for home, bundle in zip(HOME_IDS, bundles, strict=True)
            ],
        )
        hp_keys = (
            "gamma",
            "batch_size",
            "lr_actor",
            "lr_critic",
            "epsilon_start",
            "epsilon_end",
            "epsilon_decay",
            "critic_tau",
            "actor_tau",
        )
        if args.arm != "original":
            hp_keys += (
                "memory_capacity",
                "bc_rounds",
                "bc_steps",
                "bc_lr",
                "bc_weight",
                "actor_q_weight",
                "exploration_noise",
            )
        hp = {k: getattr(settings, k) for k in hp_keys}
        torch.manual_seed(settings.seed)
        clients = []
        for bundle in bundles:
            extras = (
                {}
                if args.arm == "original"
                else {
                    "active_model": "fedavg",
                    "feature_mode": settings.feature_mode,
                    "warmup_learning": settings.warmup_learning,
                }
            )
            client = client_module.Client(
                bundle,
                settings.state_dim,
                3,
                2,
                args.episodes,
                settings.epsilon,
                agent_hyperparams=hp,
                fixed_cost=5,
                device=head_device,
                **extras,
            )
            if args.arm == "original":
                _, client.test_data, client.test_dates, _ = bundle
                for name in ("actor_net", "critic_net", "actor_target_net", "critic_target_net"):
                    net = getattr(client.fedavg_agent, name)
                    net.feature_mode, net.state_dim = "raw", 9
            clients.append(client)
        atomic_json(
            output / "model_capacity.json",
            {
                "state_dim": settings.state_dim,
                "actor_head_parameters_per_home": sum(
                    p.numel() for p in clients[0].fedavg_agent.actor_net.parameters()
                ),
                "critic_head_parameters_per_home": sum(
                    p.numel() for p in clients[0].fedavg_agent.critic_net.parameters()
                ),
                "backbone_frozen": args.arm == "current_tabpfn",
            },
        )
        setup_seed(settings.fixed_seed)
        logger = MetricsLogger(output / "metrics.jsonl", HOME_IDS)
        evaluator = MatchedEvaluator(logger, args.minimum, args.checks)
        server_module.logs_root = str(output / "logs")
        cls = (
            original_server(server_module, evaluator, logger)
            if args.arm == "original"
            else server_module.Server
        )
        extras = (
            {}
            if args.arm == "original"
            else {
                "metrics_logger": logger,
                "eval_callback": evaluator,
                "batched_updates": True,
            }
        )
        server = cls(
            clients,
            args.episodes,
            0.25,
            0.25,
            settings.update,
            settings.aggregate,
            make_em_strategy(),
            p2p_config={"enabled": True, "price": 0.1},
            warmup_rounds=settings.warmup_rounds,
            **extras,
        )
        try:
            server.fedavg_train()
        except PlateauReached:
            pass
        if not evaluator.stopping.stopped and evaluator.records[-1]["episode"] != args.episodes:
            evaluator(server, args.episodes)
        training_seconds = time.monotonic() - started
        # Test dates are loaded only after training and checkpoint selection are finished.
        test = load_data(
            settings.path_data,
            "*.csv",
            choose=[f"home_{h}" for h in HOME_IDS],
            validation_days=14,
            split="test",
        )
        for client, bundle in zip(clients, test, strict=True):
            _, client.test_data, client.test_dates, _ = bundle
        atomic_json(output / "status.json", {"state": "evaluating", "pid": os.getpid()})
        evaluated = {}
        for label in ("initial", "best_feasible", "latest"):
            checkpoint = output / "checkpoints" / label / "heads.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for client, heads in zip(clients, payload["clients"], strict=True):
                client.fedavg_agent.actor_net.load_state_dict(heads["actor"])
                client.fedavg_agent.critic_net.load_state_dict(heads["critic"])
            with torch.no_grad():
                result = evaluator.evaluate(server, include_days=True, include_losses=False)
                controllers = []
                for client in clients:

                    def policy(state, agent=client.fedavg_agent, width=client.state_dim):
                        action = greedy_actions(agent, np.asarray(state[:width])[None])[0]
                        return int(action[0]), action[1:]

                    controllers.append(policy)
                physical = rollout_controllers(clients, server.em_strategy, controllers)
            if result["dates"] != physical["dates"]:
                raise ValueError("Final evaluation date mismatch")
            for left, right in zip(result["homes"], physical["homes"], strict=True):
                for key in ("reward", "comfort_pct", "elec_cost", "import"):
                    if not np.isclose(left[key], right[key], atol=1e-6, rtol=1e-6):
                        raise ValueError(f"Final replay mismatch: {key}")
            for left_days, right_days in zip(
                result["day_records"], physical["day_records"], strict=True
            ):
                for left, right in zip(left_days, right_days, strict=True):
                    for key in ("reward", "comfort_pct", "elec_cost", "import"):
                        if not np.isclose(left[key], right[key], atol=2e-5, rtol=1e-6):
                            raise ValueError(f"Final home-day replay mismatch: {key}")
            evaluated[label] = {
                "episode": payload["episode"],
                "checkpoint_sha256": file_sha256(checkpoint),
                "policy": result,
                "physical": physical,
            }
        convergence = json.loads((output / "convergence.json").read_text())
        if not convergence["stopped"]:
            convergence["reason"] = "episode cap reached; convergence not established"
        atomic_json(output / "convergence.json", convergence)
        atomic_json(
            output / "final_summary.json",
            {
                "arm": args.arm,
                "seed": args.seed,
                "training_seconds": training_seconds,
                "convergence": convergence,
                "evaluations": evaluated,
                "selected": evaluated["best_feasible"]["policy"],
                "split": "test",
            },
        )
        if args.arm == "current_tabpfn":
            atomic_json(output / "backbone.json", backbone_identity())
        atomic_json(
            output / "status.json",
            {"state": "completed", "pid": os.getpid(), "seconds": time.monotonic() - started},
        )
    except BaseException as exc:
        atomic_json(
            output / "status.json", {"state": "failed", "pid": os.getpid(), "error": str(exc)}
        )
        raise


def evaluate_reward_selection(study, gpu=0):
    """Replay completed runs' unconstrained best validation reward heads.

    It never trains or changes checkpoint selection, and uses no test feedback.
    Completed results are reused only when the exact checkpoint hash matches.
    """
    manifest = json.loads((study / "study.json").read_text())
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    configure_embedding_device(device)
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "original_eval_model", study / "original_source/model.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    for arm in ARMS:
        for seed in manifest["seeds"]:
            run = study / f"{arm}_seed{seed}"
            if not (run / "final_summary.json").exists():
                continue
            summary = json.loads((run / "final_summary.json").read_text())
            checkpoint = run / "checkpoints/best/heads.pt"
            checksum = file_sha256(checkpoint)
            destination = run / "reward_selection.json"
            if (
                destination.exists()
                and json.loads(destination.read_text())["checkpoint_sha256"] == checksum
            ):
                continue
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            feasible = summary["evaluations"]["best_feasible"]
            if feasible["checkpoint_sha256"] == checksum:
                physical = feasible["physical"]
            else:
                clients = load_clients(split="test", data_dir=manifest["data_dir"])
                controllers = []
                for heads in payload["clients"]:
                    width = 9 if arm == "original" else 17
                    if arm == "original":
                        actor, critic = original.Actor(width, 3), original.Critic(width, 3, 2)
                        actor.feature_mode = "raw"
                    else:
                        mode = "raw" if arm == "current_mlp" else "frozen"
                        actor, critic = (
                            Actor(width, 3, feature_mode=mode),
                            Critic(width, 3, 2, feature_mode=mode),
                        )
                    actor, critic = actor.to(device).eval(), critic.to(device).eval()
                    actor.load_state_dict(heads["actor"])
                    critic.load_state_dict(heads["critic"])
                    agent = SimpleNamespace(actor_net=actor, critic_net=critic, device=device)

                    def policy(state, agent=agent, width=width):
                        action = greedy_actions(agent, np.asarray(state[:width])[None])[0]
                        return int(action[0]), action[1:]

                    policy.actor_net = actor
                    controllers.append(policy)
                with torch.no_grad():
                    physical = rollout_controllers(
                        clients, compose_em_strategy(make_em_strategy(), clients), controllers
                    )
                if physical["dates"] != feasible["physical"]["dates"]:
                    raise ValueError("Sensitivity check test dates differ")
            atomic_json(
                destination,
                {
                    "episode": payload["episode"],
                    "checkpoint_sha256": checksum,
                    "selection_rule": "maximum validation reward without service/cost gate; sensitivity only",
                    "validation_feasible": payload["feasible"],
                    "physical": physical,
                },
            )


def launch(args):
    root, output = ROOT, args.output.resolve()
    if not output.is_relative_to(root / "results"):
        raise ValueError("Study output must be in results/")
    if args.minimum < 3000 and not args.smoke:
        raise ValueError("Full studies require at least 3000 episodes before stopping")
    output.mkdir(parents=True, exist_ok=False)
    source = snapshot(root, output)
    original = output / "original_source"
    revision = subprocess.check_output(
        ["git", "rev-parse", args.original], cwd=root, text=True
    ).strip()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision], cwd=root, text=True
    ).splitlines()
    for name in paths:
        if name.endswith(".py"):
            target = original / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
            )
    manifest = {
        "created_at": time.time(),
        "state": "running",
        "pid": os.getpid(),
        "study": "matched_branch",
        "original_revision": revision,
        "current_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "seeds": args.seeds,
        "arms": list(ARMS),
        "planned_runs": len(args.seeds) * len(ARMS),
        "runs": [],
        "gpu": args.gpu,
        "data_dir": str(home_data_dir),
        "original_source_hashes": {
            str(p.relative_to(original)): file_sha256(p) for p in original.rglob("*.py")
        },
        "data_hashes": {
            f"home_{h}.csv": file_sha256(home_data_dir / f"home_{h}.csv") for h in HOME_IDS
        },
        "price_weather_sha256": file_sha256(root / "dataset/temp_price_newyork.csv"),
        "runtime": {
            "python": sys.version,
            "head_device": "cuda" if torch.cuda.is_available() else "cpu",
            "cpu_threads_per_run": 1,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "tabpfn", "numpy", "scipy", "pandas")
            },
        },
        "protocol": {
            "training": "June 1-July 17",
            "validation": "July 18-31 shared complete dates",
            "test": "August shared complete dates; previously accessed in development",
            "home_ids": list(HOME_IDS),
            "minimum_episodes": args.minimum,
            "episode_cap": args.episodes,
            "eval_step": 50,
            "plateau_checks": args.checks,
            "smoke": args.smoke,
            "common": "current audited environment/em_strategy/dataset, train-only scaling/ToU, fixed-seed day order, evaluation and selection; original learning modules remain unchanged",
            "original": "pinned original three-layer actor/critic/client/server; exact independent P-DQN updates batched after original warmup, preserving Adam moments, exploration and aggregation",
            "current_mlp": "current optimized pipeline with raw observations and the same small head architecture",
            "current_tabpfn": "current optimized pipeline with frozen TabPFN embeddings",
            "selection": "best feasible validation reward, relative to each arm's initial service/cost",
            "claim": "policy-performance plateau, not mathematical or optimizer convergence",
        },
    }
    atomic_json(output / "study.json", manifest)
    atomic_json(output / "current.json", {"run": f"current_tabpfn_seed{args.seeds[0]}"})
    running = []
    try:
        for seed in args.seeds:
            for arm in ARMS:
                run = output / f"{arm}_seed{seed}"
                run.mkdir()
                log = (run / "gridpfn.experiments.train.log").open("w")
                command = [
                    sys.executable,
                    "-u",
                    "-m", "gridpfn.legacy.run_matched_study",
                    "--worker",
                    "--output",
                    str(run),
                    "--arm",
                    arm,
                    "--seed",
                    str(seed),
                    "--gpu",
                    str(args.gpu),
                    "--episodes",
                    str(args.episodes),
                    "--minimum",
                    str(args.minimum),
                    "--checks",
                    str(args.checks),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=source,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
                running.append((process, log, run))
        # Independent runs use one host thread each and share the selected GPU.
        while running:
            for item in running[:]:
                process, log, run = item
                code = process.poll()
                if code is None:
                    continue
                log.close()
                running.remove(item)
                if code:
                    raise RuntimeError(f"{run.name} failed with exit {code}")
                manifest["runs"].append(run.name)
                atomic_json(output / "study.json", manifest)
            time.sleep(1)
        inputs = [
            json.loads((output / run / "training_inputs.json").read_text())
            for run in manifest["runs"]
        ]
        if any(value != inputs[0] for value in inputs[1:]):
            raise ValueError("Training/validation/scaler inputs differ across matched runs")
        manifest["inputs_matched"] = True
        manifest["state"] = "completed"
        manifest["all_converged"] = all(
            json.loads((output / run / "convergence.json").read_text())["stopped"]
            for run in manifest["runs"]
        )
        atomic_json(output / "study.json", manifest)
        subprocess.run(
            [sys.executable, "-m", "gridpfn.legacy.plot_matched_study", str(output)],
            check=True,
            cwd=root,
        )
    except BaseException as exc:
        # Do not kill unrelated or surviving study arms on a coordinator failure.
        manifest.update(
            state="failed",
            error=str(exc),
            active_runs=[{"run": r.name, "pid": p.pid} for p, _, r in running if p.poll() is None],
        )
        atomic_json(output / "study.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/matched_branch"))
    parser.add_argument("--original", default="e34da8b68c646ae3d6c818ad13f2020e6ba06c73")
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--minimum", type=int, default=3000)
    parser.add_argument("--checks", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker(args) if args.worker else launch(args)


if __name__ == "__main__":
    main()
