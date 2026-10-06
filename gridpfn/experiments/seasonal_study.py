"""Chronological monthly selection, train+validation refit, and sealed evaluation."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def prepare_stage(root, cfg):
    from gridpfn.core.dataset import home_data_dir, load_data
    from gridpfn.core.forecasting import causal_queries, physical_series
    from gridpfn.experiments.foundation_study import verify_inputs
    from gridpfn.experiments.foundation_worker import digest, save_json
    from gridpfn.paths import ROOT

    root.mkdir(parents=True, exist_ok=True)
    if (root / "cases.json").exists():
        if verify_inputs(root)["protocol"] != cfg:
            raise ValueError("Stage protocol changed; use a fresh output")
        return
    bundles = load_data(
        home_data_dir,
        "*.csv",
        choose=[f"home_{h}" for h in cfg["home_ids"]],
        validation_days=7,
        split="test" if cfg["refit"] else "validation",
        scaler_mode="shared",
        data_period=cfg["data_period"],
    )
    cases, homes = [], []
    (root / "cases").mkdir(exist_ok=True)
    for home, (train, heldout, heldout_dates, scaler) in zip(cfg["home_ids"], bundles, strict=True):
        dates = scaler["train_dates"] + heldout_dates
        if len(set(dates)) != len(dates) or dates != sorted(dates):
            raise ValueError("Stage dates must be disjoint and ordered")
        physical = np.concatenate(
            (physical_series(train, scaler), physical_series(heldout, scaler))
        )
        path = root / f"home_{home}.npz"
        np.savez_compressed(
            path,
            physical=physical,
            dates=dates,
            train_dates=scaler["train_dates"],
            train_sha256=hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),
        )
        homes.append(
            {
                "id": home,
                "path": path.name,
                "sha256": digest(path),
                "training_days": len(train),
                "validation_days": 0 if cfg["refit"] else len(heldout),
                "test_days": len(heldout) if cfg["refit"] else 0,
            }
        )
        blocks = [*range(14, len(train), 28), len(train)]
        for i, first in enumerate(blocks):
            last = blocks[i + 1] if i + 1 < len(blocks) else len(physical)
            X, y, _ = causal_queries(physical[:first], max_lead=cfg["horizon"])
            query, _, indices = causal_queries(physical[first:last], max_lead=cfg["horizon"])
            chosen = np.random.default_rng(cfg["seed"]).choice(
                len(X), min(len(X), cfg["context_rows"]), replace=False
            )
            case_path = root / "cases" / f"home{home}_from{first}.npz"
            np.savez_compressed(
                case_path,
                X=X[chosen],
                y=y[chosen, :3],
                query=query,
                indices=indices,
                selected_context_indices=chosen,
            )
            cases.append(
                {
                    "id": case_path.stem,
                    "home": home,
                    "first": first,
                    "last": last,
                    "path": str(case_path.relative_to(root)),
                    "sha256": digest(case_path),
                    "context_dates": dates[:first],
                    "query_dates": dates[first:last],
                    "rows": len(chosen),
                    "queries": len(query),
                }
            )
    save_json(
        root / "cases.json",
        {
            "protocol": cfg,
            "normalization": None,
            "homes": homes,
            "cases": cases,
            "input_sources": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    *[home_data_dir / f"home_{h}.csv" for h in cfg["home_ids"]],
                    ROOT / "dataset/temp_price_newyork.csv",
                ]
            },
        },
    )


def selected_recipe(select_root, methods):
    from gridpfn.experiments.foundation_worker import digest

    result = {}
    for kind in methods:
        run = select_root / "policies" / kind
        if json.loads((run / "status.json").read_text())["state"] != "completed":
            raise ValueError("Every selection run must complete before refitting")
        selection = json.loads((run / "selection.json").read_text())
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        evaluations = [r for r in metrics if r.get("kind") == "eval"]
        best = max(evaluations, key=lambda r: r["reward"])
        if (
            best["episode"] != selection["best"]["episode"]
            or best["reward"] != selection["best"]["reward"]
        ):
            raise ValueError("Selection differs from recorded validation maximum")
        result[kind] = {
            "episodes": best["episode"],
            "selection_sha256": digest(run / "selection.json"),
            "heads_sha256": digest(run / "checkpoints/best/heads.pt"),
            "metrics_sha256": digest(run / "metrics.jsonl"),
        }
    return {
        "criterion": "Maximum prior-week validation reward; refit exact selected budget on all preceding dates",
        "methods": result,
    }


@dataclass(frozen=True)
class ProcessJob:
    pid: int
    started: str
    key: tuple
    resource: str


def process_identity(pid, proc=Path("/proc")):
    """Linux start ticks avoid PID reuse; zombies no longer consume resources."""
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in {"Z", "X"} else fields[19]
    except (OSError, IndexError):
        return None


def discover_jobs(root, proc=Path("/proc")):
    """Match argv tokens, never process-name substrings or shell command text."""
    root = Path(root).resolve()
    jobs = []
    for directory in proc.glob("[0-9]*"):
        try:
            pid = int(directory.name)
            started = process_identity(pid, proc)
            if started is None:
                continue
            argv = directory.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
            module_index = argv.index("-m") + 1
            module = argv[module_index]
            cwd = directory.joinpath("cwd").resolve()

            def option(name):
                # Current numerical commands use separate tokens, not aliases.
                return argv[argv.index(name) + 1]

            if module == "gridpfn.experiments.foundation_worker":
                stage = (cwd / argv[module_index + 1]).resolve()
                kind = option("--kind")
                resource = "gpu" if option("--device").startswith("cuda") else "cpu"
                key = (str(stage), kind, "forecast")
            elif module == "gridpfn.experiments.run_experiment":
                output = (cwd / option("--path_train")).resolve()
                if output.parent.name != "policies":
                    continue
                stage, kind = output.parent.parent, output.name
                resource, key = "policy", (str(stage), kind, "policy")
            else:
                continue
            if not stage.is_relative_to(root) or process_identity(pid, proc) != started:
                continue
            jobs.append(ProcessJob(pid, started, key, resource))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError("Multiple existing writers target the same study task")
    return jobs


class ResourceGate:
    """Count inherited processes before admitting any newly launched work."""

    def __init__(self, limits, jobs=(), *, identity=process_identity, progress=None, poll=0.5):
        jobs = tuple(jobs)
        if any(type(value) is not int or value < 1 for value in limits.values()):
            raise ValueError("Every worker limit must be a positive integer")
        self.limits, self.identity, self.progress = limits, identity, progress
        self.condition = threading.Condition()
        self.inherited = {job.key: job for job in jobs}
        if len(self.inherited) != len(list(jobs)):
            raise ValueError("Duplicate inherited task")
        self.active = dict(self.inherited)
        self.counts = {name: sum(job.resource == name for job in jobs) for name in limits}
        self.running = set()
        self.stop = threading.Event()
        self.poll = poll
        for job in jobs:
            self.event("adopt", job.key, pid=job.pid, started=job.started, resource=job.resource)
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def event(self, event, key, **details):
        if self.progress:
            record = {
                "time": time.time(),
                "event": event,
                "task": list(key),
                "resources": dict(self.counts),
                **details,
            }
            with self.progress.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    def _monitor(self):
        while not self.stop.wait(self.poll):
            with self.condition:
                for key, job in list(self.active.items()):
                    if self.identity(job.pid) != job.started:
                        del self.active[key]
                        self.counts[job.resource] -= 1
                        self.event("adopted_exit", key, pid=job.pid, started=job.started)
                self.condition.notify_all()

    def run(self, key, resource, execute, validate):
        with self.condition:
            if key in self.running:
                raise ValueError("Concurrent duplicate study task")
            self.running.add(key)
            inherited = key in self.inherited
            if inherited:
                while key in self.active:
                    self.condition.wait()
            else:
                self.event("queued", key, resource=resource)
                while self.counts[resource] >= self.limits[resource]:
                    self.condition.wait()
                self.counts[resource] += 1
                self.event("start", key, resource=resource)
        try:
            if not inherited:
                execute()
            validate()
            with self.condition:
                self.event("validated", key, inherited=inherited)
        finally:
            with self.condition:
                if not inherited:
                    self.counts[resource] -= 1
                self.running.remove(key)
                self.condition.notify_all()

    def close(self):
        self.stop.set()
        self.monitor.join()


@contextmanager
def scheduler_lock(root):
    """Prevent two new schedulers from racing over the same output directory."""
    with (root / "scheduler.lock").open("a") as handle:
        if sys.platform.startswith("linux"):
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another scheduler already owns this study") from None
        yield


def run_schedule(root, cfg, folds, args):
    from gridpfn.experiments.foundation_study import (
        train,
        verify_features,
        verify_inputs,
        verify_predictions,
    )
    from gridpfn.experiments.foundation_worker import save_json
    from gridpfn.paths import ROOT

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TABPFN_NO_BROWSER="1")
    gate = ResourceGate(
        {"gpu": args.gpu_workers, "cpu": args.cpu_forecast_workers, "policy": args.policy_workers},
        discover_jobs(root),
        progress=root / "scheduler.jsonl",
    )

    def execute(command, log):
        with log.open("a") as handle:
            subprocess.run(
                command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
            )

    def policy_valid(stage, kind):
        path = stage / "policies" / kind / "status.json"
        if not path.exists() or json.loads(path.read_text()).get("state") != "completed":
            raise ValueError(f"Incomplete inherited or finished policy preserved: {stage}/{kind}")
        # Existing canonical reuse checks verify feature receipts and run settings.
        train(stage, kind)

    def method(stage, kind):
        if kind not in {"history", "persistence"}:
            python = {
                "tabfm": str(args.tabfm_python.absolute()),
                "tabicl": str(args.tabicl_python.absolute()),
            }.get(kind, sys.executable)
            device = args.device if kind in {"tabpfn", "tabfm"} else "cpu"
            key = (str(stage), kind, "forecast")

            def validate():
                verify_predictions(stage, kind, verify_inputs(stage))

            if (
                key not in gate.inherited
                and (stage / "predictions" / kind / "status.json").exists()
            ):
                validate()
            else:
                gate.run(
                    key,
                    "gpu" if device.startswith("cuda") else "cpu",
                    lambda: execute(
                        [
                            python,
                            "-m",
                            "gridpfn.experiments.foundation_worker",
                            str(stage),
                            "--kind",
                            kind,
                            "--device",
                            device,
                        ],
                        stage / f"{kind}.log",
                    ),
                    validate,
                )

        def command(action):
            return [
                sys.executable,
                "-m",
                "gridpfn.experiments.foundation_study",
                action,
                "--output",
                str(stage),
                "--kind",
                kind,
            ]

        if (stage / "features" / kind / "manifest.json").exists():
            verify_features(stage, kind)
        else:
            execute(command("features"), stage / f"pipeline_{kind}.log")
        key = (str(stage), kind, "policy")
        if key not in gate.inherited and (stage / "policies" / kind).exists():
            policy_valid(stage, kind)
        else:
            gate.run(
                key,
                "policy",
                lambda: execute(command("train"), stage / f"pipeline_{kind}.log"),
                lambda: policy_valid(stage, kind),
            )
        print(f"{stage.relative_to(root)} / {kind}: complete", flush=True)

    def fold_job(item):
        month, fold = item
        for phase in ("select", "refit"):
            stage_cfg = {
                **cfg,
                "data_period": f"month_{month:02d}_{phase}",
                "refit": phase == "refit",
            }
            if phase == "refit":
                recipe = selected_recipe(root / fold["select"], cfg["methods"])
                path = root / fold["recipe"]
                if path.exists() and json.loads(path.read_text()) != recipe:
                    raise ValueError("Selected recipe changed after freeze")
                save_json(path, recipe)
                stage_cfg.update(
                    refit_budgets={k: v["episodes"] for k, v in recipe["methods"].items()},
                    refit_selection_root=str(root / fold["select"] / "policies"),
                )
            stage = root / fold[phase]
            prepare_stage(stage, stage_cfg)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg["methods"])) as pool:
                list(pool.map(lambda k: method(stage, k), cfg["methods"]))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.fold_workers) as pool:
            list(pool.map(fold_job, zip(cfg["test_months"], folds, strict=True)))
    finally:
        gate.close()
    save_json(
        root / "status.json",
        {"state": "training_completed", "folds": len(folds), "test_metrics_read": False},
    )
    print(
        "Every selection and refit completed; test evaluation is a separate frozen stage.",
        flush=True,
    )


def main(argv=None):
    from gridpfn.experiments.foundation_worker import save_json
    from gridpfn.paths import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/seasonal.json")
    parser.add_argument("--tabfm-python", type=Path, required=True)
    parser.add_argument("--tabicl-python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold-workers", type=int, default=1)
    parser.add_argument("--policy-workers", type=int, default=6)
    parser.add_argument("--gpu-workers", type=int, default=1)
    parser.add_argument("--cpu-forecast-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if any(
        getattr(args, key) < 1
        for key in ("fold_workers", "policy_workers", "gpu_workers", "cpu_forecast_workers")
    ):
        parser.error("Worker counts must be positive")
    root = args.output.resolve()
    cfg = json.loads(args.protocol.read_text())
    root.mkdir(parents=True, exist_ok=True)
    folds = [
        {
            "id": f"2019-{m:02d}",
            "select": f"folds/2019-{m:02d}/select",
            "refit": f"folds/2019-{m:02d}/refit",
            "recipe": f"folds/2019-{m:02d}/recipe.json",
            "oracle": f"oracles/2019-{m:02d}",
        }
        for m in cfg["test_months"]
    ]
    manifest = {"protocol": cfg, "folds": folds}
    with scheduler_lock(root):
        if (root / "study.json").exists() and json.loads(
            (root / "study.json").read_text()
        ) != manifest:
            raise ValueError("Frozen seasonal protocol changed")
        save_json(root / "study.json", manifest)
        run_schedule(root, cfg, folds, args)


if __name__ == "__main__":
    main()
