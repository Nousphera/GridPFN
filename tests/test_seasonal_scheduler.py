"""Orchestration adoption and resource limits, without numerical model execution."""

import concurrent.futures
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gridpfn.experiments import seasonal_study as scheduling


def proc_entry(proc, pid, argv, cwd, *, start="100", state="R"):
    directory = proc / str(pid)
    directory.mkdir()
    directory.joinpath("cmdline").write_bytes(b"\0".join(str(a).encode() for a in argv) + b"\0")
    directory.joinpath("cwd").symlink_to(cwd)
    fields = [state, *("0" for _ in range(18)), start, "0"]
    directory.joinpath("stat").write_text(
        f"{pid} (test name with ) parentheses) " + " ".join(fields)
    )


def test_exact_proc_matching_identity_zombies_and_foreign_jobs(tmp_path):
    proc, root = tmp_path / "proc", tmp_path / "study"
    proc.mkdir()
    root.mkdir()
    stage = root / "folds/2019-07/select"
    forecast = [
        "python",
        "-u",
        "-m",
        "gridpfn.experiments.foundation_worker",
        stage,
        "--kind",
        "tabfm",
        "--device",
        "cuda:0",
    ]
    proc_entry(proc, 1, forecast, tmp_path, start="333")
    proc_entry(
        proc,
        2,
        [
            "python",
            "-m",
            "gridpfn.experiments.run_experiment",
            "--path_train",
            stage / "policies/history",
        ],
        tmp_path,
        start="444",
    )
    proc_entry(proc, 3, ["bash", "-c", " ".join(map(str, forecast))], tmp_path)
    proc_entry(proc, 4, forecast, tmp_path, state="Z")
    proc_entry(
        proc,
        5,
        [
            "python",
            "-m",
            "gridpfn.experiments.foundation_worker",
            tmp_path / "other",
            "--kind",
            "tabfm",
            "--device",
            "cuda:0",
        ],
        tmp_path,
    )
    # Do not count the foundation_study parent plus run_experiment child twice.
    proc_entry(
        proc,
        6,
        [
            "python",
            "-m",
            "gridpfn.experiments.foundation_study",
            "train",
            "--output",
            stage,
            "--kind",
            "history",
        ],
        tmp_path,
    )
    found = scheduling.discover_jobs(root, proc)
    assert {(job.pid, job.started, job.resource) for job in found} == {
        (1, "333", "gpu"),
        (2, "444", "policy"),
    }
    assert scheduling.process_identity(4, proc) is None
    assert scheduling.discover_jobs(root, tmp_path / "no_proc") == []


def test_duplicate_existing_writers_fail_before_scheduling(tmp_path):
    proc, root = tmp_path / "proc", tmp_path / "study"
    proc.mkdir()
    root.mkdir()
    argv = [
        "python",
        "-m",
        "gridpfn.experiments.run_experiment",
        "--path_train",
        root / "policies/history",
    ]
    for pid in (1, 2):
        proc_entry(proc, pid, argv, tmp_path)
    with pytest.raises(ValueError, match="Multiple existing writers"):
        scheduling.discover_jobs(root, proc)


def test_inherited_jobs_above_cap_block_new_work_and_pid_reuse_releases(tmp_path):
    alive = {1: "a", 2: "b"}
    jobs = [
        scheduling.ProcessJob(pid, start, (str(pid), "tabfm", "forecast"), "gpu")
        for pid, start in alive.items()
    ]
    gate = scheduling.ResourceGate(
        {"gpu": 1}, jobs, identity=alive.get, poll=0.01, progress=tmp_path / "events.jsonl"
    )
    entered = threading.Event()
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            task = pool.submit(
                gate.run, ("new", "tabfm", "forecast"), "gpu", entered.set, lambda: None
            )
            assert not entered.wait(0.05)
            alive.pop(1)
            assert not entered.wait(0.05)  # one inherited job still occupies cap1
            alive[2] = "reused-pid"  # another start time is not the inherited job
            assert entered.wait(1)
            task.result(timeout=1)
        assert gate.counts == {"gpu": 0}
    finally:
        gate.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events[:2]] == ["adopt", "adopt"]
    assert next(row for row in events if row["event"] == "start")["resources"]["gpu"] == 1


def test_attached_job_waits_then_validates_without_relaunch():
    alive = {1: "old"}
    job = scheduling.ProcessJob(1, "old", ("stage", "tabicl", "policy"), "policy")
    gate = scheduling.ResourceGate({"policy": 1}, [job], identity=alive.get, poll=0.01)
    validated = threading.Event()
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            task = pool.submit(
                gate.run,
                job.key,
                "policy",
                lambda: pytest.fail("Relaunched inherited job"),
                validated.set,
            )
            assert not validated.wait(0.03)
            alive.clear()
            task.result(timeout=1)
            assert validated.is_set()
        with pytest.raises(ValueError, match="incomplete"):
            gate.run(
                job.key,
                "policy",
                lambda: pytest.fail("Restarted incomplete job"),
                lambda: (_ for _ in ()).throw(ValueError("incomplete")),
            )
    finally:
        gate.close()


def test_new_work_is_bounded_for_every_resource():
    caps = {"gpu": 2, "cpu": 3, "policy": 4}
    gate = scheduling.ResourceGate(caps, poll=0.01)
    active, peak = dict.fromkeys(caps, 0), dict.fromkeys(caps, 0)
    lock = threading.Lock()

    def execute(resource):
        with lock:
            active[resource] += 1
            peak[resource] = max(peak[resource], active[resource])
        time.sleep(0.02)
        with lock:
            active[resource] -= 1

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            jobs = [
                pool.submit(
                    gate.run,
                    (resource, str(i), "task"),
                    resource,
                    lambda r=resource: execute(r),
                    lambda: None,
                )
                for resource in caps
                for i in range(8)
            ]
            for job in jobs:
                job.result(timeout=2)
        assert peak == caps
        assert gate.counts == dict.fromkeys(caps, 0)
    finally:
        gate.close()


def test_folds_parallel_but_refit_follows_all_selection_policies(tmp_path, monkeypatch):
    from gridpfn import paths
    from gridpfn.experiments import foundation_study

    root = tmp_path / "study"
    root.mkdir()
    cfg = {"test_months": [6, 7], "methods": ["history", "persistence"]}
    folds = [
        {"select": f"{m}/select", "refit": f"{m}/refit", "recipe": f"{m}/recipe.json"}
        for m in cfg["test_months"]
    ]
    args = SimpleNamespace(
        fold_workers=2,
        gpu_workers=1,
        cpu_forecast_workers=1,
        policy_workers=1,
        device="cpu",
        tabfm_python=Path("python"),
        tabicl_python=Path("python"),
    )
    barrier = threading.Barrier(2)
    finished, events = set(), []
    lock = threading.Lock()

    def prepare(stage, config):
        stage.mkdir(parents=True)
        if config["refit"]:
            assert all((stage.parent / "select", kind) in finished for kind in cfg["methods"])
        else:
            barrier.wait(timeout=2)  # both folds must be scheduled concurrently

    def execute(command, **kwargs):
        action = command[3]
        stage = Path(command[command.index("--output") + 1])
        kind = command[command.index("--kind") + 1]
        with lock:
            events.append((stage, kind, action))
        if action == "train":
            run = stage / "policies" / kind
            run.mkdir(parents=True)
            (run / "status.json").write_text('{"state":"completed"}')
            finished.add((stage, kind))

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(scheduling, "prepare_stage", prepare)
    monkeypatch.setattr(scheduling, "discover_jobs", lambda _: [])
    monkeypatch.setattr(scheduling.subprocess, "run", execute)
    monkeypatch.setattr(
        scheduling,
        "selected_recipe",
        lambda stage, methods: {"methods": {kind: {"episodes": 0} for kind in methods}},
    )
    monkeypatch.setattr(foundation_study, "train", lambda stage, kind: None)
    scheduling.run_schedule(root, cfg, folds, args)
    assert len(finished) == 8
    for stage, kind in finished:
        assert events.index((stage, kind, "features")) < events.index((stage, kind, "train"))
    assert json.loads((root / "status.json").read_text())["test_metrics_read"] is False


def test_scheduler_lock_rejects_second_owner(tmp_path):
    with scheduling.scheduler_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with scheduling.scheduler_lock(tmp_path):
                pytest.fail("Double owner")
