"""Serve refreshing convergence plots using the existing evaluation plotting code."""

import argparse
import io
import json
import os
import re
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib

from gridpfn.paths import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridpfn.core.utils.plots import plot_live_convergence, plot_live_scheduling
from gridpfn.core.utils.run_io import newest_run


def read_records(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    complete = raw.rsplit(b"\n", 1)[0] if b"\n" in raw else b""
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def run_status(run):
    manifest_path, status_path = run / "run.json", run / "status.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "unmanaged"}
    if status.get("state") == "running" and status.get("pid"):
        try:
            os.kill(status["pid"], 0)
        except ProcessLookupError:
            status["state"] = "stopped without completion record"
    tail = ""
    log = run / "gridpfn.experiments.train.log"
    if log.exists():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 4096))
            tail = handle.read().decode("utf-8", errors="replace")
    progress = re.findall(
        r"\[(?:warmup|eval|embeddings|imitation|expert-critic|predictive)\] [^\n]+", tail
    )
    return {
        "run": run.name,
        "path": str(run),
        "status": status.get("state"),
        "stage": status.get("stage"),
        "mode": manifest.get("mode", "training"),
        "learning_rule": manifest.get("settings", {}).get("actor_update", "q_gradient"),
        "value_normalization": manifest.get("settings", {}).get("value_normalization", False),
        "pid": status.get("pid"),
        "gpu": manifest.get("gpu"),
        "selection": json.loads((run / "selection.json").read_text())
        if (run / "selection.json").exists()
        else {},
        "final_test": json.loads((run / "final_summary.json").read_text()).get("selected")
        if (run / "final_summary.json").exists()
        else None,
        "final_split": json.loads((run / "final_summary.json").read_text()).get("split", "test")
        if (run / "final_summary.json").exists()
        else None,
        "convergence": json.loads((run / "convergence.json").read_text())
        if (run / "convergence.json").exists()
        else {},
        "episodes": manifest.get("episodes"),
        "eval_step": manifest.get("eval_step"),
        "ac_service": manifest.get("ac_service", "energy_quota"),
        "progress": progress[-1] if progress else "",
        "log": "\n".join(tail.splitlines()[-4:]),
    }


PAGE = (ROOT / "dashboard.html").read_text()


class IncrementalRecords:
    """Read only appended bytes; retain an incomplete JSONL tail between polls."""

    def __init__(self, path):
        self.path = path
        self.records = []
        self.offset = 0
        self.pending = b""
        self.identity = None
        self.generation = uuid.uuid4().hex
        self.revision = "0"
        self.bytes_read = 0

    def poll(self):
        if not self.path.exists():
            return self.records
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            self.records, self.pending, self.offset = [], b"", 0
            self.generation = uuid.uuid4().hex
        self.identity = identity
        if stat.st_size > self.offset:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
            self.offset += len(data)
            self.bytes_read += len(data)
            pieces = (self.pending + data).split(b"\n")
            self.pending = pieces.pop()
            self.records.extend(json.loads(line) for line in pieces if line.strip())
            self.revision = f"{self.generation}:{len(self.records)}"
        return self.records


class PlotDashboard:
    def __init__(self, run, interval, window):
        self.root = run
        self.run, self.interval, self.window = run, interval, window
        self.readers = {}
        self.images = {}
        self.lock = threading.RLock()
        self.select_run("")

    def select_run(self, name):
        if name:
            selected = (self.root / name).resolve()
            if (
                not selected.is_relative_to(self.root.resolve())
                or not (selected / "run.json").exists()
            ):
                raise ValueError("Unknown study run")
        elif (self.root / "current.json").exists():
            selected = (
                self.root / json.loads((self.root / "current.json").read_text())["run"]
            ).resolve()
            if not selected.is_relative_to(self.root.resolve()):
                raise ValueError("Study run is outside its root")
        else:
            selected = self.root
        self.run = selected
        self.path = selected / "metrics.jsonl"
        self.reader = self.readers.setdefault(str(selected), IncrementalRecords(self.path))

    def revision(self):
        self.reader.poll()
        return self.reader.revision

    def status(self, include_latest=False):
        records = self.reader.poll()
        train = [r for r in records if r["kind"] == "train"]
        test = [r for r in records if r["kind"] == "eval"]
        homes = sorted({h["home_id"] for r in records for h in r.get("homes", [])})
        status = run_status(self.run)
        episode = train[-1]["episode"] if train else 0
        eta = None
        if len(train) > 1 and status.get("episodes") and status["status"] == "running":
            earlier = train[max(0, len(train) - 21)]
            rate = (train[-1]["elapsed_seconds"] - earlier["elapsed_seconds"]) / (
                episode - earlier["episode"]
            )
            eta = max(0, (status["episodes"] - episode) * rate)
        study = (
            json.loads((self.root / "study.json").read_text())
            if (self.root / "study.json").exists()
            else {}
        )
        return {
            **status,
            "revision": self.reader.revision,
            "homes": homes,
            "train_episode": episode,
            "eval_episode": test[-1]["episode"] if test else None,
            "evaluating": bool(records and records[-1]["kind"] == "eval_start"),
            "split": test[-1].get("split", "test") if test else None,
            "eta_seconds": eta,
            **(
                {
                    "latest_train": {k: v for k, v in train[-1].items() if k != "homes"}
                    if train
                    else None,
                    "latest_eval": {
                        k: v for k, v in test[-1].items() if k not in {"homes", "dates"}
                    }
                    if test
                    else None,
                }
                if include_latest
                else {}
            ),
            "runs": sorted(
                str(p.parent.relative_to(self.root)) for p in self.root.rglob("run.json")
            ),
            "study_progress": (
                f"{len(study['runs'])}/{study.get('planned_runs', len(study['seeds']) * (6 if study.get('study') == 'control' else 3))} runs complete"
                if study
                else None
            ),
        }

    def render(self, window, home, fmt=None):
        formats = [fmt] if fmt else ["png", "pdf"]
        records = self.reader.poll()
        key = (str(self.run), self.reader.revision, window, home)
        if any((*key, f) not in self.images for f in formats):
            if run_status(self.run)["mode"] == "scheduling":
                fig = plot_live_scheduling(records, home_id=home)
            else:
                fig = plot_live_convergence(records, window=window, home_id=home)
            try:
                for f in formats:
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format=f, dpi=110)
                    self.images[(*key, f)] = buffer.getvalue()
                    output = self.run / "live" / f"convergence.{f}"
                    output.parent.mkdir(exist_ok=True)
                    temporary = output.with_suffix(f".{f}.tmp")
                    temporary.write_bytes(buffer.getvalue())
                    temporary.replace(output)
            finally:
                plt.close(fig)
        result = {f: self.images[(*key, f)] for f in formats}
        self.images = {k: v for k, v in self.images.items() if k[:4] == key}
        return result

    def handler(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                request = urlparse(self.path)
                try:
                    with dashboard.lock:
                        self.respond(request)
                except (ValueError, OSError) as error:
                    self.send_error(400, str(error))

            def respond(self, request):
                args = parse_qs(request.query)
                dashboard.select_run(args.get("run", [""])[0])
                if request.path == "/":
                    body = PAGE.replace("REFRESH_MS", str(int(dashboard.interval * 1000)))
                    body = body.replace("DEFAULT_WINDOW", str(dashboard.window)).encode()
                    content_type = "text/html; charset=utf-8"
                elif request.path == "/api/events":
                    records = dashboard.reader.poll()
                    cursor = int(args.get("cursor", ["0"])[0])
                    generation = args.get("generation", [""])[0]
                    if generation != dashboard.reader.generation or not 0 <= cursor <= len(records):
                        cursor = 0
                    body = json.dumps(
                        {
                            "records": records[cursor:],
                            "cursor": len(records),
                            "generation": dashboard.reader.generation,
                            "status": dashboard.status(),
                        },
                        allow_nan=False,
                    ).encode()
                    content_type = "application/json"
                elif request.path in {"/api/status", "/api/summary"}:
                    body = json.dumps(
                        dashboard.status(include_latest=request.path == "/api/summary"),
                        allow_nan=False,
                    ).encode()
                    content_type = "application/json"
                elif request.path in {"/plot.png", "/plot.pdf"}:
                    args = parse_qs(request.query)
                    window = int(args.get("window", [dashboard.window])[0])
                    if not 1 <= window <= 1500:
                        raise ValueError("Smoothing must be 1..1500 episodes")
                    home = int(args["home"][0]) if args.get("home") else None
                    fmt = request.path.rsplit(".", 1)[1]
                    body = dashboard.render(window, home, fmt)[fmt]
                    content_type = "image/png" if fmt == "png" else "application/pdf"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_):
                pass

        return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", nargs="?", type=Path, help="Defaults to newest launched experiment."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument(
        "--window", type=int, default=10, help="Per-home training smoothing; 1 is raw."
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--once", action="store_true", help="Save one PNG/PDF and exit.")
    args = parser.parse_args()
    if args.interval <= 0 or not 1 <= args.window <= 1500:
        parser.error("Interval must be positive and window must be 1..1500")
    run = args.run_dir
    if run is None:
        run = newest_run(ROOT / "results")
        if run is None:
            parser.error("No launched experiment found; specify its run directory")

    run = run.expanduser().resolve()
    if not run.is_dir():
        parser.error(f"Missing run directory: {run}")
    dashboard = PlotDashboard(run, args.interval, args.window)
    if args.once:
        dashboard.render(args.window, None)
        print(f"Saved {dashboard.run / 'live/convergence.png'} and convergence.pdf")
        return
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), dashboard.handler())
    except OSError as error:
        parser.error(f"Could not start dashboard: {error}; try another --port")
    url = f"http://{args.host}:{httpd.server_port}"
    print(f"Live plots: {url}\nRun: {run}\nCtrl+C stops the dashboard only.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped; training continues.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
