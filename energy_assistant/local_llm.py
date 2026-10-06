"""Pinned, CPU-only local chat runtime, kept entirely in ignored results/."""

import hashlib
import json
import os
import platform
import socket
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "results/energy_assistant_llm"
RELEASE = "b11439"
REVISION = "f6d5376be1edb4d416d56da11e5397a961aca8ae"
MODEL = "Qwen3.5-2B-Q4_K_M.gguf"
MODEL_SHA = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"
MODEL_ID = "gridpfn-local"
ARCHIVE_SHA = "6c1a1c5b3a9016b07d228f1ee6d731a1b34e1dd9411ab7f35bb0a9e014ec9d06"


def cpu_threads(requested=None):
    """Respect process affinity; avoid oversubscribing a shared CPU or small device."""
    available = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    if requested is not None and not 1 <= requested <= available:
        raise ValueError(f"Choose between 1 and {available} CPU threads")
    return requested or min(4, max(1, available // 2))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, expected=None):
    if target.exists() and (expected is None or file_hash(target) == expected):
        return
    temporary = target.with_suffix(".part")
    print(f"Downloading {target.name}", flush=True)
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
    if expected and file_hash(temporary) != expected:
        temporary.unlink()
        raise ValueError("Downloaded file checksum did not match")
    temporary.replace(target)


def setup():
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ValueError(
            "Automatic setup supports Linux x64; other platforms can use an external compatible chat server"
        )
    RUNTIME.mkdir(parents=True, exist_ok=True)
    archive = RUNTIME / f"llama-{RELEASE}.tar.gz"
    download(
        f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/llama-{RELEASE}-bin-ubuntu-x64.tar.gz",
        archive,
        ARCHIVE_SHA,
    )
    if not list(RUNTIME.glob("**/llama-server")):
        with tarfile.open(archive) as tar:
            tar.extractall(RUNTIME / "runtime", filter="data")
    download(
        f"https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/{REVISION}/{MODEL}",
        RUNTIME / MODEL,
        MODEL_SHA,
    )
    (RUNTIME / "manifest.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "model_sha256": MODEL_SHA,
                "quantization_revision": REVISION,
                "runtime": RELEASE,
                "runtime_archive_sha256": file_hash(archive),
                "source": "https://huggingface.co/Qwen/Qwen3.5-2B",
                "device": "cpu",
            },
            indent=2,
        )
    )
    print("Local chat is ready: Qwen3.5-2B, 4-bit, CPU only.", flush=True)


def start(port=8771, threads=None):
    threads = cpu_threads(threads)
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise ValueError(
                f"Local chat port {port} is already in use; choose another app port"
            ) from error
    binaries = list(RUNTIME.glob("**/llama-server"))
    if not binaries or not (RUNTIME / MODEL).exists():
        raise ValueError("Run python -m energy_assistant setup-chat first")
    if file_hash(RUNTIME / MODEL) != MODEL_SHA:
        raise ValueError("Local chat model checksum changed; run setup-chat again")
    binary = binaries[0]
    log = (RUNTIME / f"server-{port}.log").open("a")
    process = subprocess.Popen(
        [
            str(binary),
            "--model",
            str(RUNTIME / MODEL),
            "--alias",
            MODEL_ID,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--n-gpu-layers",
            "0",
            "--threads",
            str(threads),
            "--threads-batch",
            str(threads),
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
            "--jinja",
            "--chat-template-kwargs",
            '{"enable_thinking": false}',
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=str(binary.parent)),
        stdout=log,
        stderr=log,
    )
    log.close()
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError(
                    f"Local chat failed to start; see results/energy_assistant_llm/server-{port}.log"
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1
                ) as response:
                    if response.status == 200:
                        return process
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("Local chat did not become ready")
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        raise


if __name__ == "__main__":
    setup()
