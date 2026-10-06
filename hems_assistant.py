"""Launch one home: python hems_assistant.py --household HOME_MODEL_DIR --llm qwen35_2b."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def chat_configuration(llm, port, api_key_env=None, endpoint=None):
    """Resolve one supported local model or an explicitly selected hosted provider."""
    if llm in {"qwen35_2b", "guided"}:
        if api_key_env or endpoint:
            raise ValueError("API options apply only to a hosted model")
        return {
            "provider": "local" if llm == "qwen35_2b" else "guided",
            "model": "gridpfn-local",
            "endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
            "api_key": "",
        }
    provider, separator, model = llm.partition(":")
    if not separator or provider not in {"openai", "anthropic", "compatible"} or not model.strip():
        raise ValueError("Use qwen35_2b, guided, openai:MODEL, anthropic:MODEL or compatible:MODEL")
    from energy_assistant.settings import ProviderSettings, SettingsStore

    endpoints = {
        "openai": "https://api.openai.com/v1/responses",
        "anthropic": "https://api.anthropic.com/v1/messages",
    }
    if endpoint and provider != "compatible":
        raise ValueError("--llm-endpoint is only for compatible:MODEL")
    key_env = (
        api_key_env
        or {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "compatible": "ENERGY_LLM_KEY",
        }[provider]
    )
    key = os.environ.get(key_env, "")
    if not key and provider == "openai" and not api_key_env:
        key = os.environ.get("OPEN_API_KEY", "")  # Accept the user's shorter spelling.
    if not key and provider != "compatible":
        raise ValueError(f"Set {key_env} before starting this provider, or choose --llm qwen35_2b")
    config = ProviderSettings(
        provider=provider,
        model=model.strip(),
        endpoint=endpoints.get(provider, endpoint or ""),
        api_key=key,
    )
    SettingsStore.validate_endpoint(config.model_dump())
    return config.model_dump(exclude={"clear_key"})


def resolve_model(path, home=None, checkpoint="best_feasible"):
    """Accept a completed cohort run or a verified local-reference home descriptor."""
    from utils.run_io import read_logged_settings

    path = Path(path).resolve()
    deployment = None
    descriptor = path / "home_model.json" if path.is_dir() else path
    if descriptor.is_file() and descriptor.suffix == ".json":
        metadata = json.loads(descriptor.read_text())
        if metadata.get("bundle_type") == "local_reference":
            from gridpfn.deployment import load_model_bundle

            deployment = load_model_bundle(descriptor.parent)
            metadata = deployment
        selected = int(metadata["home_id"])
        if home is not None and int(home) != selected:
            raise ValueError("This personalized model belongs to a different home")
        home = selected
        run = (descriptor.parent / metadata["run_dir"]).resolve()
        checkpoint = metadata.get("checkpoint_name", metadata.get("checkpoint", checkpoint))
    else:
        run = path
    if checkpoint not in {"best_feasible", "best", "latest", "initial"}:
        raise ValueError("Unsupported checkpoint selection")
    settings = read_logged_settings(run / "logs/fedavg/train_settings.txt")
    if deployment is not None:
        settings["assistant_split"] = deployment["assistant_contract"]["split"]
    if home is None:
        if len(settings["home_ids"]) != 1:
            raise ValueError(
                "This training run contains several homes. Add --home HOME_ID once at launch."
            )
        home = settings["home_ids"][0]
    if int(home) not in settings["home_ids"]:
        raise ValueError("The training run does not contain this home")
    return run, int(home), checkpoint, settings


def evidence_cache_key(paths, *, sources, home, split, days, context):
    """Bind prepared evidence to its inputs and complete implementation receipt."""
    from energy_assistant.local_llm import file_hash

    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(file_hash(path).encode())
    configuration = dict(sources=sources, home=home, split=split, days=days, context=context)
    digest.update(json.dumps(configuration, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--household", "--model", dest="model", type=Path,
        help="Home model directory (includes verified references to its household data)",
    )
    source.add_argument("--evidence", type=Path, help="Already prepared household demo")
    parser.add_argument("--home", type=int, help="Needed only if the source contains several homes")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--llm",
        help="qwen35_2b (default), guided, openai:MODEL, anthropic:MODEL or compatible:MODEL",
    )
    parser.add_argument(
        "--llm-threads",
        type=int,
        help="Local CPU threads; default adapts to available CPUs, capped at four",
    )
    parser.add_argument(
        "--api-key-env",
        help="Environment variable containing the provider key; never put the key in the command",
    )
    parser.add_argument("--llm-endpoint", help="Full chat-completions URL for compatible:MODEL")
    parser.add_argument("--chat", choices=["local", "guided"], help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint",
        choices=["best_feasible", "best", "latest", "initial"],
        default="best_feasible",
    )
    parser.add_argument(
        "--split",
        choices=["history", "validation", "test"],
        default="history",
        help="Replay all recorded history (default), or restrict to validation/test",
    )
    parser.add_argument("--days", type=int, help="Optionally prepare only the latest N replay days")
    parser.add_argument("--context", type=int, default=96)
    parser.add_argument(
        "--saved-forecasts",
        action="store_true",
        help="Use prepared forecasts instead of fresh daily predictions",
    )
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65534:
        parser.error("Choose an application port between 1024 and 65534")
    if args.days is not None and args.days < 1:
        parser.error("--days must be positive")
    if args.chat and args.llm:
        parser.error("Use --llm alone; --chat is a legacy option")
    llm = args.llm or ("guided" if args.chat == "guided" else "qwen35_2b")
    try:
        chat_config = chat_configuration(llm, args.port + 1, args.api_key_env, args.llm_endpoint)
        from energy_assistant.local_llm import cpu_threads

        threads = cpu_threads(args.llm_threads)
    except ValueError as error:
        parser.error(str(error))
    if args.model:
        from energy_assistant.prepare import ROOT, prepare, source_hashes

        run, args.home, checkpoint, settings = resolve_model(args.model, args.home, args.checkpoint)
        history = args.split == "history"
        replay_split = settings.get("assistant_split", "validation") if history else args.split
        paths = [
            run / "checkpoints" / checkpoint / "heads.pt",
            run / "logs/fedavg/train_settings.txt",
            run / "data_hashes.json",
            Path(settings["path_data"]).parent / "temp_price_newyork.csv",
        ]
        paths += [Path(settings["path_data"]) / f"home_{h}.csv" for h in settings["home_ids"]]
        if settings.get("predictive_features"):
            feature_root = Path(settings["predictive_features"])
            paths += [feature_root / "manifest.json", feature_root / f"home_{args.home}.npz"]
        cache_key = evidence_cache_key(
            paths,
            sources=source_hashes(),
            home=args.home,
            split=args.split,
            days=args.days,
            context=args.context,
        )
        evidence = ROOT / "results" / "home_assistant" / f"home_{args.home}_{cache_key}"
        if not (evidence / "evidence.json").exists():
            if evidence.exists():
                raise ValueError(
                    f"An incomplete preparation exists at {evidence}; inspect it before retrying"
                )
            prepare(
                evidence,
                homes=[args.home],
                days=args.days,
                context=args.context,
                run=run,
                checkpoint=checkpoint,
                split=replay_split,
                history=history,
            )
    else:
        evidence = args.evidence
    if llm == "qwen35_2b":
        from energy_assistant.local_llm import setup

        setup()  # Idempotent; verify cached files and repair missing downloads.
    import uvicorn

    from energy_assistant.web import create_app

    app = create_app(
        evidence,
        live=not args.saved_forecasts,
        local_chat=llm == "qwen35_2b",
        home=args.home,
        chat_port=args.port + 1,
        chat_config=chat_config,
        chat_threads=threads,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
