"""Replay identical actions against the pinned original environment and P2P rules."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from gridpfn.core.dataset import home_data_dir, load_data, share_training_scale
from gridpfn.core.em_strategy import P2P_TRADING, apply_em_strategy, compose_em_strategy
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.model import (
    configure_embedding_device,
    encoder_context_identity,
    heads_from_state,
    precompute_embeddings,
)
from gridpfn.core.predictive_features import PredictiveContext
from gridpfn.core.training_config import HOME_IDS, parse_args
from gridpfn.core.utils.agent_utils import safe_torch_load
from gridpfn.core.utils.run_io import atomic_json, read_logged_settings
from gridpfn.paths import ROOT

REFERENCE = "e34da8b68c646ae3d6c818ad13f2020e6ba06c73"


def reference_source(name, revision):
    """Verify the bundled reference so clean source exports need no old Git history."""
    directory = ROOT / "reference"
    if revision == REFERENCE and (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["revision"] != revision:
            raise ValueError("Bundled reference revision differs")
        source = (directory / f"{name}.py.txt").read_bytes()
        if hashlib.sha256(source).hexdigest() != manifest["sha256"][f"{name}.py"]:
            raise ValueError("Bundled original source hash differs")
        return source
    return subprocess.check_output(
        ["git", "show", f"{revision}:{name}.py"], cwd=ROOT
    )


def reference_module(name, revision):
    source = reference_source(name, revision)
    module = ModuleType(f"reference_{name}")
    exec(compile(source.decode("utf-8-sig"), f"{revision}:{name}.py", "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


CONSTRAINT_OPTIONS = (
    "tou",
    "tou_blocks",
    "dr_limit",
    "dr_penalty",
    "dr_incentive",
    "pv_curtail",
    "export_price",
    "fixed_cost",
    "p2p",
    "p2p_price",
)


def original_defaults(revision):
    source = reference_source("train", revision)
    defaults = {}
    for node in ast.walk(ast.parse(source.decode("utf-8-sig"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            name = node.args[0].value.removeprefix("--")
            if name in CONSTRAINT_OPTIONS:
                defaults[name] = ast.literal_eval(
                    next(k.value for k in node.keywords if k.arg == "default")
                )
    assert set(defaults) == set(CONSTRAINT_OPTIONS)
    return defaults


def audit(revision=REFERENCE):
    defaults = original_defaults(revision)
    configured = vars(parse_args([]))
    assert {key: configured[key] for key in defaults} == defaults
    original, env_hash = reference_module("environment", revision)
    market, market_hash = reference_module("em_strategy", revision)
    bundles = load_data(
        home_data_dir,
        "*.csv",
        choose=[f"home_{h}" for h in HOME_IDS],
        validation_days=14,
        split="validation",
    )
    clients = [SimpleNamespace(train_data=b[0], scaler=b[3]) for b in bundles]
    strategy = {
        "tou": {"enabled": True, "n_blocks": 5},
        "dr_limit": 5.0,
        "dr_penalty": 0.5,
        "dr_incentive": 0.025,
        "pv_curtail": 2.5,
        "export_price": 0.025,
    }
    current_strategy = compose_em_strategy(strategy, clients)
    original_strategy = market.compose_em_strategy(strategy, clients)
    assert current_strategy == original_strategy, "Training-fitted tariff changed"
    rng = np.random.default_rng(20261004)
    days = steps = 0
    largest = scale_error = 0.0
    shared = share_training_scale(bundles)
    # Physical inputs isolate constraints from the intentional inverse-scaling fix.
    for bundle, scaled_bundle in zip(bundles, shared, strict=True):
        train, validation, _, scaler = bundle
        scaled_train, scaled_validation, _, scaled_scaler = scaled_bundle
        for normalized, scaled_day in zip(
            np.concatenate((train, validation)),
            np.concatenate((scaled_train, scaled_validation)),
            strict=True,
        ):
            physical = physical_day(normalized, scaler)
            np.testing.assert_allclose(
                physical, physical_day(scaled_day, scaled_scaler), atol=1e-12, rtol=0
            )
            for policy in ("idle", "maximum", "random"):
                old = original.HOME_ENERGY_MGNT(
                    physical, scaler={"delta_t": scaler["delta_t"]}, fixed_cost=5
                )
                new = HOME_ENERGY_MGNT(
                    physical, scaler={"delta_t": scaler["delta_t"]}, fixed_cost=5, state_dim=17
                )
                scaled = HOME_ENERGY_MGNT(
                    scaled_day, scaler=scaled_scaler, fixed_cost=5, state_dim=17
                )
                market.apply_em_strategy(old, original_strategy)
                apply_em_strategy(new, current_strategy)
                apply_em_strategy(scaled, current_strategy)
                np.testing.assert_allclose(new.reset()[:9], old.reset(), atol=0, rtol=0)
                for _ in range(len(physical)):
                    control = {
                        "idle": [0, 0, 0],
                        "maximum": [20, 20, 20],
                        "random": rng.uniform([-1, -2, -5], [3, 5, 5]),
                    }[policy]
                    action = (
                        int(rng.integers(2)) if policy == "random" else int(policy == "maximum"),
                        control,
                    )
                    a, b, c = old.step(action), new.step(action), scaled.step(action)
                    np.testing.assert_allclose(a[0], b[0][:9], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(a[1:], b[1:], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(a[1:], c[1:], rtol=0, atol=1e-12)
                    scale_error = max(
                        scale_error,
                        float(np.max(np.abs(np.asarray(a[1:]) - np.asarray(c[1:])))),
                    )
                    largest = max(
                        largest, float(np.max(np.abs(np.asarray(a[1:]) - np.asarray(b[1:]))))
                    )
                    for key in (
                        "SoE_BESS",
                        "SoE_EV",
                        "ev_energy_delivered",
                        "wm_completed",
                        "ac_energy_delivered",
                        "indoor_temp",
                        "exported_kwh",
                        "net_load",
                        "power_AC",
                        "power_EV",
                        "power_BESS",
                        "power_WM",
                    ):
                        np.testing.assert_allclose(
                            getattr(old, key), getattr(new, key), rtol=0, atol=1e-12, err_msg=key
                        )
                        np.testing.assert_allclose(
                            getattr(old, key), getattr(scaled, key), rtol=0, atol=1e-12, err_msg=key
                        )
                    assert 0 <= new.SoE_BESS <= 1 and 0 <= new.SoE_EV <= 1
                    assert new.exported_kwh <= new.export_cap_kwh + 1e-6
                    assert new.time_ini_EV <= new.time_hour < new.time_end_EV or new.power_EV == 0
                    steps += 1
                assert new.ac_energy_delivered + 1e-6 >= new.ac_required_energy
                assert new.ev_energy_delivered + 1e-6 >= new.ev_required_energy
                assert not new.wm_required or new.wm_completed
                days += 1
    for _ in range(1000):
        infos = [
            {
                "net_load": float(rng.uniform(-4, 10)),
                "pv_surplus": float(rng.uniform(0, 4)),
                "price": 0.3,
                "export_price": 0.025,
                "delta_t": 1,
            }
            for _ in HOME_IDS
        ]
        for a, b in zip(
            market.P2P_TRADING.compute_adjustments(infos, 0.1),
            P2P_TRADING.compute_adjustments(infos, 0.1),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def method_asts(source):
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        return {node.name: ast.dump(node) for node in cls.body if isinstance(node, ast.FunctionDef)}

    before = method_asts(reference_source("environment", revision).decode("utf-8-sig"))
    after = method_asts((ROOT / "gridpfn/core/environment.py").read_text())
    return {
        "unchanged_environment_methods": [name for name in before if before[name] == after[name]],
        "reference_commit": revision,
        "matched_original_default_settings": defaults,
        "reference_source_sha256": {"environment.py": env_hash, "em_strategy.py": market_hash},
        "home_count": len(HOME_IDS),
        "replayed_home_days": days,
        "matched_transitions": steps,
        "max_reward_difference": largest,
        "shared_scaling": {
            "matched_original_transitions": steps,
            "max_reward_difference": scale_error,
            "note": "Training-only shared extrema preserve physical traces and identical fixed-action dynamics, rewards and original appliance constraints.",
        },
        "matched_p2p_markets": 1000,
        "ac_service": "energy_quota",
        "intentional_protocol_changes": [
            "17-state observation; first 9 coordinates unchanged",
            "July validation excluded from preprocessing and training",
            "Preserve physical held-out extremes instead of clipping to training range; invert constant columns correctly",
        ],
        "inherited_assumptions": [
            "Daily battery reset, no terminal charge equality",
            "Historical AC energy quota can conflict with thermal comfort",
            "EV demand capped at feasible capacity; WM/EV deadlines enforced by simulator",
            "Electrical objective includes DR and P2P credits",
        ],
    }


def physical_day(normalized, scaler):
    result = normalized.copy()
    for column, index in scaler["col_to_scaler_idx"].items():
        if index is not None:
            span = scaler["max"][index] - scaler["min"][index]
            result[:, column] = result[:, column] * (span or 1.0) + scaler["min"][index]
    return result


def reference_observation(env, scaler, step=None, dtype=np.float32):
    """Normalize the original simulator's physical observation; expose its live state."""
    step = env.current_step if step is None else step
    state = env._state_for_step(step)
    if step < env.max_step:
        for state_index, column in ((1, 3), (2, 1), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)):
            index = scaler["col_to_scaler_idx"][column]
            span = scaler["max"][index] - scaler["min"][index]
            state[state_index] = (state[state_index] - scaler["min"][index]) / (span or 1.0)
    live = [
        (env.indoor_temp - 20) / 10,
        env.SoE_EV,
        max(0, env.ev_required_energy - env.ev_energy_delivered) / env.max_capacity_EV,
        float(env.wm_pending),
        float(env.wm_running),
        env.wm_cycle_index / max(len(env.wm_cycle_profile), 1),
        max(0, env.ac_required_energy - env.ac_energy_delivered)
        / (env.max_power_AC * env.max_step * env.delta_t),
        1.0
        if env.export_cap_kwh is None
        else max(0, env.export_cap_kwh - env.exported_kwh) / max(env.export_cap_kwh, env.eps),
    ]
    return np.r_[state, live].astype(dtype)


def policy_evaluation_records(run, checkpoint, split, dates, home_ids, checkpoint_path):
    """Read only the declared split and bind its metrics/trace to the replay."""
    evaluation_path = run / "evaluation" / f"{split}_{checkpoint}.json"
    expected = json.loads(evaluation_path.read_text())
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if expected.get("split") != split or expected.get("dates") != dates:
        raise ValueError("Evaluation split or dates differ from the policy replay")
    if expected.get("checkpoint_sha256") != digest:
        raise ValueError("Evaluation checkpoint differs from the policy replay")
    if [home["home_id"] for home in expected["homes"]] != home_ids:
        raise ValueError("Evaluation home order differs from the policy replay")
    trace_path = evaluation_path.with_name(f"{split}_{checkpoint}_trace.json")
    traced = json.loads(trace_path.read_text()) if trace_path.exists() else expected
    if traced.get("action_records") is not None:
        if (
            traced.get("split", split) != split
            or traced.get("dates") != dates
            or traced.get("checkpoint_sha256") != digest
        ):
            raise ValueError("Action trace split, dates or checkpoint differs")
        if "homes" in traced and [home["home_id"] for home in traced["homes"]] != home_ids:
            raise ValueError("Action trace home order differs from the policy replay")
        if np.shape(traced["action_records"]) != (len(home_ids), len(dates), 24, 4):
            raise ValueError("Action trace must cover every home, day and hourly request")
    return expected, traced.get("action_records")


@torch.no_grad()
def audit_policy(
    run, revision=REFERENCE, gpu=1, allow_tariff=False, checkpoint="best_feasible", split="test"
):
    """Independent replay of the declared split using ORIGINAL dynamics and settlement.

    Does not call the production evaluation loop or its metric aggregation helpers.
    Checkpoint selection must already be complete; this cannot tune the policy.
    Recorded requests avoid accumulating scalar/batched inference roundoff; each
    request is also checked independently at the identical live state.
    """
    if split not in {"test", "validation"}:
        raise ValueError("Unknown policy audit split")
    if checkpoint not in {"best", "best_feasible", "initial", "latest"}:
        raise ValueError("Unknown policy checkpoint")
    if json.loads((run / "status.json").read_text())["state"] != "completed":
        raise ValueError("Policy audit requires a completed run")
    defaults = original_defaults(revision)
    configured = json.loads((run / "run.json").read_text())["settings"]
    if allow_tariff:
        defaults = {
            key: value
            for key, value in defaults.items()
            if key not in ("tou", "tou_blocks", "p2p_price")
        }
    assert {key: configured[key] for key in defaults} == defaults, (
        "Run altered original constraint settings"
    )
    if not allow_tariff and configured.get("grid_prices") is not None:
        raise ValueError("Use an explicit tariff-scenario audit for changed grid prices")
    original, _ = reference_module("environment", revision)
    market, _ = reference_module("em_strategy", revision)
    settings = read_logged_settings(run / "logs/fedavg/train_settings.txt")
    if not settings["em_strategy"].get("ac_energy_quota", True):
        raise ValueError("Thermal-only policies do not use the original constraints")
    checkpoint_path = run / "checkpoints" / checkpoint / "heads.pt"
    payload = safe_torch_load(checkpoint_path, "cpu")
    assert payload["home_ids"] == settings["home_ids"]
    embedding_device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") if settings.get("devices") == ["cpu"] else embedding_device
    configure_embedding_device(
        embedding_device, settings.get("embedding_cache"), settings.get("encoder_context")
    )
    if payload.get("encoder_context_sha256") != encoder_context_identity():
        raise ValueError("Encoder context differs from the selected checkpoint")
    if embedding_device.type == "cuda":
        torch.cuda.set_device(embedding_device)
        torch.cuda.set_per_process_memory_fraction(0.2, gpu)
    bundles = load_data(
        settings["path_data"],
        "*.csv",
        choose=[f"home_{h}" for h in settings["home_ids"]],
        validation_days=settings["validation_days"],
        data_period=settings.get("data_period", "legacy"),
        split=split,
        scaler_mode=settings.get("scaler_mode", "local"),
        grid_prices=settings.get("grid_prices"),
    )
    heads = [heads_from_state(row["actor"], row["critic"], device) for row in payload["clients"]]
    contexts = [
        PredictiveContext(Path(settings["predictive_features"]) / f"home_{home}.npz", b[0], b[3])
        if settings.get("predictive_features")
        else None
        for home, b in zip(settings["home_ids"], bundles, strict=True)
    ]
    for context, row in zip(contexts, payload["clients"], strict=True):
        if context is not None and context.fingerprint != row.get("predictive_context_sha256"):
            raise ValueError("Predictive features differ from the selected checkpoint")
    assert all(actor.state_dim == 17 for actor, _ in heads)
    for context, (actor, critic) in zip(contexts, heads, strict=True):
        width = context.width if context is not None else 0
        if actor.auxiliary_dim != width or critic.auxiliary_dim != width:
            raise ValueError("Predictive context width differs from the selected checkpoint")
    strategy = market.compose_em_strategy(
        settings["em_strategy"], [SimpleNamespace(train_data=b[0], scaler=b[3]) for b in bundles]
    )
    dates = sorted(set.intersection(*(set(b[2]) for b in bundles)))
    if any(b[3].get("eval_split") != split for b in bundles):
        raise ValueError("Loaded data split differs from the policy replay")
    if not dates or any(set(dates) & set(b[3]["train_dates"]) for b in bundles):
        raise ValueError("Policy replay dates must be nonempty and disjoint from training")
    if any(context is not None and not set(dates) <= set(context.dates) for context in contexts):
        raise ValueError("Predictive context does not cover the declared evaluation dates")
    expected, action_records = policy_evaluation_records(
        run, checkpoint, split, dates, settings["home_ids"], checkpoint_path
    )
    request_difference = 0.0
    envs = []
    for bundle, (actor, _) in zip(bundles, heads, strict=True):
        _, values, calendar, scaler = bundle
        home_envs = []
        for date in dates:
            env = original.HOME_ENERGY_MGNT(
                physical_day(values[calendar.index(date)], scaler),
                scaler={"delta_t": scaler["delta_t"]},
                fixed_cost=settings["fixed_cost"],
            )
            market.apply_em_strategy(env, strategy)
            env.reset()
            home_envs.append(env)
        envs.append(home_envs)
        if actor.feature_mode != "raw":
            precompute_embeddings(
                np.array(
                    [
                        reference_observation(env, scaler, t)
                        for env in home_envs
                        for t in range(env.max_step + 1)
                    ]
                ),
                device,
            )
    totals = np.zeros((len(bundles), len(dates), 10))
    peer_totals = np.zeros((len(bundles), len(dates), 2))
    for hour in range(envs[0][0].max_step):
        requests = []
        # Match the production inference batch geometry, independently of its
        # rollout and aggregation code. Scalar GEMM can round differently and
        # small differences accumulate in closed-loop controls.
        for i, (bundle, (actor, critic)) in enumerate(zip(bundles, heads, strict=True)):
            observations = [reference_observation(env, bundle[3]) for env in envs[i]]
            if contexts[i] is not None:
                observations = [
                    contexts[i].augment(state, date, hour)
                    for state, date in zip(observations, dates, strict=True)
                ]
            features = actor.prepare_features(np.asarray(observations))
            control = actor.forward_features(features)
            choice = (
                actor.discrete_features(features)
                if hasattr(actor, "discrete_fc4")
                else critic.forward_features(features, control)
            ).argmax(1)
            requests.append(torch.cat((choice[:, None], control), 1).cpu().numpy())
        for date_index, date in enumerate(dates):
            infos = []
            daily, daily_peers = totals[:, date_index], peer_totals[:, date_index]
            for i in range(len(bundles)):
                env = envs[i][date_index]
                request = requests[i][date_index]
                action = (int(request[0]), request[1:])
                if action_records is not None:
                    saved = action_records[i][date_index][hour]
                    assert action[0] == saved[0], "Policy WM choice differs from trace"
                    np.testing.assert_allclose(action[1], saved[1:], atol=2e-5, rtol=2e-5)
                    request_difference = max(
                        request_difference, float(np.max(np.abs(action[1] - saved[1:])))
                    )
                    action = (int(saved[0]), np.asarray(saved[1:], dtype=np.float32))
                _, electrical, reward, comfort, _ = env.step(action)
                infos.append(
                    {
                        "reward_elec": electrical,
                        "reward_comf": comfort,
                        "net_load": env.net_load,
                        "pv_surplus": env.pv_surplus,
                        "price": env.price,
                        "export_price": env.export_price,
                        "delta_t": env.delta_t,
                    }
                )
                deviation = max(
                    env.temperature_min - env.indoor_temp,
                    env.indoor_temp - env.temperature_max,
                    0.0,
                )
                daily[i, :4] += [
                    reward,
                    -electrical,
                    float(env.temperature_min <= env.indoor_temp <= env.temperature_max)
                    * 100
                    / env.max_step,
                    env.net_load * env.delta_t,
                ]
                daily[i, 4:8] += [
                    (max(0, env.net_load) * env.price - max(0, -env.net_load) * env.export_price)
                    * env.delta_t
                    + env.fixed_cost / 30 / env.max_step,
                    deviation**2,
                    float(deviation > 0) * env.delta_t,
                    deviation * env.delta_t,
                ]
                daily[i, 8] = max(daily[i, 8], deviation)
            if market.P2P_TRADING.is_enabled(settings["p2p_config"]):
                peer_price = market.P2P_TRADING.price(settings["p2p_config"])
                adjustments, imports, exports = market.P2P_TRADING.compute_adjustments(
                    infos, peer_price
                )
                for i in range(len(bundles)):
                    env = envs[i][date_index]
                    daily_peers[i] += np.array([imports[i], exports[i]]) * env.delta_t
                    daily[i, 0] += adjustments[i]
                    daily[i, 1] -= adjustments[i]
                    daily[i, 4] += (
                        imports[i] * (peer_price - env.price)
                        + exports[i] * (env.export_price - peer_price)
                    ) * env.delta_t
                    env.exported_kwh += exports[i] * env.delta_t
    for i, home_envs in enumerate(envs):
        for day, env in enumerate(home_envs):
            totals[i, day, 9] = 100 * float(
                env.ev_energy_delivered + env.eps >= env.ev_required_energy
                and (not env.wm_required or env.wm_completed)
            )
    keys = (
        "reward",
        "elec_cost",
        "comfort_pct",
        "net_demand",
        "energy_bill_without_dr",
        "squared_violation",
        "violation_hours",
        "degree_hours",
        "peak_violation_degrees",
        "task_success_pct",
    )
    observed = np.mean(np.asarray(totals), axis=1)
    reported = np.array([[home[key] for key in keys] for home in expected["homes"]])
    atol, rtol = (2e-8, 1e-7) if action_records is not None else (1e-3, 1e-5)
    np.testing.assert_allclose(observed, reported, atol=atol, rtol=rtol)
    # Check every home-day too: mean agreement must not conceal cancellation.
    reported_days = np.array(
        [[[day[key] for key in keys] for day in home] for home in expected["day_records"]]
    )
    np.testing.assert_allclose(totals, reported_days, atol=atol, rtol=rtol)
    peer_difference = None
    if "p2p_kwh" in expected:
        reported_peers = np.array(
            [
                [[day["p2p_kwh"], day["peer_export_kwh"]] for day in home]
                for home in expected["day_records"]
            ]
        )
        np.testing.assert_allclose(peer_totals, reported_peers, atol=atol, rtol=rtol)
        peer_difference = float(np.max(np.abs(np.asarray(peer_totals) - reported_peers)))
    return {
        "reference_commit": revision,
        "tariff_scenario": allow_tariff,
        "split": split,
        "dates": dates,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_episode": payload["episode"],
        "days": len(dates),
        "homes": len(envs),
        "matched_policy_transitions": len(envs) * len(dates) * envs[0][0].max_step,
        "max_absolute_home_mean_difference": float(np.max(np.abs(observed - reported))),
        "max_absolute_daily_difference": float(np.max(np.abs(np.asarray(totals) - reported_days))),
        "max_absolute_peer_kwh_difference": peer_difference,
        "recorded_requests": action_records is not None,
        "max_absolute_policy_request_difference": request_difference
        if action_records is not None
        else None,
        "metrics": list(keys),
        "mean": dict(zip(keys, observed.mean(0).tolist(), strict=True)),
    }


def load_scheduling_bank(run, study, clients, name):
    if name == "economic_selected":
        name = study["training_cost_selection"]["winner"]
    collection = []
    for home, client, row in zip(study["home_ids"], clients, study["forecasts"], strict=True):

        def table(kind):
            return np.load(run / f"forecast_{home}_{kind}.npz")["table"]

        if name == "selected":
            values = np.stack(
                [table(kind)[..., j] for j, kind in enumerate(row["training_selected_kind"])],
                axis=-1,
            )
        elif name == "blend":
            values = (table("tabpfn") + table("seasonal")) / 2
        else:
            values = table(name)
        collection.append(dict(zip(client.test_dates, values, strict=True)))
    return collection


def audit_scheduling(
    run, policies=("tabpfn_local", "trees_local", "economic_selected_local"), revision=REFERENCE
):
    from gridpfn.core.economic_control import EconomicCoordinator
    from gridpfn.legacy.control_benchmark import load_clients

    study = json.loads((run / "study_results.json").read_text())
    if json.loads((run / "status.json").read_text())["state"] != "completed":
        raise ValueError("Audit requires completed evaluation")
    clients = load_clients(study["home_ids"], split=study["split"])
    strategy = study["settings"]["strategy"]
    original, env_hash = reference_module("environment", revision)
    market, market_hash = reference_module("em_strategy", revision)
    audits = {}
    keys = ("reward", "elec_cost", "energy_bill_without_dr", "comfort_pct", "import", "export")
    for policy in policies:
        kind, mode = policy.rsplit("_", 1)
        expected = study["controllers"][policy]
        controller = EconomicCoordinator(
            clients, strategy, load_scheduling_bank(run, study, clients, kind), peers=mode == "peer"
        )
        records = [[] for _ in clients]
        balance_error = 0.0
        for day in expected["dates"]:
            envs = []
            for client in clients:
                env = original.HOME_ENERGY_MGNT(
                    physical_day(client.test_data[client.test_dates.index(day)], client.scaler),
                    scaler={"delta_t": 1},
                    fixed_cost=client.fixed_cost,
                )
                market.apply_em_strategy(env, strategy)
                env.reset()
                envs.append(env)
            controller.start_day(day)
            totals = np.zeros((len(clients), len(keys)))
            for _ in range(24):
                states = [
                    reference_observation(env, c.scaler, dtype=np.float64)
                    for env, c in zip(envs, clients, strict=True)
                ]
                actions = controller.dispatch(states)
                infos = []
                for env, action in zip(envs, actions, strict=True):
                    _, electrical, reward, discomfort, _ = env.step(action)
                    infos.append(
                        dict(
                            reward_elec=electrical,
                            reward=reward,
                            reward_comf=discomfort,
                            net_load=env.net_load,
                            pv_surplus=env.pv_surplus,
                            price=env.price,
                            export_price=env.export_price,
                            delta_t=env.delta_t,
                        )
                    )
                adjustments, imports, exports = market.P2P_TRADING.compute_adjustments(infos, 0.1)
                for i, (env, info) in enumerate(zip(envs, infos, strict=True)):
                    grid_in = max(env.net_load, 0) - imports[i]
                    grid_out = max(-env.net_load, 0) - exports[i]
                    bill = (
                        grid_in * env.price
                        - grid_out * env.export_price
                        + 0.1 * (imports[i] - exports[i])
                        + env.fixed_cost / 30 / 24
                    )
                    totals[i] += [
                        info["reward"] + adjustments[i],
                        -info["reward_elec"] - adjustments[i],
                        bill,
                        100 / 24 * (env.temperature_min <= env.indoor_temp <= env.temperature_max),
                        grid_in,
                        grid_out,
                    ]
                    raw = (
                        env.fixed_load
                        + env.power_AC
                        + env.power_EV
                        + env.power_WM
                        + env.power_BESS
                        - env.pv_generation
                    )
                    curtailment = max(0, env.net_load - raw)
                    balance_error = max(
                        balance_error,
                        abs(grid_in + imports[i] - grid_out - exports[i] - raw - curtailment),
                    )
                    env.exported_kwh += exports[i]
            for i, env in enumerate(envs):
                if env.ev_energy_delivered + 1e-6 < env.ev_required_energy or (
                    env.wm_required and not env.wm_completed
                ):
                    raise AssertionError("Original appliance deadline failed")
                if env.ac_energy_delivered + 1e-6 < env.ac_required_energy:
                    raise AssertionError("Original AC quota failed")
                records[i].append(totals[i])
        reported = np.array(
            [[[row[key] for key in keys] for row in days] for days in expected["day_records"]]
        )
        np.testing.assert_allclose(records, reported, atol=1e-5, rtol=1e-6)
        audits[policy] = {
            "matched_transitions": len(clients) * len(expected["dates"]) * 24,
            "max_daily_difference": float(np.max(np.abs(np.asarray(records) - reported))),
            "max_energy_balance_error_kw": balance_error,
            "solver_failures": controller.failures,
            "mean": dict(zip(keys, np.mean(records, axis=(0, 1)).tolist(), strict=True)),
        }
    return {
        "reference_commit": revision,
        "reference_source_sha256": {"environment.py": env_hash, "em_strategy.py": market_hash},
        "metrics": list(keys),
        "policies": audits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/constraint_audit.json"))
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument(
        "--policy_run",
        type=Path,
        help="Also audit the completed selected policy against original dynamics.",
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--checkpoint",
        choices=("best", "best_feasible", "initial", "latest"),
        default="best_feasible",
    )
    parser.add_argument(
        "--allow-tariff",
        action="store_true",
        help="Policy audit may use declared grid/peer tariffs; original physics stays required.",
    )
    parser.add_argument(
        "--scheduling_run", type=Path, help="Also independently replay a completed scheduling study"
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    result = audit(args.reference)
    if args.policy_run:
        result["policy_replay"] = audit_policy(
            args.policy_run.resolve(),
            args.reference,
            args.gpu,
            allow_tariff=args.allow_tariff,
            checkpoint=args.checkpoint,
            split=args.split,
        )
    if args.scheduling_run:
        result["scheduling_replay"] = audit_scheduling(
            args.scheduling_run.resolve(), revision=args.reference
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
