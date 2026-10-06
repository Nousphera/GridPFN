import argparse
import glob
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gridpfn.core.dataset import home_data_dir, load_data
from gridpfn.core.em_strategy import P2P_TRADING, apply_em_strategy, compose_em_strategy
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.model import heads_from_state, precompute_embeddings
from gridpfn.core.utils.agent_utils import safe_torch_load
from gridpfn.core.utils.plot_utils import CUSTOM_NAMES
from gridpfn.core.utils.plots import F_SIZE, LEGEND_SIZE, configure_plot_style
from gridpfn.core.utils.run_io import read_logged_settings

configure_plot_style()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-9
SCHEDULE_OUTPUT_FIELDS = (
    "time",
    "wm_power",
    "ac_power",
    "ev_power",
    "bess_power",
    "grid_import",
    "grid_export",
    "p2p_trade",
    "indoor_temp",
)


@dataclass
class ReplaySettings:
    use_p2p: bool = False
    p2p_config: dict | None = None
    em_strategy: dict | None = None
    home_ids: list[int] | None = None
    validation_days: int = 0
    data_period: str = "legacy"
    scaler_mode: str = "local"
    grid_prices: list[float] | None = None


def parse_train_settings(settings_path):
    settings = ReplaySettings()
    values = read_logged_settings(settings_path)
    settings.validation_days = int(values.get("validation_days", 0))
    settings.data_period = values.get("data_period", "legacy")
    settings.scaler_mode = values.get("scaler_mode", "local")
    settings.grid_prices = values.get("grid_prices")
    settings.use_p2p = str(values.get("use_p2p", False)).lower() in {"true", "1", "yes", "y"}
    for name in ("p2p_config", "em_strategy"):
        if values.get(name) is not None:
            setattr(settings, name, dict(values[name]))
    if values.get("home_ids") is not None:
        settings.home_ids = [int(home) for home in values["home_ids"]]

    if settings.p2p_config is not None:
        settings.p2p_config.setdefault("enabled", settings.use_p2p)
        if settings.p2p_config.get("enabled") and settings.p2p_config.get("price") is not None:
            settings.use_p2p = True
    return settings


@dataclass
class HomeBundle:
    actual_home_id: int
    train_data: np.ndarray
    test_data: np.ndarray
    test_dates: list
    scaler: dict


@dataclass
class _ClientProxy:
    train_data: np.ndarray
    scaler: dict


def load_home_bundles(
    data_path,
    num_homes=None,
    actual_home_ids=None,
    validation_days=0,
    scaler_mode="local",
    scaler_home_ids=None,
    grid_prices=None,
    data_period="legacy",
):
    data_path_sep = os.path.join(data_path, "")
    home_ids = [
        int(os.path.splitext(os.path.basename(file_path))[0].removeprefix("home_"))
        for file_path in glob.glob(os.path.join(data_path, "home_*.csv"))
    ]
    home_ids.sort()
    if actual_home_ids is not None:
        home_ids = [int(home_id) for home_id in actual_home_ids]
    elif num_homes is not None:
        home_ids = home_ids[: max(1, int(num_homes))]

    bundles = {}
    cohort = scaler_home_ids if scaler_mode == "shared" and scaler_home_ids else home_ids
    data = load_data(
        data_path_sep,
        "*.csv",
        choose=[f"home_{h}" for h in cohort],
        validation_days=validation_days,
        data_period=data_period,
        scaler_mode=scaler_mode,
        grid_prices=grid_prices,
    )
    by_home = dict(zip(cohort, data, strict=True))
    for idx, actual_home_id in enumerate(home_ids, start=1):
        train_data, test_data, test_dates, scaler = by_home[actual_home_id]
        bundles[idx] = HomeBundle(
            actual_home_id=actual_home_id,
            train_data=train_data,
            test_data=test_data,
            test_dates=test_dates,
            scaler=scaler,
        )
    return bundles


def load_actor_critic(actor_path, critic_path):
    actor_state = safe_torch_load(actor_path, DEVICE)
    critic_state = safe_torch_load(critic_path, DEVICE)
    return heads_from_state(actor_state, critic_state, DEVICE)


def resolve_model_paths(logs_root, model_name, home_id):
    base = os.path.join(logs_root, model_name, f"home_{home_id}")
    return os.path.join(base, "actor.pt"), os.path.join(base, "critic.pt")


def detect_models(logs_root):
    available = []
    for entry in sorted(os.scandir(logs_root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        model = entry.name
        if model not in CUSTOM_NAMES:
            continue
        actor_path, critic_path = resolve_model_paths(logs_root, model, 1)
        if os.path.isfile(actor_path) and os.path.isfile(critic_path):
            available.append(model)
    return available


def detect_checkpoint_homes(logs_root, model_name):
    checkpoint_homes = set()
    pattern = os.path.join(logs_root, model_name, "home_*")
    for home_dir in glob.glob(pattern):
        home_label = os.path.basename(home_dir).removeprefix("home_")
        if not home_label.isdigit():
            continue
        actor_path, critic_path = resolve_model_paths(logs_root, model_name, int(home_label))
        if os.path.isfile(actor_path) and os.path.isfile(critic_path):
            checkpoint_homes.add(int(home_label))
    return checkpoint_homes


def resolve_trained_home_ids(models, logs_root, path_settings=None):
    settings_home_ids = []
    common_checkpoint_homes = None

    for model_name in models:
        settings_path = path_settings or os.path.join(logs_root, model_name, "train_settings.txt")
        model_home_ids = parse_train_settings(settings_path).home_ids
        if model_home_ids:
            settings_home_ids.append((model_name, tuple(model_home_ids)))

        checkpoint_homes = detect_checkpoint_homes(logs_root, model_name)
        if checkpoint_homes is not None:
            common_checkpoint_homes = (
                checkpoint_homes
                if common_checkpoint_homes is None
                else common_checkpoint_homes.intersection(checkpoint_homes)
            )

    if settings_home_ids:
        expected = settings_home_ids[0][1]
        trained_home_ids = list(expected)
    else:
        trained_home_ids = None

    if common_checkpoint_homes is None:
        return trained_home_ids, None

    checkpoint_count = 0
    while checkpoint_count + 1 in common_checkpoint_homes:
        checkpoint_count += 1
    if trained_home_ids is not None:
        trained_home_ids = trained_home_ids[:checkpoint_count]
    return trained_home_ids, checkpoint_count


def replay_day(
    model_name, day_indices, home_bundles, logs_root, strategy, p2p_config, federated_p2p_models
):
    agents = {}

    for home_id in home_bundles:
        actor_path, critic_path = resolve_model_paths(logs_root, model_name, home_id)
        agents[home_id] = load_actor_critic(actor_path, critic_path)

    envs = {}
    states = {}
    done_flags = {}
    trace_fields = (
        "time",
        "price",
        "wm_power",
        "ac_power",
        "ev_power",
        "bess_power",
        "fixed_load",
        "grid_import",
        "grid_export",
        "p2p_trade",
        "indoor_temp",
        "outdoor_temperature",
    )
    traces = {home_id: {field: [] for field in trace_fields} for home_id in home_bundles}

    for home_id, bundle in home_bundles.items():
        env = HOME_ENERGY_MGNT(
            bundle.test_data[day_indices[home_id]],
            scaler=bundle.scaler,
            state_dim=agents[home_id][0].state_dim,
        )
        if strategy:
            apply_em_strategy(env, strategy)
        state = env.reset()

        static_states = np.asarray([env._state_for_step(step) for step in range(env.max_step + 1)])
        if agents[home_id][0].feature_mode != "raw":
            precompute_embeddings(static_states, DEVICE)

        envs[home_id] = env
        states[home_id] = state
        done_flags[home_id] = False

    use_p2p = (
        model_name in federated_p2p_models
        and P2P_TRADING.is_enabled(p2p_config)
        and len(home_bundles) > 1
    )
    p2p_price = P2P_TRADING.price(p2p_config) if use_p2p else None

    while not all(done_flags.values()):
        step_infos = []
        step_order = []

        for home_id in sorted(home_bundles):
            if done_flags[home_id]:
                continue

            env = envs[home_id]
            state = states[home_id]
            time_hour = env.current_step * env.delta_t

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                actor, critic = agents[home_id]
                continuous_action = actor(state_tensor)
                q_values = (
                    actor.discrete_features(actor.prepare_features(state_tensor))
                    if hasattr(actor, "discrete_fc4")
                    else critic(state_tensor, continuous_action)
                )
                discrete_action = q_values.argmax(1).item()
                continuous_action = continuous_action.squeeze(0).cpu().numpy()

            next_state, _, _, _, done = env.step((discrete_action, continuous_action))

            net_load = float(env.net_load)
            gross_import = max(0.0, net_load)
            gross_export = max(0.0, -net_load)

            trace = traces[home_id]
            trace["time"].append(time_hour)
            trace["price"].append(float(env.price))
            trace["wm_power"].append(float(env.power_WM))
            trace["ac_power"].append(float(env.power_AC))
            trace["ev_power"].append(float(env.power_EV))
            trace["bess_power"].append(float(env.power_BESS))
            trace["fixed_load"].append(float(env.fixed_load))
            trace["grid_import"].append(gross_import)
            trace["grid_export"].append(gross_export)
            trace["p2p_trade"].append(0.0)
            trace["indoor_temp"].append(float(env.expected_temp))
            trace["outdoor_temperature"].append(float(env.outdoor_temp))

            step_infos.append(
                {
                    "net_load": net_load,
                    "pv_surplus": float(env.pv_surplus),
                    "price": float(env.price),
                    "export_price": float(env.export_price),
                    "delta_t": env.delta_t,
                }
            )
            step_order.append(home_id)

            states[home_id] = next_state
            done_flags[home_id] = done

        if use_p2p:
            _, p2p_imports, p2p_exports = P2P_TRADING.compute_adjustments(step_infos, p2p_price)

            for i, home_id in enumerate(step_order):
                trace = traces[home_id]
                p2p_import = float(p2p_imports[i])
                p2p_export = float(p2p_exports[i])
                if p2p_export > 0:
                    envs[home_id].exported_kwh += p2p_export * envs[home_id].delta_t
                states[home_id] = envs[home_id]._state_for_step(envs[home_id].current_step)
                trace["p2p_trade"][-1] = p2p_import + p2p_export
                trace["grid_import"][-1] = max(0.0, trace["grid_import"][-1] - p2p_import)
                trace["grid_export"][-1] = max(0.0, trace["grid_export"][-1] - p2p_export)

    for home_id, trace in traces.items():
        for key in trace:
            trace[key] = np.asarray(trace[key], dtype=float)
        trace["temperature_min"] = float(envs[home_id].temperature_min)
        trace["temperature_max"] = float(envs[home_id].temperature_max)
    return traces


def save_schedule_data(trace, output_path):
    # Input data such as price and fixed load remain in the source home CSV.
    schedule_data = pd.DataFrame({field: trace[field] for field in SCHEDULE_OUTPUT_FIELDS})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    schedule_data.to_csv(output_path, index=False, float_format="%.4f")


def plot_schedule(trace, model_name, home_id, day, output_path):
    times = trace["time"]
    bar_times = times.astype(float)
    bar_width = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    wm_on = (trace["wm_power"] > 0.0).astype(float)
    ac_load = trace["ac_power"]
    ac_on = (ac_load > 0.0).astype(float)
    ev_on = (trace["ev_power"] > 0.0).astype(float)
    bess_charge_on = (trace["bess_power"] > EPS).astype(float)
    bess_discharge_on = (trace["bess_power"] < -EPS).astype(float)

    bess_charge = np.clip(trace["bess_power"], 0.0, None)
    bess_discharge = np.clip(trace["bess_power"], None, 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    rows = [
        ("BESS ch.", bess_charge_on, "#B279A2"),
        ("BESS dis.", bess_discharge_on, "#E45756"),
        ("EV", ev_on, "#54A24B"),
        ("AC", ac_on, "#429FCA"),
        ("WM", wm_on, "#F58518"),
    ]
    for idx, (_, on_values, color) in enumerate(rows):
        ax.bar(
            bar_times,
            0.8 * on_values,
            width=bar_width,
            align="edge",
            bottom=idx + 0.1,
            color=color,
            alpha=0.9,
        )
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([x[0] for x in rows])
    ax.invert_yaxis()
    # ax.set_ylabel("Appliance")
    ax.set_title("Appliance operation time")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    bottom = np.zeros_like(times)
    ax.bar(
        bar_times,
        trace["fixed_load"],
        width=bar_width,
        align="edge",
        bottom=bottom,
        color="#7F7F7F",
        label="Fixed",
    )
    bottom += trace["fixed_load"]
    ax.bar(
        bar_times,
        trace["wm_power"],
        width=bar_width,
        align="edge",
        bottom=bottom,
        color="#F58518",
        label="WM",
    )
    bottom += trace["wm_power"]
    ax.bar(
        bar_times,
        ac_load,
        width=bar_width,
        align="edge",
        bottom=bottom,
        color="#429FCA",
        label="AC",
    )
    bottom += ac_load
    ax.bar(
        bar_times,
        trace["ev_power"],
        width=bar_width,
        align="edge",
        bottom=bottom,
        color="#54A24B",
        label="EV",
    )
    bottom += trace["ev_power"]
    ax.bar(
        bar_times,
        bess_charge,
        width=bar_width,
        align="edge",
        bottom=bottom,
        color="#B279A2",
        label="BESS ch.",
    )
    ax.bar(
        bar_times, bess_discharge, width=bar_width, align="edge", color="#E45756", label="BESS dis."
    )
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power per appliance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=LEGEND_SIZE, columnspacing=0.8, loc="upper right")

    ax = axes[2]
    ax.plot(times, trace["indoor_temp"], label="Indoor")
    ax.plot(times, trace["outdoor_temperature"], label="Outdoor")
    ax.axhspan(
        trace["temperature_min"],
        trace["temperature_max"],
        color="tab:green",
        alpha=0.1,
        label="Comfort band",
    )
    ax.set_ylabel(r"Temp ($^\circ\mathrm{C}$)")
    ax.set_title("Home temperature")
    ax.legend(ncol=3, fontsize=LEGEND_SIZE, columnspacing=0.8)
    ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (h)")
    axes[-1].set_xticks(np.arange(0, 25, 2))
    axes[-1].set_xlim(0.0, 24.0)

    model_key = model_name.strip().lower()
    model_display_name = CUSTOM_NAMES.get(model_key, model_key.replace("_", " ").title())
    fig.suptitle(f"{model_display_name} | Home {home_id} | {day}", fontsize=F_SIZE - 1)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)


def plot_schedules(
    day,
    path_data,
    path_logs,
    path_output,
    home_id=1,
    models=None,
    path_settings=None,
    num_homes=None,
    home_ids=None,
    p2p_price=None,
    disable_p2p=False,
    fed_p2p_models=None,
):
    federated_p2p_models = {model.strip().lower() for model in (fed_p2p_models or [])}
    available_models = detect_models(path_logs)
    if models is None:
        models = available_models
    else:
        models = list(
            dict.fromkeys(
                model.strip().lower()
                for value in models
                for model in str(value).split(",")
                if model.strip()
            )
        )

    trained_home_ids, checkpoint_count = resolve_trained_home_ids(models, path_logs, path_settings)
    if home_ids is None and trained_home_ids is not None:
        home_ids = trained_home_ids[:num_homes] if num_homes is not None else trained_home_ids
        num_homes = None
    elif home_ids is None and checkpoint_count is not None:
        num_homes = min(num_homes, checkpoint_count) if num_homes is not None else checkpoint_count

    split_settings = parse_train_settings(os.path.join(path_logs, models[0], "train_settings.txt"))
    home_bundles = load_home_bundles(
        path_data,
        num_homes=num_homes,
        actual_home_ids=home_ids,
        validation_days=split_settings.validation_days,
        data_period=split_settings.data_period,
        scaler_mode=split_settings.scaler_mode,
        scaler_home_ids=split_settings.home_ids,
        grid_prices=split_settings.grid_prices,
    )
    strategy_clients = [
        _ClientProxy(train_data=bundle.train_data, scaler=bundle.scaler)
        for _, bundle in sorted(home_bundles.items())
    ]
    target_day = pd.to_datetime(day).date()
    day_indices = {
        home_id: [pd.to_datetime(value).date() for value in bundle.test_dates].index(target_day)
        for home_id, bundle in home_bundles.items()
    }
    day_tag = target_day.strftime("%Y%m%d")
    os.makedirs(path_output, exist_ok=True)

    for model_name in models:
        settings_path = path_settings or os.path.join(path_logs, model_name, "train_settings.txt")
        settings = parse_train_settings(settings_path)
        strategy = compose_em_strategy(settings.em_strategy, strategy_clients)

        if disable_p2p:
            p2p_config = {"enabled": False}
        else:
            p2p_config = (
                dict(settings.p2p_config) if settings.use_p2p and settings.p2p_config else None
            )
            if p2p_price is not None:
                p2p_config = {"enabled": True, "price": float(p2p_price)}

        traces = replay_day(
            model_name,
            day_indices,
            home_bundles,
            path_logs,
            strategy,
            p2p_config,
            federated_p2p_models,
        )
        output_stem = os.path.join(path_output, f"schedule_{model_name}_home_{home_id}_{day_tag}")
        pdf_path = f"{output_stem}.pdf"
        csv_path = f"{output_stem}.csv"
        save_schedule_data(traces[home_id], csv_path)
        plot_schedule(traces[home_id], model_name, home_id, day, pdf_path)
        print(f"Saved: {pdf_path}")
        print(f"Saved: {csv_path}")


def main(args):
    plot_schedules(
        day=args.day,
        home_id=args.home_id,
        path_data=args.path_data,
        path_logs=args.path_logs,
        path_output=args.path_output,
        models=args.run_models,
        path_settings=args.path_settings,
        num_homes=args.num_homes,
        home_ids=args.home_ids,
        p2p_price=args.p2p_price,
        disable_p2p=args.disable_p2p,
        fed_p2p_models=args.fed_p2p_models,
    )


if __name__ == "__main__":
    path_train = "eval_seed_6/train_sparsity_75"

    parser = argparse.ArgumentParser(
        description="Plot daily appliance schedules and energy flows by model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--day", default="2019-08-01", help="Day to plot schedule")
    parser.add_argument("--home_id", type=int, default=1, help="Home index to plot.")
    parser.add_argument(
        "--run_models", nargs="+", help="Models to plot; detects available checkpoints by default."
    )
    parser.add_argument("--path_data", default=str(home_data_dir), help="Home CSV directory.")
    parser.add_argument(
        "--path_logs", default=f"{path_train}/logs", help="Checkpoint and settings directory."
    )
    parser.add_argument("--path_settings", help="Training settings override.")
    parser.add_argument("--num_homes", type=int, help="Maximum trained homes to replay.")
    parser.add_argument("--home_ids", nargs="+", type=int, help="Dataset home IDs.")
    parser.add_argument("--p2p_price", type=float, help="P2P price override ($/kWh).")
    parser.add_argument("--disable_p2p", action="store_true", help="Disable P2P replay.")
    parser.add_argument("--fed_p2p_models", nargs="+", help="P2P-capable models.")
    parser.add_argument(
        "--path_output", default=f"{path_train}/schedules", help="Schedule PDF directory."
    )
    main(parser.parse_args())
