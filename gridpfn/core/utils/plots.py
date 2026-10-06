import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from gridpfn.core.utils.plot_utils import CUSTOM_NAMES, MODEL_COLORS, order_models, save_figure_pdf

F_SIZE = 18
LEGEND_SIZE = F_SIZE - 4


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "cmr10",
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": F_SIZE,
            "lines.linewidth": 2,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.labelcolor": "black",
            "axes.labelsize": F_SIZE,
            "axes.titlesize": F_SIZE,
            "xtick.labelsize": F_SIZE,
            "ytick.labelsize": F_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


configure_plot_style()


def _home_subplots(num_homes):
    columns = min(4, num_homes)
    rows = (num_homes + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 10 / 3 * rows), squeeze=False)
    return fig, axes.ravel()


def extract_metric_data(model_names, num_homes, window, metric_col, logs_base="train_test/logs"):
    data_map = {model: {} for model in model_names}
    for model in model_names:
        for hid in range(1, num_homes + 1):
            csv_path = os.path.join(logs_base, model, f"home_{hid}", "train_logs.csv")
            df = pd.read_csv(csv_path)
            metric_data = df[["episode", metric_col]].copy()
            metric_data[metric_col] = metric_data[metric_col].astype(float).rolling(window).mean()
            data_map[model][hid] = metric_data.dropna(subset=[metric_col])
    return data_map


def render_plot(data_map, model_names, num_homes, metric_col, y_label, save_path):
    fig, axes = _home_subplots(num_homes)
    handles, labels = [], []
    ordered_models = order_models(model_names)

    for hid in range(1, num_homes + 1):
        ax = axes[hid - 1]
        for model in ordered_models:
            if hid in data_map[model]:
                df = data_map[model][hid]
                series = df[metric_col]
                line = ax.plot(df["episode"], series, color=MODEL_COLORS.get(model, "black"))[0]
                if hid == 1:
                    handles.append(line)
                    labels.append(CUSTOM_NAMES.get(model, model))
        ax.set_title(f"Home {hid}")
        ax.set_xlabel("Episodes")
        ax.set_ylabel(y_label)
        ax.grid(True)

    for ax in axes[num_homes:]:
        ax.axis("off")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.969])
    if save_path:
        save_figure_pdf(save_path, fig=fig)
    plt.close(fig)


def plot_training_metric(
    metric, model_names, num_homes=10, window=1, save_path=None, logs_base="train_test/logs"
):
    config = {
        "reward": ("Reward", "train_test/results/rewards_plots_all_homes.pdf"),
        "actor_loss": ("Actor loss", "train_test/results/loss_plots_all_homes_actor.pdf"),
        "critic_loss": ("Critic loss", "train_test/results/loss_plots_all_homes_critic.pdf"),
    }
    label, default_path = config[metric]
    data = extract_metric_data(model_names, num_homes, window, metric, logs_base)
    render_plot(data, model_names, num_homes, metric, label, save_path or default_path)


def convergence_series(records, metric, kind, window=1, home_id=None):
    """Smooth each home's training curve, then compute the cross-home mean/SD.

    Evaluation points first average the same test dates within each home. The band
    is population SD across homes (ddof=0), not across dates, seeds, or episodes.
    """
    rows = {}
    for record in records:
        if record.get("kind") == kind:
            rows[record["episode"]] = {
                home["home_id"]: home.get(metric)
                for home in record["homes"]
                if home_id is None or home["home_id"] == home_id
            }
    if not rows:
        return np.array([]), np.array([]), np.array([])
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index().astype(float)
    if kind == "train":
        frame = frame.rolling(window, min_periods=1).mean()
    frame = frame.dropna(how="all")
    return (
        frame.index.to_numpy(),
        frame.mean(axis=1).to_numpy(),
        frame.std(axis=1, ddof=0).to_numpy(),
    )


def plot_live_convergence(records, window=10, home_id=None):
    """Live extension of the evaluation plots; caller saves/closes the figure."""
    panels = (
        ("actor_loss", "Actor objective / value", "train objective / eval -max Q"),
        ("critic_loss", "Critic loss", "Train TD objective / eval MSE"),
        ("reward", "Daily reward", "Reward"),
        ("task_success_pct", "EV and WM task success", "Successful home-days (%)"),
        ("elec_cost", "Electrical objective per day", "$ incl. DR/P2P"),
        ("net_demand", "Average net consumption per day", "Net energy (kWh)"),
        ("comfort", "Discomfort score per day", "Discomfort score"),
        ("comfort_pct", "Temperature within comfort bounds", "Timesteps (%)"),
    )
    if any(r.get("oracle_regret") is not None for r in records):
        panels = list(panels)
        panels[3] = ("oracle_regret", "Gap to perfect-foresight oracle", "Objective / home-day")
        panels[5] = ("excess_squared_violation", "Excess thermal violation", "Squared C / home-day")
        panels[6] = ("q_return_rmse", "Value calibration", "V/Q vs observed returns RMSE")
    if any(r.get("learning_rule") == "ppo" for r in records):
        panels = list(panels)
        panels[0] = ("actor_loss", "Actor objective", "Train PPO / eval categorical NLL")
        panels[1] = ("critic_loss", "State-value loss", "Train return MSE / eval V-TD MSE")
        if any(r.get("kind") == "ppo_update" for r in records):
            panels[6] = (
                "value_explained_variance",
                "Stochastic value fit",
                "Explained variance / same rollout",
            )
    # Reuse the fonts, labels and FedAvg color from the existing evaluation plots.
    with plt.rc_context():
        configure_plot_style()
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )
        fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
        for ax, (metric, title, label) in zip(axes.flat, panels):
            for kind, name, color in (
                ("train", "Train", MODEL_COLORS["fedavg"]),
                (
                    "eval",
                    "Validation"
                    if any(r.get("split") == "validation" for r in records)
                    else "Test",
                    "#d55e00",
                ),
            ):
                source_kind = (
                    ("ppo_update" if kind == "train" else "none")
                    if metric == "value_explained_variance"
                    else kind
                )
                x, mean, sd = convergence_series(records, metric, source_kind, window, home_id)
                if not len(x):
                    continue
                if metric == "comfort":
                    mean = -mean  # Same sign convention as plot_comparison_metrics.
                ax.plot(
                    x,
                    mean,
                    color=color,
                    label=name,
                    marker="o" if kind == "eval" else None,
                    markersize=3,
                )
                if home_id is None:
                    lower, upper = mean - sd, mean + sd
                    if metric.endswith("_pct"):
                        lower, upper = np.maximum(0, lower), np.minimum(100, upper)
                    elif metric in {"critic_loss", "comfort"}:
                        lower = np.maximum(0, lower)
                    ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
            ax.set(title=title, xlabel="Completed training episodes", ylabel=label)
            ax.grid(True, alpha=0.25)
            if ax.lines:
                ax.legend(loc="best")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Waiting for evaluation"
                    if metric not in {"actor_loss", "critic_loss", "reward"}
                    else "Waiting for training",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                )
            if metric.endswith("_pct"):
                ax.set_ylim(0, 100)
            elif metric in {"critic_loss", "comfort"}:
                ax.set_ylim(bottom=0)
        scope = "Mean +/- 1 SD across homes" if home_id is None else f"Home {home_id}"
        fig.suptitle(
            f"FedAvg convergence | {scope}\n"
            f"Train: {window}-episode mean per home; evaluation: fixed dates, no smoothing",
            fontsize=16,
        )
    return fig


def plot_comparison_metrics(results, model_names, save_path="train_test/results/"):
    os.makedirs(save_path, exist_ok=True)

    metrics_to_plot = ["elec_cost", "net_demand", "comfort"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    metric_titles = {
        "elec_cost": "Average electricity cost per day",
        "net_demand": "Average net consumption per day",
        "comfort": "Discomfort score per day",
    }
    metric_ylabels = {
        "elec_cost": "Cost ($)",
        "net_demand": "Net energy (kWh)",
        "comfort": "Discomfort score",
    }

    homes = sorted(results[model_names[0]])
    n = len(homes)
    x = np.arange(n)
    ordered_models = order_models(model_names)
    width = min(0.2, 0.8 / len(ordered_models))

    for idx, metric_key in enumerate(metrics_to_plot):
        ax = axes[idx]
        for model_idx, model_name in enumerate(ordered_models):
            values = [
                results[model_name][h]["mean"][metric_key] if h in results[model_name] else np.nan
                for h in homes
            ]
            if metric_key == "comfort":
                values = [-value for value in values]
            label = CUSTOM_NAMES.get(model_name, model_name)
            ax.bar(
                x + model_idx * width,
                values,
                width,
                label=label,
                color=MODEL_COLORS.get(model_name, "gray"),
                alpha=0.8,
            )

        ax.set_xlabel("Home ID")
        ax.set_ylabel(metric_ylabels[metric_key])
        ax.set_title(metric_titles[metric_key])
        ax.set_xticks(x + width * (len(ordered_models) - 1) / 2)
        ax.set_xticklabels(range(1, n + 1))
        ax.grid(True, alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        fontsize=LEGEND_SIZE,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.944])

    save_figure_pdf(os.path.join(save_path, "model_comparison_metrics.pdf"), fig=fig)
    plt.close(fig)
    print("Saved: model_comparison_metrics.pdf")


def plot_radar_chart(
    results, model_names, save_path="train_test/results/", logs_base="train_test/logs"
):
    os.makedirs(save_path, exist_ok=True)

    metrics_to_plot = ["elec_cost", "comfort", "comm_cost", "net_demand"]
    metric_labels = {
        "elec_cost": r"Elec. cost ($\downarrow$)",
        "comfort": r"Comfort ($\uparrow$)",
        "comm_cost": r"Comm. cost ($\downarrow$)",
        "net_demand": r"Net demand ($\downarrow$)",
    }
    num_vars = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ordered_models = order_models(model_names)

    comm_totals = {
        model_name: _read_comm_metrics(model_name, "comm_train_time.csv", logs_base)["total (MB)"]
        for model_name in ordered_models
    }

    global_min_max = {}
    for metric in metrics_to_plot:
        all_vals = []
        for model_name in ordered_models:
            if metric == "comm_cost":
                all_vals.append(comm_totals[model_name])
                continue
            all_vals.extend(
                results[model_name][home_id]["mean"][metric]
                for home_id in sorted(results[model_name])
            )
        global_min_max[metric] = (min(all_vals), max(all_vals))

    for model_name in ordered_models:
        homes = sorted(results[model_name])
        values = []

        for metric in metrics_to_plot:
            if metric == "comm_cost":
                mean_val = comm_totals[model_name]
            else:
                mean_val = np.mean(
                    [results[model_name][home_id]["mean"][metric] for home_id in homes]
                )

            vmin, vmax = global_min_max[metric]
            if abs(vmax - vmin) < 1e-9:
                normalized = 1.0
            else:
                normalized = (mean_val - vmin) / (vmax - vmin)
            normalized = float(np.clip(normalized, 0, 1))
            values.append(normalized)

        values += values[:1]
        label = CUSTOM_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, "gray")
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric_labels[m] for m in metrics_to_plot])
    label_pads = {"elec_cost": 38, "comm_cost": 48, "comfort": 0, "net_demand": 0}
    for tick, metric in zip(ax.xaxis.get_major_ticks(), metrics_to_plot):
        tick.set_pad(label_pads[metric])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_yticks([])
    ring_angles = angles[:-1] + [angles[0]]
    for radius in np.arange(0.2, 1.1, 0.2):
        ax.plot(
            ring_angles, [radius] * len(ring_angles), "-", color="gray", linewidth=0.5, alpha=0.5
        )
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], "-", color="gray", linewidth=0.5, alpha=0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0), fontsize=LEGEND_SIZE)
    ax.set_title("Model performance radar chart", fontsize=F_SIZE - 1, fontweight="bold", pad=20)

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.88)
    save_figure_pdf(os.path.join(save_path, "model_comparison_radar_chart.pdf"), fig=fig)
    plt.close(fig)
    print("Saved: model_comparison_radar_chart.pdf")


def _read_comm_metrics(model_name, csv_filename, logs_base):
    metrics = {"total (MB)": 0.0, "train_time (s)": 0.0, "test_time (s)": 0.0, "FLOPs": 0.0}
    path = os.path.join(logs_base, model_name, csv_filename)
    df = pd.read_csv(path)

    if not df.empty:
        values = pd.to_numeric(df.iloc[-1].reindex(metrics), errors="coerce").fillna(0.0)
        metrics.update(values.to_dict())
    return metrics


def plot_comm_time(
    model_names, mode="train", logs_base="train_test/logs", save_path=None, num_homes=10
):
    cfg = {
        "train": (
            "comm_train_time.csv",
            "train_time (s)",
            "Training time (s)",
            "train_test/results/model_comm_train_time.pdf",
        ),
        "test": (
            "comm_test_time.csv",
            "test_time (s)",
            "Test time (s)",
            "train_test/results/model_comm_test_time.pdf",
        ),
    }
    csv_file, time_col, time_label, default_path = cfg[mode]

    ordered = order_models(model_names)
    labels = [CUSTOM_NAMES.get(m, m) for m in ordered]
    colors = [MODEL_COLORS.get(m, "gray") for m in ordered]
    rows = {m: _read_comm_metrics(m, csv_file, logs_base) for m in ordered}

    divisors = dict.fromkeys(ordered, num_homes)
    comms = [rows[m]["total (MB)"] / divisors[m] for m in ordered]
    times = [rows[m][time_col] / divisors[m] for m in ordered]
    flops = [rows[m]["FLOPs"] / divisors[m] / 1e9 for m in ordered]

    x = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    chart_data = (
        (comms, "Communication (MB)", ".2f"),
        (times, time_label, ".2f"),
        (flops, "GFLOPs", ".1f"),
    )
    for ax, (values, ylabel, number_format) in zip(axes, chart_data):
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:{number_format}}",
                    ha="center",
                    va="bottom",
                    fontsize=F_SIZE - 7,
                )
    fig.suptitle("Computation (Average per home)", fontsize=F_SIZE)
    legend_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.6, alpha=0.85, label=label)
        for label, color in zip(labels, colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=max(1, len(labels)),
        bbox_to_anchor=(0.5, 0.92),
        frameon=True,
        fontsize=LEGEND_SIZE,
    )

    out = save_path or default_path
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure_pdf(out, fig=fig)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_model_precision(
    num_homes,
    low_bit=8,
    window=1,
    save_path="train_test/results/model_precision_FedDMPQ.pdf",
    logs_base="train_test/logs",
):
    columns = ["params_fp32", f"params_int{int(low_bit)}"]
    labels = ["FP32", f"INT{low_bit}"]
    fig, axes = _home_subplots(num_homes)
    handles = []
    for hid in range(1, num_homes + 1):
        ax = axes[hid - 1]
        ax.set_title(f"Home {hid}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Precision ratio")
        ax.grid(True, alpha=0.25)
        df = pd.read_csv(os.path.join(logs_base, "feddmpq", f"home_{hid}", "train_logs.csv"))
        counts = df[columns].astype(float).rolling(window, min_periods=1).mean()
        ratios = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        lines = ax.plot(df["episode"], ratios)
        if hid == 1:
            handles = lines
    for ax in axes[num_homes:]:
        ax.axis("off")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.969])
    save_figure_pdf(save_path, fig=fig)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_live_scheduling(records, home_id=None):
    """Categorical controller comparisons, rather than fictitious RL convergence."""
    panels = (
        ("elec_cost", "Objective including DR", "$/home/day"),
        ("energy_bill_without_dr", "Energy bill without DR", "$/home/day"),
        ("import", "Grid purchases", "kWh/home/day"),
        ("comfort_pct", "Thermal comfort", "% of timesteps"),
        ("pv_local_use_kwh", "PV retained locally", "kWh/home/day"),
        ("pv_curtailed_kwh", "PV curtailed", "kWh/home/day"),
        ("task_success_pct", "EV + WM completion", "% of home-days"),
        ("reward", "Daily reward", "Reward/home/day"),
    )
    evaluations = [r for r in records if r.get("kind") == "eval"]
    labels = [r.get("policy", str(r["episode"])) for r in evaluations]
    with plt.rc_context():
        configure_plot_style()
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "xtick.labelsize": 7})
        fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
        for ax, (key, title, unit) in zip(axes.flat, panels, strict=True):
            x, mean, sd = convergence_series(evaluations, key, "eval", 1, home_id)
            if len(x):
                ax.plot(x, mean, "o", color="#c56710", markersize=4)
                if home_id is None:
                    ax.fill_between(x, mean - sd, mean + sd, color="#c56710", alpha=0.15)
                ax.set_xticks([r["episode"] for r in evaluations], labels, rotation=75, ha="right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Waiting for controller evaluation",
                    transform=ax.transAxes,
                    ha="center",
                )
            ax.set(title=title, ylabel=unit)
            ax.grid(alpha=0.2)
        scope = "Mean +/- home SD" if home_id is None else f"Home {home_id}"
        fig.suptitle(
            f"Scheduling controller comparison | {scope} | passes are not training episodes",
            fontsize=12,
        )
    return fig
