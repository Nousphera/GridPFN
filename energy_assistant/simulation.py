"""Adapters to the project's unchanged simulator and causal controllers."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from utils.rollout_metrics import (
    calculate_metrics,
    episode_data,
    new_episode_log,
    record_environment,
)

from control_guidance import FeedbackTeacher
from economic_control import EconomicCoordinator
from em_strategy import apply_em_strategy
from environment import HOME_ENERGY_MGNT


def ledger_row(hour, imported, exported, price, export_price, fixed=0.0):
    values = np.array([imported, exported, price, export_price, fixed], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Ledger inputs must be finite and nonnegative")
    return {
        "hour": int(hour),
        "import_kwh": float(imported),
        "export_kwh": float(exported),
        "tariff": float(price),
        "import_cost": float(imported * price),
        "export_credit": float(exported * export_price),
        "fixed_cost": float(fixed),
        "net_cost": float(imported * price - exported * export_price + fixed),
    }


def replay(
    bundle,
    day_index,
    strategy,
    *,
    table=None,
    policy=None,
    target=20.0,
    fixed_cost=5.0,
    outlook_table=None,
):
    """Single-home retrospective replay. No peer trading or physical actuation.

    The simulator supplies daily appliance service requirements, as in the core
    experiment. They are declared requirements, not inferred future observations.
    Forecast tables expose only prefixes to the economic coordinator.
    """
    train, heldout, dates, scaler = bundle
    env = HOME_ENERGY_MGNT(heldout[day_index], scaler=scaler, fixed_cost=fixed_cost, state_dim=17)
    apply_em_strategy(env, strategy)
    client = SimpleNamespace(train_data=train, scaler=scaler)
    coordinator = None
    if table is not None:
        coordinator = EconomicCoordinator(
            [client], strategy, tables=[{dates[day_index]: table}], target=target
        )
        coordinator.start_day(dates[day_index])
    teacher = FeedbackTeacher(scaler, target_temp=target, quota_aware=True, storage_aware=True)
    state = env.reset()
    log = new_episode_log(env)
    ledger, actions, outlooks = [], [], {}
    for hour in range(env.max_step):
        if outlook_table is not None and hour in (0, 6, 12, 18):
            outlooks[str(hour)] = forecast_outlook(
                env, client, strategy, outlook_table[hour], dates[day_index], fixed_cost
            )
        if policy is not None:
            discrete, control = policy(state, dates[day_index], hour)
        elif coordinator is not None:
            discrete, control = coordinator.dispatch(state[None])[0]
        else:
            discrete, control = int(hour >= 10), teacher(state[None])[0]
        state, elec, _, comfort, done = env.step((int(discrete), np.asarray(control)))
        record_environment(log, env)
        log["episode_reward"] += elec + comfort
        log["episode_elec_cost"] += elec
        log["episode_comfort"] += comfort
        ledger.append(
            ledger_row(
                hour,
                max(0, env.net_load) * env.delta_t,
                max(0, -env.net_load) * env.delta_t,
                env.price,
                env.export_price,
                fixed_cost / 30 / env.max_step,
            )
        )
        actions.append(
            {
                "hour": hour,
                "washer_start_requested": int(discrete),
                "ac_kw": float(env.power_AC),
                "ev_kw": float(env.power_EV),
                "battery_kw": float(env.power_BESS),
                "washer_kw": float(env.power_WM),
                "indoor_c": float(env.indoor_temp),
                "battery_soc": float(env.SoE_BESS),
            }
        )
    metrics = calculate_metrics(episode_data(log, env))
    return {
        "evidence": "simulated",
        "date": dates[day_index],
        "bill": float(sum(row["net_cost"] for row in ledger)),
        "comfort_pct": 100 * (1 - metrics["temperature_violation_ratio"]),
        "degree_hours": metrics["degree_hours"],
        "ev_complete": bool(env.ev_energy_delivered + env.eps >= env.ev_required_energy),
        "washer_complete": bool(not env.wm_required or env.wm_completed),
        "terminal_battery_soc": float(env.SoE_BESS),
        "ledger": ledger,
        "actions": actions,
        "outlooks": outlooks,
        "solver_failures": coordinator.failures if coordinator else 0,
        "limits": "Retrospective simulator replay; daily service requirements supplied; peer trading disabled; bill excludes demand-response penalties and rewards. Battery end charge is shown, not constrained to equal the baseline.",
    }


def forecast_outlook(snapshot, client, strategy, predictions, date, fixed_cost):
    """Branch a dated state into predicted futures; never step through true future rows."""
    origin = snapshot.current_step
    projected = deepcopy(snapshot)
    projected.dataset = projected.dataset.copy()
    table = np.zeros((24, 24, 4), dtype=float)
    for hour in range(origin, 24):
        # One forecast issued at origin is reused throughout this planning branch.
        table[hour, hour:] = predictions[hour:]
        table[hour, hour:, 3] = predictions[origin, 3]  # Declared persistence price assumption.
        for physical_col, data_col in enumerate((0, 1, 4, 3)):
            value = predictions[hour, physical_col] if physical_col != 3 else predictions[origin, 3]
            projected.dataset[hour, data_col] = projected._norm_feature(data_col, value)
        # Appliance requests were declared at reset; future metered traces are not inputs.
        for col in (5, 6, 7):
            projected.dataset[hour, col] = projected._norm_feature(col, 0)
    tou = (
        strategy.get("tou", {}).get("hourly_prices")
        if strategy.get("tou", {}).get("enabled", True)
        else None
    )

    def simulate(use_forecast):
        env = deepcopy(projected)
        coordinator = EconomicCoordinator([client], strategy, tables=[{date: table}])
        coordinator.start_day(date)
        teacher = FeedbackTeacher(client.scaler, quota_aware=True, storage_aware=True)
        state = env._state_for_step(origin)
        actions, cost, violations, ledger = [], 0.0, [], []
        for hour in range(origin, 24):
            control = (
                coordinator.dispatch(state[None])[0]
                if use_forecast
                else (int(hour >= 10), teacher(state[None])[0])
            )
            state, *_ = env.step(control)
            row = ledger_row(
                hour,
                max(0, env.net_load),
                max(0, -env.net_load),
                env.price,
                env.export_price,
                fixed_cost / 30 / 24,
            )
            ledger.append(row)
            cost += row["net_cost"]
            violation = max(
                0, env.temperature_min - env.indoor_temp, env.indoor_temp - env.temperature_max
            )
            violations.append(violation)
            actions.append(
                {
                    "hour": hour,
                    "ac_kw": float(env.power_AC),
                    "ev_kw": float(env.power_EV),
                    "washer_kw": float(env.power_WM),
                    "battery_kw": float(env.power_BESS),
                    "battery_soc": float(env.SoE_BESS),
                    "indoor_c": float(env.indoor_temp),
                }
            )
        return {
            "actions": actions,
            "ledger": ledger,
            "bill": float(cost),
            "comfort_pct": 100 * sum(v <= 1e-6 for v in violations) / len(violations),
            "degree_hours": float(sum(violations)),
            "terminal_battery_soc": float(env.SoE_BESS),
            "ev_complete": bool(env.ev_energy_delivered + env.eps >= env.ev_required_energy),
            "washer_complete": bool(not env.wm_required or env.wm_completed),
            "solver_failures": coordinator.failures,
        }

    baseline, candidate = simulate(False), simulate(True)
    from .insights import passes

    better = candidate["bill"] < baseline["bill"] - 1e-6 and passes(candidate, baseline)
    selected = candidate if better else baseline
    return {
        "origin_hour": origin,
        "plan_label": "TabPFN schedule" if better else "Comfort-first schedule",
        "improvement_found": better,
        "remaining_cost": selected["bill"],
        "reference_cost": baseline["bill"],
        "actions": selected["actions"],
        "ledger": selected["ledger"],
        "comfort_pct": selected["comfort_pct"],
        "ev_complete": selected["ev_complete"],
        "washer_complete": selected["washer_complete"],
        "starting_battery_soc": float(snapshot.SoE_BESS),
        "starting_indoor_c": float(snapshot.indoor_temp),
        "ev_deadline_hour": float(snapshot.time_end_EV),
        "temperature_range": [snapshot.temperature_min, snapshot.temperature_max],
        "forecast": [
            {
                "hour": h,
                "load_kwh": float(predictions[h, 0]),
                "solar_kwh": float(predictions[h, 1]),
                "outdoor_c": float(predictions[h, 2]),
                "price_usd_kwh": float(predictions[origin, 3] + (tou[h] if tou else 0)),
            }
            for h in range(origin, 24)
        ],
    }
