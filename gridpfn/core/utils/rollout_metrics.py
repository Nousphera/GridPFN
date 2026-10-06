"""Physical daily outcomes shared by training validation and offline evaluation."""

import numpy as np


def new_episode_log(env):
    return {
        "episode_reward": 0.0,
        "episode_elec_cost": 0.0,
        "episode_comfort": 0.0,
        "powers": {"net": [], "import": [], "export": [], "AC (kW)": [], "AC baseline (kW)": []},
        "temperature": {
            "indoor": [],
            "outdoor": [],
            "min": float(env.temperature_min),
            "max": float(env.temperature_max),
        },
    }


def record_environment(log, env):
    net_load = env.net_load
    log["powers"]["net"].append(net_load)
    log["powers"]["import"].append(max(0, net_load))
    log["powers"]["export"].append(max(0, -net_load))
    log["powers"]["AC (kW)"].append(env.power_AC)
    log["powers"]["AC baseline (kW)"].append(env.ac_baseline_power)
    log["temperature"]["indoor"].append(env.indoor_temp)
    log["temperature"]["outdoor"].append(env.outdoor_temp)


def episode_data(log, env):
    return log | {
        "delta_t": env.delta_t,
        "devices": {
            "ev_required_energy": env.ev_required_energy,
            "ev_energy_delivered": env.ev_energy_delivered,
            "wm_required": env.wm_required,
            "wm_completed": env.wm_completed,
        },
    }


def calculate_metrics(episode_data):
    powers = episode_data["powers"]
    total_power = np.asarray(powers["net"])
    delta_t = float(episode_data["delta_t"])
    metrics = {
        "elec_cost": -episode_data["episode_elec_cost"],
        "reward": episode_data["episode_reward"],
    }
    if "energy_bill_without_dr" in episode_data:
        metrics["energy_bill_without_dr"] = float(episode_data["energy_bill_without_dr"])
    metrics["peak_demand"] = np.max(total_power)
    metrics["comfort"] = float(episode_data["episode_comfort"])
    metrics["import"] = float(np.sum(powers["import"])) * delta_t
    metrics["export"] = float(np.sum(powers["export"])) * delta_t
    metrics["net_demand"] = np.sum(total_power) * delta_t
    devices = episode_data["devices"]
    ev_required = float(devices.get("ev_required_energy", 0.0))
    ev_delivered = float(devices.get("ev_energy_delivered", 0.0))
    metrics["ev_completion_ratio"] = (
        min(ev_delivered / ev_required, 1.0) if ev_required > 0 else 1.0
    )
    metrics["wm_completed"] = float(
        devices.get("wm_completed", not devices.get("wm_required", False))
    )
    temperature = episode_data["temperature"]
    indoor_temperature = np.asarray(temperature["indoor"])
    violations = np.maximum(
        np.maximum(
            float(temperature["min"]) - indoor_temperature,
            indoor_temperature - float(temperature["max"]),
        ),
        0,
    )
    metrics["squared_violation"] = float(np.sum(violations**2))
    metrics["violation_hours"] = float(np.count_nonzero(violations)) * delta_t
    metrics["degree_hours"] = float(np.sum(violations)) * delta_t
    metrics["peak_violation_degrees"] = float(np.max(violations)) if violations.size else 0.0
    metrics["temperature_violation_ratio"] = (
        float(
            np.mean(
                (indoor_temperature < float(temperature["min"]))
                | (indoor_temperature > float(temperature["max"]))
            )
        )
        if indoor_temperature.size
        else 0.0
    )
    ac_power = np.asarray(powers["AC (kW)"])
    ac_baseline = np.asarray(powers["AC baseline (kW)"])
    metrics["ac_baseline_deviation"] = (
        float(np.mean(np.abs(ac_power - ac_baseline))) if ac_power.size else 0.0
    )
    return metrics
