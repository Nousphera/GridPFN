"""Certified linear thermal feasibility planning, independent of policy learning."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def thermal_limit(env, time_limit=10, require_certificate=True):
    """Maximize in-band steps with exact linear thermal dynamics and AC quota.

    Binary variables allow out-of-band temperature. Big-M values come from exact
    reachable bounds, not arbitrary constants. Return the solver gap and replayed
    temperatures so an uncertified solve cannot masquerade as an optimum.
    """
    n = env.max_step
    outdoor = np.asarray([env._denorm_feature(4, row[4]) for row in env.dataset])
    a = env.thermal_inertia
    gain = (1 - a) * env.thermal_gain
    effect = np.zeros((n, n))
    natural = np.empty(n)
    temperature = env.indoor_temp
    for t in range(n):
        temperature = a * temperature + (1 - a) * outdoor[t]
        natural[t] = temperature
        effect[t, : t + 1] = -gain * a ** np.arange(t, -1, -1)
    coldest = natural + effect @ np.full(n, env.max_power_AC)
    hottest = natural
    upper_m = np.maximum(0, hottest - env.temperature_max)
    lower_m = np.maximum(0, env.temperature_min - coldest)
    # Variables: n continuous AC powers, n binary comfort violations.
    upper = np.column_stack((effect, -np.diag(upper_m)))
    lower = np.column_stack((effect, np.diag(lower_m)))
    quota = np.r_[np.full(n, env.delta_t), np.zeros(n)][None, :]
    matrix = np.vstack((upper, lower, quota))
    required = env.ac_required_energy if env.ac_energy_quota else 0
    constraints = LinearConstraint(
        matrix,
        np.r_[np.full(n, -np.inf), env.temperature_min - natural, required],
        np.r_[env.temperature_max - natural, np.full(n + 1, np.inf)],
    )
    result = milp(
        np.r_[np.zeros(n), np.ones(n)],
        integrality=np.r_[np.zeros(n), np.ones(n)],
        bounds=Bounds(np.zeros(2 * n), np.r_[np.full(n, env.max_power_AC), np.ones(n)]),
        constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": 0},
    )
    if result.x is None or (require_certificate and (not result.success or result.mip_gap > 1e-8)):
        raise RuntimeError(f"Uncertified thermal solve: {result.message}")
    powers = result.x[:n]
    temperatures = natural + effect @ powers
    # Floating-point boundary noise must not count as physical discomfort.
    in_band = (temperatures >= env.temperature_min - 1e-7) & (
        temperatures <= env.temperature_max + 1e-7
    )
    return {
        "comfort_limit_pct": 100 * float(in_band.mean()),
        "minimum_violations": int(round(result.fun)),
        "mip_gap": float(result.mip_gap),
        "certified": bool(result.success and result.mip_gap <= 1e-8),
        "ac_required_kwh": required,
        "ac_delivered_kwh": float(powers.sum() * env.delta_t),
        "ac_power": powers.tolist(),
        "temperatures": temperatures.tolist(),
    }
