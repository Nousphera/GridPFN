"""Perfect-foresight scheduling under the unchanged home and greedy ring market.

The paper oracle minimizes its undiscounted evaluation objective. The service
oracle first certifies each home's minimum squared temperature violation, then
minimizes community electricity cost subject to every home's individual service
budget. Maximum in-band hours are a separate diagnostic. Solver
bounds and an independent simulator replay accompany every returned schedule.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from gridpfn.core.utils.thermal_planning import thermal_limit

COMFORT_TOLERANCE = 1e-6
SERVICE_TOLERANCE = 1e-5  # squared degrees C, per home/day
OBJECTIVES = ("paper_reward", "comfort_first")


def _solver(time_limit, gap):
    try:
        from pyscipopt import Model
    except ImportError as error:
        raise ImportError("The oracle requires pyscipopt==6.2.1; RL does not use it") from error

    model = Model("home-energy-oracle")
    model.hideOutput()
    model.setRealParam("limits/time", time_limit)
    model.setRealParam("limits/gap", gap)
    model.setRealParam("limits/absgap", 1e-6)
    model.setRealParam("numerics/feastol", 1e-8)
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("lp/threads", 1)
    # Avoid the bundled Ipopt/METIS native crash; equations and certificates are unchanged.
    model.setStringParam("nlpi/ipopt/optfile", str(Path(__file__).with_name("ipopt.opt")))
    model.setIntParam("randomization/randomseedshift", 0)
    return model


def _certificate(model):
    lower, upper = float(model.getDualbound()), float(model.getPrimalbound())
    return {
        "status": str(model.getStatus()),
        "lower_bound": lower,
        "upper_bound": upper,
        "absolute_gap": max(0.0, upper - lower),
        "relative_gap": float(model.getGap()),
        "seconds": float(model.getSolvingTime()),
        "nodes": int(model.getNNodes()),
        "certified": str(model.getStatus()) in ("optimal", "gaplimit")
        and np.isfinite([lower, upper]).all()
        and max(0.0, upper - lower) <= 1e-4 + 1e-6 * abs(upper),
    }


def _require_solution(model):
    if not model.getNSols():
        raise RuntimeError(f"Oracle has no feasible schedule: {model.getStatus()}")
    result = _certificate(model)
    if not result["certified"]:
        raise RuntimeError(f"Uncertified oracle solve: {result}")
    return result


def _physical(env):
    return np.array(
        [[env._denorm_feature(column, row[column]) for column in range(8)] for row in env.dataset]
    )


def _thermal_signature(env):
    digest = hashlib.sha256(np.ascontiguousarray(_physical(env)[:, 4]).tobytes())
    digest.update(
        json.dumps(
            [
                env.delta_t,
                env.indoor_temp,
                env.thermal_inertia,
                env.thermal_gain,
                env.max_power_AC,
                env.temperature_min,
                env.temperature_max,
                env.ac_required_energy,
                env.ac_energy_quota,
            ]
        ).encode()
    )
    return digest.hexdigest()


def _thermal(model, env, prefix):
    from pyscipopt import quicksum

    outdoor = _physical(env)[:, 4]
    inertia = env.thermal_inertia
    gain = (1 - inertia) * env.thermal_gain
    power, temperature, squared = [], [], []
    cold = hot = float(env.indoor_temp)
    previous = float(env.indoor_temp)
    for t, outside in enumerate(outdoor):
        cold = inertia * cold + (1 - inertia) * outside - gain * env.max_power_AC
        hot = inertia * hot + (1 - inertia) * outside
        ac = model.addVar(f"{prefix}_ac_{t}", lb=0, ub=env.max_power_AC)
        temp = model.addVar(f"{prefix}_temp_{t}", lb=cold, ub=hot)
        model.addCons(temp == inertia * previous + (1 - inertia) * outside - gain * ac)
        below = model.addVar(f"{prefix}_cold_{t}", lb=0, ub=max(0, env.temperature_min - cold))
        above = model.addVar(f"{prefix}_hot_{t}", lb=0, ub=max(0, hot - env.temperature_max))
        model.addCons(below >= env.temperature_min - temp)
        model.addCons(above >= temp - env.temperature_max)
        squared.extend((below * below, above * above))
        power.append(ac)
        temperature.append(temp)
        previous = temp
    required = env.ac_required_energy if env.ac_energy_quota else 0
    model.addCons(quicksum(power) * env.delta_t >= required)
    return power, temperature, quicksum(squared)


def comfort_frontier(env, time_limit=60, gap=1e-7):
    """Per-home lexicographic comfort certificate; no other home can offset it."""
    if env.current_step:
        raise ValueError("Oracle plans start from a reset day")
    limit = thermal_limit(env, time_limit=time_limit)
    model = _solver(time_limit, gap)
    power, temperatures, severity = _thermal(model, env, "frontier")
    objective = model.addVar("severity", lb=0)
    model.addCons(objective >= severity)
    model.setObjective(objective)
    model.optimize()
    certificate = _require_solution(model)
    actual_temperatures = np.array([model.getVal(v) for v in temperatures])
    discomfort = np.maximum(env.temperature_min - actual_temperatures, 0) ** 2
    discomfort += np.maximum(actual_temperatures - env.temperature_max, 0) ** 2
    return {
        "thermal_sha256": _thermal_signature(env),
        "maximum_comfort_pct": limit["comfort_limit_pct"],
        "minimum_violations": limit["minimum_violations"],
        "thermal_mip_gap": limit["mip_gap"],
        "minimum_squared_violation": float(discomfort.sum()),
        "comfort_at_minimum_discomfort_pct": 100
        * float(np.mean(discomfort <= COMFORT_TOLERANCE**2)),
        "severity_certificate": certificate,
        "ac_power": [float(model.getVal(v)) for v in power],
    }


class _Piecewise:
    """Exact bounded ReLU/minimum; unlike an LP relaxation, cannot invent flows."""

    def __init__(self, model):
        self.model = model
        self.index = 0

    def positive(self, expression, low, high):
        if high <= 0:
            return 0.0
        if low >= 0:
            return expression
        self.index += 1
        value = self.model.addVar(f"positive_{self.index}", lb=0, ub=high)
        active = self.model.addVar(f"sign_{self.index}", vtype="B")
        self.model.addCons(value >= expression)
        self.model.addCons(value <= expression - low * (1 - active))
        self.model.addCons(value <= high * active)
        return value

    def minimum(self, first, second, first_upper, second_upper):
        if first_upper <= 0 or second_upper <= 0:
            return 0.0
        if isinstance(first, (int, float, np.number)) and first >= second_upper:
            return second
        if isinstance(second, (int, float, np.number)) and second >= first_upper:
            return first
        return first - self.positive(first - second, -second_upper, first_upper)


def _community_model(envs, objective, frontiers, peer_price, peers, time_limit, gap):
    from pyscipopt import quicksum

    model = _solver(time_limit, gap)
    piecewise = _Piecewise(model)
    homes = []
    n, steps = len(envs), envs[0].max_step
    for i, env in enumerate(envs):
        values = _physical(env)
        fixed = np.maximum(values[:, 0], 0) / env.delta_t
        pv = np.maximum(values[:, 1], 0) / env.delta_t
        prices = [env._resolve_price(t * env.delta_t, values[t, 3]) for t in range(steps)]
        frontier = None if objective == "paper_reward" else frontiers[i]
        ac, temperature, severity = _thermal(model, env, str(i))
        if frontier is not None:
            model.addCons(severity <= frontier["minimum_squared_violation"] + SERVICE_TOLERANCE)
            if frontier["minimum_violations"] == 0:
                # A zero-discomfort home has a feasible hard comfort band. Do
                # not spend the numerical severity allowance on real violations.
                for temp in temperature:
                    model.addCons(temp >= env.temperature_min)
                    model.addCons(temp <= env.temperature_max)
        cycle = len(env.wm_cycle_profile)
        deadline = min(steps, int(round(env.time_end_WM / env.delta_t)))
        earliest = 0 if objective == "paper_reward" else int(np.ceil(env.time_ini_WM / env.delta_t))
        start = (
            {
                t: model.addVar(f"{i}_wm_start_{t}", vtype="B")
                for t in range(earliest, deadline - cycle + 1)
            }
            if env.wm_required
            else {}
        )
        if env.wm_required:
            if not start:
                raise ValueError("Washing cycle cannot finish in the required service window")
            model.addCons(quicksum(start.values()) == 1)
        wm = [
            quicksum(
                env.wm_cycle_profile[t - begin] / env.delta_t * variable
                for begin, variable in start.items()
                if begin <= t < begin + cycle
            )
            for t in range(steps)
        ]
        wm_discomfort = quicksum(
            variable
            * env.alpha
            * sum(
                max(env.time_ini_WM - t * env.delta_t, t * env.delta_t - env.time_end_WM, 0) ** 2
                for t in range(begin, begin + cycle)
            )
            for begin, variable in start.items()
        )
        ev, battery, soe = [], [], []
        last_soe = float(env.SoE_BESS)
        for t in range(steps):
            connected = env.time_ini_EV <= t * env.delta_t < env.time_end_EV
            ev.append(model.addVar(f"{i}_ev_{t}", lb=0, ub=env.max_power_EV if connected else 0))
            battery.append(
                model.addVar(f"{i}_battery_{t}", lb=-env.max_power_BESS, ub=env.max_power_BESS)
            )
            soe.append(model.addVar(f"{i}_soe_{t}", lb=0, ub=1))
            model.addCons(
                soe[-1]
                == last_soe
                + env.efficiency_BESS * battery[-1] * env.delta_t / env.max_capacity_BESS
            )
            last_soe = soe[-1]
            load = fixed[t] + ac[t] + ev[t] + wm[t] - pv[t]
            maximum = fixed[t] + env.max_power_AC + (env.max_power_EV if connected else 0)
            maximum += max(env.wm_cycle_profile, default=0) / env.delta_t
            model.addCons(
                battery[-1] >= -piecewise.positive(load, fixed[t] - pv[t], maximum - pv[t])
            )
        model.addCons(quicksum(ev) * env.delta_t == env.ev_required_energy)
        homes.append(
            {
                "env": env,
                "fixed": fixed,
                "pv": pv,
                "prices": prices,
                "ac": ac,
                "temperature": temperature,
                "severity": severity,
                "ev": ev,
                "battery": battery,
                "soe": soe,
                "wm": wm,
                "start": start,
                "wm_discomfort": wm_discomfort,
                "cost": [],
                "bill": [],
                "flows": [],
            }
        )
    exported = [0.0] * n
    for t in range(steps):
        imports, exports, surplus, maximum_import, maximum_surplus = [], [], [], [], []
        for i, home in enumerate(homes):
            env, dt = home["env"], home["env"].delta_t
            raw = (
                home["fixed"][t]
                - home["pv"][t]
                + home["ac"][t]
                + home["ev"][t]
                + home["wm"][t]
                + home["battery"][t]
            )
            connected = env.time_ini_EV <= t * dt < env.time_end_EV
            wm_possible = any(
                begin <= t < begin + len(env.wm_cycle_profile) for begin in home["start"]
            )
            upper = home["fixed"][t] + env.max_power_AC + env.max_power_BESS
            upper += env.max_power_EV if connected else 0
            upper += max(env.wm_cycle_profile, default=0) / dt if wm_possible else 0
            # Discharge cannot create exports: surplus never exceeds PV minus
            # fixed load. These bounds remove irrelevant market binaries.
            surplus_upper = max(0, home["pv"][t] - home["fixed"][t])
            surplus_lower = max(0, home["pv"][t] - upper)
            imp = piecewise.positive(raw, -surplus_upper, upper - home["pv"][t])
            potential = piecewise.positive(-raw, home["pv"][t] - upper, surplus_upper)
            if env.export_cap_kwh is None:
                exp, available = potential, potential
                available_upper = surplus_upper
            else:
                remaining = (env.export_cap_kwh - exported[i]) / dt
                cap = env.export_cap_kwh / dt
                if surplus_lower >= cap:
                    # Every possible schedule exhausts the remaining grid cap.
                    exp, available, available_upper = remaining, 0.0, 0.0
                else:
                    exp = piecewise.minimum(potential, remaining, surplus_upper, cap)
                    available = piecewise.minimum(potential, remaining - exp, surplus_upper, cap)
                    # available=min(potential, remaining-min(potential,remaining))
                    # cannot exceed half the remaining cap under original accounting.
                    available_upper = min(surplus_upper, cap / 2)
            imports.append(imp)
            exports.append(exp)
            surplus.append(available)
            maximum_import.append(max(0, upper - home["pv"][t]))
            maximum_surplus.append(available_upper)
        deficits = imports.copy()
        remaining_surplus = surplus.copy()
        peer_in, peer_out = [0.0] * n, [0.0] * n
        if peers:
            # Preserve seller order and left-neighbour-before-right priority.
            for seller in range(n):
                for buyer in ((seller - 1) % n, (seller + 1) % n):
                    if (
                        peer_price <= envs[seller].export_price
                        or peer_price >= homes[buyer]["prices"][t]
                    ):
                        continue
                    trade = piecewise.minimum(
                        remaining_surplus[seller],
                        deficits[buyer],
                        maximum_surplus[seller],
                        maximum_import[buyer],
                    )
                    remaining_surplus[seller] = remaining_surplus[seller] - trade
                    deficits[buyer] = deficits[buyer] - trade
                    peer_out[seller] = peer_out[seller] + trade
                    peer_in[buyer] = peer_in[buyer] + trade
        for i, home in enumerate(homes):
            env, dt = home["env"], home["env"].delta_t
            net = imports[i] - exports[i]
            bill = (home["prices"][t] * imports[i] - env.export_price * exports[i]) * dt
            bill -= (peer_price - env.export_price) * peer_out[i] * dt
            bill -= (home["prices"][t] - peer_price) * peer_in[i] * dt
            bill += env.fixed_cost / 30 / steps
            # SCIP expressions are mutable: keep the bill separate from DR cost.
            cost = bill + 0
            if env.dr_limit is not None:
                if env.dr_penalty >= env.dr_incentive:
                    excess = model.addVar(f"{i}_dr_excess_{t}", lb=0)
                    model.addCons(excess >= net - env.dr_limit)
                else:
                    # A concave DR tariff requires an exact kink, not an epigraph.
                    excess = piecewise.positive(
                        net - env.dr_limit,
                        -home["pv"][t] - env.dr_limit,
                        maximum_import[i] - env.dr_limit,
                    )
                cost += dt * (
                    env.dr_incentive * (net - env.dr_limit)
                    + (env.dr_penalty - env.dr_incentive) * excess
                )
            home["bill"].append(bill)
            home["cost"].append(cost)
            home["flows"].append((imports[i], exports[i], peer_in[i], peer_out[i]))
            exported[i] = exported[i] + (exports[i] + peer_out[i]) * dt
            if env.export_cap_kwh is not None:
                model.addCons(quicksum([exported[i]]) <= env.export_cap_kwh)
    total = quicksum(value for home in homes for value in home["cost"])
    if objective == "paper_reward":
        thermal_cost = model.addVar("thermal_penalty", lb=0)
        model.addCons(thermal_cost >= quicksum(h["env"].beta * h["severity"] for h in homes))
        total += thermal_cost + quicksum(h["wm_discomfort"] for h in homes)
    model.setObjective(total)
    return model, homes


def solve_oracle(
    envs,
    objective="comfort_first",
    peer_price=0.1,
    peers=True,
    time_limit=120,
    gap=1e-7,
    frontiers=None,
):
    """Return schedules and numerical global bounds, never an uncertified incumbent.

    All decisions see the realized full day. This is a diagnostic information
    advantage, not a deployable policy or a source of held-out training labels.
    Independent homes are solved separately only when no ring edge is eligible.
    """
    if objective not in OBJECTIVES or not envs:
        raise ValueError("Select a supported objective and at least one home")
    if time_limit <= 0 or gap < 0 or gap > 1e-6:
        raise ValueError("Use a positive solver budget and a gap no larger than 1e-6")
    if any(e.current_step or e.max_step != 24 or e.delta_t != 1 for e in envs):
        raise ValueError("The oracle currently requires reset hourly 24-step days")
    if any(not np.isfinite(_physical(e)).all() for e in envs):
        raise ValueError("Use finite physical traces")
    if objective == "comfort_first" and frontiers is None:
        frontiers = [comfort_frontier(e, time_limit, gap) for e in envs]
    if frontiers is not None and (
        len(frontiers) != len(envs)
        or any(
            frontier["thermal_sha256"] != _thermal_signature(env)
            or not frontier["severity_certificate"]["certified"]
            for frontier, env in zip(frontiers, envs, strict=True)
        )
    ):
        raise ValueError("Comfort frontier differs in weather, dynamics or demand")
    physical = [_physical(e) for e in envs]
    if any(not np.array_equal(values[:, 2], np.arange(24)) for values in physical):
        raise ValueError("Daily observation times must be exactly 0..23 hours")
    coupled = (
        peers
        and len(envs) > 1
        and any(
            peer_price > envs[seller].export_price
            and peer_price < envs[buyer]._resolve_price(t, physical[buyer][t, 3])
            for seller in range(len(envs))
            for buyer in ((seller - 1) % len(envs), (seller + 1) % len(envs))
            for t in range(24)
        )
    )
    groups = [list(range(len(envs)))] if coupled else [[i] for i in range(len(envs))]
    started = time.monotonic()
    certificates, solutions = [], [None] * len(envs)
    for indices in groups:
        model, homes = _community_model(
            [envs[i] for i in indices],
            objective,
            None if frontiers is None else [frontiers[i] for i in indices],
            peer_price,
            bool(coupled),
            time_limit,
            gap,
        )
        model.optimize()
        certificates.append({"homes": indices, **_require_solution(model)})

        def value(variable):
            if isinstance(variable, (int, float, np.number)):
                return float(variable)
            return float(model.getVal(variable))

        for index, home in zip(indices, homes, strict=True):
            controls = np.array([[value(v) for v in home[k]] for k in ("ac", "ev", "battery")]).T
            starts = [t for t, variable in home["start"].items() if value(variable) > 0.5]
            solutions[index] = {
                "controls": controls.tolist(),
                "wm_start": starts[0] if starts else None,
                "temperatures": [value(v) for v in home["temperature"]],
                "battery_soe": [value(v) for v in home["soe"]],
                "flows": [[value(v) for v in row] for row in home["flows"]],
                "electrical_cost": sum(value(v) for v in home["cost"]),
                "bill_before_dr": sum(value(v) for v in home["bill"]),
                "squared_violation": value(home["severity"]),
                "wm_discomfort": value(home["wm_discomfort"]),
            }
    return {
        "objective": objective,
        "perfect_foresight": True,
        "coupled_market": bool(coupled),
        "peer_price": peer_price,
        "peers": peers,
        "certified": True,
        "lower_bound": sum(c["lower_bound"] for c in certificates),
        "upper_bound": sum(c["upper_bound"] for c in certificates),
        "certificates": certificates,
        "frontiers": frontiers,
        "solutions": solutions,
        "seconds": time.monotonic() - started,
        "comfort_tolerance": COMFORT_TOLERANCE,
        "service_tolerance": SERVICE_TOLERANCE,
    }


def replay_oracle(envs, result, reference=False):
    """Audit controls, flows, objective and service in a fresh simulator.

    With reference=True, execute the pinned main-branch physics and market on
    physical traces. This shares no dispatch equations with the optimizer.
    """
    from gridpfn.core.em_strategy import P2P_TRADING
    from gridpfn.core.environment import HOME_ENERGY_MGNT

    environment_class, market = HOME_ENERGY_MGNT, P2P_TRADING
    if reference:
        from gridpfn.experiments.audit_constraints import REFERENCE, reference_module

        original, _ = reference_module("environment", REFERENCE)
        original_market, _ = reference_module("em_strategy", REFERENCE)
        environment_class, market = original.HOME_ENERGY_MGNT, original_market.P2P_TRADING
    settings = (
        "thermal_inertia",
        "thermal_gain",
        "wm_duration",
        "max_duration",
        "time_ini_WM",
        "time_end_WM",
        "time_ini_EV",
        "time_end_EV",
        "temperature_min",
        "temperature_max",
        "max_power_AC",
        "max_power_EV",
        "max_power_BESS",
        "max_capacity_EV",
        "max_capacity_BESS",
        "efficiency_EV",
        "efficiency_BESS",
        "alpha",
        "beta",
        "mu",
        "sigma",
        "tou_prices_by_hour",
        "export_price",
        "export_cap_kwh",
        "dr_limit",
        "dr_penalty",
        "dr_incentive",
        "ac_energy_quota",
    )
    replay = []
    for env in envs:
        clone = environment_class(
            _physical(env), scaler={"delta_t": env.delta_t}, fixed_cost=env.fixed_cost
        )
        for name in settings:
            setattr(clone, name, getattr(env, name))
        clone.reset()
        if not env.ac_energy_quota:
            # Main predates the quota switch. A zero required-energy parameter
            # makes its always-on lower bound exactly equivalent to no quota.
            clone.ac_required_energy = 0.0
        for name in ("indoor_temp", "SoE_BESS", "SoE_EV"):
            setattr(clone, name, getattr(env, name))
        replay.append(clone)
    records = [
        {
            "reward": 0.0,
            "elec_cost": 0.0,
            "energy_bill_without_dr": 0.0,
            "import": 0.0,
            "export": 0.0,
            "p2p_kwh": 0.0,
            "pv_local_use_kwh": 0.0,
            "pv_curtailed_kwh": 0.0,
            "squared_violation": 0.0,
            "comfortable_hours": 0,
            "strict_comfortable_hours": 0,
            "maximum_action_override_kw": 0.0,
            "temperatures": [],
            "battery_soe": [],
            "flows": [],
        }
        for _ in envs
    ]
    for t in range(envs[0].max_step):
        infos = []
        for env, solution, row in zip(replay, result["solutions"], records, strict=True):
            control = solution["controls"][t]
            _, electrical, reward, _, _ = env.step((int(solution["wm_start"] == t), control))
            actual = [env.power_AC, env.power_EV, env.power_BESS]
            row["maximum_action_override_kw"] = max(
                row["maximum_action_override_kw"],
                float(np.max(np.abs(np.asarray(control) - actual))),
            )
            row["reward"] += reward
            row["elec_cost"] -= electrical
            deviation = max(
                env.temperature_min - env.indoor_temp, env.indoor_temp - env.temperature_max, 0
            )
            row["squared_violation"] += deviation**2
            row["comfortable_hours"] += int(deviation <= COMFORT_TOLERANCE)
            row["strict_comfortable_hours"] += int(deviation == 0)
            row["temperatures"].append(env.indoor_temp)
            row["battery_soe"].append(env.SoE_BESS)
            infos.append(
                dict(
                    net_load=env.net_load,
                    pv_surplus=env.pv_surplus,
                    price=env.price,
                    export_price=env.export_price,
                    delta_t=env.delta_t,
                )
            )
        adjustment, imports, exports = (
            market.compute_adjustments(infos, result["peer_price"])
            if result["peers"]
            else ([0.0] * len(envs),) * 3
        )
        for i, (env, row) in enumerate(zip(replay, records, strict=True)):
            grid_in, grid_out = (
                max(env.net_load, 0) - imports[i],
                max(-env.net_load, 0) - exports[i],
            )
            row["reward"] += adjustment[i]
            row["elec_cost"] -= adjustment[i]
            row["energy_bill_without_dr"] += (
                grid_in * env.price
                - grid_out * env.export_price
                + result["peer_price"] * (imports[i] - exports[i])
            ) * env.delta_t + env.fixed_cost / 30 / env.max_step
            row["import"] += grid_in * env.delta_t
            row["export"] += grid_out * env.delta_t
            row["p2p_kwh"] += imports[i] * env.delta_t
            raw = (
                env.fixed_load
                + env.power_AC
                + env.power_EV
                + env.power_WM
                + env.power_BESS
                - env.pv_generation
            )
            curtailed = max(0, env.net_load - raw)
            row["pv_curtailed_kwh"] += curtailed * env.delta_t
            row["pv_local_use_kwh"] += (
                max(0, env.pv_generation - curtailed - grid_out - exports[i]) * env.delta_t
            )
            row["flows"].append(
                [max(env.net_load, 0), max(-env.net_load, 0), imports[i], exports[i]]
            )
            env.exported_kwh += exports[i] * env.delta_t
            if env.export_cap_kwh is not None and env.exported_kwh > env.export_cap_kwh + 1e-5:
                raise AssertionError("Original export budget exceeded")
    difference = 0.0
    for i, (env, row, solution) in enumerate(
        zip(replay, records, result["solutions"], strict=True)
    ):
        for key, expected in (
            ("temperatures", solution["temperatures"]),
            ("battery_soe", solution["battery_soe"]),
            ("flows", solution["flows"]),
        ):
            difference = max(difference, float(np.max(np.abs(np.asarray(row[key]) - expected))))
            np.testing.assert_allclose(row[key], expected, atol=2e-5, rtol=1e-7)
        np.testing.assert_allclose(
            row["elec_cost"], solution["electrical_cost"], atol=1e-4, rtol=1e-7
        )
        np.testing.assert_allclose(
            row["energy_bill_without_dr"], solution["bill_before_dr"], atol=1e-4, rtol=1e-7
        )
        if env.ac_energy_quota and env.ac_energy_delivered + 2e-5 < env.ac_required_energy:
            raise AssertionError("Original AC quota failed")
        if env.ev_energy_delivered + 2e-5 < env.ev_required_energy or not env.wm_completed:
            raise AssertionError("Original EV/WM completion failed")
        if row["maximum_action_override_kw"] > 2e-5:
            raise AssertionError("Oracle schedule needed a simulator action correction")
        row["comfort_pct"] = 100 * row["comfortable_hours"] / env.max_step
        row["strict_comfort_pct"] = 100 * row["strict_comfortable_hours"] / env.max_step
        row["ac_delivered_kwh"] = env.ac_energy_delivered
        row["ac_required_kwh"] = env.ac_required_energy
        row["ev_completion_ratio"] = (
            1.0
            if env.ev_required_energy == 0
            else min(1, env.ev_energy_delivered / env.ev_required_energy)
        )
        row["wm_completed"] = float(env.wm_completed)
        row["final_battery_soe"] = env.SoE_BESS
        if result["objective"] == "comfort_first":
            frontier = result["frontiers"][i]
            if (
                row["squared_violation"]
                > frontier["minimum_squared_violation"] + SERVICE_TOLERANCE + 2e-5
            ):
                raise AssertionError("Oracle exceeded a home's individual discomfort budget")
    objective = sum(
        -r["reward"] if result["objective"] == "paper_reward" else r["elec_cost"] for r in records
    )
    np.testing.assert_allclose(objective, result["upper_bound"], atol=2e-4, rtol=1e-7)
    return {
        "reference": "main:e34da8b" if reference else "current",
        "legacy_ac_quota_disabled_by_zero_budget": bool(
            reference and any(not e.ac_energy_quota for e in envs)
        ),
        "verified": True,
        "objective": objective,
        "max_trajectory_difference": difference,
        "homes": records,
    }
