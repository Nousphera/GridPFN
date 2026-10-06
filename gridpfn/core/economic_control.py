"""Fast receding-horizon economic dispatch with an unchanged thermal controller.

The LP optimizes its forecast approximation, not the original stochastic/hybrid
problem globally. Original quotas, deadlines, export accounting and peer settlement
are applied by the simulator; no realized future weather or load enters dispatch.
"""

import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from gridpfn.core.control_guidance import FeedbackTeacher


class EconomicCoordinator:
    def __init__(self, clients, strategy, tables=None, target=18.2, peers=False, forecasters=None):
        if (tables is None) == (forecasters is None):
            raise ValueError("Supply either causal replay tables or fitted online forecasters")
        if not clients or len(tables if tables is not None else forecasters) != len(clients):
            raise ValueError("One forecast source is required per home")
        self.clients, self.strategy, self.tables = clients, strategy, tables
        self.forecasters = forecasters
        self.target, self.peers = target, peers
        tou = strategy.get("tou", {})
        self.tou = tou.get("hourly_prices") if tou.get("enabled", True) else None
        self.teachers = [FeedbackTeacher(c.scaler, target) for c in clients]
        self.wm_power = []
        for client in clients:
            i = client.scaler["col_to_scaler_idx"][7]
            low, high = client.scaler["min"][i], client.scaler["max"][i]
            energy = (client.train_data[:, :, 7] * ((high - low) or 1) + low).clip(0).sum(1)
            positive = energy[energy > 1e-6]
            self.wm_power.append(float(np.mean(positive) / 3) if len(positive) else 0)
        self.solves = self.failures = 0
        self.solve_seconds = 0.0
        self.solve_times = []
        self.peer_price = 0.1

    def start_day(self, date):
        self.date = date
        self.histories = [[] for _ in self.clients]

    def _predictions(self, states, hour):
        if self.tables is not None:
            return np.stack([table[self.date][hour, hour:] for table in self.tables])
        predictions = []
        for state, client, history, model in zip(
            states, self.clients, self.histories, self.forecasters, strict=True
        ):
            if len(history) != hour:
                raise ValueError("Online dispatch must observe each hour once, in order")
            physical = []
            for coordinate, column in ((3, 0), (2, 1), (7, 4), (1, 3)):
                index = client.scaler["col_to_scaler_idx"][column]
                low, high = client.scaler["min"][index], client.scaler["max"][index]
                physical.append(state[coordinate] * ((high - low) or 1) + low)
            # State price already includes ToU; forecast inputs use base price.
            if self.tou:
                physical[3] -= self.tou[hour]
            history.append(physical)
            predictions.append(model.predict_prefix(history))
        return np.stack(predictions)

    def dispatch(self, states):
        states = np.asarray(states)
        if states.shape != (len(self.clients), 17) or not np.isfinite(states).all():
            raise ValueError("Expected one finite 17-coordinate observation per home")
        hour = int(round(states[0, 0] * 24))
        if not 0 <= hour < 24 or not np.allclose(states[:, 0] * 24, hour, atol=1e-5):
            raise ValueError("All homes must be at the same valid hourly step")
        n, horizon = len(states), 24 - hour
        # Cache tables were predicted from prefixes; this slice has no true future values.
        predictions = self._predictions(states, hour)
        if predictions.shape != (n, horizon, 4) or not np.isfinite(predictions).all():
            raise ValueError("Invalid remaining-day physical forecasts")
        prices = predictions[:, :, 3].copy()
        # All homes share the market's observed base price. Forecast the known
        # ToU changes; separate noisy home-price fits must not create arbitrage.
        prices[:] = prices[:, :1]
        if self.tou:
            prices += np.asarray(self.tou[hour:])[None]
        ac = np.zeros((n, horizon))
        wm = np.zeros_like(ac)
        starts = np.full(n, -1, dtype=int)
        for i, state in enumerate(states):
            temperature, quota = state[9] * 10 + 20, state[15] * 60
            for k in range(horizon):
                natural = 0.7 * temperature + 0.3 * predictions[i, k, 2]
                ac[i, k] = min(
                    2.5, max(0, (natural - self.target) / 3, quota - 2.5 * (horizon - k - 1))
                )
                temperature = natural - 3 * ac[i, k]
                quota = max(0, quota - ac[i, k])
            if state[13] > 0.5:
                remaining = max(0, 3 - int(round(state[14] * 3)))
                wm[i, :remaining] = self.wm_power[i]
            elif state[12] > 0.5:
                choices = list(range(max(hour, 10), 18))
                if choices:
                    # Compare forecast marginal WM cost, including the DR kink.
                    raw = predictions[i, :, 0] - predictions[i, :, 1] + ac[i]
                    marginal = []
                    for start in choices:
                        indices = np.arange(start - hour, start - hour + 3)
                        extra = self.wm_power[i]
                        excess = np.maximum(raw[indices] + extra - 5, 0) - np.maximum(
                            raw[indices] - 5, 0
                        )
                        marginal.append(np.sum(prices[i, indices] * extra + 0.5 * excess))
                    starts[i] = choices[int(np.argmin(marginal))]
                    wm[i, starts[i] - hour : starts[i] - hour + 3] = self.wm_power[i]
        base = predictions[:, :, 0] - predictions[:, :, 1] + ac + wm
        # Per home/hour: EV, battery, SoE, pre-trade import/export, DR excess, curtailment.
        block = n * horizon

        def col(kind, i, k):
            return kind * block + i * horizon + k

        edges = (
            [(i, j) for i in range(n) for j in ((i - 1) % n, (i + 1) % n)]
            if self.peers and n > 2
            else []
        )
        count = 7 * block + len(edges) * horizon
        objective = np.zeros(count)
        bounds = [(0, None)] * count
        export_price = float(self.strategy.get("export_price", 0))
        bonus = float(self.strategy.get("dr_incentive", 0))
        penalty = float(self.strategy.get("dr_penalty", 0))
        limit = float(self.strategy.get("dr_limit", 5))
        eq = lil_matrix((2 * block, count))
        rhs = np.zeros(2 * block)
        ub = lil_matrix((block + 3 * n + 2 * block, count))
        upper = np.zeros(ub.shape[0])
        for i, state in enumerate(states):
            for k in range(horizon):
                t = hour + k
                bounds[col(0, i, k)] = (0, 6 if t < 8 else 0)
                # Conservative feasible discharge: no export from the battery, even if EV plans change.
                bounds[col(1, i, k)] = (-min(2.4, max(base[i, k], 0)), 2.4)
                bounds[col(2, i, k)] = (0, 1)
                bounds[col(4, i, k)] = (0, max(0, -base[i, k]))
                bounds[col(6, i, k)] = (0, max(0, predictions[i, k, 1]))
                objective[col(3, i, k)] = prices[i, k] + bonus
                objective[col(4, i, k)] = -export_price - bonus
                objective[col(5, i, k)] = max(0, penalty - bonus)
                row = i * horizon + k
                for kind, value in ((3, 1), (4, -1), (0, -1), (1, -1), (6, -1)):
                    eq[row, col(kind, i, k)] = value
                rhs[row] = base[i, k]
                eq[block + row, col(2, i, k)] = 1
                eq[block + row, col(1, i, k)] = -0.95 / 6.4
                if k:
                    eq[block + row, col(2, i, k - 1)] = -1
                else:
                    rhs[block + row] = state[8]
                ub[row, col(3, i, k)] = 1
                ub[row, col(4, i, k)] = -1
                ub[row, col(5, i, k)] = -1
                upper[row] = limit
                ub[block + i, col(0, i, k)] = 1
                ub[block + n + i, col(0, i, k)] = -1
                ub[block + 2 * n + i, col(4, i, k)] = 1
                # Total peer exports cannot exceed pre-trade surplus; imports cannot exceed deficit.
                ub[block + 3 * n + row, col(4, i, k)] = -1
                ub[2 * block + 3 * n + row, col(3, i, k)] = -1
            remaining = min(max(0, state[11] * 24), max(0, 1 - state[10]) * 24 / 0.95)
            upper[block + i] = remaining
            upper[block + n + i] = -remaining if hour < 8 else 0
            upper[block + 2 * n + i] = max(0, state[16]) * float(
                self.strategy.get("pv_curtail", 2.5)
            )
        for edge, (seller, buyer) in enumerate(edges):
            for k in range(horizon):
                variable = 7 * block + edge * horizon + k
                if prices[buyer, k] <= self.peer_price or export_price >= self.peer_price:
                    bounds[variable] = (0, 0)
                # Payment adjustments preserve the original pre-trade DR accounting.
                objective[variable] = -(prices[buyer, k] - export_price) + 1e-7
                ub[block + 3 * n + seller * horizon + k, variable] = 1
                ub[2 * block + 3 * n + buyer * horizon + k, variable] = 1
                # Original simulator charges pre-trade grid export AND peer export against cap.
                ub[block + 2 * n + seller, variable] = 1
        started = time.monotonic()
        result = linprog(
            objective,
            A_eq=eq.tocsr(),
            b_eq=rhs,
            A_ub=ub.tocsr(),
            b_ub=upper,
            bounds=bounds,
            method="highs",
        )
        if result.success:
            # Resolve any simultaneous pre-trade import/export allowed by the
            # relaxation. Fix its predicted net-flow signs and solve again.
            ambiguous = any(
                min(result.x[col(3, i, k)], result.x[col(4, i, k)]) > 1e-6
                for i in range(n)
                for k in range(horizon)
            )
            if ambiguous:
                for i in range(n):
                    for k in range(horizon):
                        exporting = result.x[col(4, i, k)] > result.x[col(3, i, k)]
                        bounds[col(3 if exporting else 4, i, k)] = (0, 0)
                result = linprog(
                    objective,
                    A_eq=eq.tocsr(),
                    b_eq=rhs,
                    A_ub=ub.tocsr(),
                    b_ub=upper,
                    bounds=bounds,
                    method="highs",
                )
        elapsed = time.monotonic() - started
        self.solves += 1
        self.solve_seconds += elapsed
        self.solve_times.append(elapsed)
        controls = np.array(
            [teacher(state[None])[0] for teacher, state in zip(self.teachers, states, strict=True)]
        )
        controls[:, 0] = ac[:, 0]
        if result.success:
            for i in range(n):
                controls[i, 1:] = result.x[[col(0, i, 0), col(1, i, 0)]]
        else:
            self.failures += 1
        return [(int(starts[i] == hour), controls[i]) for i in range(n)]
