import os

import numpy as np

STATE_DIM = 9
CONTROL_STATE_DIM = 17
STATIC_STATE_DIM = 8
CONTINUOUS_ACTION_DIM = 3
DISCRETE_ACTION_DIM = 2
CONTINUOUS_ACTION_MIN = (0.0, 0.0, -2.4)
CONTINUOUS_ACTION_MAX = (2.5, 6.0, 2.4)
ACTION_AC = 0
ACTION_EV = 1
ACTION_BESS = 2
WM_WAIT = 0
WM_START = 1


class HOME_ENERGY_MGNT:
    def __init__(self, dataset, scaler=None, fixed_cost=0.0, state_dim=STATE_DIM):
        if state_dim not in (STATE_DIM, CONTROL_STATE_DIM):
            raise ValueError("state_dim must be 9 (legacy) or 17 (control state)")
        self.dataset = dataset
        self.max_step = len(dataset)
        self.current_step = 0
        self.scaler = scaler or {}
        self.scaler_min = self.scaler.get("min")
        self.scaler_max = self.scaler.get("max")
        self.col_to_scaler_idx = self.scaler.get("col_to_scaler_idx", {})

        self.IDX_FIXED = 0
        self.IDX_PV = 1
        self.IDX_T = 2
        self.IDX_PRICE = 3
        self.IDX_TEMP = 4
        self.IDX_AC = 5
        self.IDX_EV = 6
        self.IDX_WM = 7

        self.STATE_T = 0
        self.STATE_PRICE = 1
        self.STATE_PV = 2
        self.STATE_FIXED = 3
        self.STATE_AC = 4
        self.STATE_EV = 5
        self.STATE_WM = 6
        self.STATE_TEMP = 7
        self.STATE_SOE_BESS = 8

        self.delta_t = float(self.scaler.get("delta_t", 1.0))
        # Device and comfort settings.
        self.thermal_inertia = 0.7
        self.thermal_gain = 10.0
        self.wm_duration = 3.0  # WM cycle hours
        self.max_duration = self.wm_duration
        self.time_ini_WM = 10.0
        self.time_end_WM = 20.0
        self.time_ini_EV = 0.0
        self.time_end_EV = 8.0
        self.temperature_min = 18.0
        self.temperature_max = 22.0
        self.ac_energy_quota = True  # Legacy service model; thermal mode disables this quota.
        self.max_power_AC = CONTINUOUS_ACTION_MAX[ACTION_AC]
        self.max_power_EV = CONTINUOUS_ACTION_MAX[ACTION_EV]
        self.max_power_BESS = CONTINUOUS_ACTION_MAX[ACTION_BESS]
        self.max_capacity_EV = 24.0
        self.max_capacity_BESS = 6.4
        self.efficiency_EV = 0.95
        self.efficiency_BESS = 0.95

        # HEMS objective coefficients.
        self.alpha = 0.005
        self.beta = 0.08
        self.mu = 0.05
        self.sigma = 0.01

        self.eps = 1e-6
        self.tou_prices_by_hour = None
        self.export_price = 0.0
        self.export_cap_kwh = float(os.environ.get("EXPORT_CAP_KWH", 5.0))
        self.fixed_cost = float(fixed_cost)
        self.dr_limit = None
        self.dr_penalty = 0.0
        self.dr_incentive = 0.0

        self.discrete_action_space = DISCRETE_ACTION_DIM
        self.continuous_action_space = CONTINUOUS_ACTION_DIM
        self.state_space = state_dim
        self.reset()

    def _denorm_feature(self, idx, value):
        if self.scaler_min is None or self.scaler_max is None:
            return float(value)
        scaler_idx = self.col_to_scaler_idx.get(idx)
        if scaler_idx is None:
            return float(value)
        vmin = float(self.scaler_min[scaler_idx])
        vmax = float(self.scaler_max[scaler_idx])
        if vmax == vmin:
            # MinMaxScaler uses a unit denominator for a constant training column.
            return float(value) + vmin
        return float(value) * (vmax - vmin) + vmin

    def _norm_feature(self, idx, value):
        if self.scaler_min is None or self.scaler_max is None:
            return float(value)
        scaler_idx = self.col_to_scaler_idx.get(idx)
        if scaler_idx is None:
            return float(value)
        vmin = float(self.scaler_min[scaler_idx])
        vmax = float(self.scaler_max[scaler_idx])
        if vmax == vmin:
            return float(value) - vmin
        return (float(value) - vmin) / (vmax - vmin)

    def _energy_trace(self, idx, max_power=None):
        trace = np.array(
            [max(0.0, self._denorm_feature(idx, row[idx])) for row in self.dataset], dtype=float
        )
        if max_power is not None:
            trace = np.minimum(trace, max_power * self.delta_t)
        return trace

    def _prepare_day_demands(self):
        self.ac_required_energy = float(self._energy_trace(self.IDX_AC, self.max_power_AC).sum())

        ev_trace = self._energy_trace(self.IDX_EV, self.max_power_EV)
        measured_ev_energy = float(ev_trace.sum())
        connection_hours = max(0.0, self.time_end_EV - self.time_ini_EV)
        feasible_ev_energy = min(
            self.max_capacity_EV / self.efficiency_EV, self.max_power_EV * connection_hours
        )
        self.ev_required_energy = min(measured_ev_energy, feasible_ev_energy)

        wm_trace = self._energy_trace(self.IDX_WM)
        cycle_steps = max(1, int(round(self.wm_duration / self.delta_t)))
        self.wm_cycle_profile = np.full(cycle_steps, wm_trace.sum() / cycle_steps)

    def _state_for_step(self, step):
        state = np.zeros(self.state_space, dtype=float)
        state[self.STATE_T] = min(step * self.delta_t / 24.0, 1.0)
        state[self.STATE_SOE_BESS] = self.SoE_BESS
        if self.state_space == CONTROL_STATE_DIM:
            # Physical, home-independent scales. These values bypass TabPFN so
            # action-dependent state never invalidates cached exogenous features.
            export_remaining = (
                1.0
                if self.export_cap_kwh is None
                else max(0.0, self.export_cap_kwh - self.exported_kwh)
                / max(self.export_cap_kwh, self.eps)
            )
            state[STATE_DIM:] = (
                (self.indoor_temp - 20.0) / 10.0,
                self.SoE_EV,
                max(0.0, self.ev_required_energy - self.ev_energy_delivered) / self.max_capacity_EV,
                float(self.wm_pending),
                float(self.wm_running),
                self.wm_cycle_index / max(len(self.wm_cycle_profile), 1),
                (
                    max(0.0, self.ac_required_energy - self.ac_energy_delivered)
                    / (self.max_power_AC * self.max_step * self.delta_t)
                )
                if self.ac_energy_quota
                else 0.0,
                export_remaining,
            )
        if step >= self.max_step:
            return state

        row = self.dataset[step]
        time_hour = float(row[self.IDX_T]) * self.delta_t
        effective_price = self._resolve_price(
            time_hour, self._denorm_feature(self.IDX_PRICE, row[self.IDX_PRICE])
        )
        state[self.STATE_PRICE] = self._norm_feature(self.IDX_PRICE, effective_price)
        state[self.STATE_PV] = row[self.IDX_PV]
        state[self.STATE_FIXED] = row[self.IDX_FIXED]
        state[self.STATE_AC] = row[self.IDX_AC]
        state[self.STATE_EV] = row[self.IDX_EV]
        state[self.STATE_WM] = row[self.IDX_WM]
        state[self.STATE_TEMP] = row[self.IDX_TEMP]
        return state

    def reset(self):
        self.current_step = 0
        self.exported_kwh = 0.0
        self._prepare_day_demands()

        self.indoor_temp = 23.0
        self.expected_temp = self.indoor_temp
        self.ev_energy_delivered = 0.0
        self.SoE_EV = max(
            0.0, 1.0 - self.efficiency_EV * self.ev_required_energy / self.max_capacity_EV
        )
        self.ev_departure_scored = False

        self.wm_required = bool(self.wm_cycle_profile.sum() > self.eps)
        self.wm_pending = self.wm_required
        self.wm_running = False
        self.wm_completed = not self.wm_required
        self.wm_cycle_index = 0
        self.wm_operating_duration = 0.0
        self.identify = 0

        self.SoE_BESS = 0.2
        self.power_AC = 0.0
        self.power_EV = 0.0
        self.power_WM = 0.0
        self.power_BESS = 0.0
        self.ac_energy = 0.0
        self.ac_energy_delivered = 0.0
        self.ev_energy = 0.0
        self.wm_energy = 0.0
        return self._state_for_step(0)

    def step(self, action):
        self.action = self._constrain_action(action)
        self.state_transition(*self.action)
        reward, reward_elec, reward_comf = self.calculate_reward()
        self.current_step += 1
        done = self.check_termination()
        next_state = self._state_for_step(self.current_step)
        return next_state, reward_elec, reward, reward_comf, done

    def _constrain_action(self, action):
        discrete_action, continuous_action = action
        actions = np.asarray(continuous_action, dtype=float).reshape(-1).copy()
        if self.ac_energy_quota:
            future_ac_capacity = (
                (self.max_step - self.current_step - 1) * self.max_power_AC * self.delta_t
            )
            actions[ACTION_AC] = max(
                actions[ACTION_AC],
                (self.ac_required_energy - self.ac_energy_delivered - future_ac_capacity)
                / self.delta_t,
            )

        time_hour = float(self.dataset[self.current_step][self.IDX_T]) * self.delta_t
        if self.time_ini_EV <= time_hour < self.time_end_EV:
            future_ev_capacity = (
                max(0.0, self.time_end_EV - time_hour - self.delta_t) * self.max_power_EV
            )
            actions[ACTION_EV] = max(
                actions[ACTION_EV],
                (self.ev_required_energy - self.ev_energy_delivered - future_ev_capacity)
                / self.delta_t,
            )

        wm_end_step = min(self.max_step, int(round(self.time_end_WM / self.delta_t)))
        if self.wm_pending and self.current_step + len(self.wm_cycle_profile) >= wm_end_step:
            discrete_action = WM_START
        return int(discrete_action), actions

    def _resolve_price(self, time_hour, price_raw):
        base_price = max(0.0, float(price_raw))
        tou_tariff = 0.0
        if (
            isinstance(self.tou_prices_by_hour, (list, tuple, np.ndarray))
            and len(self.tou_prices_by_hour) >= 24
        ):
            hour = int(np.clip(np.floor(time_hour), 0, 23))
            tou_tariff = max(0.0, float(self.tou_prices_by_hour[hour]))
        return base_price + tou_tariff

    def _run_washing_machine(self, discrete_action):
        if self.wm_pending and not self.wm_running and int(discrete_action) == WM_START:
            self.wm_pending = False
            self.wm_running = True
            self.wm_cycle_index = 0

        self.wm_energy = 0.0
        self.identify = int(self.wm_running)
        if self.wm_running:
            self.wm_energy = float(self.wm_cycle_profile[self.wm_cycle_index])
            self.wm_operating_duration += self.delta_t
            self.wm_cycle_index += 1
            if self.wm_cycle_index >= len(self.wm_cycle_profile):
                self.wm_running = False
                self.wm_completed = True
        self.power_WM = self.wm_energy / self.delta_t

    def state_transition(self, discrete_action, continuous_action):
        row = self.dataset[self.current_step]
        step_of_day = float(row[self.IDX_T])
        self.time_hour = step_of_day * self.delta_t
        self.price = self._resolve_price(
            self.time_hour, self._denorm_feature(self.IDX_PRICE, row[self.IDX_PRICE])
        )
        self.pv_energy = max(0.0, self._denorm_feature(self.IDX_PV, row[self.IDX_PV]))
        self.fixed_load_energy = max(0.0, self._denorm_feature(self.IDX_FIXED, row[self.IDX_FIXED]))
        self.pv_generation = self.pv_energy / self.delta_t
        self.fixed_load = self.fixed_load_energy / self.delta_t
        self.outdoor_temp = self._denorm_feature(self.IDX_TEMP, row[self.IDX_TEMP])
        self.temperature = self.outdoor_temp

        measured_ac_energy = max(0.0, self._denorm_feature(self.IDX_AC, row[self.IDX_AC]))
        self.ac_baseline_power = min(measured_ac_energy / self.delta_t, self.max_power_AC)

        actions = np.asarray(continuous_action, dtype=float).reshape(-1)
        self.power_AC = float(
            np.clip(actions[ACTION_AC], CONTINUOUS_ACTION_MIN[ACTION_AC], self.max_power_AC)
        )
        self.ac_energy = self.power_AC * self.delta_t
        self.ac_energy_delivered += self.ac_energy
        self.expected_temp = self.thermal_inertia * self.indoor_temp + (
            1.0 - self.thermal_inertia
        ) * (self.outdoor_temp - self.thermal_gain * self.power_AC)
        self.indoor_temp = self.expected_temp

        remaining_ev_energy = max(0.0, self.ev_required_energy - self.ev_energy_delivered)
        ev_connected = self.time_ini_EV <= self.time_hour < self.time_end_EV
        requested_ev_power = (
            float(np.clip(actions[ACTION_EV], 0.0, self.max_power_EV)) if ev_connected else 0.0
        )
        capacity_limited_power = (
            (1.0 - self.SoE_EV) * self.max_capacity_EV / (self.efficiency_EV * self.delta_t)
        )
        self.power_EV = min(
            requested_ev_power, remaining_ev_energy / self.delta_t, capacity_limited_power
        )
        self.ev_energy = self.power_EV * self.delta_t
        self.ev_energy_delivered += self.ev_energy
        self.SoE_EV = float(
            np.clip(
                self.SoE_EV + self.efficiency_EV * self.ev_energy / self.max_capacity_EV, 0.0, 1.0
            )
        )

        self._run_washing_machine(discrete_action)

        self.power_BESS = float(
            np.clip(actions[ACTION_BESS], -self.max_power_BESS, self.max_power_BESS)
        )

        # Battery discharge may serve on-site demand, but it must not create an export.
        if self.power_BESS < 0.0:
            onsite_demand = self.fixed_load + self.power_AC + self.power_EV + self.power_WM
            max_onsite_discharge = max(0.0, onsite_demand - self.pv_generation)
            self.power_BESS = max(self.power_BESS, -max_onsite_discharge)

        next_soe = (
            self.SoE_BESS
            + self.efficiency_BESS * self.power_BESS * self.delta_t / self.max_capacity_BESS
        )
        if next_soe > 1.0:
            self.power_BESS = (
                (1.0 - self.SoE_BESS)
                * self.max_capacity_BESS
                / (self.efficiency_BESS * self.delta_t)
            )
        elif next_soe < 0.0:
            self.power_BESS = (
                -self.SoE_BESS * self.max_capacity_BESS / (self.efficiency_BESS * self.delta_t)
            )
        self.SoE_BESS = float(
            np.clip(
                self.SoE_BESS
                + self.efficiency_BESS * self.power_BESS * self.delta_t / self.max_capacity_BESS,
                0.0,
                1.0,
            )
        )

    def calculate_reward(self):
        load_without_pv = (
            self.fixed_load + self.power_AC + self.power_EV + self.power_WM + self.power_BESS
        )
        elec = load_without_pv - self.pv_generation

        if elec < 0 and self.export_cap_kwh is not None:
            cap_remaining = max(0.0, self.export_cap_kwh - self.exported_kwh)
            potential_export_kwh = -elec * self.delta_t
            actual_export_kwh = min(potential_export_kwh, cap_remaining)
            self.exported_kwh += actual_export_kwh
            if actual_export_kwh < potential_export_kwh:
                elec = -actual_export_kwh / max(self.delta_t, self.eps)

        if elec > 0:
            reward_elec = -elec * self.price * self.delta_t
        elif self.export_price > 0:
            reward_elec = -elec * self.export_price * self.delta_t
        else:
            reward_elec = 0.0

        if self.dr_limit is not None:
            reward_elec += (
                -self.dr_penalty * max(0.0, elec - self.dr_limit)
                + self.dr_incentive * max(0.0, self.dr_limit - elec)
            ) * self.delta_t

        fixed_cost_per_step = self.fixed_cost / 30.0 / max(self.max_step, 1)
        reward_elec -= fixed_cost_per_step
        if self.identify:
            if self.time_hour < self.time_ini_WM:
                reward_pre_wm = -self.alpha * (self.time_ini_WM - self.time_hour) ** 2
            elif self.time_hour > self.time_end_WM:
                reward_pre_wm = -self.alpha * (self.time_hour - self.time_end_WM) ** 2
            else:
                reward_pre_wm = 0.0
        else:
            reward_pre_wm = 0.0

        if self.current_step == self.max_step - 1 and self.wm_required:
            missing_duration = max(0.0, self.max_duration - self.wm_operating_duration)
            reward_use_wm = -self.sigma * missing_duration**2
        else:
            reward_use_wm = 0.0

        if self.expected_temp < self.temperature_min:
            reward_temp = -self.beta * (self.temperature_min - self.expected_temp) ** 2
        elif self.expected_temp > self.temperature_max:
            reward_temp = -self.beta * (self.expected_temp - self.temperature_max) ** 2
        else:
            reward_temp = 0.0

        interval_end = self.time_hour + self.delta_t
        if not self.ev_departure_scored and interval_end >= self.time_end_EV:
            reward_travel = -self.mu * (1.0 - self.SoE_EV)
            self.ev_departure_scored = True
        else:
            reward_travel = 0.0

        reward_comf = reward_pre_wm + reward_use_wm + reward_temp + reward_travel

        self.net_load = elec
        self.pv_surplus = max(0.0, self.pv_generation - max(load_without_pv, 0.0))
        if self.export_cap_kwh is not None:
            cap_remaining = max(0.0, self.export_cap_kwh - self.exported_kwh)
            self.pv_surplus = min(self.pv_surplus * self.delta_t, cap_remaining) / max(
                self.delta_t, self.eps
            )

        reward = reward_elec + reward_comf
        return reward, reward_elec, reward_comf

    def check_termination(self):
        return self.current_step >= self.max_step
