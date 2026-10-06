import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gridpfn.core.control_guidance import FeedbackTeacher
from gridpfn.core.economic_control import EconomicCoordinator
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.forecasting import DirectForecaster, causal_queries
from gridpfn.legacy.control_benchmark import rollout_controllers
from gridpfn.legacy.run_energy_study import comparison


class EnergySchedulingTests(unittest.TestCase):
    def test_cost_gate_requires_bill_savings_and_preserves_every_home(self):
        def record(cost, bill, reward, comfort=92):
            return {
                "homes": [
                    {
                        "elec_cost": cost,
                        "energy_bill_without_dr": bill,
                        "reward": reward,
                        "comfort_pct": comfort,
                        "ev_completion_ratio": 1,
                        "wm_completed": 1,
                        "energy_balance_max_abs_kw": 0,
                    }
                ]
            }

        reference = record(1, 1.5, -20)
        self.assertFalse(comparison(reference, record(0.5, 1.6, -19))["gate"])
        self.assertTrue(comparison(reference, record(0.5, 1.3, -19))["gate"])
        self.assertFalse(comparison(reference, record(0.5, 1.3, -19, 91.8))["gate"])

    def test_online_prefix_matches_batch_replay_for_every_origin(self):
        days = np.random.default_rng(12).uniform(0, 2, (4, 24, 4))
        for kind in ("persistence", "seasonal", "trees"):
            model = DirectForecaster(kind).fit(days[:3])
            table = model.predict_table(days[3:])[0]
            for hour in range(24):
                np.testing.assert_allclose(
                    model.predict_prefix(days[3, : hour + 1]),
                    table[hour, hour:],
                    atol=1e-12,
                    rtol=0,
                )
        with self.assertRaises(ValueError):
            model.predict_prefix([[np.nan] * 4])
        with self.assertRaises(RuntimeError):
            DirectForecaster().predict_prefix([[1] * 4])

    def test_forecast_origin_never_reads_future_observations(self):
        rng = np.random.default_rng(18)
        days = rng.uniform(0, 1, (3, 24, 4))
        fitted = DirectForecaster("trees").fit(days[:2])
        before = fitted.predict_table(days[2:])
        changed = days[2:].copy()
        changed[:, 9:] += 100
        after = fitted.predict_table(changed)
        np.testing.assert_allclose(before[:, 8, 8:], after[:, 8, 8:], atol=1e-12, rtol=0)
        x, _, indices = causal_queries(days[2:])
        modified, _, _ = causal_queries(changed)
        np.testing.assert_array_equal(x[indices[:, 1] <= 8], modified[indices[:, 1] <= 8])

    def fixture(self):
        day = np.zeros((24, 8))
        day[:, 0], day[:, 3], day[:, 4], day[:, 5] = 2, 0.3, 30, 1
        day[:, 2] = np.arange(24)
        day[:8, 6] = 1
        day[10:13, 7] = 0.3
        scaler = {
            "col_to_scaler_idx": {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},
            "min": [0] * 7,
            "max": [1] * 7,
            "delta_t": 1,
        }
        client = SimpleNamespace(
            train_data=np.stack([day] * 3),
            test_data=day[None],
            test_dates=["day"],
            scaler=scaler,
            fixed_cost=5,
        )
        truth = day[None][:, :, [0, 1, 4, 3]]
        table = (
            DirectForecaster("persistence").fit(np.repeat(truth, 2, axis=0)).predict_table(truth)
        )
        return [client] * 3, [{"day": table[0]}] * 3

    def test_dispatch_preserves_comfort_deadlines_and_energy_conservation(self):
        clients, tables = self.fixture()
        strategy = {
            "dr_limit": 5,
            "dr_penalty": 0.5,
            "dr_incentive": 0.025,
            "pv_curtail": 2.5,
            "export_price": 0.025,
        }
        coordinator = EconomicCoordinator(clients, strategy, tables, peers=True)
        result = rollout_controllers(clients, strategy, coordinator)
        self.assertEqual(coordinator.failures, 0)
        for home in result["homes"]:
            self.assertEqual(home["comfort_pct"], 100)
            self.assertAlmostEqual(home["ev_completion_ratio"], 1, places=9)
            self.assertEqual(home["wm_completed"], 1)
            self.assertLess(home["energy_balance_max_abs_kw"], 1e-9)
            self.assertLess(home["billing_reconciliation_error"], 1e-9)

    def test_positive_peer_trades_reconcile_physical_flows_and_payments(self):
        clients, _ = self.fixture()
        solar = clients[0].test_data.copy()
        solar[0, 12, 1] = 4.5
        clients[0] = SimpleNamespace(**(vars(clients[0]) | {"test_data": solar}))
        teachers = [FeedbackTeacher(c.scaler, 18.2) for c in clients]
        policies = [
            lambda state, t=t: (int(state[0] * 24 >= 10), t(state[None])[0]) for t in teachers
        ]
        record = rollout_controllers(clients, {"pv_curtail": 2.5, "export_price": 0.025}, policies)
        self.assertGreater(sum(h["p2p_kwh"] for h in record["homes"]), 0)
        for home in record["homes"]:
            self.assertLess(home["energy_balance_max_abs_kw"], 1e-9)
            self.assertLess(home["billing_reconciliation_error"], 1e-9)

    def test_live_forecasts_and_causal_replay_produce_same_rollout(self):
        clients, tables = self.fixture()
        truth = clients[0].train_data[:, :, [0, 1, 4, 3]]
        models = [DirectForecaster("persistence").fit(truth) for _ in clients]
        strategy = {
            "pv_curtail": 2.5,
            "tou": {"hourly_prices": np.linspace(0.01, 0.05, 24).tolist()},
        }
        replay = rollout_controllers(
            clients, strategy, EconomicCoordinator(clients, strategy, tables)
        )
        online = rollout_controllers(
            clients, strategy, EconomicCoordinator(clients, strategy, forecasters=models)
        )
        for a, b in zip(replay["homes"], online["homes"], strict=True):
            for key in ("elec_cost", "comfort_pct", "import", "export", "reward"):
                self.assertAlmostEqual(a[key], b[key], places=5)

    def test_failed_solver_uses_safe_feedback_and_completes_tasks(self):
        clients, tables = self.fixture()
        coordinator = EconomicCoordinator(clients, {"pv_curtail": 2.5}, tables)
        with patch("gridpfn.core.economic_control.linprog", return_value=SimpleNamespace(success=False)):
            record = rollout_controllers(clients, {"pv_curtail": 2.5}, coordinator)
        self.assertEqual(coordinator.failures, 24)
        self.assertTrue(all(h["task_success_pct"] == 100 for h in record["homes"]))

    def test_planner_immediate_control_uses_observed_weather_and_quota(self):
        clients, tables = self.fixture()
        coordinator = EconomicCoordinator(clients, {"pv_curtail": 2.5}, tables)
        coordinator.start_day("day")
        env = HOME_ENERGY_MGNT(clients[0].test_data[0], scaler=clients[0].scaler, state_dim=17)
        state = env.reset()
        actions = coordinator.dispatch([state] * 3)
        self.assertAlmostEqual(actions[0][1][0], (0.7 * 23 + 0.3 * 30 - 18.2) / 3, places=6)


if __name__ == "__main__":
    unittest.main()
