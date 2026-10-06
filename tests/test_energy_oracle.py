"""Analytical scheduling cases and independent main-branch market replay."""

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from gridpfn.core.environment import HOME_ENERGY_MGNT
from oracle import OracleConfig, load_config
from oracle.report import matched_policies
from oracle.runner import _digest, _inputs, _solve_day, read_record
from oracle.solver import (
    SERVICE_TOLERANCE,
    _require_solution,
    comfort_frontier,
    replay_oracle,
    solve_oracle,
)


@unittest.skipUnless(
    importlib.util.find_spec("pyscipopt"), "Optional oracle solver is not installed"
)
class EnergyOracleTests(unittest.TestCase):
    def environment(self, price=0.2, temperature=20):
        day = np.zeros((24, 8))
        day[:, 2] = np.arange(24)
        day[:, 3] = price
        day[:, 4] = temperature
        env = HOME_ENERGY_MGNT(day, state_dim=17)
        env.indoor_temp = temperature
        env.max_power_BESS = 0
        env.export_price = 0.025
        env.export_cap_kwh = 2.5
        return env

    def solve_and_audit(self, envs, objective="paper_reward", **options):
        result = solve_oracle(envs, objective, time_limit=20, **options)
        audit = replay_oracle(envs, result, reference=True)
        self.assertTrue(result["certified"])
        self.assertTrue(audit["verified"])
        return result, audit

    def test_washing_machine_selects_analytically_cheapest_complete_cycle(self):
        env = self.environment()
        env.dataset[10:13, 3] = 0.01
        env.dataset[0, 7] = 6
        env.reset()
        env.indoor_temp = 20
        result, audit = self.solve_and_audit([env])
        self.assertEqual(result["solutions"][0]["wm_start"], 10)
        self.assertAlmostEqual(audit["objective"], 0.06, places=5)
        self.assertEqual(audit["homes"][0]["wm_completed"], 1)

    def test_ev_charges_at_cheapest_hours_and_respects_departure(self):
        env = self.environment()
        env.dataset[:2, 3] = 0.01
        env.dataset[:2, 6] = 6
        env.reset()
        env.indoor_temp = 20
        result, audit = self.solve_and_audit([env])
        powers = np.array(result["solutions"][0]["controls"])[:, 1]
        np.testing.assert_allclose(powers[:2], [6, 6], atol=2e-5)
        np.testing.assert_allclose(powers[8:], 0, atol=1e-7)
        self.assertAlmostEqual(audit["objective"], 0.12, places=5)

    def test_reward_optimum_can_rationally_violate_comfort(self):
        env = self.environment()
        env.indoor_temp = 23
        reward, reward_audit = self.solve_and_audit([env])
        service, service_audit = self.solve_and_audit([env], "comfort_first")
        self.assertAlmostEqual(reward["solutions"][0]["controls"][0][0], 0, places=5)
        self.assertAlmostEqual(reward_audit["objective"], 0.08 * 0.1**2, places=6)
        self.assertLessEqual(
            service_audit["homes"][0]["squared_violation"], SERVICE_TOLERANCE + 2e-5
        )
        self.assertGreater(
            service_audit["homes"][0]["energy_bill_without_dr"],
            reward_audit["homes"][0]["energy_bill_without_dr"],
        )
        # The second-stage service tolerance is explicit, not an exact 100% claim.
        self.assertGreater(service_audit["homes"][0]["comfort_pct"], 90)

    def test_original_quota_can_make_comfort_unachievable(self):
        env = self.environment()
        env.dataset[:, 5] = 2.5
        env.reset()
        frontier = comfort_frontier(env, time_limit=20)
        self.assertEqual(frontier["maximum_comfort_pct"], 0)
        self.assertGreater(frontier["minimum_squared_violation"], 100)
        result, audit = self.solve_and_audit([env], "comfort_first", frontiers=[frontier])
        self.assertAlmostEqual(audit["homes"][0]["ac_delivered_kwh"], 60, places=4)
        self.assertEqual(audit["homes"][0]["comfort_pct"], 0)

    def market(self, solar):
        seller, buyer = self.environment(), self.environment()
        seller.dataset[12, 1] = solar
        buyer.dataset[12, 0] = 2
        for env in (seller, buyer):
            env.max_power_AC = 0
            env.state_space = 9  # No AC observation normalization in this market-only fixture.
            env.reset()
            env.indoor_temp = 20
        return [seller, buyer]

    def test_greedy_peer_market_and_legacy_double_export_accounting(self):
        result, audit = self.solve_and_audit(self.market(2))
        self.assertTrue(result["coupled_market"])
        self.assertAlmostEqual(audit["homes"][1]["p2p_kwh"], 0.5, places=5)
        self.assertAlmostEqual(audit["objective"], 0.2 * 1.5 - 0.025 * 1.5, places=5)

    def test_cannot_withhold_grid_exports_to_invent_peer_trade(self):
        _, audit = self.solve_and_audit(self.market(4))
        self.assertAlmostEqual(audit["homes"][1]["p2p_kwh"], 0, places=5)
        self.assertAlmostEqual(audit["objective"], 0.2 * 2 - 0.025 * 2.5, places=5)
        self.assertAlmostEqual(audit["homes"][0]["pv_curtailed_kwh"], 1.5, places=5)

    def test_ineligible_peer_tariff_factorizes_the_community(self):
        envs = self.market(2)
        for env in envs:
            env.dataset[:, 3] = 0.05
        result, audit = self.solve_and_audit(envs)
        self.assertFalse(result["coupled_market"])
        self.assertEqual(len(result["certificates"]), 2)
        self.assertAlmostEqual(audit["objective"], 0.05 * 2 - 0.025 * 2, places=5)

    def test_three_home_ring_keeps_left_neighbour_priority(self):
        seller, right = self.market(2)
        left = copy.deepcopy(right)
        _, audit = self.solve_and_audit([seller, right, left])
        self.assertAlmostEqual(audit["homes"][1]["p2p_kwh"], 0, places=5)
        self.assertAlmostEqual(audit["homes"][2]["p2p_kwh"], 0.5, places=5)

    def test_battery_uses_original_efficiency_and_free_daily_initial_charge(self):
        env = self.environment()
        env.dataset[12, 0] = 3
        env.max_power_BESS = 2.4
        env.reset()
        env.indoor_temp = 20
        _, audit = self.solve_and_audit([env])
        available = 0.2 * 6.4 / 0.95
        self.assertAlmostEqual(audit["objective"], 0.2 * (3 - available), places=5)
        self.assertAlmostEqual(audit["homes"][0]["final_battery_soe"], 0, places=5)

    def test_dr_incentive_and_penalty_match_analytical_cost(self):
        for load, daily_cost in ((4, 18.6), (6, 40.8)):
            with self.subTest(load=load):
                env = self.environment()
                env.dataset[:, 0] = load
                env.dr_limit, env.dr_incentive, env.dr_penalty = 5, 0.025, 0.5
                env.reset()
                env.indoor_temp = 20
                _, audit = self.solve_and_audit([env])
                self.assertAlmostEqual(audit["objective"], daily_cost, places=5)

    def test_independent_replay_rejects_a_modified_schedule(self):
        env = self.environment()
        result = solve_oracle([env], time_limit=20)
        changed = copy.deepcopy(result)
        changed["solutions"][0]["controls"][0][0] = 2.5
        with self.assertRaises(AssertionError):
            replay_oracle([env], changed, reference=True)

    def test_frontier_cannot_be_reused_for_different_weather(self):
        env = self.environment()
        frontier = comfort_frontier(env, time_limit=20)
        env.dataset[:, 4] = 35
        with self.assertRaisesRegex(ValueError, "frontier differs"):
            solve_oracle([env], frontiers=[frontier], time_limit=20)

    def test_cached_frontiers_skip_resolution_but_reject_thermal_changes(self):
        config = replace(
            OracleConfig(),
            home_ids=(27,),
            objectives=("paper_reward",),
            tou_enabled=False,
            workers=1,
            time_limit=20,
        )
        day = np.zeros((24, 8))
        day[:, 2], day[:, 3], day[:, 4] = np.arange(24), 0.2, 20
        original = _solve_day(("test", [day], {}, config))
        with patch("oracle.runner.comfort_frontier", side_effect=AssertionError("Recomputed")):
            repeated = _solve_day(("test", [day], {}, config, original["frontiers"]))
            self.assertAlmostEqual(
                original["oracles"]["paper_reward"]["audit"]["objective"],
                repeated["oracles"]["paper_reward"]["audit"]["objective"],
                places=5,
            )
            changed = day.copy()
            changed[12, 4] += 1
            with self.assertRaisesRegex(ValueError, "frontier differs"):
                _solve_day(("test", [changed], {}, config, original["frontiers"]))


class OracleCertificateTests(unittest.TestCase):
    def test_timeout_incumbent_is_not_reported_as_optimal(self):
        model = Mock()
        model.getNSols.return_value = 1
        model.getDualbound.return_value = 0
        model.getPrimalbound.return_value = 1
        model.getGap.return_value = 1
        model.getSolvingTime.return_value = 60
        model.getNNodes.return_value = 50
        model.getStatus.return_value = "timelimit"
        with self.assertRaisesRegex(RuntimeError, "Uncertified"):
            _require_solution(model)


class OracleConfigurationTests(unittest.TestCase):
    def test_default_file_matches_default_physical_settings(self):
        config = load_config()
        defaults = OracleConfig()
        self.assertEqual(replace(config, output=defaults.output), defaults)

    def test_unknown_key_is_rejected_and_paths_are_relative_to_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenario.toml"
            path.write_text('output="trial"\ntemperatur_min=19\n')
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(path)
            path.write_text('output="trial"\n')
            self.assertEqual(load_config(path).output, path.parent / "trial")

    def test_conflicting_prices_and_invalid_physics_are_rejected(self):
        invalid = (
            {"flat_price_per_kwh": 0.2},
            {"tou_enabled": False, "hourly_prices_per_kwh": [0.2] * 23},
            {"tou_enabled": False, "flat_price_per_kwh": -0.2},
            {"temperature_min": 23, "temperature_max": 22},
            {"wm_duration": 11},
            {"initial_battery_soe": 2},
            {"efficiency_BESS": 0},
            {"base_price_multiplier": float("nan")},
            {"peer_trading": "false"},
            {"home_ids": [27, 27]},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                OracleConfig(**options)

    def test_price_multiplier_scales_both_base_and_training_fitted_tou(self):
        import pandas as pd

        from demo import generate_data
        from gridpfn.core.dataset import _construct_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_data(root)
            bundle = _construct_dataset(
                pd.read_csv(root / "split_homes_clean/home_101.csv"),
                temp_price_path=root / "temp_price_newyork.csv",
                validation_days=14,
                split="validation",
            )
            base = OracleConfig(data_dir=root / "split_homes_clean", home_ids=(101,), days=1)
            with patch("oracle.runner.load_data", return_value=[bundle]):
                original, strategy = _inputs(base)
                doubled, scaled = _inputs(replace(base, base_price_multiplier=2))
                np.testing.assert_allclose(
                    doubled[0].test_data[..., 3], 2 * original[0].test_data[..., 3]
                )
                np.testing.assert_allclose(
                    scaled["tou"]["hourly_prices"], 2 * np.array(strategy["tou"]["hourly_prices"])
                )
                flat, tariff = _inputs(replace(base, tou_enabled=False, flat_price_per_kwh=0.2))
                np.testing.assert_allclose(flat[0].test_data[..., 3], 0.2)
                self.assertFalse(tariff["tou"]["enabled"])
                hourly, _ = _inputs(
                    replace(
                        base,
                        tou_enabled=False,
                        hourly_prices_per_kwh=np.linspace(0.1, 0.3, 24).tolist(),
                    )
                )
                np.testing.assert_allclose(hourly[0].test_data[0, :, 3], np.linspace(0.1, 0.3, 24))

    def test_corrupted_date_receipt_is_rejected(self):
        protocol = {"scenario": "test"}
        record = {"date": "2019-08-01", "protocol_sha256": _digest(protocol)}
        record["receipt_sha256"] = _digest(record)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "day.json"
            path.write_text(json.dumps(record))
            self.assertEqual(read_record(path, protocol)["date"], "2019-08-01")
            record["date"] = "2019-08-02"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                read_record(path, protocol)

    def test_changed_scenario_cannot_be_compared_with_old_rl_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            (study / "study.json").write_text("{}")
            protocol = {"scenario": OracleConfig(temperature_min=19).settings()}
            with self.assertRaisesRegex(ValueError, "re-evaluated"):
                matched_policies(study, protocol, [])

    @unittest.skipUnless(importlib.util.find_spec("pyscipopt"), "Oracle solver is optional")
    def test_configured_thermal_battery_and_nonconvex_dr_scenario_replays(self):
        config = OracleConfig(
            home_ids=(27,),
            temperature_min=19,
            temperature_max=23,
            initial_temperature=21,
            initial_battery_soe=0.5,
            ac_energy_quota=False,
            dr_limit=1,
            dr_penalty=0.01,
            dr_incentive=0.1,
            fixed_cost=0,
            tou_enabled=False,
            max_power_BESS=0,
            time_limit=20,
        )
        day = np.zeros((24, 8))
        day[:, 0], day[:, 2], day[:, 3], day[:, 4] = 2, np.arange(24), 0.2, 21
        day[:, 5] = 2.5  # A real nonzero quota must be disabled in both simulators.
        strategy = {
            "dr_limit": 1,
            "dr_penalty": 0.01,
            "dr_incentive": 0.1,
            "pv_curtail": 2.5,
            "export_price": 0.025,
        }
        result = _solve_day(("synthetic", [day], strategy, config))
        for objective in config.objectives:
            audit = result["oracles"][objective]["audit"]
            self.assertTrue(audit["verified"])
            self.assertTrue(audit["legacy_ac_quota_disabled_by_zero_budget"])
            self.assertAlmostEqual(audit["homes"][0]["elec_cost"], 9.84, places=5)
            self.assertEqual(audit["homes"][0]["comfort_pct"], 100)


if __name__ == "__main__":
    unittest.main()
