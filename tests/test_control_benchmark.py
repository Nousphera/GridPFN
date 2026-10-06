import unittest

import numpy as np

from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.utils.rollout_metrics import calculate_metrics
from gridpfn.legacy.control_benchmark import thermal_limit


class ControlBenchmarkTests(unittest.TestCase):
    def environment(self, temperature=30, ac=0):
        day = np.zeros((24, 8))
        day[:, 2] = np.arange(24)
        day[:, 4] = temperature
        day[:, 5] = ac
        return HOME_ENERGY_MGNT(day, state_dim=17)

    def test_certified_schedule_replays_exactly_and_meets_original_quota(self):
        env = self.environment(ac=1)
        result = thermal_limit(env)
        actual = []
        for power in result["ac_power"]:
            env.step((0, [power, 0, 0]))
            actual.append(env.indoor_temp)
        np.testing.assert_allclose(actual, result["temperatures"], atol=1e-7)
        self.assertGreaterEqual(env.ac_energy_delivered + 1e-7, env.ac_required_energy)
        self.assertEqual(result["mip_gap"], 0)
        self.assertEqual(result["comfort_limit_pct"], 100)

    def test_unavoidable_cold_and_maximum_quota_are_not_hidden(self):
        for temperature, ac in [(0, 0), (20, 2.5)]:
            result = thermal_limit(self.environment(temperature, ac))
            self.assertEqual(result["minimum_violations"], 24)
            self.assertEqual(result["comfort_limit_pct"], 0)

    def test_comfort_metrics_distinguish_duration_severity_and_peak(self):
        def metrics(temperature):
            return calculate_metrics(
                {
                    "episode_elec_cost": 0,
                    "episode_reward": 0,
                    "episode_comfort": 0,
                    "delta_t": 1,
                    "devices": {},
                    "temperature": {"indoor": temperature, "min": 18, "max": 22},
                    "powers": {
                        k: [0] * 4
                        for k in ("net", "import", "export", "AC (kW)", "AC baseline (kW)")
                    },
                }
            )

        long, severe = metrics([17, 17, 17, 17]), metrics([16, 20, 20, 20])
        self.assertEqual(long["squared_violation"], severe["squared_violation"])
        self.assertEqual((long["violation_hours"], severe["violation_hours"]), (4, 1))
        self.assertEqual((long["peak_violation_degrees"], severe["peak_violation_degrees"]), (1, 2))
        self.assertEqual((long["degree_hours"], severe["degree_hours"]), (4, 2))


if __name__ == "__main__":
    unittest.main()
