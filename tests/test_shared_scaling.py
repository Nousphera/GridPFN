import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from gridpfn.core.dataset import _construct_dataset, feature_columns, share_training_scale
from gridpfn.core.em_strategy import compose_em_strategy, make_em_strategy
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.evaluate import ModelEvaluator
from gridpfn.core.schedule import load_home_bundles


class SharedScalingTests(unittest.TestCase):
    def bundles(self, grid_prices=None):
        bundles = []
        for magnitude, pv in ((1, 0), (4, 2)):
            frames = []
            for day, load in (
                ("2019-06-01", magnitude),
                ("2019-06-02", magnitude * 2),
                ("2019-07-20", 100),
            ):
                frame = pd.DataFrame(
                    {"datetime": pd.date_range(day, periods=96, freq="15min"), "t": np.arange(96)}
                )
                for col in feature_columns:
                    frame[col] = 0.0
                frame["fixed_load (kWh)"], frame["pv (kWh)"] = load, pv
                frame["temp (C)"], frame["price ($/kWh)"] = 30, 0.2
                frame["ac (kWh)"] = 0.25
                frame.loc[frame["t"] < 16, "ev (kWh)"] = 0.2
                frame.loc[frame["t"].between(40, 51), "wm (kWh)"] = 0.1
                frames.append(frame)
            with patch("gridpfn.core.dataset.merge_temp_price", side_effect=lambda data, **_: data):
                bundles.append(
                    _construct_dataset(
                        pd.concat(frames),
                        validation_days=14,
                        split="validation",
                        grid_prices=grid_prices,
                    )
                )
        return bundles

    def test_explicit_tariff_round_trips_without_changing_physical_appliances(self):
        prices = [0.1] * 8 + [0.3] * 8 + [0.2] * 8
        original = self.bundles()
        changed = self.bundles(grid_prices=prices)
        for old, new in zip(original, changed, strict=True):
            self.assertEqual(new[3]["grid_prices"], prices)
            for split in (0, 1):
                a = HOME_ENERGY_MGNT(old[split][0], scaler=old[3], state_dim=17)
                b = HOME_ENERGY_MGNT(new[split][0], scaler=new[3], state_dim=17)
                for hour in range(24):
                    self.assertAlmostEqual(b._denorm_feature(3, b.dataset[hour, 3]), prices[hour])
                    a.step((int(hour == 10), [1, 2, 0]))
                    b.step((int(hour == 10), [1, 2, 0]))
                    for key in (
                        "indoor_temp",
                        "SoE_BESS",
                        "SoE_EV",
                        "power_AC",
                        "power_EV",
                        "power_WM",
                        "ac_energy_delivered",
                    ):
                        self.assertAlmostEqual(getattr(a, key), getattr(b, key), places=10)
        for invalid in ([0.1] * 23, [-0.1] * 24, [float("nan")] * 24):
            with self.assertRaisesRegex(ValueError, "24 finite nonnegative"):
                self.bundles(grid_prices=invalid)

    def test_training_extrema_and_constant_columns_preserve_heldout_physics(self):
        local = self.bundles()
        shared = share_training_scale(local)
        fixed_index = shared[0][3]["col_to_scaler_idx"][0]
        self.assertEqual(shared[0][3]["max"][fixed_index], 32)
        self.assertEqual(shared[0][3]["min"], shared[1][3]["min"])
        self.assertEqual(shared[0][3]["max"], shared[1][3]["max"])
        self.assertGreater(shared[0][1][0, 0, 0], 1)  # held-out 400kWh is not clipped
        for before, after in zip(local, shared, strict=True):
            self.assertEqual(before[2], after[2])
            for split in (0, 1):
                old = HOME_ENERGY_MGNT(before[split][0], scaler=before[3], state_dim=17)
                new = HOME_ENERGY_MGNT(after[split][0], scaler=after[3], state_dim=17)
                for col in (0, 1, 3, 4, 5, 6, 7):
                    for a, b in zip(old.dataset[:, col], new.dataset[:, col], strict=True):
                        self.assertAlmostEqual(
                            old._denorm_feature(col, a), new._denorm_feature(col, b), places=10
                        )
                actions = np.random.default_rng(42).uniform([0, 0, -2.4], [2.5, 6, 2.4], (24, 3))
                for action in actions:
                    a, b = old.step((0, action)), new.step((0, action))
                    np.testing.assert_allclose(a[1:4], b[1:4], atol=1e-10)
                    for key in (
                        "indoor_temp",
                        "SoE_BESS",
                        "SoE_EV",
                        "net_load",
                        "power_AC",
                        "power_EV",
                        "power_BESS",
                    ):
                        self.assertAlmostEqual(getattr(old, key), getattr(new, key), places=10)
        strategies = [
            compose_em_strategy(
                make_em_strategy(), [SimpleNamespace(train_data=b[0], scaler=b[3]) for b in group]
            )
            for group in (local, shared)
        ]
        np.testing.assert_allclose(
            strategies[0]["tou"]["hourly_prices"], strategies[1]["tou"]["hourly_prices"], atol=1e-10
        )

    def test_same_physical_load_maps_to_same_shared_coordinate(self):
        shared = share_training_scale(self.bundles())
        coords = []
        for _, _, _, scaler in shared:
            env = HOME_ENERGY_MGNT(np.zeros((24, 8)), scaler=scaler)
            coords.append(env._norm_feature(0, 10))
        self.assertEqual(coords[0], coords[1])

    def test_subset_evaluators_fit_the_original_training_cohort(self):
        data = share_training_scale(self.bundles())
        with patch("gridpfn.core.evaluate.load_data", return_value=data) as load:
            evaluator = ModelEvaluator(
                home_ids=[950], scaler_mode="shared", scaler_home_ids=[27, 950]
            )
            self.assertEqual(load.call_args.kwargs["choose"], ["home_27", "home_950"])
            self.assertIs(evaluator.homes[1].scaler, data[1][3])
        with patch("gridpfn.core.schedule.load_data", return_value=data) as load:
            bundles = load_home_bundles(
                "unused", actual_home_ids=[950], scaler_mode="shared", scaler_home_ids=[27, 950]
            )
            self.assertEqual(load.call_args.kwargs["choose"], ["home_27", "home_950"])
            self.assertIs(bundles[1].scaler, data[1][3])


if __name__ == "__main__":
    unittest.main()
