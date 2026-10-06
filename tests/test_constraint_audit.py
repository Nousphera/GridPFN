"""Verify the independent reference observation used for the original-policy audit."""

import unittest

import numpy as np

from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.experiments.audit_constraints import physical_day, reference_observation


class ConstraintAuditTests(unittest.TestCase):
    def test_reference_observation_matches_at_hourly_and_quarter_hourly_resolution(self):
        for steps in (24, 96):
            with self.subTest(steps=steps):
                dt = 24 / steps
                rng = np.random.default_rng(71)
                raw = rng.uniform(0, 0.3, size=(steps, 8))
                raw[:, 2] = np.arange(steps)
                raw[:, 4] = 25
                low = np.array([0, 0, 0.1, 20, 0, 0, 0])
                high = np.array([1, 1, 0.5, 20, 2.5, 6, 1])  # Constant weather fit.
                mapping = {0: 0, 1: 1, 2: None, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
                scaler = {
                    "min": low.tolist(),
                    "max": high.tolist(),
                    "col_to_scaler_idx": mapping,
                    "delta_t": dt,
                }
                normalized = raw.copy()
                for column, index in mapping.items():
                    if index is not None:
                        normalized[:, column] = (raw[:, column] - low[index]) / (
                            (high[index] - low[index]) or 1
                        )
                np.testing.assert_allclose(physical_day(normalized, scaler), raw)
                actual = HOME_ENERGY_MGNT(normalized, scaler=scaler, state_dim=17)
                reference = HOME_ENERGY_MGNT(raw, scaler={"delta_t": dt})
                for _ in range(steps):
                    np.testing.assert_allclose(
                        reference_observation(reference, scaler),
                        actual._state_for_step(actual.current_step),
                        atol=1e-6,
                    )
                    np.testing.assert_allclose(
                        reference_observation(reference, scaler, dtype=np.float64),
                        actual._state_for_step(actual.current_step),
                        atol=1e-12,
                        rtol=0,
                    )
                    action = (1, [0.5, 2.0, -0.5])
                    actual.step(action)
                    reference.step(action)


if __name__ == "__main__":
    unittest.main()
