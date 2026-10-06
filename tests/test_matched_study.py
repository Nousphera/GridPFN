import copy
import importlib.util
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gridpfn.core.utils.original_learning import batch_original_agents
from gridpfn.legacy.plot_matched_study import paired_difference
from gridpfn.legacy.run_matched_study import stability


class MatchedStudyTests(unittest.TestCase):
    def records(self):
        return [
            {"episode": i * 50, "reward": -10, "comfort_pct": 90, "elec_cost": 1, "import": 20}
            for i in range(61)
        ]

    def test_stable_policy_requires_minimum_duration_and_history(self):
        records = self.records()
        self.assertTrue(stability(records)["stopped"])
        self.assertFalse(stability(records[:-1])["stopped"])
        self.assertFalse(stability(records[-20:])["stopped"])

    def test_improving_reward_does_not_count_as_convergence(self):
        records = self.records()
        records[-1]["reward"] += 1
        self.assertFalse(stability(records)["stopped"])

    def test_service_and_energy_drift_prevent_false_plateau(self):
        for key, change in (("comfort_pct", 1), ("elec_cost", 0.2), ("import", 2), ("reward", -2)):
            records = self.records()
            for row in records[-10:]:
                row[key] += change
            self.assertFalse(stability(records)["stopped"], key)

    def test_paired_statistics_use_seed_differences(self):
        result = paired_difference([1, 100, 1000], [2, 101, 1001])
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["seed_sd"], 0)
        self.assertEqual(result["ci95"], [1, 1])
        self.assertIsNone(paired_difference([1], [2])["ci95"])

    def test_paired_statistics_reject_unmatched_or_invalid_seeds(self):
        for left, right in (([1], [2, 3]), ([], []), ([[1]], [[2]]), ([1], [np.nan])):
            with self.assertRaises(ValueError):
                paired_difference(left, right)

    def test_large_oscillations_cannot_hide_behind_equal_window_means(self):
        records = self.records()
        records[0]["reward"] = 10
        for index, row in enumerate(records[-20:]):
            row["reward"] = -10 + (5 if index % 2 else -5)
        self.assertFalse(stability(records)["stopped"])

    def test_original_update_moments_actions_rng_and_aggregation_are_preserved(self):
        self.original_parity("cpu")

    @unittest.skipUnless(os.environ.get("FED_HEMS_GPU_TEST") == "1", "GPU parity is opt-in")
    def test_original_gpu_update_parity(self):
        self.original_parity("cuda:0")

    def original_parity(self, device):
        root = Path(__file__).resolve().parents[1]
        revision = "e34da8b68c646ae3d6c818ad13f2020e6ba06c73"
        saved = {
            name: sys.modules.get(name) for name in ("model", "gridpfn.core.utils.agent_utils", "original_agent")
        }
        with tempfile.TemporaryDirectory() as directory:
            try:
                for name, path in (
                    ("model", "model.py"),
                    ("gridpfn.core.utils.agent_utils", "utils/agent_utils.py"),
                    ("original_agent", "agents/agent.py"),
                ):
                    result = subprocess.run(
                        ["git", "show", f"{revision}:{path}"], cwd=root, capture_output=True
                    )
                    if result.returncode:
                        self.skipTest("Original revision not available in this checkout")
                    file = Path(directory) / (name + ".py")
                    file.write_bytes(result.stdout)
                    spec = importlib.util.spec_from_file_location(name, file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                original_model, original_agent = sys.modules["model"], sys.modules["original_agent"]
            finally:
                for name, module in saved.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module
            torch.set_num_threads(1)
            torch.manual_seed(19)
            references = []
            rng = np.random.default_rng(8)
            for _ in range(3):
                agent = original_agent.P_DQN(
                    original_model.Actor(9, 3).to(device),
                    original_model.Critic(9, 3, 2).to(device),
                    2,
                    3,
                    9,
                    {"batch_size": 8, "epsilon_start": 0.4},
                )
                for index in range(40):
                    state = rng.random(9).astype("float32")
                    agent.store_transition(
                        state,
                        (index % 2, np.array([1, 2, 0])),
                        float(-index % 4),
                        state.copy(),
                        index % 24 == 23,
                    )
                for _ in range(5):
                    agent.learn()
                references.append(agent)
            agents = copy.deepcopy(references)
            batch = batch_original_agents(agents)
            states = rng.random((3, 9)).astype("float32")
            for step in range(6):
                random.seed(step)
                np.random.seed(step)
                expected_actions = [
                    agent.choose_action(s) for agent, s in zip(references, states, strict=True)
                ]
                python_rng, numpy_rng = random.getstate(), np.random.get_state()
                random.seed(step)
                np.random.seed(step)
                actual_actions = batch.choose_actions(states)
                self.assertEqual(random.getstate(), python_rng)
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
                for left, right in zip(expected_actions, actual_actions, strict=True):
                    self.assertEqual(left[0], right[0])
                    np.testing.assert_allclose(left[1], right[1], atol=2e-6, rtol=2e-5)
                random.seed(step)
                expected = [agent.learn() for agent in references]
                random.seed(step)
                actual = batch.learn()
                np.testing.assert_allclose(expected, actual, atol=2e-6, rtol=2e-5)
                for left, right in zip(references, agents, strict=True):
                    for network in (
                        "actor_net",
                        "critic_net",
                        "actor_target_net",
                        "critic_target_net",
                    ):
                        for p, q in zip(
                            getattr(left, network).parameters(),
                            getattr(right, network).parameters(),
                            strict=True,
                        ):
                            torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
                if step == 2:
                    for group in (references, agents):
                        for network in ("actor_net", "critic_net"):
                            states_dict = [getattr(agent, network).state_dict() for agent in group]
                            average = {
                                name: sum(row[name] for row in states_dict) / len(group)
                                for name in states_dict[0]
                            }
                            for agent in group:
                                getattr(agent, network).load_state_dict(average)


if __name__ == "__main__":
    unittest.main()
