# Oracle

Perfect-foresight, multi-home scheduling with numerical optimality bounds and independent simulator replay. The package reuses the repository's data and audited home/market physics; it never trains TabPFN or consumes GPU memory.

## Run a scenario

Use the project environment from the repository root. Edit **[config.toml](config.toml)**; give each experiment a fresh `output` directory.

```bash
python -m oracle validate --config oracle/config.toml
python -m oracle run --config oracle/config.toml
python -m oracle status results/oracle/original_validation
python -m oracle report results/oracle/original_validation
```

Reports contain two rows of daily plots, with home means and shaded home SD. Results include `oracle.json` (complete settings and provenance), `days/*.json` (plans, certificates and independent audits), `summary.json`, `status.json` and append-only progress. Raw runs remain in ignored `results/`.

```bash
python -m oracle run --split test --output results/oracle/original_test
python -m oracle run --split train --output results/oracle/original_train
python -m oracle run --resume
```

Resume requires identical settings, sources, data and solver versions; date receipts detect corruption or mixed scenarios. Completed dates are reused. Matched RL comparisons reject changed physical or tariff settings: evaluate the policies under the new scenario first. The default run covers 13 shared July validation dates; `test` covers 31 August dates and `train` covers the 47 earlier complete days. Only an explicit training run can supply expert labels. CLI overrides are limited to output, split, day count, workers and solve time; scenario parameters belong in the config.

## Configuration

`--frontier-reference` reuses completed, receipt-checked thermal certificates,
validating exact home/weather/physics signatures before solving. Missing
homes or dates are solved normally.

All prices are $/kWh, power is kW, energy is kWh and temperatures are °C. Paths in TOML resolve relative to that file.

| Settings | Meaning |
|---|---|
| `base_price_multiplier`, `tou_enabled`, `tou_blocks` | Scale recorded prices and, when enabled, fit additive ToU prices on training data only |
| `flat_price_per_kwh` or `hourly_prices_per_kwh` | Override the **total** grid price; disable ToU, retain multiplier 1; hourly overrides need 24 nonnegative values |
| `export_price`, `export_cap_*`, `peer_price`, `peer_trading` | Original greedy ring settlement and explicit export budget |
| `dr_*`, `fixed_cost` | DR threshold, incentive/penalty and monthly fixed charge |
| `temperature_min/max`, `initial_temperature`, `alpha`, `beta` | Comfort band, initial indoor temperature and original discomfort weights |
| `ac_energy_quota`, device power/capacity/efficiency, time windows | Explicit physical scenario changes; default values preserve the original model |
| `objectives`, `workers`, `time_limit`, `gap` | Benchmark choice and CPU solver budgets |

For a flat $0.15 tariff, set `tou_enabled = false` and uncomment `flat_price_per_kwh = 0.15`. To change comfort, edit both band endpoints. The full scenario is recorded; modified scenarios must not be presented as improvements under original constraints. Unknown keys, conflicting tariffs, invalid bounds/windows and non-finite numbers are rejected. The solver supports both convex and concave DR tariffs exactly.

## Python API

```python
from dataclasses import replace
from oracle import load_config, run

config = replace(
    load_config(),
    name="flat_15c",
    tou_enabled=False,
    flat_price_per_kwh=0.15,
    temperature_min=19,
    temperature_max=23,
    output="results/oracle/flat_15c",
)
if __name__ == "__main__":
    summary = run(config)
```

`OracleConfig` also supports direct construction. Advanced callers can use `build_homes(physical_days, config, resolved_strategy)`, `comfort_frontier(env)`, `solve_oracle(envs, ...)` and `replay_oracle(envs, solution, reference=True)` for custom physical 24×8 daily arrays. A resolved strategy contains explicit tariffs; it must agree with the scenario. Input hours must be exactly 0–23. Solver schedules are returned only after numerical certification; independent replay remains mandatory before reporting them as verified.

## What is optimal?

Let `C_i` be electrical cost including DR and peer settlement, `D_i` the sum of squared temperature-band violations, and `W_i` the original washing timing penalty.

- **`paper_reward`:** minimize the community sum of `C_i + beta_i D_i + W_i`, exactly negative undiscounted evaluation reward.
- **`comfort_first`:** certify each home's minimum feasible `D_i`, then minimize community cost while every home respects its own minimum plus 1e−5 squared °C. EV charging and a preferred-window complete washing cycle are required. Enforce the hard comfort band whenever zero discomfort is feasible.

Maximum comfortable-hour count is a **separate** certified thermal diagnostic. It can concentrate quota-driven overcooling into fewer, more severe violations, whereas minimum squared discomfort can spread smaller violations. Neither criterion subsumes the other. The squared sum already combines duration and degree: four hours at 1°C outside the band and one hour at 2°C both contribute 4 squared °C in hourly episodes. A separate violation-duration penalty would define another objective and require its own oracle. Hour counts are recorded both strictly and with 1e−6°C numerical tolerance. Oracle discount is 1; the original Q-gradient preset uses 0.99 and the PPO preset uses 1.

Peer payments cancel in the community objective; individual bill fairness is not imposed. Trades keep seller order and left-neighbour priority. Export clipping, mandatory curtailment and legacy counting of pre-grid and peer exports remain exact. No battery-created exports, terminal battery equality or degradation cost is introduced. Default daily reset and the same-sign battery efficiency equation remain original. Changed appliance limits also change the simulator's clipped trace-derived demand budgets; these are different scenarios.

The pinned main simulator predates the quota-disable flag. When a scenario explicitly disables the AC quota, its independent replay uses a zero required-energy parameter and records that adaptation. Default-scenario audits apply the original nonzero quotas.

SCIP solves the hybrid linear/quadratic mixed-integer problem. Accepted solves require `optimal`/`gaplimit` and a primal/dual objective gap at most `1e-4 + 1e-6 |upper bound|`; timeout incumbents are rejected. Every schedule is replayed in fresh current and pinned [main:e34da8b](https://github.com/dcdube/FedDMPQ/tree/e34da8b68c646ae3d6c818ad13f2020e6ba06c73) simulators with the **same configured parameters**, checking controls, temperatures, SoE, flows, cost and appliance completion. This is numerical certification for the configured simulator, not real-world optimality or a symbolic proof. Held-out oracle actions never become training labels.

## Final submission comparison

The [GridPFN study](../docs/protocol.md) compares TabPFN-3.5, TabFM, TabICLv2,
Extra Trees, persistence and history-only inputs using the same PPO architecture
across 25 homes and five forward monthly folds. Each method selects its training
budget on the preceding validation week, then refits on all eligible training
and validation dates. The standalone scenario above is separate from this study.

After all 60 selection/refit stages complete:

```bash
python -m gridpfn.experiments.finalize_study results/seasonal --workers 4
```

The finalizer freezes all 30 final controllers before opening test outcomes. It
constructs the matching monthly `paper_reward` oracle for every test date,
requires numerical certificates and independent original-simulator replays,
and exports the [complete comparison](../site/performance.html). Resume with the
same worker count and settings. Full instructions are in [reproduction](../docs/REPRODUCE.md).

The oracle knows future demand, generation and weather. Its bounds apply to the
combined objective; bill and discomfort components are not independent lower
bounds. The complete date coverage, six methods and remaining objective gaps are
reported together, including outcomes where TabPFN does not win.

The [perfect-foresight dispatch principle](https://web.stanford.edu/~boyd/papers/hbd.html)
motivates this benchmark. [PySCIPOpt](https://pyscipopt.readthedocs.io/en/stable/tutorials/expressions.html)
provides the optimization interface. The original simulator and evaluation
assumptions remain explicit in the [architecture](../docs/architecture.md).

## Native solver reproducibility

`ipopt.opt` selects AMD ordering in MUMPS (`mumps_pivot_order 0`) to avoid the
bundled Ipopt/METIS heap-corruption path reported for PySCIPOpt 6.2.1 in
[the upstream issue](https://github.com/scipopt/PySCIPOpt/issues/1234). The options
file is part of each oracle's source receipt. This changes numerical factorization,
not the objective, physical constraints or certificate tolerances. Any source or
option change requires fresh oracle artifacts; do not combine old and new receipts.
