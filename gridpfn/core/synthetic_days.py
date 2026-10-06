"""Training-only, synchronized synthetic daily markets; simulator supplies rewards.

Whole-day mixtures preserve time ordering and share weather/price donors across
homes. Mild scenario perturbations vary exogenous inputs, never appliance physics,
comfort penalties or tariffs. Real preprocessing and imitation remain unchanged.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gridpfn.core.dataset import load_data
from gridpfn.core.environment import CONTINUOUS_ACTION_MAX


@dataclass(frozen=True)
class ScenarioConfig:
    days: int = 64
    seed: int = 4242
    method: str = "mix"
    alpha: float = 0.4
    load_std: float = 0.10
    pv_std: float = 0.10
    temperature_std: float = 1.0

    def __post_init__(self):
        if (
            not isinstance(self.days, int)
            or not isinstance(self.seed, int)
            or isinstance(self.days, bool)
            or isinstance(self.seed, bool)
            or self.days < 1
            or self.seed < 0
            or self.method not in ("mix", "jitter", "mixed_jitter")
            or not np.isfinite([self.alpha, self.load_std, self.pv_std, self.temperature_std]).all()
            or self.alpha <= 0
            or min(self.load_std, self.pv_std, self.temperature_std) < 0
        ):
            raise ValueError("Invalid synthetic scenario configuration")


def training_identity(data, scaler):
    digest = hashlib.sha256(np.ascontiguousarray(data, dtype=np.float64).tobytes())
    digest.update(
        json.dumps(
            {
                k: scaler[k]
                for k in ("min", "max", "cols", "delta_t", "train_dates", "train_end_exclusive")
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def generate(bundles, home_ids, config):
    """Return normalized [home, scenario, step, column] rows and donor provenance."""
    if len(bundles) != len(home_ids) or not bundles or len(set(home_ids)) != len(home_ids):
        raise ValueError("Synthetic generation requires a unique matching home cohort")
    dates = sorted(set.intersection(*(set(b[3]["train_dates"]) for b in bundles)))
    if len(dates) < 2:
        raise ValueError("At least two complete shared training dates are required")
    donors, lows, spans = [], [], []
    for train, _, _, scaler in bundles:
        if train.ndim != 3 or train.shape[1:] != (24, 8) or scaler["delta_t"] != 1:
            raise ValueError("Synthetic scenarios require complete hourly 24x8 days")
        if any(date >= scaler["train_end_exclusive"] for date in scaler["train_dates"]):
            raise ValueError("Synthetic donors must precede the training cutoff")
        lookup = {date: i for i, date in enumerate(scaler["train_dates"])}
        selected = np.asarray(train[[lookup[d] for d in dates]], dtype=np.float64).copy()
        low = np.zeros(selected.shape[-1])
        span = np.ones_like(low)
        for column, index in scaler["col_to_scaler_idx"].items():
            if index is not None:
                column = int(column)
                low[column] = scaler["min"][index]
                width = scaler["max"][index] - low[column]
                span[column] = width if width else 1
        donors.append(selected * span + low)
        lows.append(low)
        spans.append(span)
    physical = np.asarray(donors)
    if not np.isfinite(physical).all():
        raise ValueError("Training donors must be finite")
    if not np.array_equal(physical[..., 2], np.broadcast_to(np.arange(24), physical[..., 2].shape)):
        raise ValueError("Synthetic donors must retain ordered hours 0-23")
    # Weather and raw spot prices are shared environmental signals.
    if not np.allclose(physical[..., [3, 4]], physical[:1, ..., [3, 4]], atol=1e-5):
        raise ValueError("Synchronized scenarios require shared weather and spot prices")
    rng = np.random.default_rng(config.seed)
    samples, provenance = [], []
    for _ in range(config.days):
        first, second = rng.choice(len(dates), 2, replace=False).tolist()
        weight = float(rng.beta(config.alpha, config.alpha)) if config.method != "jitter" else 1.0
        day = weight * physical[:, first] + (1 - weight) * physical[:, second]
        # The clock is an index, not a continuous scenario feature.
        day[..., 2] = np.arange(24)
        temperature_shift, pv_factor = 0.0, 1.0
        load_factors = np.ones(len(bundles))
        if config.method != "mix":
            temperature_shift = float(np.clip(rng.normal(0, config.temperature_std), -2.5, 2.5))
            pv_factor = float(np.clip(np.exp(rng.normal(0, config.pv_std)), 0.75, 1.25))
            load_factors = np.clip(np.exp(rng.normal(0, config.load_std, len(bundles))), 0.75, 1.25)
            day[..., 4] += temperature_shift
            day[..., 1] *= pv_factor
            day[..., [0, 5, 6, 7]] *= load_factors[:, None, None]
            # Historical AC/EV traces specify service demand; retain feasible ratings.
            dt = bundles[0][3]["delta_t"]
            day[..., 5] = np.clip(day[..., 5], 0, CONTINUOUS_ACTION_MAX[0] * dt)
            day[..., 6] = np.clip(day[..., 6], 0, CONTINUOUS_ACTION_MAX[1] * dt)
        day[..., [0, 1, 3, 5, 6, 7]] = np.maximum(day[..., [0, 1, 3, 5, 6, 7]], 0)
        samples.append(day)
        provenance.append(
            {
                "donors": [dates[first], dates[second]],
                "weight": weight,
                "temperature_shift": temperature_shift,
                "pv_factor": pv_factor,
                "load_factors": load_factors.tolist(),
            }
        )
    rows = (np.stack(samples, axis=1) - np.asarray(lows)[:, None, None]) / np.asarray(spans)[
        :, None, None
    ]
    manifest = {
        "schema": 1,
        "config": asdict(config),
        "home_ids": list(home_ids),
        "source_dates": dates,
        "training_identities": [training_identity(b[0], b[3]) for b in bundles],
        "provenance": provenance,
        "training_only": True,
        "physical_summary": {
            "temperature_min": float(np.min(np.asarray(samples)[..., 4])),
            "temperature_max": float(np.max(np.asarray(samples)[..., 4])),
        },
    }
    return rows, manifest


def attach_synthetic_days(clients, directory):
    """Validate original training identities before extending private rollout pools."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("training_only") is not True:
        raise ValueError("Synthetic dataset is not a training-only scenario archive")
    if [c.home_id for c in clients] != manifest["home_ids"]:
        raise ValueError("Synthetic home cohort differs from training")
    path = directory / "days.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["data_sha256"]:
        raise ValueError("Synthetic archive digest mismatch")
    with np.load(path, allow_pickle=False) as archive:
        rows = archive["days"].copy()
    if (
        rows.ndim != 4
        or rows.shape[:2] != (len(clients), manifest["config"]["days"])
        or rows.shape[2:] != (24, 8)
        or len(manifest["training_identities"]) != len(clients)
        or not np.isfinite(rows).all()
    ):
        raise ValueError("Invalid synthetic rows")
    if not np.array_equal(rows[..., 2], np.broadcast_to(np.arange(24), rows[..., 2].shape)):
        raise ValueError("Synthetic archives must retain exact ordered hours 0-23")
    for client, days, identity in zip(clients, rows, manifest["training_identities"], strict=True):
        if (
            training_identity(client.train_data, client.scaler) != identity
            or days.shape[1:] != client.train_data.shape[1:]
        ):
            raise ValueError("Synthetic donors or preprocessing differ from real training")
    for client, days in zip(clients, rows, strict=True):
        client.real_train_count = len(client.train_data)
        client.train_data = np.concatenate((client.train_data, days))


def main():
    from gridpfn.core.dataset import home_data_dir
    from gridpfn.core.training_config import HOME_IDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--path_data", type=Path, default=home_data_dir)
    parser.add_argument("--home_ids", type=int, nargs="+", default=HOME_IDS)
    parser.add_argument("--validation_days", type=int, default=14)
    parser.add_argument("--days", type=int, help="Number of scenarios; default 64.")
    parser.add_argument("--seed", type=int, help="Generator seed; default 4242.")
    parser.add_argument("--method", choices=("mix", "jitter", "mixed_jitter"))
    args = parser.parse_args()
    if args.validation_days < 1:
        parser.error("Synthetic generation requires a held-out validation period")
    values = json.loads(args.config.read_text()) if args.config else {}
    values.update(
        {k: getattr(args, k) for k in ("days", "seed", "method") if getattr(args, k) is not None}
    )
    config = ScenarioConfig(**values)
    bundles = load_data(
        args.path_data,
        "*.csv",
        choose=[f"home_{h}" for h in args.home_ids],
        validation_days=args.validation_days,
        split="validation",
        scaler_mode="shared",
    )
    rows, manifest = generate(bundles, args.home_ids, config)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "days.npz"
    np.savez_compressed(path, days=rows)
    manifest["data_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Generated {config.days} synchronized training days for {len(bundles)} homes: {args.output}"
    )


if __name__ == "__main__":
    main()
