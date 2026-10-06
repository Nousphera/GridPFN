"""Strict, explicit scenario settings; the default preserves the paper simulator."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomllib

from gridpfn.core.dataset import DATA_PERIODS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class OracleConfig:
    name: str = "original"
    data_dir: Path = ROOT / "dataset/split_homes_clean"
    output: Path = ROOT / "results/oracle/original"
    split: str = "validation"
    data_period: str = "legacy"
    home_ids: tuple = (27, 950, 1222, 3000, 3488, 3517, 5587, 5679, 5997, 9053)
    days: int = 0
    objectives: tuple = ("paper_reward", "comfort_first")
    workers: int = 4
    time_limit: float = 600
    gap: float = 1e-7
    base_price_multiplier: float = 1.0
    flat_price_per_kwh: float | None = None
    hourly_prices_per_kwh: tuple | None = None
    tou_enabled: bool = True
    tou_blocks: int = 5
    export_price: float = 0.025
    export_cap_enabled: bool = True
    export_cap_kwh: float = 2.5
    peer_trading: bool = True
    peer_price: float = 0.10
    dr_enabled: bool = True
    dr_limit: float = 5.0
    dr_penalty: float = 0.50
    dr_incentive: float = 0.025
    fixed_cost: float = 5.0
    temperature_min: float = 18.0
    temperature_max: float = 22.0
    initial_temperature: float = 23.0
    initial_battery_soe: float = 0.2
    ac_energy_quota: bool = True
    alpha: float = 0.005
    beta: float = 0.08
    thermal_inertia: float = 0.7
    thermal_gain: float = 10.0
    max_power_AC: float = 2.5
    max_power_EV: float = 6.0
    max_power_BESS: float = 2.4
    max_capacity_EV: float = 24.0
    max_capacity_BESS: float = 6.4
    efficiency_EV: float = 0.95
    efficiency_BESS: float = 0.95
    time_ini_EV: int = 0
    time_end_EV: int = 8
    time_ini_WM: int = 10
    time_end_WM: int = 20
    wm_duration: int = 3

    def __post_init__(self):
        for name in ("data_dir", "output"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        for name in ("home_ids", "objectives", "hourly_prices_per_kwh"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Give the scenario a nonempty name")
        for name in (
            "tou_enabled",
            "export_cap_enabled",
            "peer_trading",
            "dr_enabled",
            "ac_energy_quota",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        for name in (
            "days",
            "workers",
            "tou_blocks",
            "time_ini_EV",
            "time_end_EV",
            "time_ini_WM",
            "time_end_WM",
            "wm_duration",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer (hourly simulation)")
        if self.data_period not in DATA_PERIODS:
            raise ValueError("data_period must be a declared chronological period")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        if (
            not self.home_ids
            or len(set(self.home_ids)) != len(self.home_ids)
            or any(type(h) is not int or h < 0 for h in self.home_ids)
        ):
            raise ValueError("home_ids must be unique nonnegative integers in ring order")
        if (
            not self.objectives
            or len(set(self.objectives)) != len(self.objectives)
            or any(o not in ("paper_reward", "comfort_first") for o in self.objectives)
        ):
            raise ValueError("Choose unique paper_reward and/or comfort_first objectives")
        numeric = [f.name for f in fields(self) if f.type in (float, int, "float", "int")]
        for name in numeric + ["flat_price_per_kwh"]:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.days < 0 or self.workers < 1 or self.time_limit <= 0 or not 0 <= self.gap <= 1e-6:
            raise ValueError("Use positive workers/time, nonnegative days and gap <= 1e-6")
        if not 1 <= self.tou_blocks <= 24:
            raise ValueError("tou_blocks must be between 1 and 24")
        if self.flat_price_per_kwh is not None and self.hourly_prices_per_kwh is not None:
            raise ValueError("Choose a flat price or 24 hourly prices, not both")
        if (self.flat_price_per_kwh is not None or self.hourly_prices_per_kwh is not None) and (
            self.tou_enabled or self.base_price_multiplier != 1
        ):
            raise ValueError(
                "Explicit grid prices require tou_enabled=false and base_price_multiplier=1"
            )
        prices = () if self.flat_price_per_kwh is None else (self.flat_price_per_kwh,)
        if self.hourly_prices_per_kwh is not None:
            if len(self.hourly_prices_per_kwh) != 24:
                raise ValueError("hourly_prices_per_kwh requires exactly 24 entries")
            prices += self.hourly_prices_per_kwh
        if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in prices):
            raise ValueError("Original price resolution supports finite nonnegative prices")
        if self.temperature_min >= self.temperature_max:
            raise ValueError("temperature_min must be below temperature_max")
        if not 0 <= self.initial_battery_soe <= 1 or not 0 <= self.thermal_inertia < 1:
            raise ValueError("Initial SoE must be in [0,1] and thermal inertia in [0,1)")
        for name in (
            "max_power_AC",
            "max_power_EV",
            "max_capacity_EV",
            "max_capacity_BESS",
            "thermal_gain",
            "efficiency_EV",
            "efficiency_BESS",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.efficiency_EV > 1 or self.efficiency_BESS > 1:
            raise ValueError("Efficiencies must be at most one")
        for name in (
            "base_price_multiplier",
            "export_price",
            "export_cap_kwh",
            "peer_price",
            "dr_limit",
            "dr_penalty",
            "dr_incentive",
            "fixed_cost",
            "alpha",
            "beta",
            "max_power_BESS",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.time_ini_EV < self.time_end_EV <= 24:
            raise ValueError("EV hours must define a positive window within 0..24")
        if not 0 <= self.time_ini_WM < self.time_end_WM <= 24 or not (
            1 <= self.wm_duration <= self.time_end_WM - self.time_ini_WM
        ):
            raise ValueError("The washing cycle must fit its preferred window within 0..24")

    def settings(self):
        value = asdict(self)
        for name in ("data_dir", "output"):
            value[name] = str(value[name])
        for name in ("home_ids", "objectives", "hourly_prices_per_kwh"):
            if value[name] is not None:
                value[name] = list(value[name])
        return value


def load_config(path=DEFAULT_CONFIG):
    """Read TOML, reject unknown keys and resolve paths relative to that file."""
    path = Path(path).resolve()
    settings = tomllib.loads(path.read_text())
    unknown = settings.keys() - {f.name for f in fields(OracleConfig)}
    if unknown:
        raise ValueError(f"Unknown oracle settings: {sorted(unknown)}")
    for name in ("data_dir", "output"):
        if name in settings:
            settings[name] = path.parent / settings[name]
    return OracleConfig(**settings)
