import copy
from functools import cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gridpfn.core.environment import (
    CONTINUOUS_ACTION_MAX,
    CONTINUOUS_ACTION_MIN,
    CONTROL_STATE_DIM,
    DISCRETE_ACTION_DIM,
    STATE_DIM,
    STATIC_STATE_DIM,
)


@cache
def _tabpfn_backbone(device):
    """Share one frozen TabPFN 3.5 per device, outside the federated head state.

    Keeping the backbone out of each policy's registered modules prevents sparse
    masks, parameter averaging and target updates from modifying pretrained weights.
    It also avoids allocating a 219M-parameter copy for every client and target.
    """
    # A first call can come from evaluation's inference_mode. The shared weights
    # must remain ordinary tensors so later policy updates can differentiate inputs.
    with torch.inference_mode(False), torch.no_grad(), torch.random.fork_rng(devices=[]):
        if device == "cpu":
            from tabpfn.model_loading import load_model_criterion_config

            models, _, _, _ = load_model_criterion_config(
                model_path=None,
                check_bar_distribution_criterion=True,
                cache_trainset_representation=False,
                estimator_type="regressor",
                version="v3.5",
                download_if_not_exists=True,
            )
            backbone = models[0]
        else:
            backbone = copy.deepcopy(_tabpfn_backbone("cpu")).to(device)
        return backbone.requires_grad_(False).eval()


_feature_device = None
_feature_directory = None
_encoder_context = None
_disk_stats = {"disk_cache_hits": 0, "disk_cache_misses": 0}


def configure_embedding_device(device=None, cache_directory=None, encoder_context=None):
    """Optionally extract frozen features on a GPU while small heads run on CPU."""
    global _feature_device, _feature_directory, _encoder_context
    _feature_device = None if device is None else str(torch.device(device))
    _feature_directory = None if cache_directory is None else Path(cache_directory)
    _encoder_context = None
    if encoder_context is not None:
        path = Path(encoder_context)
        with np.load(path, allow_pickle=False) as saved:
            rows, labels = saved["features"].copy(), saved["labels"].copy()
            import json

            metadata = json.loads(str(saved["metadata"]))
        if (
            rows.ndim != 2
            or rows.shape[1] != STATIC_STATE_DIM
            or len(rows) < 2
            or labels.shape != (len(rows),)
            or not np.isfinite(rows).all()
            or not np.isfinite(labels).all()
            or metadata.get("split") != "train"
        ):
            raise ValueError(
                "Encoder context requires finite labelled training-only exogenous rows"
            )
        from gridpfn.core.utils.run_io import file_sha256

        _encoder_context = (rows, labels, metadata, file_sha256(path))
    _embedding_cache.cache_clear()
    _feature_store.cache_clear()
    _disk_stats.update(disk_cache_hits=0, disk_cache_misses=0)


def validate_encoder_context(clients, strategy, market):
    """Bind supervised encoder context to the exact training cohort and scenario."""
    if _encoder_context is None:
        return
    import hashlib
    import json

    metadata = _encoder_context[2]
    digest = hashlib.sha256()
    for client in clients:
        digest.update(np.ascontiguousarray(client.train_data).tobytes())
    dates = set.intersection(*(set(c.scaler["train_dates"]) for c in clients))
    if (
        metadata["home_ids"] != [c.home_id for c in clients]
        or metadata["training_sha256"] != digest.hexdigest()
        or not set(metadata["dates"]).issubset(dates)
        or metadata["strategy"] != json.loads(json.dumps(strategy))
        or metadata["market"] != market
    ):
        raise ValueError("Encoder context training cohort, normalization or scenario differs")


def encoder_context_identity():
    return _encoder_context[3] if _encoder_context is not None else None


@cache
def _feature_store(compute_device):
    if _feature_directory is None:
        return None
    from gridpfn.core.utils.feature_cache import FrozenFeatureCache
    from gridpfn.core.utils.run_io import backbone_identity, file_sha256

    identity = {
        "backbone": backbone_identity(),
        "encoder_source": file_sha256(Path(__file__)),
        "encoder_context": _encoder_context[3] if _encoder_context is not None else None,
        "torch": str(torch.__version__),
        "tabpfn": version("tabpfn"),
        "device": compute_device,
        "hardware": torch.cuda.get_device_name(compute_device)
        if compute_device.startswith("cuda")
        else "cpu",
        "cuda": torch.version.cuda,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    return FrozenFeatureCache(_feature_directory, identity)


def release_embedding_models():
    """Keep cached feature rows, release unused frozen weights after preparation.

    A later cache miss reloads the same frozen checkpoint through the normal API.
    """
    gpu = torch.cuda.is_available()
    before = torch.cuda.memory_allocated() if gpu else 0
    _tabpfn_backbone.cache_clear()
    if gpu:
        torch.cuda.empty_cache()
    return {
        "torch_gpu_bytes_before": before,
        "torch_gpu_bytes_after": torch.cuda.memory_allocated() if gpu else 0,
        **_disk_stats,
    }


@cache
def _embedding_cache(device):
    return _EmbeddingTable()


class _EmbeddingTable:
    """One dense feature matrix; resolve rows with a single indexed gather."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.indices, self.matrix = {}, None

    def __len__(self):
        return len(self.indices)

    def __contains__(self, key):
        return key in self.indices

    def extend(self, keys, embeddings):
        offset = len(self)
        self.indices.update((key, offset + i) for i, key in enumerate(keys))
        self.matrix = embeddings if self.matrix is None else torch.cat((self.matrix, embeddings))

    def gather(self, keys):
        index = torch.tensor(
            [self.indices[key] for key in keys], dtype=torch.long, device=self.matrix.device
        )
        return self.matrix.index_select(0, index)


def _static_embeddings(features):
    """Cache exact float32 exogenous rows, shared by all clients and networks.

    Live control state and actions are deliberately excluded: they go through the
    trainable heads. No gradient through the frozen backbone is needed anymore.
    """
    return embedding_rows(features.detach().float().cpu().contiguous().numpy(), features.device)


def embedding_rows(rows, device):
    """Look up CPU replay rows directly, without a GPU-to-CPU synchronization."""
    from tabpfn.architectures.interface import PerformanceOptions

    device = str(device)
    table = _embedding_cache(device)
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    keys = [(rows.shape[1], row.tobytes()) for row in rows]
    missing = {key: index for index, key in enumerate(keys) if key not in table}
    missing_items = list(missing.items())
    compute_device = _feature_device or device
    if not missing_items:
        return table.gather(keys)
    # Cache ordinary tensors even when the first request comes from inference_mode;
    # autograd must be able to save these inputs when training the output heads.
    with torch.inference_mode(False), torch.no_grad():
        for start in range(0, len(missing_items), 256):
            chunk = missing_items[start : start + 256]
            indices = [index for _, index in chunk]
            x = torch.tensor(rows[indices], dtype=torch.float32, device=compute_device)
            if _encoder_context is None:
                context = torch.stack((x.new_zeros(x.shape[1]), x.new_ones(x.shape[1])))
                labels = x.new_zeros((2, 1))
            else:
                context = torch.as_tensor(_encoder_context[0], device=x.device, dtype=x.dtype)
                labels = torch.as_tensor(_encoder_context[1], device=x.device, dtype=x.dtype)[
                    :, None
                ]
            output = (
                _tabpfn_backbone(compute_device)(
                    torch.cat((context, x), dim=0).unsqueeze(1),
                    labels,
                    task_type="regression",
                    only_return_standard_out=False,
                    performance_options=PerformanceOptions(),
                )["test_embeddings"][:, 0, :]
                .contiguous()
                .to(device)
            )
            table.extend([key for key, _ in chunk], output)
    return table.gather(keys)


def precompute_embeddings(states, device):
    """Prepare exogenous state embeddings once before repeated RL updates.

    Cache keys contain the first eight exogenous inputs, including the applied
    tariff and normalization. All remaining coordinates are live control state.
    Optional labelled context is fitted exclusively from the declared training split.
    """
    states = torch.as_tensor(states, dtype=torch.float32, device=device)
    table = _embedding_cache(str(states.device))
    before = len(table)
    rows = states[:, :STATIC_STATE_DIM].detach().cpu().contiguous().numpy()
    if not len(rows):
        return 0
    if all((STATIC_STATE_DIM, row.tobytes()) in table for row in rows):
        return 0
    store = _feature_store(_feature_device or str(states.device))
    if store is not None:
        saved = store.get(rows, _tabpfn_backbone("cpu").embedding_dim)
        if saved is not None:
            keys = [(STATIC_STATE_DIM, row.tobytes()) for row in rows]
            missing = {key: i for i, key in enumerate(keys) if key not in table}
            if missing:
                with torch.inference_mode(False), torch.no_grad():
                    table.extend(
                        list(missing), torch.as_tensor(saved[list(missing.values())]).to(device)
                    )
            _disk_stats["disk_cache_hits"] += 1
            return len(table) - before
        _disk_stats["disk_cache_misses"] += 1
    for chunk in states.split(256):
        if len(chunk):
            _static_embeddings(chunk[:, :STATIC_STATE_DIM])
    if store is not None:
        store.put(rows, embedding_rows(rows, device).detach().float().cpu().numpy())
    return len(table) - before


class _TabPFNHead(nn.Module):
    def __init__(
        self,
        state_dim,
        output_dim,
        action_dim=0,
        feature_mode="frozen",
        hidden_dim=256,
        embedding_weight=1.0,
        auxiliary_dim=0,
    ):
        super().__init__()
        if state_dim not in (STATE_DIM, CONTROL_STATE_DIM):
            raise ValueError("state_dim must be 9 (legacy) or 17 (control state)")
        if feature_mode not in ("frozen", "normalized", "hybrid", "raw"):
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        self.state_dim, self.feature_mode = state_dim, feature_mode
        self.auxiliary_dim = int(auxiliary_dim)
        if self.auxiliary_dim < 0 or (self.auxiliary_dim and state_dim != CONTROL_STATE_DIM):
            raise ValueError("Auxiliary features require the original 17-value control state")
        if self.auxiliary_dim:
            self.register_buffer("auxiliary_width", torch.tensor(self.auxiliary_dim))
        self.action_dim = action_dim
        self.embedding_dim = 0 if feature_mode == "raw" else _tabpfn_backbone("cpu").embedding_dim
        input_dim = (
            (
                state_dim
                if feature_mode == "raw"
                else self.embedding_dim
                + state_dim
                - (0 if feature_mode == "hybrid" else STATIC_STATE_DIM)
            )
            + action_dim
            + self.auxiliary_dim
        )
        if feature_mode in ("normalized", "hybrid"):
            self.register_buffer("feature_mean", torch.zeros(self.embedding_dim))
            self.register_buffer("feature_scale", torch.ones(self.embedding_dim))
        if feature_mode == "hybrid":
            # Expose the existing observation alongside normalized frozen features.
            # This skip path changes representation, not the information contract.
            self.register_buffer("raw_skip", torch.ones(1))
            if not np.isfinite(embedding_weight) or embedding_weight < 0:
                raise ValueError("embedding_weight must be finite and nonnegative")
            if embedding_weight != 1:
                self.register_buffer("embedding_weight", torch.tensor(float(embedding_weight)))
        elif embedding_weight != 1:
            raise ValueError("embedding_weight applies only to hybrid features")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def prepare_features(self, states):
        """Resolve a replay/environment batch once for all compatible heads."""
        if isinstance(states, torch.Tensor):
            return self._state_features(states)
        states = np.asarray(states, dtype=np.float32)
        tensor = torch.as_tensor(states, device=self.fc1.weight.device, dtype=self.fc1.weight.dtype)
        if tensor.ndim != 2 or tensor.shape[1] != self.state_dim + self.auxiliary_dim:
            raise ValueError("Observation does not match the model's physical/auxiliary schema")
        if self.feature_mode == "raw":
            return torch.cat((tensor[:, self.state_dim :], tensor[:, : self.state_dim]), -1)
        embeddings = embedding_rows(states[:, :STATIC_STATE_DIM], self.fc1.weight.device)
        return self._join_features(tensor, embeddings)

    def _join_features(self, state, embeddings):
        embeddings = embeddings.to(dtype=self.fc1.weight.dtype)
        if self.feature_mode in ("normalized", "hybrid"):
            embeddings = (embeddings - self.feature_mean) / self.feature_scale
        if hasattr(self, "embedding_weight"):
            embeddings = embeddings * self.embedding_weight
        return torch.cat(
            (
                embeddings,
                state[:, self.state_dim :].to(dtype=self.fc1.weight.dtype),
                state[
                    :, 0 if self.feature_mode == "hybrid" else STATIC_STATE_DIM : self.state_dim
                ].to(dtype=self.fc1.weight.dtype),
            ),
            dim=1,
        )

    def _state_features(self, state):
        if state.ndim != 2 or state.shape[1] != self.state_dim + self.auxiliary_dim:
            raise ValueError(
                f"Expected [batch, {self.state_dim + self.auxiliary_dim}] input, got {tuple(state.shape)}"
            )
        if self.feature_mode == "raw":
            state = state.to(dtype=self.fc1.weight.dtype)
            return torch.cat((state[:, self.state_dim :], state[:, : self.state_dim]), -1)
        if state.shape[0] == 0:
            return state.new_empty(
                (0, self.fc1.in_features - getattr(self, "action_dim", 0)),
                dtype=self.fc1.weight.dtype,
            )
        return self._join_features(state, _static_embeddings(state[:, :STATIC_STATE_DIM]))

    def _uncontrolled_temperature(self, features):
        state = features[:, -CONTROL_STATE_DIM:]
        low, high = self.thermal_bounds.unbind()
        outdoor = state[:, 7] * torch.where(high == low, torch.ones_like(low), high - low) + low
        return 0.7 * (state[:, 9] * 10 + 20) + 0.3 * outdoor


def checkpoint_feature_mode(actor_state):
    if "raw_skip" in actor_state:
        return "hybrid"
    if actor_state["fc1.weight"].shape[1] - int(actor_state.get("auxiliary_width", 0)) in (
        STATE_DIM,
        CONTROL_STATE_DIM,
    ):
        return "raw"
    return "normalized" if "feature_mean" in actor_state else "frozen"


def checkpoint_state_dim(actor_state):
    """Recover the observation contract of legacy or control-state head checkpoints."""
    width = actor_state["fc1.weight"].shape[1] - int(actor_state.get("auxiliary_width", 0))
    state_dim = (
        width
        if checkpoint_feature_mode(actor_state) == "raw"
        else width
        - _tabpfn_backbone("cpu").embedding_dim
        + (0 if checkpoint_feature_mode(actor_state) == "hybrid" else STATIC_STATE_DIM)
    )
    if state_dim not in (STATE_DIM, CONTROL_STATE_DIM):
        raise ValueError(f"Unsupported actor checkpoint input size: inferred state_dim={state_dim}")
    return state_dim


class Actor(_TabPFNHead):
    def __init__(
        self,
        state_dim,
        continuous_action_dim,
        action_min=CONTINUOUS_ACTION_MIN,
        action_max=CONTINUOUS_ACTION_MAX,
        feature_mode="frozen",
        residual_scale=None,
        thermal_bounds=None,
        target_temperature_bounds=None,
        quota_actor=False,
        feasible_dt=None,
        hidden_dim=256,
        learned_discrete=False,
        stochastic_std=None,
        stochastic_ac_std=None,
        thermal_conditioning=False,
        embedding_weight=1.0,
        quota_correction=2.0,
        auxiliary_dim=0,
        hidden_layers=1,
        activation="relu",
    ):
        super().__init__(
            state_dim,
            continuous_action_dim,
            feature_mode=feature_mode,
            hidden_dim=hidden_dim,
            embedding_weight=embedding_weight,
            auxiliary_dim=auxiliary_dim,
        )
        if (
            type(hidden_layers) is not int
            or hidden_layers < 1
            or activation not in ("relu", "tanh")
        ):
            raise ValueError("Actor requires positive hidden_layers and relu/tanh activation")
        if residual_scale is not None and (hidden_layers != 1 or activation != "relu"):
            raise ValueError("Residual actor requires the original single ReLU layer")
        self.hidden = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)
        )
        self.activation = activation
        if activation != "relu":
            self.register_buffer("activation_code", torch.tensor(1))
        self.set_action_bounds(action_min, action_max)
        if learned_discrete:
            self.discrete_fc4 = nn.Linear(hidden_dim, DISCRETE_ACTION_DIM)
        if stochastic_std is not None:
            if not learned_discrete or not 0 < stochastic_std <= 1:
                raise ValueError("Stochastic actors require categorical actions and std in (0,1]")
            self.log_std = nn.Parameter(
                torch.full((continuous_action_dim,), np.log(stochastic_std))
            )
            if stochastic_ac_std is not None:
                if not np.isfinite(stochastic_ac_std) or not np.exp(-4) <= stochastic_ac_std <= 1:
                    raise ValueError("AC exploration std must be in [exp(-4), 1]")
                with torch.no_grad():
                    self.log_std[0] = np.log(stochastic_ac_std)
        elif stochastic_ac_std is not None:
            raise ValueError("AC exploration requires a stochastic actor")
        if thermal_bounds is not None:
            if (
                state_dim != CONTROL_STATE_DIM
                or feature_mode not in ("raw", "hybrid")
                or residual_scale is not None
            ):
                raise ValueError(
                    "Temperature actor requires 17-state raw/hybrid features without a power residual"
                )
            bounds = torch.as_tensor(thermal_bounds, dtype=self.fc1.weight.dtype)
            if bounds.shape != (2,) or not torch.isfinite(bounds).all() or bounds[1] < bounds[0]:
                raise ValueError("thermal_bounds must be finite training temperature min/max")
            self.register_buffer("thermal_bounds", bounds)
            target_bounds = torch.as_tensor(
                target_temperature_bounds
                if target_temperature_bounds is not None
                else [18.2, 21.8],
                dtype=self.fc1.weight.dtype,
            )
            if (
                target_bounds.shape != (2,)
                or not torch.isfinite(target_bounds).all()
                or target_bounds[0] >= target_bounds[1]
            ):
                raise ValueError("target_temperature_bounds must be two finite increasing values")
            self.register_buffer(
                "target_temperature_bounds",
                target_bounds,
                persistent=target_temperature_bounds is not None,
            )
            if target_temperature_bounds is not None:
                # Start near the original 20C controller: a wide quota-aware
                # range otherwise saturates AC at 2.5kW and kills its gradient.
                fraction = ((20 - target_bounds[0]) / (target_bounds[1] - target_bounds[0])).clamp(
                    0.01, 0.99
                )
                with torch.no_grad():
                    self.fc4.weight[0].zero_()
                    self.fc4.bias[0].copy_(torch.atanh(2 * fraction - 1))
        elif target_temperature_bounds is not None:
            raise ValueError("Target temperature bounds require a temperature actor")
        if residual_scale is not None:
            self.enable_residual(residual_scale)
        if feasible_dt is not None:
            if (
                state_dim != CONTROL_STATE_DIM
                or feature_mode not in ("raw", "hybrid")
                or residual_scale is not None
            ):
                raise ValueError(
                    "Feasible controls require 17-state raw/hybrid features without a power residual"
                )
            dt = torch.as_tensor(feasible_dt, dtype=self.fc1.weight.dtype).reshape(1)
            if not torch.isfinite(dt).all() or dt.item() <= 0:
                raise ValueError("feasible_dt must be positive")
            self.register_buffer("control_dt", dt)
        if quota_actor:
            if thermal_bounds is None or feasible_dt is None:
                raise ValueError("Quota residual actor requires temperature and feasible controls")
            self.register_buffer("quota_actor", torch.ones(1))
            if not np.isfinite(quota_correction) or quota_correction <= 0:
                raise ValueError("quota_correction must be finite and positive")
            if quota_correction != 2:
                self.register_buffer("quota_correction", torch.tensor(float(quota_correction)))
            with torch.no_grad():
                self.fc4.weight[0].zero_()
                self.fc4.bias[0].zero_()
        if thermal_conditioning:
            if not quota_actor or stochastic_std is None:
                raise ValueError("Thermal conditioning requires a stochastic quota actor")
            scale = torch.ones(continuous_action_dim)
            scale[0] = 2 / quota_correction
            self.register_buffer("latent_scale", scale)

    def enable_residual(self, scale):
        """Freeze the initialized controller and learn bounded corrections."""
        import copy

        scale = torch.as_tensor(scale, dtype=self.fc1.weight.dtype, device=self.fc1.weight.device)
        if (
            scale.shape != self.action_min.shape
            or not torch.isfinite(scale).all()
            or (scale < 0).any()
        ):
            raise ValueError("residual_scale needs one finite nonnegative fraction per actuator")
        self.base_fc1 = copy.deepcopy(self.fc1).requires_grad_(False)
        self.base_fc4 = copy.deepcopy(self.fc4).requires_grad_(False)
        self.register_buffer("residual_scale", scale)
        nn.init.zeros_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def reference_features(self, features):
        normalized = torch.tanh(self.base_fc4(F.relu(self.base_fc1(features))))
        return self.action_min + 0.5 * (normalized + 1) * (self.action_max - self.action_min)

    def project_action(self, features, action):
        if hasattr(self, "residual_scale"):
            base = self.reference_features(features)
            radius = self.residual_scale * (self.action_max - self.action_min)
            action = action.clamp(base - radius, base + radius)
        action = action.clamp(self.action_min, self.action_max)
        if hasattr(self, "thermal_bounds"):
            natural = self._uncontrolled_temperature(features)
            low = ((natural - self.target_temperature_bounds[1]) / 3).clamp(0, 2.5)
            high = ((natural - self.target_temperature_bounds[0]) / 3).clamp(0, 2.5)
            if hasattr(self, "control_dt"):
                quota = self._feasible_bounds(features)[0][:, 0]
                low, high = torch.maximum(low, quota), torch.maximum(high, quota)
            action = torch.cat(
                (action[:, :1].clamp(low[:, None], high[:, None]), action[:, 1:]), dim=1
            )
        if hasattr(self, "control_dt"):
            low, high = self._feasible_bounds(features)
            action = action.clamp(low, high)
        return action

    def _feasible_bounds(self, features):
        state = features[:, -CONTROL_STATE_DIM:]
        dt = self.control_dt[0]
        hour = state[:, 0] * 24
        ac_min = ((state[:, 15] * 60 - 2.5 * (24 - hour - dt).clamp_min(0)) / dt).clamp(0, 2.5)
        ev_max = torch.minimum(
            (state[:, 11] * 24 / dt).clamp(0, 6),
            ((1 - state[:, 10]) * 24 / (0.95 * dt)).clamp(0, 6),
        )
        ev_max = torch.where((hour >= 0) & (hour < 8), ev_max, torch.zeros_like(ev_max))
        ev_min = ((state[:, 11] * 24 - 6 * (8 - hour - dt).clamp_min(0)) / dt).clamp_min(0)
        ev_min = torch.minimum(ev_min, ev_max)
        battery_min = (-state[:, 8] * 6.4 / (0.95 * dt)).clamp(-2.4, 0)
        battery_max = ((1 - state[:, 8]) * 6.4 / (0.95 * dt)).clamp(0, 2.4)
        return (
            torch.stack((ac_min, ev_min, battery_min), dim=1),
            torch.stack((torch.full_like(hour, 2.5), ev_max, battery_max), dim=1),
        )

    def set_action_bounds(self, action_min, action_max):
        device = self.fc4.weight.device
        dtype = self.fc4.weight.dtype
        action_min = torch.as_tensor(action_min, device=device, dtype=dtype)
        action_max = torch.as_tensor(action_max, device=device, dtype=dtype)

        if "action_min" in self._buffers:
            self.action_min = action_min
            self.action_max = action_max
        else:
            self.register_buffer("action_min", action_min, persistent=False)
            self.register_buffer("action_max", action_max, persistent=False)

    def forward(self, x):
        return self.forward_features(self._state_features(x))

    def discrete_features(self, features):
        return self.discrete_fc4(self.hidden_features(features))

    def hidden_features(self, features):
        activate = torch.tanh if self.activation == "tanh" else F.relu
        hidden = activate(self.fc1(features))
        for layer in self.hidden:
            hidden = activate(layer(hidden))
        return hidden

    def forward_features(self, features):
        latent = self.fc4(self.hidden_features(features)) * getattr(self, "latent_scale", 1)
        return self.latent_features(features, latent)

    def distribution_features(self, features):
        hidden = self.hidden_features(features)
        return (
            self.fc4(hidden) * getattr(self, "latent_scale", 1),
            self.log_std.clamp(-4, 0).exp().expand(features.shape[0], -1)
            * getattr(self, "latent_scale", 1),
            self.discrete_fc4(hidden),
        )

    def latent_features(self, features, latent):
        """Map a sampled latent action through the same physical control contract."""
        normalized_action = torch.tanh(latent)
        if hasattr(self, "residual_scale"):
            base = self.reference_features(features)
            return (
                base + normalized_action * self.residual_scale * (self.action_max - self.action_min)
            ).clamp(self.action_min, self.action_max)
        action = self.action_min + 0.5 * (normalized_action + 1.0) * (
            self.action_max - self.action_min
        )
        if hasattr(self, "control_dt"):
            low, high = self._feasible_bounds(features)
            action = low + 0.5 * (normalized_action + 1) * (high - low)
        if hasattr(self, "thermal_bounds"):
            target = self.target_temperature_bounds[0] + 0.5 * (normalized_action[:, 0] + 1) * (
                self.target_temperature_bounds[1] - self.target_temperature_bounds[0]
            )
            if hasattr(self, "quota_actor"):
                state = features[:, -CONTROL_STATE_DIM:]
                quota_power = (state[:, 15] * 60 / (24 - state[:, 0] * 24).clamp_min(1)).clamp(
                    0, 2.5
                )
                base = torch.minimum(
                    torch.full_like(quota_power, 20),
                    self._uncontrolled_temperature(features) - 3 * quota_power,
                )
                target = (
                    base + getattr(self, "quota_correction", 2) * normalized_action[:, 0]
                ).clamp(self.target_temperature_bounds[0], self.target_temperature_bounds[1])
            ac = ((self._uncontrolled_temperature(features) - target) / 3).clamp(0, 2.5)
            action = torch.cat((ac[:, None], action[:, 1:]), dim=1)
        return self.project_action(features, action) if hasattr(self, "control_dt") else action


class Critic(_TabPFNHead):
    def __init__(
        self,
        state_dim,
        continuous_action_dim,
        discrete_action_dim,
        feature_mode="frozen",
        twin=False,
        hidden_dim=256,
        value_head=False,
        embedding_weight=1.0,
        thermal_bounds=None,
        value_hidden_dim=None,
        value_normalization=False,
        auxiliary_dim=0,
    ):
        super().__init__(
            state_dim,
            discrete_action_dim,
            action_dim=continuous_action_dim,
            feature_mode=feature_mode,
            hidden_dim=hidden_dim,
            embedding_weight=embedding_weight,
            auxiliary_dim=auxiliary_dim,
        )
        self.twin = twin
        if thermal_bounds is not None:
            if state_dim != CONTROL_STATE_DIM or feature_mode not in ("raw", "hybrid"):
                raise ValueError("Physics critic requires 17-state raw/hybrid coordinates")
            bounds = torch.as_tensor(thermal_bounds, dtype=self.fc1.weight.dtype)
            if bounds.shape != (2,) or not torch.isfinite(bounds).all() or bounds[1] < bounds[0]:
                raise ValueError("thermal_bounds must be finite training temperature min/max")
            self.register_buffer("thermal_bounds", bounds)
        if value_head:
            value_width = hidden_dim if value_hidden_dim is None else value_hidden_dim
            if value_width < 1:
                raise ValueError("value_hidden_dim must be positive")
            self.value_fc1 = nn.Linear(self.fc1.in_features - continuous_action_dim, value_width)
            self.value_fc4 = nn.Linear(value_width, 1)
            if value_normalization:
                self.register_buffer("value_mean", torch.zeros(1))
                self.register_buffer("value_scale", torch.ones(1))
                self.register_buffer("value_second", torch.ones(1))
        if twin:
            self.twin_fc1 = nn.Linear(self.fc1.in_features, hidden_dim)
            self.twin_fc4 = nn.Linear(hidden_dim, discrete_action_dim)

    def forward(self, state, continuous_action):
        return self.forward_features(self._state_features(state), continuous_action)

    def value_features(self, features):
        value = self.value_fc4(F.relu(self.value_fc1(features))).squeeze(-1)
        return value * getattr(self, "value_scale", 1) + getattr(self, "value_mean", 0)

    def forward_features(self, features, continuous_action):
        x = torch.cat((features, continuous_action), dim=1)
        return self.fc4(F.relu(self.fc1(x))) + self.thermal_reward(features, continuous_action)

    def thermal_reward(self, features, control):
        """Exact known immediate thermal reward; the head learns the remaining Q.

        Q = known_thermal_reward + residual is an algebraic reparameterization,
        with unchanged Bellman targets, physical comfort limits and reward beta.
        Feasible AC powers are required so this term uses the executed action.
        """
        if not hasattr(self, "thermal_bounds"):
            return 0
        natural = self._uncontrolled_temperature(features)
        temperature = natural - 3 * control[:, 0]
        violation = torch.maximum(18 - temperature, temperature - 22).clamp_min(0)
        return (-0.08 * violation.square())[:, None]

    def both_features(self, features, continuous_action):
        first = self.forward_features(features, continuous_action)
        if not self.twin:
            return first, first
        x = torch.cat((features, continuous_action), dim=1)
        return first, self.twin_fc4(F.relu(self.twin_fc1(x))) + self.thermal_reward(
            features, continuous_action
        )

    def minimum_features(self, features, continuous_action):
        first, second = self.both_features(features, continuous_action)
        return torch.minimum(first, second)


def heads_from_state(actor_state, critic_state, device):
    """Reconstruct evaluation heads from their explicit tensor contract."""
    width, mode = checkpoint_state_dim(actor_state), checkpoint_feature_mode(actor_state)
    actor = Actor(
        width,
        3,
        feature_mode=mode,
        residual_scale=actor_state.get("residual_scale"),
        thermal_bounds=actor_state.get("thermal_bounds"),
        target_temperature_bounds=actor_state.get("target_temperature_bounds"),
        quota_actor="quota_actor" in actor_state,
        feasible_dt=actor_state.get("control_dt"),
        hidden_dim=actor_state["fc1.weight"].shape[0],
        learned_discrete="discrete_fc4.weight" in actor_state,
        stochastic_std=float(actor_state["log_std"].exp()[0]) if "log_std" in actor_state else None,
        thermal_conditioning="latent_scale" in actor_state,
        embedding_weight=float(actor_state.get("embedding_weight", 1.0)),
        quota_correction=float(actor_state.get("quota_correction", 2.0)),
        auxiliary_dim=int(actor_state.get("auxiliary_width", 0)),
        hidden_layers=1
        + sum(k.startswith("hidden.") and k.endswith(".weight") for k in actor_state),
        activation="tanh" if "activation_code" in actor_state else "relu",
    ).to(device)
    critic = Critic(
        width,
        3,
        2,
        feature_mode=mode,
        twin="twin_fc1.weight" in critic_state,
        hidden_dim=critic_state["fc1.weight"].shape[0],
        value_head="value_fc1.weight" in critic_state,
        embedding_weight=float(critic_state.get("embedding_weight", 1.0)),
        thermal_bounds=critic_state.get("thermal_bounds"),
        value_hidden_dim=critic_state["value_fc1.weight"].shape[0]
        if "value_fc1.weight" in critic_state
        else None,
        value_normalization="value_mean" in critic_state,
        auxiliary_dim=int(critic_state.get("auxiliary_width", 0)),
    ).to(device)
    actor.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
    return actor.eval(), critic.eval()
