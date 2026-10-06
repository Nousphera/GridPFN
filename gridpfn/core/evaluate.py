
import argparse
import io
import os
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import torch

from gridpfn.paths import ROOT

matplotlib.use("Agg")
from gridpfn.core.dataset import STEPS_PER_DAY, home_data_dir, load_data, setup_seed
from gridpfn.core.em_strategy import (
    P2P_TRADING,
    apply_em_strategy,
    compose_em_strategy,
    make_em_strategy,
)
from gridpfn.core.environment import HOME_ENERGY_MGNT, STATE_DIM
from gridpfn.core.model import configure_embedding_device, heads_from_state, precompute_embeddings
from gridpfn.core.schedule import detect_models, resolve_trained_home_ids
from gridpfn.core.utils.agent_utils import quantization_bounds, safe_torch_load, validate_low_bit
from gridpfn.core.utils.plot_utils import CUSTOM_NAMES
from gridpfn.core.utils.plots import (
    plot_comm_time,
    plot_comparison_metrics,
    plot_model_precision,
    plot_radar_chart,
    plot_training_metric,
)
from gridpfn.core.utils.rollout_metrics import (
    calculate_metrics,
    episode_data,
    new_episode_log,
    record_environment,
)
from gridpfn.core.utils.run_io import read_logged_settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CommunicationCostEvaluator:
    BYTES_PER_MB = 1_000_000

    def __init__(self, results_dir="train_test/logs", compression_level=9):
        self.results_dir = results_dir
        self.compression_level = compression_level

    def _compressed_size_bytes(self, obj):
        # Serialize payload and compress with DEFLATE to estimate transfer size
        buf = io.BytesIO()
        torch.save(obj, buf)
        return len(zlib.compress(buf.getvalue(), level=self.compression_level))

    @staticmethod
    def _pack_mask(mask):
        mask = mask.flatten().bool()
        padding = (-mask.numel()) % 8
        if padding:
            mask = torch.cat((mask, torch.zeros(padding, dtype=torch.bool, device=mask.device)))
        packed = torch.zeros(mask.numel() // 8, dtype=torch.uint8, device=mask.device)
        for offset in range(8):
            packed |= mask[offset::8].to(torch.uint8) << offset
        return packed

    @staticmethod
    def _make_sparse_payload(state_dict):
        # A bit-packed mask avoids the 32-bit-index overhead of COO payloads.
        payload = {}
        for key, tensor in state_dict.items():
            flat = tensor.flatten().float()
            mask = flat != 0
            payload[key + ".values"] = flat[mask]
            payload[key + ".mask"] = CommunicationCostEvaluator._pack_mask(mask)
            payload[key + ".shape"] = torch.tensor(tensor.shape)
        return payload

    @staticmethod
    def _masked_state(state_dict, masks, invert=False):
        masked = {}
        for name, tensor in state_dict.items():
            mask = masks.get(name)
            if mask is None:
                masked[name] = tensor
            else:
                mask = mask.bool()
                masked[name] = tensor * (~mask if invert else mask)
        return masked

    @staticmethod
    def _make_quantized_sparse_payload(state_dict, bits=8, clip_value=1.0, eps=1e-8):
        bits = validate_low_bit(bits)
        payload = {}
        qmin, qmax = quantization_bounds(bits)
        values_per_byte = 8 // bits if bits <= 8 else None

        for key, tensor in state_dict.items():
            flat = tensor.flatten().float()
            mask = flat != 0
            sparse_vals = flat[mask]
            if clip_value is not None and clip_value > 0:
                sparse_vals = torch.clamp(sparse_vals, -clip_value, clip_value)

            if sparse_vals.numel() == 0:
                packed_values = torch.empty(0, dtype=torch.int16 if bits == 16 else torch.uint8)
                scale = torch.tensor(1.0, dtype=torch.float32)
            else:
                max_abs = torch.max(torch.abs(sparse_vals)).item()
                scale = torch.tensor(max(max_abs / max(float(qmax), 1.0), eps), dtype=torch.float32)
                quantized = torch.round(sparse_vals / scale).clamp(qmin, qmax).to(torch.int16)
                if bits == 16:
                    packed_values = quantized
                else:
                    unsigned = (quantized - qmin).to(torch.uint8)
                    padding = (-unsigned.numel()) % values_per_byte
                    if padding:
                        unsigned = torch.cat((unsigned, torch.zeros(padding, dtype=torch.uint8)))
                    packed_values = torch.zeros(
                        unsigned.numel() // values_per_byte, dtype=torch.uint8
                    )
                    for offset in range(values_per_byte):
                        packed_values |= unsigned[offset::values_per_byte] << (offset * bits)

            payload[key + ".qvalues"] = packed_values
            payload[key + ".scale"] = scale
            payload[key + ".bits"] = torch.tensor(bits, dtype=torch.uint8)
            payload[key + ".num_values"] = torch.tensor(sparse_vals.numel(), dtype=torch.int32)
            payload[key + ".mask"] = CommunicationCostEvaluator._pack_mask(mask)
            payload[key + ".shape"] = torch.tensor(tensor.shape)
        return payload

    @staticmethod
    def _compute_flops_per_step(state_dict):
        return 2 * sum(
            tensor.count_nonzero().item()
            for key, tensor in state_dict.items()
            if key.endswith(".weight")
        )

    @staticmethod
    def _train_step_flops(actor_flops, critic_flops, batch_size):
        return actor_flops + critic_flops + batch_size * (4 * actor_flops + 7 * critic_flops)

    def _read_train_time(self, model_name):
        path = os.path.join(self.results_dir, model_name, "comm_train_time.csv")
        if not os.path.isfile(path):
            return 0.0
        df = pd.read_csv(path)
        return float(df["train_time (s)"].iloc[-1])

    def _read_test_time(self, model_name):
        path = os.path.join(self.results_dir, model_name, "comm_test_time.csv")
        if not os.path.isfile(path):
            return 0.0
        df = pd.read_csv(path)
        return float(df["test_time (s)"].iloc[-1])

    def compute(
        self,
        model_names,
        num_homes,
        episode,
        aggregate,
        test_times=None,
        num_test_days=0,
        steps_per_day=STEPS_PER_DAY,
        non_comm_models=None,
        sparse_models=None,
        sparse_quant_models=None,
        low_bit=8,
        batch_size=32,
        update=10,
        fedfit_t_end=None,
    ):
        low_bit = validate_low_bit(low_bit)
        topology_updates = (episode - 1) // update
        fedfit_updates = (
            min(episode - 1, episode if fedfit_t_end is None else fedfit_t_end) // update
        )
        preserve_test_times = test_times is None
        test_times = test_times or {}
        non_comm_models = set(non_comm_models or ())
        sparse_models = set(sparse_models or {"edgehem", "fedfit", "pffdst"})
        sparse_quant_models = set(sparse_quant_models or {"feddmpq"})

        for model_name in model_names:
            model_dir = os.path.join(self.results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            # Historical runs aggregated after episodes 21, 41, ...; new FedAvg
            # runs aggregate after 20, 40, ... . Preserve old artifact accounting.
            settings_path = os.path.join(model_dir, "train_settings.txt")
            settings = read_logged_settings(settings_path) if os.path.isfile(settings_path) else {}
            offset = settings.get("fedavg_aggregation_offset", 1) if model_name == "fedavg" else 1
            num_rounds = max(0, (episode - offset) // aggregate)

            state_cache = {}

            def load_states(directory):
                if directory not in state_cache:
                    state_cache[directory] = (
                        safe_torch_load(os.path.join(directory, "actor.pt"), "cpu"),
                        safe_torch_load(os.path.join(directory, "critic.pt"), "cpu"),
                    )
                return state_cache[directory]

            train_time_s = self._read_train_time(model_name) or 0.0
            test_time_s = (
                self._read_test_time(model_name)
                if preserve_test_times
                else float(test_times.get(model_name, 0.0) or 0.0)
            )

            is_non_comm = model_name in non_comm_models
            is_sparse = model_name in sparse_models
            is_quantized_sparse = model_name in sparse_quant_models
            frozen_path = os.path.join(model_dir, "server", "frozen_masks.pt")
            pffdst_masks = (
                safe_torch_load(frozen_path, "cpu")
                if model_name == "pffdst" and os.path.isfile(frozen_path)
                else None
            )

            # ── Training & deployment communication ───────────────────────────
            if is_non_comm:
                train_up = train_down = test_up = test_down = 0.0
            else:
                # Per-round client upload (sum over all homes)
                per_round_upload_bytes = 0
                pffdst_upload_bytes = [0, 0]
                for hid in range(1, num_homes + 1):
                    home_dir = os.path.join(model_dir, f"home_{hid}")
                    actor_sd, critic_sd = load_states(home_dir)
                    if pffdst_masks is not None:
                        for index, invert in enumerate((False, True)):
                            payload = {
                                "actor": self._make_sparse_payload(
                                    self._masked_state(actor_sd, pffdst_masks["actor"], invert)
                                ),
                                "critic": self._make_sparse_payload(
                                    self._masked_state(critic_sd, pffdst_masks["critic"], invert)
                                ),
                            }
                            pffdst_upload_bytes[index] += self._compressed_size_bytes(payload)
                    elif is_quantized_sparse:
                        payload = {
                            "actor": self._make_quantized_sparse_payload(actor_sd, bits=low_bit),
                            "critic": self._make_quantized_sparse_payload(critic_sd, bits=low_bit),
                        }
                    elif is_sparse:
                        payload = {
                            "actor": self._make_sparse_payload(actor_sd),
                            "critic": self._make_sparse_payload(critic_sd),
                        }
                    else:
                        payload = {"actor": actor_sd, "critic": critic_sd}
                    if pffdst_masks is None:
                        per_round_upload_bytes += self._compressed_size_bytes(payload)

                # Server model for broadcast
                server_dir = os.path.join(model_dir, "server")
                server_actor_sd, server_critic_sd = load_states(server_dir)

                if is_quantized_sparse:
                    server_payload = {
                        "actor": self._make_quantized_sparse_payload(server_actor_sd, bits=low_bit),
                        "critic": self._make_quantized_sparse_payload(
                            server_critic_sd, bits=low_bit
                        ),
                    }
                elif is_sparse:
                    server_payload = {
                        "actor": self._make_sparse_payload(server_actor_sd),
                        "critic": self._make_sparse_payload(server_critic_sd),
                    }
                else:
                    server_payload = {"actor": server_actor_sd, "critic": server_critic_sd}

                full_download_bytes = num_homes * self._compressed_size_bytes(server_payload)
                if pffdst_masks is not None:
                    stage1_rounds = min(num_rounds, max(1, num_rounds // 2))
                    stage2_rounds = num_rounds - stage1_rounds
                    pffdst_download_bytes = []
                    for invert in (False, True):
                        payload = {
                            "actor": self._make_sparse_payload(
                                self._masked_state(server_actor_sd, pffdst_masks["actor"], invert)
                            ),
                            "critic": self._make_sparse_payload(
                                self._masked_state(server_critic_sd, pffdst_masks["critic"], invert)
                            ),
                        }
                        pffdst_download_bytes.append(
                            num_homes * self._compressed_size_bytes(payload)
                        )
                    train_up = (
                        pffdst_upload_bytes[0] * stage1_rounds
                        + pffdst_upload_bytes[1] * stage2_rounds
                    ) / self.BYTES_PER_MB
                    train_down = (
                        pffdst_download_bytes[0] * stage1_rounds
                        + pffdst_download_bytes[1] * stage2_rounds
                    ) / self.BYTES_PER_MB
                else:
                    train_up = (per_round_upload_bytes * num_rounds) / self.BYTES_PER_MB
                    train_down = (full_download_bytes * num_rounds) / self.BYTES_PER_MB

                # Test (deployment) comm: one-time model broadcast to all homes
                test_up = 0.0
                test_down = full_download_bytes / self.BYTES_PER_MB

            train_flops = 0
            test_flops = 0

            for hid in range(1, num_homes + 1):
                home_dir = os.path.join(model_dir, f"home_{hid}")
                actor_sd, critic_sd = load_states(home_dir)
                actor_flops = self._compute_flops_per_step(actor_sd)
                critic_flops = self._compute_flops_per_step(critic_sd)
                if pffdst_masks is None:
                    train_flops += (
                        self._train_step_flops(actor_flops, critic_flops, batch_size)
                        * episode
                        * steps_per_day
                    )
                    if model_name in {"edgehem", "fedfit", "feddmpq"}:
                        updates = fedfit_updates if model_name == "fedfit" else topology_updates
                        train_flops += batch_size * (4 * actor_flops + 7 * critic_flops) * updates
                else:
                    frozen_actor = self._compute_flops_per_step(
                        self._masked_state(actor_sd, pffdst_masks["actor"])
                    )
                    frozen_critic = self._compute_flops_per_step(
                        self._masked_state(critic_sd, pffdst_masks["critic"])
                    )
                    stage1_episodes = min(episode, max(1, num_rounds // 2) * aggregate)
                    stage2_step = (
                        actor_flops
                        + critic_flops
                        + batch_size
                        * (
                            3 * actor_flops
                            + 5 * critic_flops
                            + actor_flops
                            - frozen_actor
                            + 2 * (critic_flops - frozen_critic)
                        )
                    )
                    train_flops += (
                        self._train_step_flops(frozen_actor, frozen_critic, batch_size)
                        * stage1_episodes
                        + stage2_step * (episode - stage1_episodes)
                    ) * steps_per_day
                test_flops += (actor_flops + critic_flops) * num_test_days * steps_per_day

            train_csv_path = os.path.join(model_dir, "comm_train_time.csv")
            test_csv_path = os.path.join(model_dir, "comm_test_time.csv")

            pd.DataFrame(
                [
                    {
                        "uplink (MB)": round(train_up, 4),
                        "downlink (MB)": round(train_down, 4),
                        "total (MB)": round(train_up + train_down, 4),
                        "train_time (s)": round(train_time_s, 3),
                        "FLOPs": int(train_flops),
                    }
                ]
            ).to_csv(train_csv_path, index=False, float_format="%.4f")
            pd.DataFrame(
                [
                    {
                        "uplink (MB)": round(test_up, 4),
                        "downlink (MB)": round(test_down, 4),
                        "total (MB)": round(test_up + test_down, 4),
                        "test_time (s)": round(test_time_s, 3),
                        "FLOPs": int(test_flops),
                    }
                ]
            ).to_csv(test_csv_path, index=False, float_format="%.4f")


class ModelEvaluator:
    @staticmethod
    def get_home_ids(path_data=str(home_data_dir)):
        labels = (path.stem.removeprefix("home_") for path in Path(path_data).glob("home_*.csv"))
        return sorted(int(label) for label in labels if label.isdigit())

    def __init__(
        self,
        num_homes=10,
        path_data=str(home_data_dir),
        fixed_cost=0.0,
        home_ids=None,
        logs_root="train_test/logs",
        validation_days=0,
        scaler_mode="local",
        scaler_home_ids=None,
        grid_prices=None,
        data_period="legacy",
    ):
        self.device = device
        self.results = {}
        self.test_times = {}
        self.fixed_cost = fixed_cost
        self.logs_root = logs_root
        available_home_ids = self.get_home_ids(path_data)
        requested = (
            available_home_ids[:num_homes]
            if home_ids is None
            else [int(home_id) for home_id in home_ids]
        )
        self.home_ids = requested
        self.num_homes = len(self.home_ids)
        self.home_index_map = {i + 1: home_id for i, home_id in enumerate(self.home_ids)}
        self.homes = {}
        cohort = scaler_home_ids if scaler_mode == "shared" and scaler_home_ids else self.home_ids
        data = load_data(
            path_data,
            "*.csv",
            choose=[f"home_{h}" for h in cohort],
            validation_days=validation_days,
            data_period=data_period,
            scaler_mode=scaler_mode,
            grid_prices=grid_prices,
        )
        by_home = dict(zip(cohort, data, strict=True))
        for index, actual_home_id in self.home_index_map.items():
            train_data, test_data, test_dates, scaler = by_home[actual_home_id]
            self.homes[index] = SimpleNamespace(
                actual_home_id=actual_home_id,
                train_data=train_data,
                test_data=test_data,
                test_dates=test_dates,
                scaler=scaler,
            )

    def load_model(self, actor_path, critic_path):
        actor_state = safe_torch_load(actor_path, self.device)
        critic_state = safe_torch_load(critic_path, self.device)
        return heads_from_state(actor_state, critic_state, self.device)

    def _model_for_home(self, model_name, home_id):
        model_dir = Path(self.logs_root) / model_name / f"home_{home_id}"
        return self.load_model(model_dir / "actor.pt", model_dir / "critic.pt")

    @torch.inference_mode()
    def _choose_action(self, actor_net, critic_net, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        continuous_action = actor_net(state_tensor)
        values = (
            actor_net.discrete_features(actor_net.prepare_features(state_tensor))
            if hasattr(actor_net, "discrete_fc4")
            else critic_net(state_tensor, continuous_action)
        )
        discrete_action = values.argmax(1).item()
        return discrete_action, continuous_action.squeeze(0).cpu().numpy()

    _new_episode_log = staticmethod(new_episode_log)

    _record_environment = staticmethod(record_environment)

    _episode_data = staticmethod(episode_data)

    def _make_env(self, home, day_idx, em_strategy, state_dim=STATE_DIM):
        env = HOME_ENERGY_MGNT(
            home.test_data[day_idx],
            scaler=home.scaler,
            fixed_cost=self.fixed_cost,
            state_dim=state_dim,
        )
        if em_strategy:
            apply_em_strategy(env, em_strategy)
        states = np.asarray([env._state_for_step(step) for step in range(env.max_step + 1)])
        precompute_embeddings(states, self.device)
        return env

    def evaluate_episode(self, actor_net, critic_net, env):
        state = env.reset()
        log = self._new_episode_log(env)
        while True:
            action = self._choose_action(actor_net, critic_net, state)
            next_state, reward_elec, reward, reward_comf, done = env.step(action)
            self._record_environment(log, env)
            log["episode_reward"] += reward
            log["episode_elec_cost"] += reward_elec
            log["episode_comfort"] += reward_comf
            state = next_state
            if done:
                break
        return self._episode_data(log, env)

    calculate_metrics = staticmethod(calculate_metrics)

    @staticmethod
    def _summarize(metrics, actual_home_id):
        metrics_df = pd.DataFrame(metrics)
        numeric = metrics_df.select_dtypes(include=[np.number]).columns
        return {
            "mean": metrics_df[numeric].mean().to_dict(),
            "std": metrics_df[numeric].std().to_dict(),
            "all": metrics_df,
            "actual_home_id": actual_home_id,
        }

    def _evaluate_batched(self, model_name, em_strategy, home_ids, p2p_price=None):
        # Reuse the same batched rollout as live validation, avoiding two subtly
        # different simulation/evaluation loops. Final reports need no TD targets.
        from gridpfn.core.training_metrics import PeriodicEvaluator

        clients = []
        for home_id in home_ids:
            home = self.homes[home_id]
            actor, critic = self._model_for_home(model_name, home_id)
            clients.append(
                SimpleNamespace(
                    test_data=home.test_data,
                    test_dates=home.test_dates,
                    scaler=home.scaler,
                    fixed_cost=self.fixed_cost,
                    device=self.device,
                    state_dim=getattr(actor, "state_dim", STATE_DIM),
                    fedavg_agent=SimpleNamespace(
                        actor_net=actor, critic_net=critic, device=self.device
                    ),
                )
            )
        evaluator = PeriodicEvaluator(SimpleNamespace(home_ids=list(home_ids)), interval=1)
        server = SimpleNamespace(
            clients=clients,
            em_strategy=em_strategy or {},
            p2p_config={"enabled": p2p_price is not None, "price": p2p_price},
        )
        with torch.no_grad():
            record = evaluator.evaluate(server, include_days=True, include_losses=False)
        return {
            home_id: self._summarize(days, self.homes[home_id].actual_home_id)
            for home_id, days in zip(home_ids, record["day_records"], strict=True)
        }

    def evaluate_model(self, model_name, home_id, em_strategy=None):
        return self._evaluate_batched(model_name, em_strategy, [home_id])[home_id]

    def _evaluate_p2p_model(self, model_name, em_strategy, p2p_price):
        return self._evaluate_batched(model_name, em_strategy, list(self.homes), p2p_price)

    def evaluate_all_models(
        self, model_names, em_strategy=None, p2p_config=None, fed_p2p_models=None
    ):
        results = {}
        p2p_enabled = P2P_TRADING.is_enabled(p2p_config)
        p2p_price = P2P_TRADING.price(p2p_config) if p2p_enabled else None
        fed_p2p_models = set(fed_p2p_models or [])

        for model_name in model_names:
            print(f"--- Evaluating {CUSTOM_NAMES[model_name]} ---")
            started = time.perf_counter()
            strategy = (
                em_strategy.get(model_name)
                if isinstance(em_strategy, dict) and model_name in em_strategy
                else em_strategy
            )
            if p2p_enabled and model_name in fed_p2p_models:
                results[model_name] = self._evaluate_p2p_model(model_name, strategy, p2p_price)
            else:
                results[model_name] = {
                    home_id: self.evaluate_model(model_name, home_id, em_strategy=strategy)
                    for home_id in self.homes
                }
            self.test_times[model_name] = time.perf_counter() - started

        self.results = results
        return results


# ====================================================================================================
#                           Main Evaluation Function for Models
# ====================================================================================================


def _compose_evaluation_strategy_map(
    model_names, evaluator, em_strategy=None, em_strategy_models=None
):
    strategy_clients = list(evaluator.homes.values())

    model_strategy_keys = (
        set(model_names).intersection(em_strategy or {}) if isinstance(em_strategy, dict) else set()
    )
    if model_strategy_keys:
        return {
            model: compose_em_strategy(strategy, strategy_clients)
            for model, strategy in em_strategy.items()
            if model in model_names and strategy is not None
        }

    strategy = compose_em_strategy(em_strategy or make_em_strategy(), strategy_clients)
    strategy_targets = set(em_strategy_models or model_names)
    return {model: strategy for model in model_names if model in strategy_targets}


def run_evaluation(
    model_names=None,
    num_homes=10,
    path_data=str(home_data_dir),
    save_path="train_test/results/",
    generate_plots=True,
    verbose=True,
    em_strategy=None,
    p2p_config=None,
    fixed_cost=0.0,
    fed_p2p_models=None,
    em_strategy_models=None,
    home_ids=None,
    logs_root="train_test/logs",
    evaluation_results_root=None,
):
    model_names = model_names or list(CUSTOM_NAMES)

    if verbose:
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        print(f"Models: {model_names}")
        print(f"Homes: {num_homes}")
        print(f"Data path: {path_data}")
        print(f"Save path: {save_path}")
        print("=" * 80 + "\n")

    settings_path = Path(logs_root) / model_names[0] / "train_settings.txt"
    saved_settings = read_logged_settings(settings_path) if settings_path.exists() else {}
    # Reconstruct the same training-fitted scaler for the untouched test dates.
    evaluator = ModelEvaluator(
        num_homes=num_homes,
        path_data=path_data,
        fixed_cost=fixed_cost,
        home_ids=home_ids,
        logs_root=logs_root,
        validation_days=saved_settings.get("validation_days", 0),
        data_period=saved_settings.get("data_period", "legacy"),
        scaler_mode=saved_settings.get("scaler_mode", "local"),
        scaler_home_ids=saved_settings.get("home_ids"),
        grid_prices=saved_settings.get("grid_prices"),
    )

    if verbose:
        print(f"Home ID mapping: {evaluator.home_index_map}\n")

    em_strategy_map = _compose_evaluation_strategy_map(
        model_names, evaluator, em_strategy, em_strategy_models
    )

    results = evaluator.evaluate_all_models(
        model_names,
        em_strategy=em_strategy_map,
        p2p_config=p2p_config,
        fed_p2p_models=fed_p2p_models,
    )

    if verbose:
        print("\n" + "=" * 80)
        print("SAVING EVALUATION RESULTS")
        print("=" * 80)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    output_columns = [
        "day",
        "home_id",
        "elec_cost",
        "tot_reward",
        "peak_demand",
        "import",
        "export",
        "net_demand",
        "comfort",
    ]
    for model_name, home_results in results.items():
        frames = [
            result["all"].assign(home_id=home_id)
            for home_id, result in sorted(home_results.items())
        ]
        if not frames:
            continue
        output = pd.concat(frames, ignore_index=True).rename(columns={"reward": "tot_reward"})
        csv_path = (
            Path(evaluation_results_root or logs_root) / model_name / "evaluation_results.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        output[output_columns].to_csv(csv_path, index=False, float_format="%.4f")
        if verbose:
            print(f"Saved: {csv_path}")

    if generate_plots:
        plot_comparison_metrics(results, model_names, save_path)

    return results, evaluator


def configure_evaluation_runtime(settings, gpu=None, cpu_threads=2):
    global device
    setup_seed(int(settings["fixed_seed"]))
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(min(2, cpu_threads))
    saved_device = str((settings.get("devices") or ["cuda:0"])[0])
    gpu_id = (
        gpu if gpu is not None else int(saved_device.split(":")[-1]) if ":" in saved_device else 0
    )
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    feature_device = settings.get("embedding_device", str(device))
    configure_embedding_device(feature_device if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}; CPU threads: {cpu_threads}")


def generate_logged_plots(
    model_names,
    results,
    logs_root,
    results_root,
    num_homes,
    episode,
    aggregate,
    window,
    evaluator,
    low_bit,
    batch_size=32,
    update=10,
    fedfit_t_end=None,
):
    distributed = list(model_names)
    if distributed:
        for metric, filename in {
            "reward": "rewards_plots_all_homes.pdf",
            "actor_loss": "loss_plots_all_homes_actor.pdf",
            "critic_loss": "loss_plots_all_homes_critic.pdf",
        }.items():
            plot_training_metric(
                metric,
                distributed,
                num_homes=num_homes,
                window=window,
                logs_base=logs_root,
                save_path=os.path.join(results_root, filename),
            )

    if "feddmpq" in model_names:
        plot_model_precision(
            num_homes,
            low_bit=low_bit,
            window=window,
            logs_base=logs_root,
            save_path=os.path.join(results_root, "model_precision_FedDMPQ.pdf"),
        )

    num_test_days = min(
        len(home_result["all"])
        for model_results in results.values()
        for home_result in model_results.values()
    )
    CommunicationCostEvaluator(results_dir=logs_root).compute(
        model_names,
        num_homes,
        episode,
        aggregate,
        test_times=evaluator.test_times,
        num_test_days=num_test_days,
        low_bit=low_bit,
        batch_size=batch_size,
        update=update,
        fedfit_t_end=fedfit_t_end,
    )
    plot_comm_time(
        model_names,
        logs_base=logs_root,
        num_homes=num_homes,
        save_path=os.path.join(results_root, "model_comm_train_time.pdf"),
    )
    plot_comm_time(
        model_names,
        mode="test",
        logs_base=logs_root,
        num_homes=num_homes,
        save_path=os.path.join(results_root, "model_comm_test_time.pdf"),
    )
    plot_radar_chart(results, model_names, save_path=results_root, logs_base=logs_root)


def main(args, eval_models, federated_models):
    path_train = Path(args.path_train).expanduser().resolve()
    logs_root, results_root = str(path_train / "logs"), str(path_train / "results")
    available_models = [model for model in eval_models if model in detect_models(logs_root)]

    settings = read_logged_settings(
        os.path.join(logs_root, available_models[0], "train_settings.txt")
    )
    saved_evaluation = settings.get("evaluation_settings")
    saved_evaluation = saved_evaluation if isinstance(saved_evaluation, dict) else None
    default_models = (saved_evaluation or {}).get("model_names") or available_models
    model_names = (
        list(default_models)
        if args.run_models is None
        else list(
            dict.fromkeys(
                model.strip().lower()
                for value in args.run_models
                for model in str(value).split(",")
                if model.strip()
            )
        )
    )

    inferred_home_ids, checkpoint_count = resolve_trained_home_ids(model_names, logs_root)
    home_ids = args.home_ids or (saved_evaluation or {}).get("home_ids") or inferred_home_ids
    if args.num_homes is not None:
        home_ids = home_ids[: args.num_homes] if home_ids else None
    num_homes = len(home_ids) if home_ids else checkpoint_count

    path_data = (
        args.path_data
        or (saved_evaluation or {}).get("path_data")
        or (saved_evaluation or {}).get("data_path")
        or settings.get("path_data", str(home_data_dir))
    )
    path_data = Path(path_data).expanduser()
    if not path_data.is_absolute() and not path_data.exists():
        path_data = ROOT / path_data
    path_data = str(path_data.resolve())
    fixed_cost = (
        (saved_evaluation or {}).get("fixed_cost", settings.get("fixed_cost", 0.0))
        if args.fixed_cost is None
        else args.fixed_cost
    )
    episode = args.episode or int(settings.get("episode", 2000))
    aggregate = args.aggregate or int(settings.get("aggregate", 40))
    window = int(settings.get("ma_window", 1)) if args.ma_window is None else args.ma_window
    low_bit = int(settings.get("feddmpq_params", {}).get("low_bit", 8))
    batch_size = int(settings.get("agent_params", {}).get("batch_size", 32))
    update = int(settings.get("update", 10))
    fedfit_t_end = settings.get("fedfit_params", {}).get("t_end")

    p2p_config = dict(
        (saved_evaluation or {}).get("p2p_config") or settings.get("p2p_config") or {}
    )
    if args.p2p is not None:
        p2p_config["enabled"] = args.p2p
    if args.p2p_price is not None:
        p2p_config.update(enabled=True, price=args.p2p_price)
    saved_fed_p2p = (saved_evaluation or {}).get("fed_p2p_models")
    p2p_capable_models = set(saved_fed_p2p) if saved_fed_p2p is not None else set(federated_models)
    fed_p2p_models = [
        model
        for model in model_names
        if model in p2p_capable_models and P2P_TRADING.is_enabled(p2p_config)
    ]

    evaluation_strategy = (
        saved_evaluation.get("em_strategy") if saved_evaluation is not None else None
    )

    configure_evaluation_runtime(settings, gpu=args.gpu, cpu_threads=args.cpu_threads)
    results, evaluator = run_evaluation(
        model_names=model_names,
        num_homes=num_homes,
        path_data=path_data,
        save_path=results_root,
        fixed_cost=fixed_cost,
        p2p_config=p2p_config,
        fed_p2p_models=fed_p2p_models,
        em_strategy=evaluation_strategy,
        home_ids=home_ids,
        logs_root=logs_root,
        evaluation_results_root=logs_root,
    )
    generate_logged_plots(
        model_names,
        results,
        logs_root,
        results_root,
        num_homes,
        episode,
        aggregate,
        window,
        evaluator,
        low_bit,
        batch_size,
        update,
        fedfit_t_end,
    )


if __name__ == "__main__":
    path_train = "results/eval_seed_6/train_sparsity_75"
    EVAL_MODELS = ("fedavg", "edgehem", "fedfit", "pffdst", "feddmpq")
    FEDERATED_MODELS = EVAL_MODELS

    parser = argparse.ArgumentParser(
        description="Evaluate logged models and regenerate result plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path_train", default=path_train, help="Directory containing logs and results."
    )
    parser.add_argument(
        "--run_models",
        nargs="+",
        choices=EVAL_MODELS,
        help="Models to evaluate; detects saved checkpoints by default.",
    )
    parser.add_argument("--path_data", help="Dataset override.")
    parser.add_argument("--home_ids", nargs="+", type=int, help="Dataset home-ID override.")
    parser.add_argument("--num_homes", type=int, help="Maximum trained homes to evaluate.")
    parser.add_argument("--fixed_cost", type=float, help="Monthly fixed-cost override.")
    parser.add_argument("--episode", type=int, help="Training-episode override for FLOPs.")
    parser.add_argument("--aggregate", type=int, help="Aggregation-interval override.")
    parser.add_argument(
        "--ma_window", type=int, default=100, help="Plot with moving-average window."
    )
    parser.add_argument(
        "--p2p", action=argparse.BooleanOptionalAction, default=None, help="P2P override."
    )
    parser.add_argument("--p2p_price", type=float, help="P2P-price override.")
    parser.add_argument("--gpu", type=int, help="CUDA device override.")
    parser.add_argument("--cpu_threads", type=int, default=5, help="CPU worker threads.")
    main(parser.parse_args(), EVAL_MODELS, FEDERATED_MODELS)
