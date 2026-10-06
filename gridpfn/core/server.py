import copy
import os
import time

import pandas as pd
import torch

from gridpfn.core.em_strategy import P2P_TRADING, compose_em_strategy
from gridpfn.core.utils.agent_utils import hard_update_target_network, low_precision_param_column

logs_root = os.environ.get("FED_HEMS_LOGS_ROOT", os.path.join("train_test", "logs"))


class TrainLogs:
    def log_feddmpq_params(logs, hid, params_fp32, params_low_precision, episode):
        logs["params_fp32"][hid].append((episode, params_fp32))
        logs["params_low_precision"][hid].append((episode, params_low_precision))

    def save_train_time(model_name, total_training_time):
        base_dir = os.path.join(logs_root, model_name)
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, "comm_train_time.csv")

        df = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()
        if df.empty:
            df.loc[0, "train_time (s)"] = round(float(total_training_time), 3)
        else:
            df.loc[df.index[-1], "train_time (s)"] = round(float(total_training_time), 3)

        df.to_csv(path, index=False, float_format="%.4f")

    def save_model_logs(model_name, logs):
        base_dir = os.path.join(logs_root, model_name)
        os.makedirs(base_dir, exist_ok=True)

        for hid, episodes in logs["rewards"].items():
            home_dir = os.path.join(base_dir, f"home_{hid}")
            os.makedirs(home_dir, exist_ok=True)

            actor_loss_by_ep = dict(logs["actor_losses"][hid])
            critic_loss_by_ep = dict(logs["critic_losses"][hid])
            params_fp32_by_ep = dict(logs.get("params_fp32", {}).get(hid, []))
            low_precision_by_ep = dict(logs.get("params_low_precision", {}).get(hid, []))
            precision_column = (
                low_precision_param_column(logs["low_bit"]) if low_precision_by_ep else None
            )
            df_data = []
            for episode, reward in episodes:
                actor_loss = actor_loss_by_ep.get(episode)
                critic_loss = critic_loss_by_ep.get(episode)
                row = {
                    "episode": episode,
                    "reward": reward,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                }
                if params_fp32_by_ep:
                    row["params_fp32"] = params_fp32_by_ep.get(episode)
                if precision_column:
                    row[precision_column] = low_precision_by_ep.get(episode)
                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(os.path.join(home_dir, "train_logs.csv"), index=False, float_format="%.4f")

        total_training_time = logs["end_time"] - logs["start_time"]
        TrainLogs.save_train_time(model_name, total_training_time)

    def save_best_models(model_name, hid, actor_net, critic_net):
        base_dir = os.path.join(logs_root, model_name)
        home_dir = os.path.join(base_dir, f"home_{hid}")
        os.makedirs(home_dir, exist_ok=True)

        actor = {
            name: value.detach().cpu().clone() for name, value in actor_net.state_dict().items()
        }
        critic = {
            name: value.detach().cpu().clone() for name, value in critic_net.state_dict().items()
        }
        torch.save(actor, os.path.join(home_dir, "actor.pt"))
        torch.save(critic, os.path.join(home_dir, "critic.pt"))

    def save_global_models(model_name, actor_params, critic_params):
        base_dir = os.path.join(logs_root, model_name)
        server_dir = os.path.join(base_dir, "server")
        os.makedirs(server_dir, exist_ok=True)

        torch.save(actor_params, os.path.join(server_dir, "actor.pt"))
        if critic_params is not None:
            torch.save(critic_params, os.path.join(server_dir, "critic.pt"))

    def save_pffdst_frozen_masks(actor_masks, critic_masks):
        server_dir = os.path.join(logs_root, "pffdst", "server")
        os.makedirs(server_dir, exist_ok=True)
        torch.save(
            {"actor": actor_masks, "critic": critic_masks},
            os.path.join(server_dir, "frozen_masks.pt"),
        )

    def count_feddmpq_params(client):
        agent = client.feddmpq_agent
        fp32 = low_precision = 0
        actor_masks, critic_masks = agent.get_model_masks(device=agent.device)
        for masks, precision in (
            (actor_masks, agent.actor_precision),
            (critic_masks, agent.critic_precision),
        ):
            for name, mask in masks.items():
                active = mask.bool()
                precision_mask = precision.get(name)
                if precision_mask is None:
                    fp32 += int(active.sum().item())
                    continue
                precision_mask = precision_mask.bool().to(active.device)
                low_precision += int((active & precision_mask).sum().item())
                fp32 += int((active & ~precision_mask).sum().item())
        return fp32, low_precision


class Server:
    def __init__(
        self,
        clients,
        episode,
        actor_sparsity,
        critic_sparsity,
        update,
        aggregate,
        em_strategy,
        p2p_config=None,
        warmup_rounds=75,
        metrics_logger=None,
        eval_callback=None,
        batched_updates=False,
        local_critics=False,
    ) -> None:

        self.clients = clients
        self.episode = episode
        self.warmup_rounds = int(warmup_rounds)
        self.actor_sparsity = actor_sparsity
        self.critic_sparsity = critic_sparsity
        self.update = update
        self.aggregate = aggregate
        self.em_strategy = compose_em_strategy(em_strategy, clients)
        self.p2p_config = p2p_config or {"enabled": False, "price": None}
        self.metrics_logger = metrics_logger
        self.eval_callback = eval_callback
        self.batched_updates = batched_updates
        self.batched_learner = None
        self.local_critics = local_critics

    def _make_client_logs(self, low_bit=None):
        home_ids = range(1, len(self.clients) + 1)
        fields = ("rewards", "actor_losses", "critic_losses")
        logs = {field: {home_id: [] for home_id in home_ids} for field in fields}
        if low_bit is not None:
            logs["params_fp32"] = {home_id: [] for home_id in home_ids}
            logs["params_low_precision"] = {home_id: [] for home_id in home_ids}
        logs["start_time"] = time.time()
        logs["low_bit"] = low_bit
        return logs

    def _collect_episode_results(self, episode, train_fn, use_fed_for_p2p=False):
        if P2P_TRADING.is_enabled(self.p2p_config):
            return P2P_TRADING.run_episode(
                self.clients,
                episode,
                use_fed=use_fed_for_p2p,
                p2p_config=self.p2p_config,
                learner=self.batched_learner,
            )

        return [train_fn(client, episode % len(client.train_data)) for client in self.clients]

    def _warmup_clients(self, get_agent_fn):
        for warmup_round in range(self.warmup_rounds):
            for client in self.clients:
                agent = get_agent_fn(client)
                index = warmup_round % len(client.train_data)
                client.warmup_episode(agent, index)

    def _log_episode_results(self, logs, episode_results, episode, tag):
        started = time.perf_counter()
        if self.metrics_logger is not None:
            self.metrics_logger.training(
                tag.lower(),
                episode + 1,
                episode_results,
                timings=getattr(self.batched_learner, "last_episode_timings", None),
                scenario=("synthetic" if self.batched_learner.last_synthetic else "real")
                if hasattr(self.clients[0], "real_train_count")
                else None,
            )
        for hid, result in enumerate(episode_results):
            reward, actor_loss, critic_loss = result
            client_id = hid + 1
            logs["rewards"][client_id].append((episode, f"{reward:.4f}"))
            if actor_loss is not None and critic_loss is not None:
                logs["actor_losses"][client_id].append((episode, f"{actor_loss:.4f}"))
                logs["critic_losses"][client_id].append((episode, f"{critic_loss:.4f}"))
                loss_str = f"actor_loss: {actor_loss:.6f}, critic_loss: {critic_loss:.6f}"
            else:
                loss_str = "actor_loss: N/A, critic_loss: N/A"
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{tag}] {timestamp} episode: {episode}, home: {client_id}, "
                f"reward: {reward:.6f}, {loss_str}",
                flush=True,
            )
        if self.metrics_logger is not None:
            self.metrics_logger.write(
                {
                    "kind": "logging",
                    "episode": episode + 1,
                    "seconds": time.perf_counter() - started,
                }
            )

    # Average federated agent models across all clients (dense baseline).
    def fedavg_model_average(self):
        if self.local_critics:
            actors = [
                {
                    name: value.detach().cpu().clone()
                    for name, value in client.fedavg_agent.actor_net.state_dict().items()
                }
                for client in self.clients
            ]
            return self._average_parameters(actors), None
        actors, critics = zip(*(client.get_fedavg_model_params() for client in self.clients))
        return self._average_parameters(actors), self._average_parameters(critics)

    @staticmethod
    def _average_parameters(parameter_sets):
        averaged = copy.deepcopy(parameter_sets[0])
        for name in averaged:
            if not averaged[name].is_floating_point():
                if any(
                    not torch.equal(averaged[name], params[name]) for params in parameter_sets[1:]
                ):
                    raise ValueError(f"Federated model metadata differs: {name}")
                continue
            if name in {
                "feature_mean",
                "feature_scale",
                "residual_scale",
                "thermal_bounds",
                "target_temperature_bounds",
                "quota_actor",
                "control_dt",
                "embedding_weight",
                "quota_correction",
            } or name.startswith("base_"):
                if any(
                    not torch.equal(averaged[name], parameters[name])
                    for parameters in parameter_sets[1:]
                ):
                    raise ValueError("FedAvg requires a shared feature normalization")
                continue
            for parameters in parameter_sets[1:]:
                averaged[name] += parameters[name]
            averaged[name] /= len(parameter_sets)
        return averaged

    @staticmethod
    def _masked_average(parameter_sets, mask_sets):
        averaged = {}
        for name in parameter_sets[0]:
            count = torch.zeros_like(mask_sets[0][name], device="cpu")
            total = torch.zeros_like(parameter_sets[0][name])
            for parameters, masks in zip(parameter_sets, mask_sets):
                count += masks[name].cpu()
                total += parameters[name]
            averaged[name] = total * torch.where(
                count != 0, count.reciprocal(), torch.zeros_like(count)
            )
        return averaged

    def _sparse_model_average(self, parameter_getter, mask_getter):
        actors, critics = zip(*(parameter_getter(client) for client in self.clients))
        actor_masks, critic_masks = zip(*(mask_getter(client) for client in self.clients))
        return self._masked_average(actors, actor_masks), self._masked_average(
            critics, critic_masks
        )

    # Average sparse agent models accounting for per-client mask differences (weighted aggregation).
    def sparse_fed_model_average(self):
        return self._sparse_model_average(
            lambda client: client.get_model_params(), lambda client: client.get_model_masks()
        )

    def fedfit_sparse_fed_model_average(self):
        return self._sparse_model_average(
            lambda client: client.get_fedfit_model_params(),
            lambda client: client.get_fedfit_model_masks(),
        )

    def pffdst_sparse_fed_model_average(self):
        actor, critic = self._sparse_model_average(
            lambda client: client.get_pffdst_model_params(),
            lambda client: client.get_pffdst_communication_masks(),
        )
        frozen = self.clients[0].pffdst_agent.get_frozen_state()
        if frozen is not None:
            actor_masks, critic_masks, actor_values, critic_values = frozen
            for name, mask in actor_masks.items():
                actor[name][mask] = actor_values[name][mask]
            for name, mask in critic_masks.items():
                critic[name][mask] = critic_values[name][mask]
        return actor, critic

    @staticmethod
    def _fedfit_consensus(mask_sets, score_sets, target_counts):
        if not score_sets:
            return copy.deepcopy(mask_sets[0])
        consensus = {}
        for name in mask_sets[0]:
            if mask_sets[0][name].ndim < 2:
                consensus[name] = mask_sets[0][name].clone()
                continue
            union = mask_sets[0][name].bool().clone()
            for masks in mask_sets[1:]:
                union.logical_or_(masks[name].bool())
            candidates = union.flatten().nonzero(as_tuple=False).squeeze(1)
            keep_count = min(target_counts[name], candidates.numel())
            scores = sum(score[name] for score in score_sets) / len(score_sets)
            keep = candidates[
                torch.topk(scores.flatten()[candidates], keep_count, sorted=False).indices
            ]
            consensus[name] = torch.zeros_like(mask_sets[0][name])
            consensus[name].flatten()[keep] = 1
        return consensus

    def fedfit_global_masks(self, target_counts):
        actor_masks, critic_masks = zip(
            *(client.get_fedfit_model_masks() for client in self.clients)
        )
        scores = [
            client.fedfit_agent.scores
            for client in self.clients
            if client.fedfit_agent.scores is not None
        ]
        actor_scores = [score[0]["prune"] for score in scores]
        critic_scores = [score[1]["prune"] for score in scores]
        return (
            self._fedfit_consensus(actor_masks, actor_scores, target_counts[0]),
            self._fedfit_consensus(critic_masks, critic_scores, target_counts[1]),
        )

    # ====================================================================================================
    #                                         Energy Management Models
    # ====================================================================================================
    # Train federated (dense) models with periodic server-side aggregation.
    def fedavg_train(self):
        from gridpfn.core.model import validate_encoder_context

        validate_encoder_context(self.clients, self.em_strategy, self.p2p_config)
        # FedAvg averages updates from a shared starting point. Independently
        # initialized hidden units are not aligned for parameter averaging.
        actor_params, critic_params = self.clients[0].get_fedavg_model_params()
        for client in self.clients:
            if "thermal_bounds" in actor_params and not torch.equal(
                actor_params["thermal_bounds"], client.fedavg_agent.actor_net.thermal_bounds.cpu()
            ):
                raise ValueError("Temperature actors require shared weather scaling across homes")
            client.set_fedavg_model_params(actor_params, critic_params)
            agent = client.fedavg_agent
            hard_update_target_network(agent.actor_net, agent.actor_target_net)
            hard_update_target_network(agent.critic_net, agent.critic_target_net)
            client.set_em_strategy(self.em_strategy)
        if getattr(self.clients[0].fedavg_agent.actor_net, "feature_mode", "frozen") in (
            "normalized",
            "hybrid",
        ):
            # One shared coordinate system; fit only on the training days.
            from gridpfn.core.model import embedding_rows

            total = square = None
            count = 0
            for client in self.clients:
                features = embedding_rows(client.training_states[:, :8], client.device)
                block_sum, block_square = features.sum(0).cpu(), features.square().sum(0).cpu()
                total = block_sum if total is None else total + block_sum
                square = block_square if square is None else square + block_square
                count += len(features)
            mean = total / count
            scale = (square / count - mean.square()).clamp_min(0).sqrt().clamp_min(0.01)
            for client in self.clients:
                agent = client.fedavg_agent
                for net in (
                    agent.actor_net,
                    agent.critic_net,
                    agent.actor_target_net,
                    agent.critic_target_net,
                ):
                    net.feature_mean.copy_(mean)
                    net.feature_scale.copy_(scale)
        from gridpfn.core.control_guidance import initialize_guided_actors

        onpolicy = getattr(self.clients[0].fedavg_agent, "actor_update", "q_gradient") == "ppo"
        if onpolicy and all(c.device.type == "cpu" for c in self.clients):
            from gridpfn.core.model import release_embedding_models

            if hasattr(self.eval_callback, "prepare_features"):
                self.eval_callback.prepare_features(self)
            freed = release_embedding_models()
            if self.metrics_logger is not None:
                self.metrics_logger.write({"kind": "feature_cache", "episode": 0, **freed})
        initialize_guided_actors(self)
        if onpolicy and (self.warmup_rounds or not self.batched_updates or not self.local_critics):
            raise ValueError("PPO requires zero warmup, synchronized rollouts and local critics")
        self._warmup_clients(lambda c: c.fedavg_agent)
        if self.batched_updates:
            agents = [c.fedavg_agent for c in self.clients]
            if onpolicy:
                from gridpfn.core.agents.onpolicy import OnPolicyLearner

                self.batched_learner = OnPolicyLearner(
                    agents, agents[0].ppo_rollout_days or self.aggregate
                )
                days, steps = getattr(self, "value_warmup", (0, 256))
                if days:
                    warmup = self.batched_learner.warmup_values(
                        self.clients, self.p2p_config, days, steps
                    )
                    if self.metrics_logger is not None:
                        self.metrics_logger.write({"kind": "value_warmup", **warmup})
            else:
                from gridpfn.core.batched_learning import BatchedLearner

                self.batched_learner = BatchedLearner(agents)
        if self.eval_callback is not None:
            self.eval_callback(self, 0)
        logs = self._make_client_logs()
        global_actor_params, global_critic_params = None, None

        for episode in range(self.episode):
            episode_results = self._collect_episode_results(
                episode, train_fn=lambda c, idx: c.fedavg_train(idx), use_fed_for_p2p=True
            )
            self._log_episode_results(logs, episode_results, episode, "FedAvg")
            if (episode + 1) % self.aggregate == 0:
                if onpolicy and hasattr(self.eval_callback, "before_broadcast"):
                    self.eval_callback.before_broadcast(self, episode + 1)
                global_actor_params, global_critic_params = self.fedavg_model_average()
                for client in self.clients:
                    if self.local_critics:
                        mode = getattr(client.fedavg_agent, "ppo_federation", "actor")
                        if mode == "none":
                            continue
                        actor_params = global_actor_params
                        if mode == "trunk":
                            private = client.fedavg_agent.actor_net.state_dict()
                            actor_params = {
                                name: private[name]
                                if name == "log_std" or name.startswith(("fc4.", "discrete_fc4."))
                                else value
                                for name, value in global_actor_params.items()
                            }
                        client.fedavg_agent.actor_net.load_state_dict(actor_params)
                    else:
                        client.set_fedavg_model_params(global_actor_params, global_critic_params)
                if onpolicy:
                    self.batched_learner.after_broadcast()
            if (
                onpolicy
                and self.metrics_logger is not None
                and (episode + 1) % self.batched_learner.interval == 0
            ):
                diagnostic = self.batched_learner.last_diagnostics
                self.metrics_logger.write(
                    {
                        "kind": "ppo_update",
                        "episode": episode + 1,
                        "homes": [
                            {"home_id": c.home_id, **{k: v[i] for k, v in diagnostic.items()}}
                            for i, c in enumerate(self.clients)
                        ],
                    }
                )
            if self.eval_callback is not None:
                self.eval_callback(self, episode + 1)
                if getattr(getattr(self.eval_callback, "stopping", None), "stopped", False):
                    print(f"[stop] Validation plateau at episode {episode + 1}", flush=True)
                    break
        logs["end_time"] = time.time()
        # Include updates since the last communication round in the final artifact.
        global_actor_params, global_critic_params = self.fedavg_model_average()
        TrainLogs.save_model_logs("fedavg", logs)
        for hid, client in enumerate(self.clients):
            client_id = hid + 1
            TrainLogs.save_best_models(
                "fedavg", client_id, client.fedavg_actor_net, client.fedavg_critic_net
            )
        TrainLogs.save_global_models("fedavg", global_actor_params, global_critic_params)
        return logs

    # Train sparse models with periodic federated aggregation and dynamic sparsity updates.
    def edgehem_train(self):
        actor_mask, critic_mask = self.clients[0].initialize(
            self.actor_sparsity, self.critic_sparsity
        )
        for client in self.clients:
            client.set_model_masks(actor_mask, critic_mask)
            client.set_em_strategy(self.em_strategy)
        self._warmup_clients(lambda c: c.agent)
        logs = self._make_client_logs()

        global_actor_params, global_critic_params = None, None

        for episode in range(self.episode):
            if episode % self.aggregate == 0 and episode > 0:
                global_actor_params, global_critic_params = self.sparse_fed_model_average()
                for c in self.clients:
                    c.set_model_params(global_actor_params, global_critic_params)
            episode_results = self._collect_episode_results(
                episode, train_fn=lambda c, idx: c.train(idx), use_fed_for_p2p=False
            )
            self._log_episode_results(logs, episode_results, episode, "EdgeHEM")
            if episode % self.update == 0 and episode > 0:
                for client in self.clients:
                    client.dynamic_update(episode)
        logs["end_time"] = time.time()
        global_actor_params, global_critic_params = self.sparse_fed_model_average()
        TrainLogs.save_model_logs("edgehem", logs)
        for hid, client in enumerate(self.clients):
            client_id = hid + 1
            TrainLogs.save_best_models("edgehem", client_id, client.actor_net, client.critic_net)
        TrainLogs.save_global_models("edgehem", global_actor_params, global_critic_params)
        return logs

    # Average FedDMPQ sparse models using mixed-precision client payloads and sparse masks.
    def feddmpq_sparse_fed_model_average(self):
        return self._sparse_model_average(
            lambda client: client.get_feddmpq_mixed_precision_params(),
            lambda client: client.get_feddmpq_model_masks(),
        )

    # Train FedDMPQ with soft-pruning via precision reduction.
    def feddmpq_train(self):
        actor_mask, critic_mask = self.clients[0].feddmpq_initialize(
            self.actor_sparsity, self.critic_sparsity
        )
        for client in self.clients[1:]:
            client.feddmpq_agent.init_soft_prune_state()
            client.set_feddmpq_model_masks(copy.deepcopy(actor_mask), copy.deepcopy(critic_mask))
        for client in self.clients:
            client.set_em_strategy(self.em_strategy)
        self._warmup_clients(lambda c: c.feddmpq_agent)
        logs = self._make_client_logs(low_bit=self.clients[0].feddmpq_agent.low_bit)

        global_actor_params, global_critic_params = None, None

        for episode in range(self.episode):
            episode_results = self._collect_episode_results(
                episode, train_fn=lambda c, idx: c.feddmpq_train(idx), use_fed_for_p2p="feddmpq"
            )
            if episode % self.update == 0 and episode > 0:
                for client in self.clients:
                    client.dynamic_update_feddmpq(self.actor_sparsity, self.critic_sparsity)

            self._log_episode_results(logs, episode_results, episode, "FedDMPQ")
            for hid, client in enumerate(self.clients, start=1):
                fp32, low_precision = TrainLogs.count_feddmpq_params(client)
                TrainLogs.log_feddmpq_params(logs, hid, fp32, low_precision, episode)
            if (episode + 1) % self.aggregate == 0 and episode < self.episode - 1:
                global_actor_params, global_critic_params = self.feddmpq_sparse_fed_model_average()
                for c in self.clients:
                    c.set_feddmpq_model_params(global_actor_params, global_critic_params)
        logs["end_time"] = time.time()
        TrainLogs.save_model_logs("feddmpq", logs)
        for hid, client in enumerate(self.clients):
            client_id = hid + 1
            TrainLogs.save_best_models(
                "feddmpq", client_id, client.feddmpq_actor_net, client.feddmpq_critic_net
            )
        global_actor_params, global_critic_params = self.feddmpq_sparse_fed_model_average()
        TrainLogs.save_global_models("feddmpq", global_actor_params, global_critic_params)
        return logs

    def fedfit_train(self):
        actor_mask, critic_mask = self.clients[0].fedfit_initialize(
            self.actor_sparsity, self.critic_sparsity
        )
        for client in self.clients[1:]:
            client.set_fedfit_model_masks(actor_mask, critic_mask)
        for client in self.clients:
            client.set_em_strategy(self.em_strategy)
        target_counts = (
            {name: int(mask.sum()) for name, mask in actor_mask.items()},
            {name: int(mask.sum()) for name, mask in critic_mask.items()},
        )
        self._warmup_clients(lambda client: client.fedfit_agent)
        logs = self._make_client_logs()

        for episode in range(self.episode):
            results = self._collect_episode_results(
                episode,
                train_fn=lambda client, index: client.fedfit_train(index),
                use_fed_for_p2p="fedfit",
            )
            self._log_episode_results(logs, results, episode, "FedFit")
            if episode % self.update == 0 and episode > 0:
                for client in self.clients:
                    client.dynamic_update_fedfit(episode)
            if (episode + 1) % self.aggregate == 0 and episode < self.episode - 1:
                actor_params, critic_params = self.fedfit_sparse_fed_model_average()
                actor_mask, critic_mask = self.fedfit_global_masks(target_counts)
                for client in self.clients:
                    client.set_fedfit_model_masks(actor_mask, critic_mask)
                    client.set_fedfit_model_params(actor_params, critic_params)

        logs["end_time"] = time.time()
        TrainLogs.save_model_logs("fedfit", logs)
        for hid, client in enumerate(self.clients, start=1):
            TrainLogs.save_best_models(
                "fedfit", hid, client.fedfit_actor_net, client.fedfit_critic_net
            )
        actor_params, critic_params = self.fedfit_sparse_fed_model_average()
        actor_mask, critic_mask = self.fedfit_global_masks(target_counts)
        for name, mask in actor_mask.items():
            actor_params[name].mul_(mask)
        for name, mask in critic_mask.items():
            critic_params[name].mul_(mask)
        TrainLogs.save_global_models("fedfit", actor_params, critic_params)
        return logs

    def pffdst_train(self):
        actor_mask, critic_mask = self.clients[0].pffdst_initialize(
            self.actor_sparsity, self.critic_sparsity
        )
        actor_params, critic_params = self.clients[0].get_pffdst_model_params()
        for client in self.clients[1:]:
            client.set_pffdst_model_params(actor_params, critic_params)
            client.pffdst_initialize(
                self.actor_sparsity, self.critic_sparsity, (actor_mask, critic_mask)
            )
        for client in self.clients:
            client.set_em_strategy(self.em_strategy)
        self._warmup_clients(lambda client: client.pffdst_agent)
        logs = self._make_client_logs()

        communication_rounds = max(0, (self.episode - 1) // self.aggregate)
        stage_rounds = max(1, communication_rounds // 2)
        readjust_end = max(1, stage_rounds // 4)
        interval = self.clients[0].pffdst_agent.readjust_interval
        round_idx = 0

        for episode in range(self.episode):
            results = self._collect_episode_results(
                episode,
                train_fn=lambda client, index: client.pffdst_train(index),
                use_fed_for_p2p="pffdst",
            )
            self._log_episode_results(logs, results, episode, "PFFDST")
            if (episode + 1) % self.aggregate != 0 or episode >= self.episode - 1:
                continue

            actor_params, critic_params = self.pffdst_sparse_fed_model_average()
            for client in self.clients:
                client.set_pffdst_model_params(actor_params, critic_params)
            round_idx += 1
            leader = self.clients[0].pffdst_agent
            update = None
            if round_idx <= stage_rounds:
                if round_idx == readjust_end:
                    update = leader.server_readjust("stage1", grow=False)
                elif round_idx < readjust_end and round_idx % interval == 0:
                    update = leader.server_readjust("stage1")
                if round_idx == stage_rounds:
                    if update is not None:
                        actor_params, critic_params, actor_mask, critic_mask, survivors = update
                        for client in self.clients[1:]:
                            client.set_pffdst_model_params(actor_params, critic_params)
                            client.set_pffdst_model_masks(actor_mask, critic_mask, survivors)
                    leader.freeze_subnetwork()
                    frozen = leader.get_frozen_state()
                    for client in self.clients[1:]:
                        client.pffdst_agent.freeze_subnetwork(frozen)
                    update = leader.server_readjust("stage2")
            else:
                local_round = round_idx - stage_rounds
                if local_round == readjust_end:
                    update = leader.server_readjust("stage2", grow=False)
                elif local_round < readjust_end and local_round % interval == 0:
                    update = leader.server_readjust("stage2")

            if update is not None:
                actor_params, critic_params, actor_mask, critic_mask, survivors = update
                for client in self.clients[1:]:
                    client.set_pffdst_model_params(actor_params, critic_params)
                    client.set_pffdst_model_masks(actor_mask, critic_mask, survivors)

        logs["end_time"] = time.time()
        TrainLogs.save_model_logs("pffdst", logs)
        for hid, client in enumerate(self.clients, start=1):
            TrainLogs.save_best_models(
                "pffdst", hid, client.pffdst_actor_net, client.pffdst_critic_net
            )
        actor_params, critic_params = self.pffdst_sparse_fed_model_average()
        TrainLogs.save_global_models("pffdst", actor_params, critic_params)
        frozen = self.clients[0].pffdst_agent.get_frozen_state()
        if frozen is not None:
            TrainLogs.save_pffdst_frozen_masks(frozen[0], frozen[1])
        return logs
