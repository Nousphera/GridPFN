import copy
import math

import numpy as np
import torch

from gridpfn.core.agents.agent import P_DQN
from gridpfn.core.agents.agent_feddmpq import P_DQN_FedDMPQ
from gridpfn.core.agents.agent_fedfit import P_DQN_FedFit
from gridpfn.core.agents.agent_pffdst import P_DQN_PFFDST
from gridpfn.core.agents.agent_sparse import P_DQN_sparse
from gridpfn.core.agents.onpolicy import OnPolicyAgent
from gridpfn.core.em_strategy import apply_em_strategy
from gridpfn.core.environment import HOME_ENERGY_MGNT
from gridpfn.core.model import Actor, Critic, precompute_embeddings
from gridpfn.core.predictive_features import PredictiveContext


# Client wrapper that manages sparse and dense agents for each household.
class Client:
    def __init__(
        self,
        data,
        state_dim,
        continuous_action_dim,
        discrete_action_dim,
        episode,
        epsilon,
        agent_hyperparams=None,
        fedavg_agent_hyperparams=None,
        sparse_agent_hyperparams=None,
        fedfit_agent_hyperparams=None,
        pffdst_agent_hyperparams=None,
        feddmpq_agent_hyperparams=None,
        fixed_cost=0.0,
        device=None,
        warmup_learning=True,
        active_model=None,
        feature_mode="frozen",
        predictive_features=None,
    ) -> None:

        self.warmup_learning = warmup_learning
        self.active_model, self.feature_mode = active_model, feature_mode
        if feature_mode != "frozen" and active_model != "fedavg":
            raise ValueError("Alternative representations currently require active_model=fedavg")
        self.round_max = episode
        self.epsilon = epsilon
        self.fixed_cost = fixed_cost
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_dim = state_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dim = discrete_action_dim
        self.train_data, self.test_data, self.test_dates, self.scaler = data
        self.predictive_context = (
            PredictiveContext(predictive_features, self.train_data, self.scaler)
            if predictive_features
            else None
        )
        auxiliary_dim = self.predictive_context.width if self.predictive_context else 0

        # Construction order preserves the original initialization RNG sequence.
        specifications = (
            ("fedavg", "fedavg_", P_DQN, "fedavg_agent_hyperparams", fedavg_agent_hyperparams),
            ("edgehem", "", P_DQN_sparse, "sparse_agent_hyperparams", sparse_agent_hyperparams),
            (
                "feddmpq",
                "feddmpq_",
                P_DQN_FedDMPQ,
                "feddmpq_agent_hyperparams",
                feddmpq_agent_hyperparams,
            ),
            (
                "fedfit",
                "fedfit_",
                P_DQN_FedFit,
                "fedfit_agent_hyperparams",
                fedfit_agent_hyperparams,
            ),
            (
                "pffdst",
                "pffdst_",
                P_DQN_PFFDST,
                "pffdst_agent_hyperparams",
                pffdst_agent_hyperparams,
            ),
        )
        if active_model is not None and active_model not in {item[0] for item in specifications}:
            raise ValueError(f"Unknown active model: {active_model}")
        for name, prefix, agent_type, attribute, overrides in specifications:
            hp = copy.deepcopy(agent_hyperparams or {})
            hp.update(copy.deepcopy(overrides or {}))
            setattr(self, attribute, hp)
            if active_model is not None and active_model != name:
                continue
            actor_options, critic_options = {}, {}
            if name == "fedavg":
                onpolicy = hp.get("actor_update") == "ppo"
                if onpolicy:
                    agent_type = OnPolicyAgent
                if hp.get("physics_critic") and not hp.get("feasible_actions"):
                    raise ValueError("Physics critic requires feasible executed AC controls")
                actor_options = dict(
                    feature_mode=feature_mode,
                    thermal_bounds=(
                        self.scaler["min"][self.scaler["col_to_scaler_idx"][4]],
                        self.scaler["max"][self.scaler["col_to_scaler_idx"][4]],
                    )
                    if hp.get("temperature_actor")
                    else None,
                    feasible_dt=self.scaler.get("delta_t", 1)
                    if hp.get("feasible_actions")
                    else None,
                    target_temperature_bounds=hp.get("target_temperature_bounds")
                    if hp.get("temperature_actor")
                    else None,
                    quota_actor=hp.get("quota_actor", False),
                    hidden_dim=int(hp.get("head_width", 256)),
                    learned_discrete=hp.get("actor_update") in ("implicit", "ppo"),
                    stochastic_std=hp.get("ppo_std", 0.1) if onpolicy else None,
                    stochastic_ac_std=hp.get("ppo_ac_std") if onpolicy else None,
                    thermal_conditioning=hp.get("thermal_conditioning", False),
                    embedding_weight=hp.get("embedding_weight", 1.0),
                    quota_correction=hp.get("quota_correction", 2.0),
                    auxiliary_dim=auxiliary_dim,
                    hidden_layers=int(hp.get("actor_depth", 1)),
                    activation=hp.get("actor_activation", "relu"),
                )
                critic_options = dict(
                    feature_mode=feature_mode,
                    twin=bool(hp.get("twin_critic", False)),
                    hidden_dim=int(hp.get("head_width", 256)),
                    value_head=hp.get("actor_update") in ("implicit", "ppo"),
                    value_hidden_dim=hp.get("value_width"),
                    value_normalization=hp.get("value_normalization", False),
                    auxiliary_dim=auxiliary_dim,
                    embedding_weight=hp.get("embedding_weight", 1.0),
                    thermal_bounds=(
                        self.scaler["min"][self.scaler["col_to_scaler_idx"][4]],
                        self.scaler["max"][self.scaler["col_to_scaler_idx"][4]],
                    )
                    if hp.get("physics_critic")
                    else None,
                )
            actor = Actor(self.state_dim, self.continuous_action_dim, **actor_options).to(
                self.device
            )
            critic = Critic(
                self.state_dim,
                self.continuous_action_dim,
                self.discrete_action_dim,
                **critic_options,
            ).to(self.device)
            setattr(self, prefix + "actor_net", actor)
            setattr(self, prefix + "critic_net", critic)
            setattr(
                self,
                prefix + "agent",
                agent_type(
                    actor_net=actor,
                    critic_net=critic,
                    state_dim=self.state_dim,
                    continuous_action_dim=self.continuous_action_dim,
                    discrete_action_dim=self.discrete_action_dim,
                    hyperparams=hp,
                ),
            )

        # Current energy-management strategy (set by server)
        self.em_strategy = {}

    # Compute and return initial masks for actor and critic given target sparsities.
    def initialize(self, actor_sparsity, critic_sparsity):
        actor_params, critic_params = self.agent.get_trainable_params()
        a_sp = self.agent.calculate_sparsities(actor_params, sparse=actor_sparsity)
        c_sp = self.agent.calculate_sparsities(critic_params, sparse=critic_sparsity)
        a_mask = self.agent.init_masks(actor_params, a_sp)
        c_mask = self.agent.init_masks(critic_params, c_sp)

        return a_mask, c_mask

    # Receive and store an energy-management strategy dict from the server
    def set_em_strategy(self, em_strategy: dict):
        if em_strategy is None:
            self.em_strategy = {}
        else:
            self.em_strategy = copy.deepcopy(em_strategy)

        # Only the first eight exogenous inputs enter the frozen feature cache.
        states = []
        for day in self.train_data[: getattr(self, "real_train_count", len(self.train_data))]:
            env = HOME_ENERGY_MGNT(
                day, scaler=self.scaler, fixed_cost=self.fixed_cost, state_dim=self.state_dim
            )
            apply_em_strategy(env, self.em_strategy)
            states.extend(env._state_for_step(step) for step in range(env.max_step + 1))
        self.training_states = np.asarray(states)
        added = (
            precompute_embeddings(self.training_states, self.device)
            if self.feature_mode != "raw"
            else 0
        )
        print(
            f"[embeddings] Prepared {len(states)} training state rows; {added} new embeddings",
            flush=True,
        )
        if hasattr(self, "real_train_count") and self.feature_mode != "raw":
            synthetic_states = []
            for day in self.train_data[self.real_train_count :]:
                env = HOME_ENERGY_MGNT(day, scaler=self.scaler, state_dim=self.state_dim)
                apply_em_strategy(env, self.em_strategy)
                synthetic_states.extend(env._state_for_step(t) for t in range(env.max_step + 1))
            precompute_embeddings(np.asarray(synthetic_states), self.device)

    # Shared episode loop used by train and fedavg_train.
    def _run_episode(self, agent, index):
        dataset = self.train_data[index]
        env = HOME_ENERGY_MGNT(
            dataset,
            scaler=self.scaler,
            fixed_cost=self.fixed_cost,
            state_dim=getattr(self, "state_dim", 9),
        )
        if self.em_strategy:
            apply_em_strategy(env, self.em_strategy)
        state = env.reset()
        epo_reward = 0
        actor_losses = []
        critic_losses = []
        while True:
            a = agent.choose_action(state)
            s_, _, r, _, done = env.step(a)
            # Q represents the requested action; environment constraints are part
            # of its transition, just as for the actor and bootstrap targets.
            agent.store_transition(
                state,
                agent.replay_action(env, a) if hasattr(agent, "replay_action") else a,
                r,
                s_,
                done,
            )
            state = s_
            result = agent.learn()
            loss_actor, loss_critic = result if result is not None else (None, None)
            if loss_actor is not None:
                actor_losses.append(loss_actor)
            if loss_critic is not None:
                critic_losses.append(loss_critic)
            epo_reward += r
            if done:
                break
        del env
        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else None
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else None
        return epo_reward, avg_actor_loss, avg_critic_loss

    def warmup_episode(self, agent, index):
        dataset = self.train_data[index]
        env = HOME_ENERGY_MGNT(
            dataset,
            scaler=self.scaler,
            fixed_cost=self.fixed_cost,
            state_dim=getattr(self, "state_dim", 9),
        )
        if self.em_strategy:
            apply_em_strategy(env, self.em_strategy)
        state = env.reset()
        while True:
            a = agent.choose_action(state)
            s_, _, r, _, done = env.step(a)
            agent.store_transition(
                state,
                agent.replay_action(env, a) if hasattr(agent, "replay_action") else a,
                r,
                s_,
                done,
            )
            if getattr(self, "warmup_learning", True):
                agent.learn()
            state = s_
            if done:
                break
        del env

    # Run one training episode using the sparse agent on train_data[index].
    def train(self, index):
        return self._run_episode(self.agent, index)

    # Retrieve sparse agent actor and critic model parameters.
    def get_model_params(self):
        return self.agent.get_model_params()

    # Load actor and critic parameters into the sparse agent.
    def set_model_params(self, actor_params, critic_params):
        self.agent.set_model_params(actor_params, critic_params)

    # Get current sparsity masks from the sparse agent.
    def get_model_masks(self):
        return self.agent.get_model_masks()

    # Set sparsity masks on the sparse agent.
    def set_model_masks(self, actor_masks, critic_masks):
        self.agent.set_model_masks(actor_masks, critic_masks)

    # Prune (fire) a fraction of active weights based on smallest magnitude (cosine-annealed drop ratio).
    def fire_mask(self, weights, masks, round):
        drop_ratio = self.epsilon / 2 * (1 + np.cos((round * np.pi) / self.round_max))
        new_masks = {name: mask.clone() for name, mask in masks.items()}

        num_remove = {}
        for name, mask in masks.items():
            active_indices = mask.flatten().bool().nonzero(as_tuple=False).squeeze(1)
            remove_count = min(
                math.ceil(drop_ratio * active_indices.numel()), active_indices.numel()
            )
            num_remove[name] = remove_count
            if remove_count == 0:
                continue
            active_weights = weights[name].detach().flatten()[active_indices].abs()
            selected = torch.topk(
                active_weights, k=remove_count, largest=False, sorted=False
            ).indices
            new_masks[name].flatten()[active_indices[selected]] = 0
        return new_masks, num_remove

    # Regrow previously pruned connections at positions with largest absolute gradient magnitude.
    def regrow_mask(self, masks, num_remove, gradient):
        new_masks = {name: mask.clone() for name, mask in masks.items()}
        for name, mask in masks.items():
            dead_indices = (~mask.flatten().bool()).nonzero(as_tuple=False).squeeze(1)
            regrow_count = min(num_remove[name], dead_indices.numel())
            if regrow_count == 0:
                continue
            dead_gradients = gradient[name].detach().flatten()[dead_indices].abs()
            selected = torch.topk(
                dead_gradients, k=regrow_count, largest=True, sorted=False
            ).indices
            new_masks[name].flatten()[dead_indices[selected]] = 1
        return new_masks

    def _dynamic_update_for_agent(self, agent, round):
        actor_weights, critic_weights = agent.get_trainable_params()
        actor_masks, critic_masks = agent.get_model_masks(device=agent.device)
        actor_gradient, critic_gradient = agent.screen_gradients()
        actor_new_masks, actor_num_remove = self.fire_mask(actor_weights, actor_masks, round)
        actor_new_masks = self.regrow_mask(actor_new_masks, actor_num_remove, actor_gradient)
        critic_new_masks, critic_num_remove = self.fire_mask(critic_weights, critic_masks, round)
        critic_new_masks = self.regrow_mask(critic_new_masks, critic_num_remove, critic_gradient)
        agent.set_model_masks(actor_new_masks, critic_new_masks)

    # Perform a single dynamic-sparsity update: prune then regrow for actor and critic.
    def dynamic_update(self, round):
        self._dynamic_update_for_agent(self.agent, round)

    # Run one training episode using the federated (dense) agent on train_data[index].
    def fedavg_train(self, index):
        return self._run_episode(self.fedavg_agent, index)

    # Get actor and critic parameters from the federated agent.
    def get_fedavg_model_params(self):
        return self.fedavg_agent.get_model_params()

    # Set actor and critic parameters on the federated agent.
    def set_fedavg_model_params(self, actor_params, critic_params):
        self.fedavg_agent.set_model_params(actor_params, critic_params)

    # ----------------------------------------------------------------
    # FedDMPQ API
    # ----------------------------------------------------------------

    # Compute and return initial masks for FedDMPQ actor and critic.
    def feddmpq_initialize(self, actor_sparsity, critic_sparsity):
        actor_params, critic_params = self.feddmpq_agent.get_trainable_params()
        a_sp = self.feddmpq_agent.calculate_sparsities(actor_params, sparse=actor_sparsity)
        c_sp = self.feddmpq_agent.calculate_sparsities(critic_params, sparse=critic_sparsity)
        a_mask = self.feddmpq_agent.init_masks(actor_params, a_sp)
        c_mask = self.feddmpq_agent.init_masks(critic_params, c_sp)
        self.feddmpq_agent.set_model_masks(a_mask, c_mask)
        self.feddmpq_agent.init_soft_prune_state()
        return a_mask, c_mask

    # Run one training episode with FedDMPQ.
    def feddmpq_train(self, index):
        return self._run_episode(self.feddmpq_agent, index)

    # Load actor and critic parameters into the FedDMPQ agent.
    def set_feddmpq_model_params(self, actor_params, critic_params):
        self.feddmpq_agent.set_model_params(actor_params, critic_params)

    # Retrieve mixed-precision parameters for federated upload.
    def get_feddmpq_mixed_precision_params(self):
        return self.feddmpq_agent.get_mixed_precision_params()

    # Get current FedDMPQ sparsity masks.
    def get_feddmpq_model_masks(self):
        return self.feddmpq_agent.get_model_masks()

    # Set FedDMPQ sparsity masks.
    def set_feddmpq_model_masks(self, actor_masks, critic_masks):
        self.feddmpq_agent.set_model_masks(actor_masks, critic_masks)

    # Perform one FedDMPQ soft-prune update (replaces fire/regrow).
    def dynamic_update_feddmpq(self, actor_sparsity, critic_sparsity):
        self.feddmpq_agent.soft_prune_step(actor_sparsity, critic_sparsity)

    def fedfit_initialize(self, actor_sparsity, critic_sparsity):
        actor_params, critic_params = self.fedfit_agent.get_trainable_params()
        actor_mask = self.fedfit_agent.init_masks(
            actor_params,
            self.fedfit_agent.calculate_sparsities(
                actor_params,
                tabu={name for name, param in actor_params.items() if param.ndim < 2},
                sparse=actor_sparsity,
            ),
        )
        critic_mask = self.fedfit_agent.init_masks(
            critic_params,
            self.fedfit_agent.calculate_sparsities(
                critic_params,
                tabu={name for name, param in critic_params.items() if param.ndim < 2},
                sparse=critic_sparsity,
            ),
        )
        self.fedfit_agent.set_model_masks(actor_mask, critic_mask)
        return actor_mask, critic_mask

    def fedfit_train(self, index):
        return self._run_episode(self.fedfit_agent, index)

    def get_fedfit_model_params(self):
        return self.fedfit_agent.get_model_params()

    def set_fedfit_model_params(self, actor_params, critic_params):
        self.fedfit_agent.set_model_params(actor_params, critic_params)

    def get_fedfit_model_masks(self):
        return self.fedfit_agent.get_model_masks()

    def set_fedfit_model_masks(self, actor_masks, critic_masks):
        self.fedfit_agent.set_model_masks(actor_masks, critic_masks)

    def dynamic_update_fedfit(self, round_idx):
        return self.fedfit_agent.adjust_topology(round_idx)

    def pffdst_initialize(self, actor_sparsity, critic_sparsity, masks=None):
        return self.pffdst_agent.initialize(actor_sparsity, critic_sparsity, masks)

    def pffdst_train(self, index):
        return self._run_episode(self.pffdst_agent, index)

    def get_pffdst_model_params(self):
        return self.pffdst_agent.get_model_params()

    def set_pffdst_model_params(self, actor_params, critic_params):
        self.pffdst_agent.set_model_params(actor_params, critic_params)

    def get_pffdst_model_masks(self):
        return self.pffdst_agent.get_model_masks()

    def get_pffdst_communication_masks(self):
        return self.pffdst_agent.get_communication_masks()

    def set_pffdst_model_masks(self, actor_masks, critic_masks, survivor_masks=None):
        self.pffdst_agent.set_model_masks(actor_masks, critic_masks, survivor_masks)
