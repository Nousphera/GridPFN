from datetime import datetime
from pathlib import Path

import matplotlib
import torch

import gridpfn.core.evaluate as evaluate_module
import gridpfn.core.schedule as schedule_module
import gridpfn.core.server as server_module
from gridpfn.core.client import Client
from gridpfn.core.dataset import PRICE_SCALE_FACTOR, load_data, setup_seed
from gridpfn.core.evaluate import CommunicationCostEvaluator, run_evaluation
from gridpfn.core.model import configure_embedding_device
from gridpfn.core.server import Server
from gridpfn.core.training_config import MODEL_ORDER, apply_refit_mode
from gridpfn.core.training_metrics import (
    MetricsLogger,
    PeriodicEvaluator,
    refit_selection_contract,
    save_refit_checkpoint,
)
from gridpfn.core.utils.plots import (
    plot_comm_time,
    plot_model_precision,
    plot_radar_chart,
    plot_training_metric,
)

matplotlib.use("Agg")


def configure_runtime(args, logs_root):
    # Synchronize module-level output and evaluator device settings with client allocation.
    server_module.logs_root = str(logs_root)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(2, args.cpu_threads))

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device_ids = [args.gpu]
        devices = [torch.device(f"cuda:{index}") for index in device_ids]
        for index in device_ids:
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, index)
        names = ", ".join(f"{device}={torch.cuda.get_device_name(device)}" for device in devices)
        print(f"CUDA devices: {names}")
    else:
        print("CUDA unavailable; using CPU")
    args.embedding_device = str(devices[0])
    if args.embedding_cache:
        args.embedding_cache = str(Path(args.embedding_cache).expanduser().resolve())
    if args.encoder_context:
        args.encoder_context = str(Path(args.encoder_context).expanduser().resolve())
    configure_embedding_device(devices[0], args.embedding_cache, args.encoder_context)
    if args.head_device == "cpu":
        devices = [torch.device("cpu")]
    print(f"Feature extraction={args.embedding_device}; head training={devices[0]}")
    evaluate_module.device = devices[0]
    schedule_module.DEVICE = devices[0]
    allocation = "round-robin" if len(devices) > 1 else devices[0]
    print(
        f"CPU threads={args.cpu_threads}; GPU memory fraction={args.gpu_memory_fraction:.2f}; "
        f"client allocation={allocation}\n"
    )
    return devices


def save_train_settings(path, sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("DATE: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
        for title, settings in sections.items():
            handle.write(f"\n--- {title} ---\n")
            handle.writelines(f"{key} = {value!r}\n" for key, value in settings.items())


def main(args, federated_models):
    apply_refit_mode(args)
    refit_contract = None
    if args.refit:
        if args.run_models != ["fedavg"] or args.actor_update != "ppo":
            raise ValueError("Monthly fixed-budget refit requires the FedAvg PPO controller")
        refit_contract = refit_selection_contract(
            args.refit_selection, args.episode, args.data_period, args.home_ids
        )
    if args.grid_prices is not None and args.tou:
        raise ValueError("Explicit grid prices require --no-tou")
    if args.encoder_context and (args.feature_mode != "hybrid" or args.synthetic_data):
        raise ValueError("Labelled encoder context requires real-only hybrid inputs")
    if args.ppo_oracle_init and (args.actor_update != "ppo" or not args.bc_rounds):
        raise ValueError("Oracle actor initialization requires PPO and positive BC rounds")
    if args.synthetic_data and (
        args.actor_update != "ppo" or not args.ppo_shuffle_days or args.predictive_features
    ):
        raise ValueError("Synthetic scenarios require mixed-day PPO without precomputed forecasts")
    if args.ppo_value_warmup_days < 0 or args.ppo_value_warmup_steps < 1:
        raise ValueError("Invalid value warmup days or steps")
    if args.ppo_value_warmup_days and args.actor_update != "ppo":
        raise ValueError("Value warmup requires PPO")
    if args.predictive_features and (args.actor_update != "ppo" or not args.p2p):
        raise ValueError("Predictive context currently requires the synchronized PPO/P2P pipeline")
    if args.predictive_features and not args.skip_final_evaluation:
        raise ValueError(
            "Predictive PPO heads use evaluate_checkpoint.py; enable --skip_final_evaluation"
        )
    if args.actor_update == "ppo" and (
        args.state_dim != 17
        or args.feature_mode not in ("raw", "hybrid")
        or not args.feasible_actions
        or not args.aggregate
        or not args.batched_updates
        or not args.local_critics
        or args.warmup_rounds
        or args.twin_critic
        or args.physics_critic
        or args.oracle_demonstrations
        or args.return_mode != "td"
        or args.policy_delay != 1
        or args.target_noise
        or args.residual_scale is not None
        or args.episode % args.aggregate
        or (args.eval_step and args.eval_step % args.aggregate)
    ):
        raise ValueError(
            "PPO requires local state-value critics, batched rollouts, zero warmup, no twin/physics Q or expert replay; episode/evaluation intervals must align with aggregation"
        )
    if (
        args.actor_update == "ppo"
        and args.ppo_rollout_days
        and (args.ppo_rollout_days < 1 or args.aggregate % args.ppo_rollout_days)
    ):
        raise ValueError("PPO rollout days must divide the aggregation interval")
    if args.ppo_clip_ac_likelihood and args.ac_service != "energy_quota":
        raise ValueError("Clipped AC likelihood currently requires original energy quotas")
    if args.batched_updates and (
        set(args.run_models) != {"fedavg"} or not args.p2p or args.warmup_learning
    ):
        raise ValueError("Batched updates require FedAvg, P2P and --no-warmup_learning")
    if args.eval_step < 0 or args.eval_days < 0:
        raise ValueError("--eval_step and --eval_days must be nonnegative")
    if args.eval_step and set(args.run_models) != {"fedavg"}:
        raise ValueError("Periodic evaluation currently supports --run_models fedavg only")
    if args.feature_mode != "frozen" and set(args.run_models) != {"fedavg"}:
        raise ValueError("Alternative representations currently support FedAvg only")
    if args.scaler_mode != "local" and set(args.run_models) != {"fedavg"}:
        raise ValueError("Shared observation scaling currently supports FedAvg only")
    if (
        args.bc_rounds
        or args.bc_weight
        or args.policy_delay != 1
        or args.actor_q_weight != 1
        or args.twin_critic
        or args.local_critics
        or args.critic_huber_delta
        or args.temperature_actor
        or args.feedback_target != 20
        or args.feasible_actions
        or args.head_width != 256
        or args.target_noise
        or args.exploration_noise is not None
        or args.residual_scale is not None
        or args.return_mode != "td"
        or args.oracle_demonstrations
        or args.actor_update != "q_gradient"
        or args.quota_actor
        or args.quota_guidance
        or args.storage_guidance
        or args.target_temperature_bounds is not None
        or args.embedding_weight != 1
        or args.physics_critic
    ) and set(args.run_models) != {"fedavg"}:
        raise ValueError("Guided/delayed policy updates currently support FedAvg only")
    has_validation = not args.refit and (args.data_period != "legacy" or bool(args.validation_days))
    if args.patience and (not has_validation or not args.eval_step):
        raise ValueError("Early stopping requires periodic validation")
    setup_seed(args.fixed_seed)
    fedfit_t_end = args.episode if args.fedfit_t_end is None else args.fedfit_t_end
    logs_root = Path(args.path_train).expanduser().resolve() / "logs"
    results_root = logs_root.parent / "results"
    schedules_root = logs_root.parent / "schedules"
    results_root.mkdir(parents=True, exist_ok=True)
    logs_dir, results_dir = str(logs_root), str(results_root)

    devices = configure_runtime(args, logs_root)
    aggregate = args.aggregate or max(1, args.episode // 50)
    home_ids = list(args.home_ids)
    run_models = list(dict.fromkeys(args.run_models))
    eval_models = list(dict.fromkeys(args.eval_models))
    eval_federated = [model for model in eval_models if model in federated_models]
    num_homes = len(home_ids)
    metrics_logger = MetricsLogger(logs_root.parent / "metrics.jsonl", home_ids)

    datas = load_data(
        args.path_data,
        "*.csv",
        choose=[f"home_{home_id}" for home_id in home_ids],
        validation_days=args.validation_days,
        data_period=args.data_period,
        split="validation" if has_validation or args.refit else "test",
        scaler_mode=args.scaler_mode,
        grid_prices=args.grid_prices,
    )

    agent_keys = (
        "gamma",
        "batch_size",
        "lr_actor",
        "lr_critic",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay",
        "critic_tau",
        "actor_tau",
        "memory_capacity",
        "bc_rounds",
        "bc_steps",
        "bc_lr",
        "demonstration_cache",
        "oracle_demonstrations",
        "expert_fraction",
        "expert_objective",
        "expert_critic_steps",
        "bc_weight",
        "guidance_weights",
        "actor_q_weight",
        "actor_update",
        "ppo_epochs",
        "ppo_value_epochs",
        "ppo_clip",
        "ppo_gae_lambda",
        "ppo_entropy",
        "ppo_clip_ac_likelihood",
        "ppo_rollout_days",
        "ppo_lr_decay_days",
        "ppo_bc_decay_days",
        "ppo_federation",
        "ppo_team_reward",
        "ppo_advantage_scale",
        "ppo_shared_gradient_clip",
        "ppo_std",
        "ppo_ac_std",
        "thermal_conditioning",
        "synthetic_fraction",
        "synthetic_until",
        "ppo_target_kl",
        "ppo_shuffle_days",
        "ppo_compile_mapping",
        "ppo_mask_inactive",
        "ppo_reset_momentum",
        "value_width",
        "value_normalization",
        "quota_correction",
        "expectile",
        "physics_critic",
        "embedding_weight",
        "advantage_temperature",
        "policy_delay",
        "twin_critic",
        "target_noise",
        "exploration_noise",
        "residual_scale",
        "return_mode",
        "critic_huber_delta",
        "temperature_actor",
        "target_temperature_bounds",
        "quota_actor",
        "quota_guidance",
        "storage_guidance",
        "feedback_target",
        "feasible_actions",
        "head_width",
        "actor_depth",
        "actor_activation",
        "ppo_oracle_init",
    )
    agent_params = {key: getattr(args, key) for key in agent_keys}
    agent_params["training_seed"] = args.fixed_seed
    feddmpq_params = {
        "alpha_w": args.feddmpq_alpha_w,
        "beta_g": args.feddmpq_beta_g,
        "gamma_e": args.feddmpq_gamma_e,
        "K": args.feddmpq_k,
        "low_bit": args.feddmpq_low_bit,
    }
    fedfit_params = {"adjust_fraction": args.fedfit_adjust_fraction, "t_end": fedfit_t_end}
    pffdst_params = {
        "differential_ratio": args.pffdst_differential_ratio,
        "readjust_interval": args.pffdst_readjust_interval,
    }
    em_strategy = {
        "tou": {"enabled": args.tou, "n_blocks": args.tou_blocks},
        "ac_energy_quota": args.ac_service == "energy_quota",
        "dr_limit": args.dr_limit,
        "dr_penalty": args.dr_penalty,
        "dr_incentive": args.dr_incentive,
        "pv_curtail": args.pv_curtail,
        "export_price": args.export_price,
    }
    p2p_config = {"enabled": args.p2p, "price": args.p2p_price}
    setup_keys = (
        "episode",
        "warmup_rounds",
        "seed",
        "fixed_seed",
        "update",
        "state_dim",
        "continuous_action_dim",
        "discrete_action_dim",
        "actor_sparsity",
        "critic_sparsity",
        "epsilon",
        "ma_window",
    )
    experimental_setup = {key: getattr(args, key) for key in setup_keys} | {
        "aggregate": aggregate,
        "fedavg_aggregation_offset": 0,
        "validation_days": args.validation_days,
        "data_period": args.data_period,
        "refit": args.refit,
        "refit_selection": refit_contract,
        "feature_mode": args.feature_mode,
        "encoder_context": args.encoder_context,
        "predictive_features": args.predictive_features,
        "scaler_mode": args.scaler_mode,
        "grid_prices": args.grid_prices,
        "warmup_learning": args.warmup_learning,
    }
    experimental_setup["batched_updates"] = args.batched_updates
    experimental_setup["local_critics"] = args.local_critics
    sections = {
        "EXPERIMENTAL SETUP": experimental_setup,
        "MODEL PARAMETERS": {
            "agent_params": agent_params,
            "fedfit_params": fedfit_params,
            "pffdst_params": pffdst_params,
            "feddmpq_params": feddmpq_params,
        },
        "ENERGY MANAGEMENT STRATEGY": {
            "fixed_cost": args.fixed_cost,
            "use_p2p": args.p2p,
            "p2p_config": p2p_config,
            "price_scale_factor": PRICE_SCALE_FACTOR,
            "em_strategy": em_strategy,
        },
        "RUNTIME": {
            "path_data": args.path_data,
            "home_ids": home_ids,
            "devices": [str(device) for device in devices],
            "embedding_device": args.embedding_device,
            "embedding_cache": args.embedding_cache,
            "eval_step": args.eval_step,
            "eval_days": args.eval_days,
        },
    }

    for model_name in run_models:
        torch.manual_seed(args.seed)
        clients = [
            Client(
                data=data,
                state_dim=args.state_dim,
                continuous_action_dim=args.continuous_action_dim,
                discrete_action_dim=args.discrete_action_dim,
                epsilon=args.epsilon,
                episode=args.episode,
                agent_hyperparams=agent_params,
                fedfit_agent_hyperparams=fedfit_params,
                pffdst_agent_hyperparams=pffdst_params,
                feddmpq_agent_hyperparams=feddmpq_params,
                fixed_cost=args.fixed_cost,
                device=devices[index % len(devices)],
                active_model=model_name,
                feature_mode=args.feature_mode,
                warmup_learning=args.warmup_learning,
                predictive_features=Path(args.predictive_features) / f"home_{home_ids[index]}.npz"
                if args.predictive_features
                else None,
            )
            for index, data in enumerate(datas)
        ]
        for home_id, client in zip(home_ids, clients, strict=True):
            client.home_id = home_id
        if args.synthetic_data:
            from gridpfn.core.synthetic_days import attach_synthetic_days

            attach_synthetic_days(clients, args.synthetic_data)
        setup_seed(args.fixed_seed)
        server = Server(
            clients=clients,
            episode=args.episode,
            actor_sparsity=args.actor_sparsity,
            critic_sparsity=args.critic_sparsity,
            update=args.update,
            aggregate=aggregate,
            em_strategy=em_strategy,
            p2p_config=p2p_config,
            warmup_rounds=args.warmup_rounds,
            metrics_logger=metrics_logger,
            batched_updates=args.batched_updates,
            local_critics=args.local_critics,
            eval_callback=PeriodicEvaluator(
                metrics_logger,
                args.eval_step,
                args.eval_days,
                split="validation" if has_validation else "test",
                min_episodes=args.min_episodes,
                patience=args.patience,
                min_delta=args.min_delta,
                oracle_reference=args.oracle_reference,
                strict_convergence=args.strict_convergence,
                selection_reference=args.selection_reference,
            )
            if args.eval_step
            else None,
        )
        server.value_warmup = (args.ppo_value_warmup_days, args.ppo_value_warmup_steps)
        save_train_settings(logs_root / model_name / "train_settings.txt", sections)
        getattr(server, f"{model_name}_train")()
        if args.refit:
            save_refit_checkpoint(
                server, metrics_logger, args.episode, refit_contract, args.data_period
            )
        del server, clients
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("--- Training complete ---")

    if args.skip_final_evaluation:
        return

    for metric in ("reward", "actor_loss", "critic_loss"):
        filename = {
            "reward": "rewards_plots_all_homes.pdf",
            "actor_loss": "loss_plots_all_homes_actor.pdf",
            "critic_loss": "loss_plots_all_homes_critic.pdf",
        }[metric]
        plot_training_metric(
            metric,
            eval_models,
            num_homes=num_homes,
            window=args.ma_window,
            logs_base=logs_dir,
            save_path=str(results_root / filename),
        )

    if "feddmpq" in eval_models:
        plot_model_precision(
            num_homes,
            low_bit=args.feddmpq_low_bit,
            window=args.ma_window,
            logs_base=logs_dir,
            save_path=str(results_root / "model_precision_FedDMPQ.pdf"),
        )

    setup_seed(args.fixed_seed)
    results, evaluator = run_evaluation(
        model_names=eval_models,
        num_homes=num_homes,
        path_data=args.path_data,
        save_path=results_dir,
        fixed_cost=args.fixed_cost,
        p2p_config=p2p_config,
        fed_p2p_models=eval_federated,
        em_strategy=em_strategy,
        home_ids=home_ids,
        logs_root=logs_dir,
    )
    num_test_days = min(
        len(home_result["all"])
        for model_results in results.values()
        for home_result in model_results.values()
    )
    CommunicationCostEvaluator(results_dir=logs_dir).compute(
        eval_models,
        num_homes,
        args.episode,
        aggregate,
        test_times=evaluator.test_times,
        num_test_days=num_test_days,
        low_bit=args.feddmpq_low_bit,
        batch_size=args.batch_size,
        update=args.update,
        fedfit_t_end=fedfit_t_end,
    )
    plot_comm_time(
        eval_models,
        logs_base=logs_dir,
        num_homes=num_homes,
        save_path=str(results_root / "model_comm_train_time.pdf"),
    )
    plot_comm_time(
        eval_models,
        mode="test",
        logs_base=logs_dir,
        num_homes=num_homes,
        save_path=str(results_root / "model_comm_test_time.pdf"),
    )
    plot_radar_chart(results, eval_models, save_path=results_dir, logs_base=logs_dir)
    schedule_module.plot_schedules(
        day=args.day,
        models=eval_models,
        path_data=args.path_data,
        path_logs=logs_dir,
        path_output=str(schedules_root),
        home_id=args.home_id,
        home_ids=home_ids,
        disable_p2p=not args.p2p,
        fed_p2p_models=eval_federated,
    )


if __name__ == "__main__":
    from gridpfn.core.training_config import parse_args

    main(parse_args(), tuple(MODEL_ORDER))
