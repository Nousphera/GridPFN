"""Training options and the validated default preset; CLI overrides always win."""

import argparse

from gridpfn.core.dataset import DATA_PERIODS, home_data_dir
from gridpfn.core.environment import CONTINUOUS_ACTION_DIM, DISCRETE_ACTION_DIM, STATE_DIM

MODEL_ORDER = ("fedavg", "edgehem", "fedfit", "pffdst", "feddmpq")

HOME_IDS = (27, 950, 1222, 3000, 3488, 3517, 5587, 5679, 5997, 9053)
OPTIMIZED_DEFAULTS = dict(
    path_train="results/final",
    run_models=["fedavg"],
    eval_models=["fedavg"],
    state_dim=17,
    validation_days=14,
    skip_final_evaluation=True,
    warmup_learning=False,
    memory_capacity=2048,
    bc_rounds=20,
    bc_steps=50,
    bc_weight=1.0,
    actor_q_weight=0.1,
    exploration_noise=0.05,
    epsilon_start=0.2,
    batched_updates=True,
    episode=6000,
    eval_step=50,
    warmup_rounds=3,
    aggregate=5,
    actor_sparsity=0.25,
    critic_sparsity=0.25,
    gpu=1,
    cpu_threads=3,
    gpu_memory_fraction=0.2,
    ac_service="energy_quota",
    min_episodes=3000,
    patience=20,
    min_delta=0.01,
)
THERMAL_DEFAULTS = {
    **OPTIMIZED_DEFAULTS,
    "path_train": "results/thermal_final",
    "feature_mode": "hybrid",
    "temperature_actor": True,
    "feedback_target": 18.2,
    "local_critics": True,
    "critic_huber_delta": 1.0,
    "twin_critic": True,
    "policy_delay": 2,
    "target_noise": 0.03,
    "head_device": "cpu",
    "cpu_threads": 6,
}
PPO_DEFAULTS = {
    **OPTIMIZED_DEFAULTS,
    "path_train": "results/federated_ppo",
    "feature_mode": "hybrid",
    "embedding_weight": 0.1,
    "head_width": 64,
    "head_device": "cpu",
    "cpu_threads": 1,
    "scaler_mode": "shared",
    "temperature_actor": True,
    "target_temperature_bounds": (-5, 21.8),
    "quota_actor": True,
    "quota_guidance": True,
    "storage_guidance": True,
    "feasible_actions": True,
    "local_critics": True,
    "actor_update": "ppo",
    "gamma": 1.0,
    "warmup_rounds": 0,
    "lr_actor": 0.0003,
    "lr_critic": 0.001,
    "bc_weight": 0.1,
    "guidance_weights": (1, 0, 0),
    "ppo_gae_lambda": 1.0,
    "episode": 8000,
    "strict_convergence": True,
}
PPO_REFERENCE_DEFAULTS = dict(PPO_DEFAULTS)
PPO_DEFAULTS.update(
    bc_rounds=60,
    bc_weight=0.0,
    ppo_shuffle_days=True,
    ppo_compile_mapping=True,
    embedding_cache="results/frozen_features",
    quota_correction=6.0,
    target_temperature_bounds=(-5, 26),
    ppo_ac_std=1 / 30,
)
PRESETS = ("optimized", "thermal", "ppo", "ppo_reference", "legacy")


def build_parser(preset="optimized"):
    ALL_MODELS = tuple(MODEL_ORDER)

    parser = argparse.ArgumentParser(
        description="Train, evaluate, and plot a complete Fed-HEMS experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    parser.add_argument("--path_train", default="train_test", help="Training output directory.")
    parser.add_argument("--path_data", default=str(home_data_dir), help="Home CSV directory.")
    parser.add_argument(
        "--home_ids", nargs="+", type=int, default=list(HOME_IDS), help="Dataset home IDs."
    )
    parser.add_argument(
        "--run_models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
        help="Models to train.",
    )
    parser.add_argument(
        "--eval_models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--feature_mode", choices=("frozen", "normalized", "hybrid", "raw"), default="frozen"
    )
    parser.add_argument(
        "--embedding_cache",
        default=None,
        help="Optional persistent frozen features, keyed by model, software and exact inputs.",
    )
    parser.add_argument(
        "--encoder_context",
        default=None,
        help="Optional immutable labelled training context for the frozen TabPFN encoder.",
    )
    parser.add_argument(
        "--scaler_mode",
        choices=("local", "shared"),
        default="local",
        help="Optional common observation units using training extrema across the selected homes.",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="Experiment runner: complete a development run without accessing the test period.",
    )
    parser.add_argument(
        "--data_period",
        choices=DATA_PERIODS,
        default="legacy",
        help="Legacy/full chronological periods, or monthly selection/refit folds from June to October.",
    )
    parser.add_argument(
        "--refit_selection",
        default=None,
        help="Monthly refit: prior selection.json whose best episode fixes this run's budget.",
    )
    parser.add_argument(
        "--validation_days",
        type=int,
        default=0,
        help="Legacy period only: hold out the last N July days; full period always validates on August.",
    )
    parser.add_argument(
        "--skip_final_evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For validation-only studies: do not evaluate the test period or generate final reports.",
    )
    parser.add_argument(
        "--warmup_learning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable to fill replay during warmup without optimizing networks.",
    )
    parser.add_argument(
        "--memory_capacity", type=int, default=1000, help="Replay transitions per home."
    )
    parser.add_argument(
        "--bc_rounds",
        type=int,
        default=0,
        help="Federated actor-imitation rounds before RL (17-state FedAvg).",
    )
    parser.add_argument(
        "--bc_steps", type=int, default=50, help="Local imitation updates per pretraining round."
    )
    parser.add_argument(
        "--bc_lr", type=float, default=0.001, help="Imitation-only pretraining learning rate."
    )
    parser.add_argument(
        "--demonstration_cache",
        default=None,
        help="Optional directory for source/data-hashed training labels shared across seeds.",
    )
    parser.add_argument(
        "--oracle_demonstrations",
        default=None,
        help="Certified oracle run with split=train, used only for local expert initialization/replay.",
    )
    parser.add_argument(
        "--expert_fraction",
        type=float,
        default=0.25,
        help="Training batch fraction of oracle transitions.",
    )
    parser.add_argument(
        "--expert_objective",
        choices=("paper_reward", "comfort_first"),
        default="paper_reward",
        help="Which certified training schedules provide demonstrations; RL still uses the original reward.",
    )
    parser.add_argument(
        "--expert_critic_steps",
        type=int,
        default=1000,
        help="Local critic initialization updates on expert returns.",
    )
    parser.add_argument(
        "--bc_weight", type=float, default=0, help="Online feedback-imitation actor loss weight."
    )
    parser.add_argument(
        "--guidance_weights",
        nargs=3,
        type=float,
        default=(1, 1, 1),
        help="Online AC/EV/BESS guidance weights; pretraining still initializes all actuators.",
    )
    parser.add_argument(
        "--actor_q_weight",
        type=float,
        default=1,
        help="Q-gradient actor weight; zero gives an imitation control. Inactive for implicit/PPO.",
    )
    parser.add_argument(
        "--actor_update",
        choices=("q_gradient", "implicit", "ppo"),
        default="q_gradient",
        help="Direct Q gradient, IQL-style regression or on-policy PPO (FedAvg).",
    )
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument(
        "--ppo_value_epochs",
        type=int,
        default=None,
        help="Independent value updates per fresh rollout; defaults to actor epochs.",
    )
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument(
        "--ppo_value_warmup_days",
        type=int,
        default=0,
        help="Optional private-critic pretraining on fresh real-day returns; actor stays fixed.",
    )
    parser.add_argument("--ppo_value_warmup_steps", type=int, default=256)
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_entropy", type=float, default=0.001)
    parser.add_argument(
        "--ppo_clip_ac_likelihood",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Integrate latent AC tails collapsed by target, power and quota clipping.",
    )
    parser.add_argument(
        "--ppo_rollout_days",
        type=int,
        default=0,
        help="Fresh days per local PPO update; 0 uses --aggregate. Must divide aggregation.",
    )
    parser.add_argument(
        "--ppo_lr_decay_days",
        type=int,
        default=0,
        help="Linearly reduce actor LR to 10%% over this many days; 0 keeps it fixed.",
    )
    parser.add_argument(
        "--ppo_bc_decay_days",
        type=int,
        default=0,
        help="Linearly remove online feedback guidance over this many days; 0 keeps it fixed.",
    )
    parser.add_argument(
        "--ppo_federation",
        choices=("actor", "trunk", "gradient", "none"),
        default="actor",
        help="Average actor weights, share a trunk, average gradients per PPO epoch, or local-only diagnostic.",
    )
    parser.add_argument(
        "--ppo_team_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use mean settled home reward for cooperative PPO targets; evaluation stays per home.",
    )
    parser.add_argument(
        "--ppo_advantage_scale",
        choices=("home", "cohort"),
        default="home",
        help="Normalize advantages per home or preserve relative reward scales across the cohort.",
    )
    parser.add_argument(
        "--ppo_shared_gradient_clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For gradient federation, clip after averaging rather than per home.",
    )
    parser.add_argument("--ppo_std", type=float, default=0.1)
    parser.add_argument(
        "--ppo_ac_std",
        type=float,
        default=None,
        help="Initial cooling latent std; None uses the common --ppo_std.",
    )
    parser.add_argument(
        "--thermal_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale cooling mean and std together to preserve sensitivity with a wider map.",
    )
    parser.add_argument(
        "--synthetic_data",
        type=str,
        default=None,
        help="Optional synchronized training-only archive built by synthetic_days.py.",
    )
    parser.add_argument(
        "--synthetic_fraction",
        type=float,
        default=0.5,
        help="Probability of a synthetic joint day; real tariffs and preprocessing stay fixed.",
    )
    parser.add_argument(
        "--synthetic_until",
        type=int,
        default=0,
        help="Use synthetic days only before this episode; zero keeps mixing throughout training.",
    )
    parser.add_argument("--ppo_target_kl", type=float, default=0.0)
    parser.add_argument(
        "--ppo_shuffle_days",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mix shared complete training dates across weather regimes.",
    )
    parser.add_argument(
        "--ppo_compile_mapping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile cached rollout actuator math; amortize startup on long runs.",
    )
    parser.add_argument("--ppo_mask_inactive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ppo_reset_momentum", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--value_width", type=int, default=None)
    parser.add_argument(
        "--value_normalization", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--quota_correction", type=float, default=2.0)
    parser.add_argument("--predictive_features", type=str, default=None)
    parser.add_argument(
        "--strict_convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require stable validation windows, not just stale best-checkpoint scores.",
    )
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument(
        "--physics_critic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use exact original immediate thermal reward plus a learned residual Q.",
    )
    parser.add_argument(
        "--quota_actor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn a bounded thermal correction around the causal quota-aware controller.",
    )
    parser.add_argument(
        "--quota_guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use current-state remaining AC quota in local feedback guidance.",
    )
    parser.add_argument(
        "--storage_guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize battery control with causal current-PV self-consumption guidance.",
    )
    parser.add_argument(
        "--embedding_weight",
        type=float,
        default=1.0,
        help="Scale normalized TabPFN coordinates relative to the raw-state skip (hybrid mode).",
    )
    parser.add_argument("--advantage_temperature", type=float, default=3.0)
    parser.add_argument(
        "--oracle_reference",
        default=None,
        help="Matched validation oracle directory for monitoring regret; never used as training labels.",
    )
    parser.add_argument(
        "--policy_delay",
        type=int,
        default=1,
        help="Critic updates per actor/target update; 2 is a TD3-inspired ablation.",
    )
    parser.add_argument(
        "--twin_critic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Independent second Q head with clipped minimum bootstrap (FedAvg only).",
    )
    parser.add_argument(
        "--local_critics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aggregate the actor while retaining each home's value heads and targets locally.",
    )
    parser.add_argument(
        "--critic_huber_delta",
        type=float,
        default=0,
        help="Positive: use Huber TD loss with this delta; zero retains squared TD loss.",
    )
    parser.add_argument(
        "--temperature_actor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn an 18.2–21.8C target and derive AC power from the original thermal equation (raw/hybrid).",
    )
    parser.add_argument(
        "--target_temperature_bounds",
        nargs=2,
        type=float,
        default=None,
        help="Optional learned target range for quota-aware temperature actors; simulator comfort limits remain 18–22C.",
    )
    parser.add_argument(
        "--feedback_target",
        type=float,
        default=20,
        help="Temperature target for training-only feedback labels, inside the original comfort band.",
    )
    parser.add_argument(
        "--target_noise",
        type=float,
        default=0,
        help="Target-policy noise std as a fraction of each actuator range.",
    )
    parser.add_argument(
        "--exploration_noise",
        type=float,
        default=None,
        help="Use local Gaussian continuous exploration; epsilon then applies to WM only.",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        nargs=3,
        default=None,
        help="Bound corrections to the pretrained actor as AC/EV/BESS action-range fractions.",
    )
    parser.add_argument(
        "--return_mode",
        choices=("td", "episode"),
        default="td",
        help="Experimental full-episode behavior-return critic targets instead of bootstrapping.",
    )
    parser.add_argument(
        "--batched_updates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Vectorize independent FedAvg home updates on one device (replay-only warmup, P2P).",
    )
    parser.add_argument(
        "--feasible_actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Parameterize AC quota, EV availability and battery charge limits using existing live state (raw/hybrid).",
    )
    parser.add_argument(
        "--head_width",
        type=int,
        default=256,
        help="Hidden units per trainable head (FedAvg); checkpoint loaders infer this shape.",
    )
    parser.add_argument(
        "--actor_depth",
        type=int,
        default=1,
        help="Number of trainable actor hidden layers; critic architecture is independent.",
    )
    parser.add_argument("--actor_activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument(
        "--ppo_oracle_init",
        default=None,
        help="Optional certified training-only oracle directory for actor initialization; PPO stays on-policy.",
    )
    parser.add_argument(
        "--selection_reference",
        default=None,
        help="Optional fixed validation service reference shared across architecture/initialization comparisons.",
    )
    # Global experimental settings
    parser.add_argument("--episode", type=int, default=1500, help="Training episodes per model.")
    parser.add_argument(
        "--eval_step",
        type=int,
        default=0,
        help="Evaluate FedAvg every N completed episodes, plus episode 0 and the final episode; 0 disables.",
    )
    parser.add_argument(
        "--eval_days",
        type=int,
        default=0,
        help="Fixed number of shared test dates per periodic evaluation; 0 uses all dates.",
    )
    parser.add_argument("--warmup_rounds", type=int, default=75, help="Warmup rounds per agent.")
    parser.add_argument("--seed", type=int, default=6, help="Model initialization seed.")
    parser.add_argument(
        "--fixed_seed", type=int, default=100, help="Seed for all other randomness."
    )
    parser.add_argument("--update", type=int, default=10, help="Episodes between mask updates.")
    parser.add_argument(
        "--aggregate", type=int, default=20, help="Episodes per communication round."
    )
    parser.add_argument(
        "--ma_window", type=int, default=100, help="Moving-average window for training plots."
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        choices=(9, 17),
        default=STATE_DIM,
        help="9: legacy observation; 17: also expose live indoor temperature, EV, WM and budgets.",
    )
    parser.add_argument(
        "--continuous_action_dim",
        type=int,
        default=CONTINUOUS_ACTION_DIM,
        help="Continuous action size.",
    )
    parser.add_argument(
        "--discrete_action_dim", type=int, default=DISCRETE_ACTION_DIM, help="Discrete action size."
    )
    parser.add_argument(
        "--actor_sparsity", type=float, default=0.75, help="Initial actor sparsity."
    )
    parser.add_argument(
        "--critic_sparsity", type=float, default=0.75, help="Initial critic sparsity."
    )
    parser.add_argument("--epsilon", type=float, default=0.3, help="EdgeHEM rewiring rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward discount factor.")
    parser.add_argument("--batch_size", type=int, default=64, help="Replay batch size.")
    parser.add_argument("--lr_actor", type=float, default=1e-5, help="Actor learning rate.")
    parser.add_argument("--lr_critic", type=float, default=1e-4, help="Critic learning rate.")
    parser.add_argument(
        "--epsilon_start", type=float, default=1.0, help="Initial exploration rate."
    )
    parser.add_argument("--epsilon_end", type=float, default=0.005, help="Final exploration rate.")
    parser.add_argument("--epsilon_decay", type=int, default=10000, help="Exploration decay steps.")
    parser.add_argument("--critic_tau", type=float, default=0.01, help="Critic target-update rate.")
    parser.add_argument("--actor_tau", type=float, default=0.001, help="Actor target-update rate.")
    # FedFit model settings
    parser.add_argument(
        "--fedfit_adjust_fraction",
        type=float,
        default=0.2,
        help="Initial fraction of active weights exchanged per topology update.",
    )
    parser.add_argument(
        "--fedfit_t_end",
        type=int,
        help="Last episode for FedFit topology updates; defaults to --episode.",
    )
    # PFFDST model settings
    parser.add_argument(
        "--pffdst_differential_ratio",
        type=float,
        default=0.5,
        help="Temporary extra-density ratio during PFFDST server mask readjustment.",
    )
    parser.add_argument(
        "--pffdst_readjust_interval",
        type=int,
        default=1,
        help="Communication rounds between PFFDST server mask readjustments.",
    )
    # FedDMPQ model settings
    parser.add_argument(
        "--feddmpq_alpha_w", type=float, default=0.25, help="FedDMPQ weight-magnitude coefficient."
    )
    parser.add_argument(
        "--feddmpq_beta_g", type=float, default=0.25, help="FedDMPQ gradient coefficient."
    )
    parser.add_argument(
        "--feddmpq_gamma_e", type=float, default=0.50, help="FedDMPQ error coefficient."
    )
    parser.add_argument("--feddmpq_k", type=int, default=10, help="FedDMPQ pruning patience.")
    parser.add_argument(
        "--feddmpq_low_bit",
        type=int,
        choices=(2, 4, 8, 16),
        default=8,
        help="FedDMPQ low-precision bit width.",
    )
    # Energy management settings
    parser.add_argument(
        "--tou", action=argparse.BooleanOptionalAction, default=True, help="Enable the ToU tariff."
    )
    parser.add_argument(
        "--tou_blocks", type=int, default=5, help="Number of daily time-of-use price blocks."
    )
    grid_tariff = parser.add_mutually_exclusive_group()
    grid_tariff.add_argument(
        "--grid_prices",
        type=float,
        nargs=24,
        help="Explicit hourly grid prices ($/kWh); requires --no-tou.",
    )
    grid_tariff.add_argument(
        "--flat_grid_price", type=float, help="Explicit flat grid price ($/kWh); requires --no-tou."
    )
    parser.add_argument(
        "--ac_service",
        choices=("energy_quota", "thermal"),
        default="energy_quota",
        help="Thermal mode serves AC comfort without forcing historical daily AC consumption.",
    )
    parser.add_argument(
        "--dr_limit",
        type=float,
        default=5.0,
        help="Demand-response import limit (kW), or 'none' to disable.",
    )
    parser.add_argument(
        "--dr_penalty",
        type=float,
        default=0.50,
        help="Penalty for demand above the DR limit ($/kWh).",
    )
    parser.add_argument(
        "--dr_incentive",
        type=float,
        default=0.025,
        help="Incentive for demand below the DR limit ($/kWh).",
    )
    parser.add_argument(
        "--pv_curtail",
        type=float,
        default=2.5,
        help="Daily PV export cap (kWh), or 'None' for no cap.",
    )
    parser.add_argument(
        "--export_price", type=float, default=0.025, help="Grid export compensation ($/kWh)."
    )
    parser.add_argument("--fixed_cost", type=float, default=5.0, help="Monthly fixed cost ($).")
    parser.add_argument(
        "--p2p", action=argparse.BooleanOptionalAction, default=True, help="Enable P2P trading."
    )
    parser.add_argument(
        "--p2p_price", type=float, default=0.10, help="P2P price ($/kWh) homes pay each other."
    )
    parser.add_argument("--day", default="2019-08-01", help="Day to plot schedule")
    parser.add_argument("--home_id", type=int, default=1, help="Home index to plot schedule.")
    # GPU settings
    parser.add_argument(
        "--head_device",
        choices=("auto", "cpu"),
        default="auto",
        help="Optionally train small heads on CPU while keeping feature extraction on --gpu.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--cpu_threads", type=int, default=5, help="CPU worker threads.")
    parser.add_argument(
        "--gpu_memory_fraction", type=float, default=1.0, help="GPU memory fraction."
    )
    parser.add_argument("--preset", choices=PRESETS, default=preset)
    parser.add_argument(
        "--min_episodes",
        type=int,
        default=0,
        help="Minimum episodes before validation early stopping.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Validation checks without meaningful feasible reward improvement; 0 disables.",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.01,
        help="Minimum feasible reward improvement to reset patience.",
    )
    if preset == "optimized":
        parser.set_defaults(**OPTIMIZED_DEFAULTS)
    elif preset == "thermal":
        parser.set_defaults(**THERMAL_DEFAULTS)
    elif preset == "ppo":
        parser.set_defaults(**PPO_DEFAULTS)
    elif preset == "ppo_reference":
        parser.set_defaults(**PPO_REFERENCE_DEFAULTS)
    return parser


def apply_refit_mode(args):
    """A monthly refit never evaluates or selects on its overlapping training view."""
    args.refit = args.data_period.endswith("_refit")
    if args.refit:
        args.eval_step = 0
        args.validation_only = True
        args.skip_final_evaluation = True
        args.patience = 0
        args.strict_convergence = False
    return args


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--preset", choices=PRESETS, default="optimized")
    preset, _ = selector.parse_known_args(argv)
    parser = build_parser(preset.preset)
    args = parser.parse_args(argv)
    if args.flat_grid_price is not None:
        args.grid_prices = [args.flat_grid_price] * 24
    if args.grid_prices is not None and args.tou:
        parser.error("Explicit grid prices require --no-tou")
    return apply_refit_mode(args)
