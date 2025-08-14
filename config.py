from dataclasses import asdict, dataclass, field


@dataclass
class PPOConfig:
    # Generator parameters
    generator_type: str = "R"  # Generator type ["R", "C"]
    dimension: int = 21  # Number of customers
    capacity: int = 64  # Vehicle capacity
    max_demand: int = 16  # Maximum demand
    num_vehicles: int = dimension  # Number of vehicles
    min_window_width: int = 5  # Minimum time window width
    max_window_width: int = 15  # Maximum time window width
    max_travel_time: int = 50  # Maximum travel time

    # Train Config
    train_seed: int = 3333  # Training random seed
    num_envs: int = 32  # Number of parallel environments
    n_steps: int = 1000  # Experience pool size
    total_timesteps: int = 1000 * 32 * 50  # Total timesteps
    save_freq: int = 1000 * 32 * 5  # Save frequency
    batch_size: int = 128  # Training batch size
    n_epochs: int = 10  # Number of epochs per update
    learning_rate: float = 1e-3  # Initial learning rate
    progress_bar: bool = True  # Whether to show progress bar
    device: str = "cuda:0"  # Device
    training_pool_size: int = 128  # Training instance pool size
    training_refresh_threshold: int = (
        128 * 32
    )  # Training instance pool refresh threshold
    training_sampling_memory: int = 128  # Training instance pool sampling memory
    training_chunk_size: int = 32  # Training instance pool batch size
    training_max_workers: int = 4  # Training instance pool max workers
    training_no_refresh: bool = (
        False  # Whether training instance pool should not refresh
    )

    # Eval Config
    n_eval_envs: int = 1  # Number of evaluation environments
    eval_seed: int = 123  # Evaluation random seed
    eval_freq: int = 2000  # Evaluation frequency
    eval_pool_size: int = 100  # Evaluation instance pool size
    eval_sampling_memory: int = 100  # Evaluation instance pool sampling memory
    eval_chunk_size: int = 20  # Evaluation instance pool batch size
    eval_max_workers: int = 2  # Evaluation instance pool max workers
    eval_no_refresh: bool = True  # Whether evaluation instance pool should not refresh

    # Test Config
    test_seed: int = 321  # Test random seed
    test_instances: int = 10  # Number of test instances

    # Other Config
    num_flag: int = dimension  # Experiment number
    tensorboard_log: str = f"./logs/tb_log{num_flag}/"
    model_save_path: str = (
        f"./models/model{num_flag}_{generator_type}/capacity_{capacity}"
    )


@dataclass
class ALNSConfig:
    seed: int = 520  # Algorithm random seed
    num_iterations: int = 100  # Number of iterations
    num_destroy: int = 5  # Number of destroy operators
    num_repair: int = 3  # Number of repair operators
    roulette_wheel_scores: list[float] = field(
        default_factory=lambda: [25, 5, 1, 0]
    )  # Roulette wheel score list
    roulette_wheel_decay: float = 0.9  # Roulette wheel decay coefficient
    autofit_start_gap: float = 0.05  # Initially only accept solutions within 0.* coefficient of optimal solution
    autofit_end_gap: float = 0  # Finally accept solutions within 0.* coefficient of optimal solution (usually 0)

    def get_alns_params_dict(self):
        return asdict(self)
