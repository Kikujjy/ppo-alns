import argparse
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch
from core import LightLogger, VRPTWGeneratorC, VRPTWGeneratorR
from config import PPOConfig
from ppo.vrptwenv import VRPTWEnvironment
from ppo.instance_pool import InstancePool
from core import format_vrptw_routes, LightLogger

ppo_config = PPOConfig()
parser = argparse.ArgumentParser(description="VRPTW Solver")
parser.add_argument("-a", "--algorithm", type=str, default="ppo", help="Algorithm")
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["train", "test"],
    default="train",
    help="Run mode",
)
parser.add_argument(
    "-p",
    "--model_path",
    type=str,
    default=f"models/model{ppo_config.dimension}_{ppo_config.generator_type}/capacity_{ppo_config.capacity}/final_model.zip",
    help="Model loading path",
)
args = parser.parse_args()

logger = LightLogger(name="Main")
# Create generator
if ppo_config.generator_type == "R":
    generator = VRPTWGeneratorR(
        dimension=ppo_config.dimension,
        capacity=ppo_config.capacity,
        max_demand=ppo_config.max_demand,
        num_vehicles=ppo_config.num_vehicles,
        min_window_width=ppo_config.min_window_width,
        max_window_width=ppo_config.max_window_width,
        max_travel_time=ppo_config.max_travel_time,
    )
elif ppo_config.generator_type == "C":
    generator = VRPTWGeneratorC(
        dimension=ppo_config.dimension,
        capacity=ppo_config.capacity,
        max_demand=ppo_config.max_demand,
        num_vehicles=ppo_config.num_vehicles,
        min_window_width=ppo_config.min_window_width,
        max_window_width=ppo_config.max_window_width,
        max_travel_time=ppo_config.max_travel_time,
    )


def make_env(in_pool):
    """
    Helper function to create environment
    """

    def _init():
        env = VRPTWEnvironment(instance_pool=in_pool)
        return env

    return _init


def create_subproc_env(n_envs, in_pool):
    """
    Create vectorized environment
    Args:
        n_envs: Number of parallel environments
    """
    env = SubprocVecEnv([make_env(in_pool) for _ in range(n_envs)])
    return env


def create_dummy_env(n_envs, in_pool):
    env = DummyVecEnv([make_env(in_pool) for _ in range(n_envs)])
    return env


def train_ppo(
    n_envs: int = 8,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    total_timesteps: int = 256 * 64 * 10,
    progress_bar: bool = False,
    save_path: str = "models",
    model_path: str = None,
):
    instance_pool = InstancePool(
        generator,
        pool_size=ppo_config.training_pool_size,
        refresh_threshold=ppo_config.training_refresh_threshold,
        sampling_memory=ppo_config.training_sampling_memory,
        chunk_size=ppo_config.training_chunk_size,
        max_workers=ppo_config.training_max_workers,
        seed=ppo_config.train_seed,
    )
    eval_instance_pool = InstancePool(
        generator,
        pool_size=ppo_config.eval_pool_size,
        sampling_memory=ppo_config.eval_sampling_memory,
        chunk_size=ppo_config.eval_chunk_size,
        max_workers=ppo_config.eval_max_workers,
        no_refresh=ppo_config.eval_no_refresh,
        seed=ppo_config.eval_seed,
    )

    train_env = create_subproc_env(n_envs, instance_pool)
    train_env = VecMonitor(train_env, f"{ppo_config.tensorboard_log}/train")

    if args.mode == "train":
        # Create model
        def linear_schedule(initial_value: float):
            def func(progress_remaining: float) -> float:
                return progress_remaining * initial_value

            return func

        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=linear_schedule(learning_rate),
            verbose=1,
            ent_coef=0.001,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=ppo_config.tensorboard_log,
            device=ppo_config.device,
        )
    else:
        model = PPO.load(model_path)
        model.set_env(train_env)

    # Create evaluation environment
    eval_env = create_dummy_env(
        n_envs=ppo_config.n_eval_envs, in_pool=eval_instance_pool
    )
    eval_env = VecMonitor(eval_env, f"{ppo_config.tensorboard_log}/eval")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=ppo_config.tensorboard_log,
        eval_freq=ppo_config.eval_freq,
        deterministic=True,
        render=False,
    )

    # Create callback function
    checkpoint_callback = CheckpointCallback(
        save_freq=ppo_config.save_freq, save_path=save_path, name_prefix="ppo4vrptw"
    )

    try:
        # Start training
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=progress_bar,
            tb_log_name=f"ppo4vrptw",
        )
    finally:
        train_env.close()
        eval_env.close()

    # Save final model
    model.save(f"{save_path}/final_model")
    logger.info(f"The model have saved to {save_path}/final_model")
    return model


def evaluate_model(model_path):
    """
    Evaluate the trained model
    """
    test_pool = InstancePool(
        generator,
        pool_size=ppo_config.test_instances,
        sampling_memory=ppo_config.test_instances,
        max_workers=1,
        chunk_size=1,
        seed=ppo_config.test_seed,
    )
    env = VRPTWEnvironment(instance_pool=test_pool)
    env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path)
    model.policy = model.policy.to(ppo_config.device)
    model.policy.eval()
    for _ in range(ppo_config.test_instances):
        obs = env.reset()
        done = False
        cumulate_reward = 0
        with torch.no_grad():
            steps = 0
            time_start = time.time()
            while not done:
                steps += 1
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                cumulate_reward += reward
                logger.info(
                    f"action: {action}, reward: {reward}, cumulate_reward: {cumulate_reward}"
                )
            time_end = time.time()
            logger.info(f"Time Cost: {time_end - time_start}, Steps: {steps}")
            logger.info(
                f"PPO Result Routes: {format_vrptw_routes(info[0]['best_solution'].routes)}"
            )
            logger.info(f"PPO Result Costs: {info[0]['best_cost']}")
    return


if __name__ == "__main__":
    logger = LightLogger(name="main")
    if args.algorithm.lower() == "ppo":
        if args.mode == "train":
            train_ppo(
                n_envs=ppo_config.num_envs,
                n_steps=ppo_config.n_steps,
                batch_size=ppo_config.batch_size,
                n_epochs=ppo_config.n_epochs,
                learning_rate=ppo_config.learning_rate,
                total_timesteps=ppo_config.total_timesteps,
                progress_bar=ppo_config.progress_bar,
                save_path=ppo_config.model_save_path,
                model_path=args.model_path,
            )
        elif args.mode == "test":
            evaluate_model(args.model_path)
