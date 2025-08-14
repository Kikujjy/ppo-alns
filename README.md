# Reinforcement Learning-guided Adaptive Large Neighborhood Search for Vehicle Routing Problem with Time Windows

A novel hybrid framework that integrates Proximal Policy Optimization (PPO) with Adaptive Large Neighborhood Search (ALNS) for solving the Vehicle Routing Problem with Time Windows (VRPTW).

## Features

- **PPO-based RL**: Uses stable-baselines3 PPO implementation for route optimization
- **ALNS Integration**: Combines with ALNS for enhanced solution quality
- **VRPTW Support**: Handles time window constraints and vehicle capacity
- **Parallel Training**: Multi-environment training with configurable parameters
- **Flexible Generation**: Supports both random (R) and custom (C) instance generators

## Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `torch==2.5.1+cu121` - PyTorch with CUDA support
- `stable_baselines3==2.6.0` - RL algorithms implementation
- `gymnasium==1.1.1` - RL environment interface
- `numpy==2.3.2` - Numerical computing
- `matplotlib==3.10.5` - Plotting and visualization

## Quick Start

### Training

```bash
python main.py --algorithm ppo --mode train
```

**Training Parameters:**
- `--algorithm`: Algorithm type (ppo)
- `--mode`: Run mode (train/test)
- `--model_path`: Path to load existing model (optional)

### Testing

```bash
python main.py --algorithm ppo --mode test --model_path models/model21/capacity_64/final_model.zip
```

## Configuration

### Generator Configuration

The system supports two types of instance generators:

- **`generator_type`**: Generator type selection
  - `"R"`: Random generator
  - `"C"`: Custom generator

**Problem Scale Parameters:**
- `dimension`: Number of customers (default: 21)
- `capacity`: Vehicle capacity (default: 64)
- `max_demand`: Maximum demand per customer (default: 16)
- `num_vehicles`: Number of available vehicles (default: equals dimension)
- `min_window_width`: Minimum time window width (default: 5)
- `max_window_width`: Maximum time window width (default: 15)
- `max_travel_time`: Maximum travel time between locations (default: 50)

### Training Hyperparameters

**Basic Training Settings:**
- `train_seed`: Training random seed (default: 3333)
- `num_envs`: Number of parallel environments (default: 32)
- `n_steps`: Experience pool size (default: 1000)
- `total_timesteps`: Total training steps (default: 1,600,000)
- `save_freq`: Model save frequency (default: 160,000 steps)

**Network Training Parameters:**
- `batch_size`: Training batch size (default: 128)
- `n_epochs`: Number of epochs per update (default: 10)
- `learning_rate`: Initial learning rate (default: 1e-3)
- `progress_bar`: Show training progress bar (default: True)
- `device`: Training device (default: "cuda:0")

**Instance Pool Configuration:**
- `training_pool_size`: Training instance pool size (default: 128)
- `training_refresh_threshold`: Refresh threshold for training pool (default: 4,096)
- `training_sampling_memory`: Sampling memory size (default: 128)
- `training_chunk_size`: Batch size for instance processing (default: 32)
- `training_max_workers`: Maximum worker threads (default: 4)

### Evaluation Configuration

- `n_eval_envs`: Number of evaluation environments (default: 1)
- `eval_seed`: Evaluation random seed (default: 123)
- `eval_freq`: Evaluation frequency (default: 2000 steps)
- `eval_pool_size`: Evaluation instance pool size (default: 100)
- `eval_sampling_memory`: Evaluation sampling memory (default: 100)
- `eval_chunk_size`: Evaluation batch size (default: 20)
- `eval_max_workers`: Evaluation worker threads (default: 2)
- `eval_no_refresh`: Disable evaluation pool refresh (default: True)

### Test Configuration

- `test_seed`: Test random seed (default: 321)
- `test_instances`: Number of test instances (default: 128)

### Output Configuration

- `num_flag`: Experiment identifier (default: equals dimension)
- `tensorboard_log`: Tensorboard log directory (default: `./logs/tb_log{dimension}/`)
- `model_save_path`: Model save directory (default: `./models/model{dimension}_{generator_type}/capacity_{capacity}`)
