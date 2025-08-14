import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import PPOConfig, ALNSConfig
from .instance_pool import InstancePool
import alns.vrp as vrp
from alns.alns4ppo import ALNS4PPO

ppo_config = PPOConfig()
alns_config = ALNSConfig()

# Reward weights
MAX_ITERATIONS = alns_config.num_iterations  # Maximum iterations
STOP_THRESHOLD = 2  # Consecutive stop signal threshold


class VRPTWEnvironment(gym.Env):
    def __init__(
        self, instance_pool: InstancePool, record_data: bool = False, alns_init: int = 0
    ):
        """
        Initialize VRPTW environment
        Args:
            data: VRPTWDataset data
            alns: ALNS algorithm instance
        """
        super(VRPTWEnvironment, self).__init__()

        self.instance_pool = instance_pool
        self.alns = ALNS4PPO()
        self.accept = None
        self.dimension = ppo_config.dimension

        self.d_op_num = alns_config.num_destroy  # Number of destroy operators
        self.r_op_num = alns_config.num_repair  # Number of repair operators
        self.destroy_usage = np.zeros(self.d_op_num, dtype=np.float32)
        self.repair_usage = np.zeros(self.r_op_num, dtype=np.float32)

        self.init_solution = None
        self.current_solution = None
        self.best_solution = None
        self.initial_temperature = 1.0
        self.final_temperature = 0.01

        self.iters = 0
        self.stop_counter = 0

        self.record_data = record_data
        self.alns_init = alns_init

        # Action space: [destroy operator index, repair operator index, whether to accept new solution]
        self.action_space = spaces.MultiDiscrete(
            [
                self.d_op_num,  # Destroy operator
                self.r_op_num,  # Repair operator
                2,  # Whether to accept new solution
                2,  # Whether to terminate algorithm
            ]
        )

        # State space
        self.observation_space = spaces.Dict(
            {
                # Search progress
                "search_progress": spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),
                "solution_delta": spaces.Box(
                    low=-1, high=1, shape=(1,), dtype=np.float32
                ),
                "init_cost": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "best_cost": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                # Operator usage history
                "destroy_usage": spaces.Box(
                    low=0, high=1, shape=(self.d_op_num,), dtype=np.float32
                ),
                "repair_usage": spaces.Box(
                    low=0, high=1, shape=(self.r_op_num,), dtype=np.float32
                ),
                # Problem features
                "demand": spaces.Box(
                    low=0, high=1, shape=(self.dimension,), dtype=np.float32
                ),
                "time_windows": spaces.Box(
                    low=0, high=1, shape=(self.dimension, 2), dtype=np.float32
                ),
                "service_times": spaces.Box(
                    low=0, high=1, shape=(self.dimension,), dtype=np.float32
                ),
                "travel_times": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.dimension, self.dimension),
                    dtype=np.float32,
                ),
            }
        )

        # Initialize state
        self.state = {
            "search_progress": np.zeros(1, dtype=np.float32),
            "solution_delta": np.zeros(1, dtype=np.float32),
            "init_cost": np.zeros(1, dtype=np.float32),
            "best_cost": np.zeros(1, dtype=np.float32),
            "destroy_usage": np.zeros(self.d_op_num, dtype=np.float32),
            "repair_usage": np.zeros(self.r_op_num, dtype=np.float32),
        }

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment"""
        data = self.instance_pool.sample()
        if self.record_data:
            self.data = data
        self.alns.reset_opt()
        d_opt_list = []
        r_opt_list = []
        d_opt_list.append(vrp.create_random_customer_removal_operator(data))
        d_opt_list.append(vrp.create_random_route_removal_operator())
        d_opt_list.append(vrp.create_string_removal_operator(data))
        d_opt_list.append(vrp.create_worst_removal_operator(data))
        d_opt_list.append(vrp.create_sequence_removal_operator(data))
        d_opt_list.append(vrp.create_related_removal_operator(data))

        r_opt_list.append(vrp.create_greedy_repair_operator(data))
        r_opt_list.append(vrp.create_criticality_repair_operator(data))
        r_opt_list.append(vrp.create_regret_repair_operator(data))

        for i in range(self.d_op_num):
            self.alns.add_destroy_operator(d_opt_list[i])

        for i in range(self.r_op_num):
            self.alns.add_repair_operator(r_opt_list[i])

        if self.alns_init == 0:
            self.init_solution = vrp.clarke_wright_tw(data=data)
        elif self.alns_init == 1:
            self.init_solution = vrp.initial.nearest_neighbor_tw(data=data)

        self.max_cost = np.sum(data["travel_times"][0, 1:]) * 2
        # min-max normalise the costs
        init_cost = self.init_solution.objective() / self.max_cost
        self.state["search_progress"][:] = 0
        self.state["init_cost"][:] = init_cost
        self.state["best_cost"][:] = init_cost
        self.state["solution_delta"][:] = 0
        self.state["destroy_usage"][:] = 0
        self.state["repair_usage"][:] = 0

        self.state["demand"] = np.array(data["demand"], dtype=np.float32)
        self.state["time_windows"] = np.array(data["time_windows"], dtype=np.float32)
        self.state["service_times"] = np.array(data["service_times"], dtype=np.float32)
        self.state["travel_times"] = np.array(data["travel_times"], dtype=np.float32)

        self.destroy_usage[:] = 0
        self.repair_usage[:] = 0

        # Get initial solution
        self.current_solution = self.init_solution.copy()
        self.pre_solution = self.init_solution.copy()
        self.best_solution = self.init_solution.copy()

        self.iters = 0
        self.stop_counter = 0

        self.alns._rng = np.random.default_rng()

        return self.state, {}

    def _update_state(self):
        """Update environment state"""
        self.state["search_progress"][0] = self.iters / MAX_ITERATIONS
        # min-max normalise the costs
        self.state["best_cost"][0] = self.best_solution.objective() / self.max_cost
        # Update operator usage frequency, normalize self.destroy_usage and self.repair_usage
        self.state["destroy_usage"] = self.destroy_usage / (
            np.sum(self.destroy_usage) + 1
        )
        self.state["repair_usage"] = self.repair_usage / (np.sum(self.repair_usage) + 1)

    def _check_done(self, done_num):
        # if done_num == 1 and self.iters % 10 == 0 :
        # return True
        if done_num == 1:
            self.stop_counter += 1
            if self.stop_counter >= STOP_THRESHOLD:
                return True
        else:
            self.stop_counter = 0

        if self.iters >= MAX_ITERATIONS:
            return True
        return False

    def step(self, action):
        """
        Execute one step of ALNS iteration
        Args:
            action: [destroy operator index, repair operator index, whether to accept new solution]
        """
        d_idx, r_idx, accept, stop = action

        self.iters += 1
        reward = 0
        # Update operator usage count
        self.destroy_usage[d_idx] += 1
        self.repair_usage[r_idx] += 1

        improvement = self.state["solution_delta"][0]
        # Penalty for rejecting better solutions
        if improvement > 0 and accept == 0:
            reward -= 0.8 * improvement
        # Penalty for accepting worse solutions
        if improvement < 0 and accept == 1:
            reward += 0.8 * improvement

        # Execute ALNS iteration
        self.pre_solution, self.current_solution = self.alns.iterate(
            self.current_solution, self.pre_solution, d_idx, r_idx, accept
        )
        # Record best_solution
        best_cost = self.best_solution.objective()
        pre_cost = self.pre_solution.objective()
        cur_cost = self.current_solution.objective()
        # Step improvement reward
        improvement = (pre_cost - cur_cost) / pre_cost
        # reward += 0.1 * improvement
        if cur_cost < best_cost:
            self.best_solution = self.current_solution
            # Reward for finding optimal solution
            relative_to_best = (best_cost - cur_cost) / best_cost
            reward += 2 * relative_to_best
        # Calculate solution quality improvement
        self.state["solution_delta"][:] = improvement

        # Update state
        self._update_state()

        # reward += STEP_PENALTY

        # Check if termination conditions are met
        done = self._check_done(stop)
        info = {}
        if done:
            # Final reward is the quality of the solution
            reward += (
                4
                * (self.state["init_cost"][0] - self.state["best_cost"][0])
                / self.state["init_cost"][0]
            )

            # Reward for early termination
            reward -= 0.04 * self.state["search_progress"][0]
            info = {
                "best_solution": self.best_solution,
                "best_cost": self.best_solution.objective(),
            }

        return self.state, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass
