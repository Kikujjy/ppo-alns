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


class VRPTWEnvironmentDestroyOnly(gym.Env):
    """
    Ablation experiment environment: PPO only selects destroy operators, repair operators are intelligently selected by ALNS

    This environment maintains the same action space and observation space as the original VRPTWEnvironment,
    but ignores the action input of repair operators, making it directly compatible with the original model for ablation experiments.
    """

    def __init__(
        self, instance_pool: InstancePool, record_data: bool = False, alns_init: int = 0
    ):
        """
        Initialize VRPTW environment (only select destroy operators)
        Args:
            instance_pool: Instance pool
            record_data: Whether to record data
            alns_init: Initialization method (0: Clarke-Wright, 1: Nearest neighbor)
        """
        super(VRPTWEnvironmentDestroyOnly, self).__init__()

        self.instance_pool = instance_pool
        self.alns = ALNS4PPO()
        self.accept = None
        self.dimension = ppo_config.dimension

        self.d_op_num = alns_config.num_destroy  # Number of destroy operators
        self.r_op_num = alns_config.num_repair  # Number of repair operators
        self.destroy_usage = np.zeros(self.d_op_num, dtype=np.float32)
        self.repair_usage = np.zeros(self.r_op_num, dtype=np.float32)

        # Add weight management needed for intelligent selection
        self.repair_weights = np.ones(
            self.r_op_num, dtype=np.float32
        )  # Repair operator weights
        self.scores = alns_config.roulette_wheel_scores  # [25, 5, 1, 0] corresponding to new global optimum, better solution, accepted solution, rejected solution
        self.decay = alns_config.roulette_wheel_decay  # Weight decay coefficient 0.9

        self.init_solution = None
        self.current_solution = None
        self.best_solution = None
        self.initial_temperature = 1.0
        self.final_temperature = 0.01

        self.iters = 0
        self.stop_counter = 0

        self.record_data = record_data
        self.alns_init = alns_init

        # Action space: [destroy operator index, repair operator index(ignored), whether to accept new solution, whether to terminate algorithm]
        # Maintain the same action space as the original environment, but ignore repair operator selection
        self.action_space = spaces.MultiDiscrete(
            [
                self.d_op_num,  # Destroy operator
                self.r_op_num,  # Repair operator (will be ignored in this environment)
                2,  # Whether to accept new solution
                2,  # Whether to terminate algorithm
            ]
        )

        # State space (maintain consistency with original environment, including repair operator usage history)
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
        self.repair_weights[:] = 1.0  # Reset repair operator weights

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
        if done_num == 1:
            self.stop_counter += 1
            if self.stop_counter >= STOP_THRESHOLD:
                return True
        else:
            self.stop_counter = 0

        if self.iters >= MAX_ITERATIONS:
            return True
        return False

    def _select_repair_operator_intelligently(self):
        """
        Intelligently select repair operator based on weights (roulette wheel algorithm)
        """
        # Calculate probability distribution
        probs = self.repair_weights / np.sum(self.repair_weights)
        # Select operator based on probability
        r_idx = self.alns._rng.choice(range(self.r_op_num), p=probs)
        return r_idx

    def _update_repair_operator_weight(self, r_idx, outcome):
        """
        Update repair operator weights based on operation results
        Args:
            r_idx: Repair operator index
            outcome: Operation result (0: new global optimum, 1: better solution, 2: accepted solution, 3: rejected solution)
        """
        self.repair_weights[r_idx] = (
            self.decay * self.repair_weights[r_idx]
            + (1 - self.decay) * self.scores[outcome]
        )

    def iterate_destroy_only(self, cur_solution, pre_solution, d_idx, accept):
        """
        Execute single ALNS iteration, only select destroy operator, repair operator intelligently selected
        """
        r_idx = None  # Used to record the selected repair operator index

        if accept == 0:
            cur_solution = pre_solution
        elif accept == 1:
            # PPO selects destroy operator
            d_name, d_operator = self.alns.destroy_operators[d_idx]
            # Intelligently select repair operator
            r_idx = self._select_repair_operator_intelligently()
            r_name, r_operator = self.alns.repair_operators[r_idx]

            destroyed = d_operator(cur_solution, self.alns._rng)
            cand = r_operator(destroyed, self.alns._rng)

            pre_solution = cur_solution
            cur_solution = cand

            # Update repair operator usage count
            self.repair_usage[r_idx] += 1

        return pre_solution, cur_solution, r_idx

    def step(self, action):
        """
        Execute one step of ALNS iteration
        Args:
            action: [destroy operator index, repair operator index (ignored), whether to accept new solution, whether to terminate algorithm]
        """
        d_idx, r_idx, accept, stop = (
            action  # r_idx will be ignored, repair operator through intelligent selection
        )

        self.iters += 1
        reward = 0
        # Update destroy operator usage count
        self.destroy_usage[d_idx] += 1

        improvement = self.state["solution_delta"][0]
        # Rejecting better solutions requires penalty
        if improvement > 0 and accept == 0:
            reward -= 0.8 * improvement
        # Accepting worse solutions requires penalty
        if improvement < 0 and accept == 1:
            reward += 0.8 * improvement

        # Execute ALNS iteration (only select destroy operator, repair operator intelligently selected)
        # Note: r_idx parameter is ignored, repair operator is determined by intelligent selection algorithm
        self.pre_solution, self.current_solution, selected_r_idx = (
            self.iterate_destroy_only(
                self.current_solution, self.pre_solution, d_idx, accept
            )
        )

        # Record best_solution
        best_cost = self.best_solution.objective()
        pre_cost = self.pre_solution.objective()
        cur_cost = self.current_solution.objective()
        # Step-by-step improvement reward
        improvement = (pre_cost - cur_cost) / pre_cost
        if cur_cost < best_cost:
            self.best_solution = self.current_solution
            # Discovery of optimal solution reward
            relative_to_best = (best_cost - cur_cost) / best_cost
            reward += 2 * relative_to_best
        # Calculate solution quality improvement
        self.state["solution_delta"][:] = improvement

        # Update repair operator weights (only when repair operator is selected)
        if selected_r_idx is not None and accept == 1:
            # Determine operation result
            if cur_cost < best_cost:
                outcome = 0  # New global optimum
            elif cur_cost < pre_cost:
                outcome = 1  # Better solution
            else:
                # Simplified handling here, treat other cases as accepted solutions
                # In actual ALNS, there would be acceptance criteria like simulated annealing
                outcome = 2  # Accepted solution

            # Update weights
            self._update_repair_operator_weight(selected_r_idx, outcome)

        # Update state
        self._update_state()

        # Check if termination condition is reached
        done = self._check_done(stop)
        info = {}
        if done:
            # Final reward is the quality of the solution
            reward += (
                4
                * (self.state["init_cost"][0] - self.state["best_cost"][0])
                / self.state["init_cost"][0]
            )

            # Early termination reward
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
