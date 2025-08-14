import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import torch


@dataclass
class VRPTWInstance:
    dimension: int
    capacity: int
    num_vehicles: int
    max_travel_time: int
    depot: int
    demand: np.ndarray
    time_windows: np.ndarray
    service_times: np.ndarray
    travel_times: np.ndarray
    node_coord: np.ndarray
    edge_weight: np.ndarray

    def to_tensor(
        self, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert instance data to PyTorch tensors
        Args:
            device: Device type
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing tensorized data
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "dimension": torch.tensor(self.dimension, device=device),
            "capacity": torch.tensor(self.capacity, dtype=torch.float32, device=device),
            "num_vehicles": torch.tensor(self.num_vehicles, device=device),
            "max_travel_time": torch.tensor(
                self.max_travel_time, dtype=torch.float32, device=device
            ),
            "depot": torch.tensor(self.depot, device=device),
            "demand": torch.FloatTensor(self.demand).to(device),
            "time_windows": torch.FloatTensor(self.time_windows).to(device),
            "service_times": torch.FloatTensor(self.service_times).to(device),
            "travel_times": torch.FloatTensor(self.travel_times).to(device),
            "node_coord": torch.FloatTensor(self.node_coord).to(device),
            "edge_weight": torch.FloatTensor(self.edge_weight).to(device),
        }

    def get_data(self):
        return {
            "dimension": self.dimension,
            "capacity": self.capacity,
            "num_vehicles": self.num_vehicles,
            "max_travel_time": self.max_travel_time,
            "depot": self.depot,
            "demand": self.demand,
            "time_windows": self.time_windows,
            "service_times": self.service_times,
            "travel_times": self.travel_times,
            "node_coord": self.node_coord,
            "edge_weight": self.edge_weight,
        }

    def print_details(self):
        """Print detailed information"""
        print(f"Dimension: {self.dimension}")
        print(f"Capacity: {self.capacity}")
        print(f"Number of Vehicles: {self.num_vehicles}")
        print(f"Max Travel Time: {self.max_travel_time}")
        print(f"Depot: {self.depot}")
        print("Demand:", self.demand)
        print("Time Windows:\n", self.time_windows)
        print("Service Times:", self.service_times)
        print("Travel Times:\n", self.travel_times)
        print("Node Coordinates:\n", self.node_coord)
        print("Edge Weight:\n", self.edge_weight)

    def plot_nodes(self):
        """Visualize node coordinates"""
        plt.figure(figsize=(8, 8))
        plt.scatter(
            self.node_coord[:, 0], self.node_coord[:, 1], c="blue", label="Nodes"
        )
        plt.scatter(
            self.node_coord[self.depot, 0],
            self.node_coord[self.depot, 1],
            c="red",
            label="Depot",
            marker="x",
        )
        plt.title("Node Coordinates")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()


class BaseVRPTWGenerator:
    SERVICE_TIME_MIN = 0.05
    SERVICE_TIME_MAX = 0.1

    def __init__(
        self,
        dimension,
        capacity,
        max_demand,
        num_vehicles,
        min_window_width,
        max_window_width,
        max_travel_time,
        unit_cost=1.0,
    ):
        assert dimension > 0, "Dimension must be positive"
        assert capacity > 0, "Capacity must be positive"
        assert max_demand <= capacity, "Maximum demand cannot exceed capacity"
        self.dimension = dimension
        self.capacity = capacity
        self.max_demand = max_demand
        self.num_vehicles = num_vehicles
        self.max_travel_time = max_travel_time
        self.min_window_width = min_window_width
        self.max_window_width = max_window_width
        self.unit_cost = unit_cost
        self.depot = 0

    def _calculate_edge_weight(self, node_coord):
        # Edge weight calculation logic
        diff = node_coord[:, np.newaxis, :] - node_coord[np.newaxis, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))


class VRPTWGeneratorR(BaseVRPTWGenerator):
    def __init__(
        self,
        dimension: int,
        capacity: int,
        max_demand: int,
        num_vehicles: int,
        min_window_width: int,
        max_window_width: int,
        max_travel_time: int,
        unit_cost: float = 1.0,
    ):
        super().__init__(
            dimension,
            capacity,
            max_demand,
            num_vehicles,
            min_window_width,
            max_window_width,
            max_travel_time,
            unit_cost,
        )

    def generate(
        self, seed: Optional[int] = None, is_window: bool = True
    ) -> VRPTWInstance:
        if seed is not None:
            np.random.seed(seed)
        # Generate node coordinates
        node_coord = np.random.uniform(0, 1, size=(self.dimension, 2))
        # Depot location is set to the weighted center of all customer points
        node_coord[0] = np.mean(node_coord[1:], axis=0)

        # Calculate edge weights (Euclidean distance)
        edge_weight = self._calculate_edge_weight(node_coord)

        # Calculate travel times
        travel_times = edge_weight * self.unit_cost
        # Normalize
        travel_times /= float(self.max_travel_time)

        # Generate demands
        demand = np.random.randint(1, self.max_demand, size=self.dimension)
        demand[self.depot] = 0
        demand = demand.astype(np.float32)
        # Normalize
        demand /= float(self.capacity)

        # Maximum capacity normalization
        capacity = 1.0

        # Generate service times
        service_times = np.random.uniform(
            self.SERVICE_TIME_MIN, self.SERVICE_TIME_MAX, size=self.dimension
        )
        service_times[self.depot] = 0

        # Generate time windows
        time_windows = np.zeros((self.dimension, 2))
        if is_window:
            for i in range(1, self.dimension):
                t_start = np.random.uniform(
                    0, self.max_travel_time - self.max_window_width
                )
                t_end = t_start + np.random.uniform(
                    self.min_window_width, self.max_window_width
                )
                time_windows[i] = [t_start, t_end]
            time_windows[self.depot] = [0, self.max_travel_time]
            # Normalize
            time_windows /= float(self.max_travel_time)
        else:
            time_windows[:] = [0, 1]
            time_windows[self.depot] = [0, 1]
        # Maximum travel time normalization
        max_travel_time = 1.0

        return VRPTWInstance(
            dimension=self.dimension,
            capacity=capacity,
            num_vehicles=self.num_vehicles,
            max_travel_time=max_travel_time,
            depot=self.depot,
            demand=demand,
            time_windows=time_windows,
            service_times=service_times,
            travel_times=travel_times,
            node_coord=node_coord,
            edge_weight=edge_weight,
        )


class VRPTWGeneratorC(BaseVRPTWGenerator):
    def __init__(
        self,
        dimension: int,
        capacity: int,
        max_demand: int,
        num_vehicles: int,
        min_window_width: int,
        max_window_width: int,
        max_travel_time: int,
        unit_cost: float = 1.0,
    ):
        super().__init__(
            dimension,
            capacity,
            max_demand,
            num_vehicles,
            min_window_width,
            max_window_width,
            max_travel_time,
            unit_cost,
        )
        self.n_clusters = max(3, self.dimension // 5)
        self.min_cluster_distance = 0.2  # Minimum cluster center distance

    def generate(
        self, seed: Optional[int] = None, is_window: bool = True
    ) -> VRPTWInstance:
        """Generate C-type (clustered distribution) instance
        Args:
            n_clusters: Number of clusters
            seed: Random seed
            is_window: Whether to generate time windows
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate cluster centers
        cluster_centers = np.random.uniform(0.2, 0.8, size=(self.n_clusters, 2))

        # 2. Generate node coordinates (based on clusters)
        node_coord = np.zeros((self.dimension, 2))

        # Assign clusters to each node
        nodes_per_cluster = (self.dimension - 1) // self.n_clusters
        remaining_nodes = (self.dimension - 1) % self.n_clusters

        customer_nodes = []
        current_idx = 1  # Start from 1 (skip depot)
        for i in range(self.n_clusters):
            cluster_size = nodes_per_cluster + (1 if i < remaining_nodes else 0)
            cluster_nodes = np.random.normal(
                loc=cluster_centers[i], scale=0.02, size=(cluster_size, 2)
            )
            cluster_nodes = np.clip(cluster_nodes, 0, 1)
            customer_nodes.extend(cluster_nodes)
            node_coord[current_idx : current_idx + cluster_size] = cluster_nodes
            current_idx += cluster_size

        node_coord[0] = np.mean(customer_nodes, axis=0)

        # 3. Calculate edge weights (Euclidean distance)
        edge_weight = self._calculate_edge_weight(node_coord)

        # 4. Calculate travel times
        travel_times = edge_weight * self.unit_cost
        travel_times /= float(self.max_travel_time)

        # 5. Generate demands (based on clusters)
        demand = np.zeros(self.dimension)
        current_idx = 1
        for i in range(self.n_clusters):
            if i < remaining_nodes:
                cluster_size = nodes_per_cluster + 1
            else:
                cluster_size = nodes_per_cluster

            # Fix: ensure base demand is at least 2 to avoid range overlap
            base_demand = np.random.randint(2, self.max_demand)

            # Calculate demand range
            min_demand = max(1, int(0.8 * base_demand))
            max_demand = max(
                min_demand + 1, min(self.max_demand, int(1.2 * base_demand))
            )

            # Generate cluster demands
            cluster_demand = np.random.randint(
                min_demand, max_demand, size=cluster_size
            )

            demand[current_idx : current_idx + cluster_size] = cluster_demand
            current_idx += cluster_size

        demand[self.depot] = 0
        demand = demand.astype(np.float32)
        demand /= float(self.capacity)

        # 6. Generate service times (based on clusters)
        service_times = np.zeros(self.dimension)
        current_idx = 1
        for i in range(self.n_clusters):
            if i < remaining_nodes:
                cluster_size = nodes_per_cluster + 1
            else:
                cluster_size = nodes_per_cluster

            # Service times within the same cluster are similar
            base_service = np.random.uniform(
                self.SERVICE_TIME_MIN, self.SERVICE_TIME_MAX
            )
            cluster_service = np.random.uniform(
                0.9 * base_service, 1.1 * base_service, size=cluster_size
            )

            service_times[current_idx : current_idx + cluster_size] = cluster_service
            current_idx += cluster_size

        service_times[self.depot] = 0

        # 7. Generate time windows (based on clusters)
        time_windows = np.zeros((self.dimension, 2))
        if is_window:
            current_idx = 1
            for i in range(self.n_clusters):
                if i < remaining_nodes:
                    cluster_size = nodes_per_cluster + 1
                else:
                    cluster_size = nodes_per_cluster

                # Time windows within the same cluster are similar
                base_start = np.random.uniform(
                    0, self.max_travel_time - self.max_window_width
                )
                cluster_starts = np.random.uniform(
                    max(0, base_start - 0.1 * self.max_travel_time),
                    min(
                        self.max_travel_time - self.max_window_width,
                        base_start + 0.1 * self.max_travel_time,
                    ),
                    size=cluster_size,
                )

                for j in range(cluster_size):
                    t_start = cluster_starts[j]
                    t_end = t_start + np.random.uniform(
                        self.min_window_width, self.max_window_width
                    )
                    time_windows[current_idx + j] = [t_start, t_end]

                current_idx += cluster_size

            time_windows[self.depot] = [0, self.max_travel_time]
            time_windows /= float(self.max_travel_time)
        else:
            time_windows[:] = [0, 1]
            time_windows[self.depot] = [0, 1]

        # Normalize
        capacity = 1.0
        max_travel_time = 1.0

        return VRPTWInstance(
            dimension=self.dimension,
            capacity=capacity,
            num_vehicles=self.num_vehicles,
            max_travel_time=max_travel_time,
            depot=self.depot,
            demand=demand,
            time_windows=time_windows,
            service_times=service_times,
            travel_times=travel_times,
            node_coord=node_coord,
            edge_weight=edge_weight,
        )


if __name__ == "__main__":
    # Create generators
    gr = VRPTWGeneratorR(
        dimension=11,
        capacity=20,
        max_demand=8,
        num_vehicles=10,
        min_window_width=2,
        max_window_width=8,
        max_travel_time=48,
    )
    gc = VRPTWGeneratorC(
        dimension=11,
        capacity=20,
        max_demand=8,
        num_vehicles=10,
        min_window_width=2,
        max_window_width=8,
        max_travel_time=48,
    )
    # Generate instances
    time_start = time.time()
    instance = gr.generate()
    time_end = time.time()
    print(f"R-type instance generation time: {time_end - time_start} seconds")
    # instance.print_details()
    instance.plot_nodes()
    time_start = time.time()
    c_instance = gc.generate()
    time_end = time.time()
    print(f"C-type instance generation time: {time_end - time_start} seconds")
    # c_instance.print_details()
    c_instance.plot_nodes()
