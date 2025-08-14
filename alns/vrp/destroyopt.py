import numpy as np
from .initial import neighbors


def get_destroy_params(dimension):
    """
    Set hyperparameters for destroy operators based on problem scale

    Parameters:
        dimension: Number of problem nodes (including depot)
    Returns:
        Dictionary containing various hyperparameters
    """
    # Actual number of customers (excluding depot)
    n_customers = dimension - 1

    # Base destruction ratio (percentage of total customers)
    if n_customers <= 20:
        base_destroy_ratio = 0.2  # Small-scale problems can destroy larger proportion
    elif n_customers <= 50:
        base_destroy_ratio = 0.15
    elif n_customers <= 100:
        base_destroy_ratio = 0.1
    else:
        base_destroy_ratio = 0.05  # Large-scale problems need smaller destruction ratio

    params = {
        # Random Removal
        "random_removal": {
            "min_destruction": base_destroy_ratio * 0.5,
            "max_destruction": base_destroy_ratio,
        },
        # String Removal
        "string_removal": {
            "max_string_removals": max(2, int(n_customers * 0.04)),  # Minimum 2 routes
            "max_string_size": max(4, int(n_customers * 0.12)),  # Minimum 4 customers
        },
        # Related Removal
        "related_removal": {
            "min_removals": max(3, int(n_customers * base_destroy_ratio * 0.5)),
            "max_removals": max(5, int(n_customers * base_destroy_ratio)),
        },
        # Worst Removal
        "worst_removal": {
            "min_removals": max(3, int(n_customers * base_destroy_ratio * 0.5)),
            "max_removals": max(5, int(n_customers * base_destroy_ratio)),
            "randomization_factor": 0.6
            if n_customers < 50
            else 0.4,  # Large-scale problems need more randomness
        },
        # sequence removal
        "sequence_removal": {
            "min_seq_len": max(3, int(n_customers * 0.1)),
            "max_seq_len": max(5, int(n_customers * 0.2)),
        },
    }

    return params


def remove_empty_routes(state):
    """
    Remove empty routes after applying the destroy operator.
    """
    state.routes = [route for route in state.routes if len(route) != 0]
    return state


def create_random_customer_removal_operator(data):
    params = get_destroy_params(data["dimension"])

    def random_removal(state, rng):
        """
        Removes a number of randomly selected customers from the passed-in solution.
        """
        destroyed = state.copy()

        degree_of_destruction = rng.uniform(
            params["random_removal"]["min_destruction"],
            params["random_removal"]["max_destruction"],
        )

        customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)

        for customer in rng.choice(
            range(1, data["dimension"]), customers_to_remove, replace=False
        ):
            destroyed.unassigned.append(customer)
            route = destroyed.find_route(customer)
            route.remove(customer)

        return remove_empty_routes(destroyed)

    return random_removal


def create_string_removal_operator(data):
    params = get_destroy_params(data["dimension"])
    MAX_STRING_SIZE = params["string_removal"]["max_string_size"]
    MAX_STRING_REMOVALS = params["string_removal"]["max_string_removals"]

    def string_removal(state, rng):
        """
        Remove partial routes around a randomly chosen customer.
        """
        destroyed = state.copy()

        avg_route_size = int(np.mean([len(route) for route in state.routes]))
        max_string_size = max(MAX_STRING_SIZE, avg_route_size)
        max_string_removals = min(len(state.routes), MAX_STRING_REMOVALS)

        destroyed_routes = []
        center = rng.integers(1, data["dimension"])

        for customer in neighbors(data["edge_weight"], center):
            if len(destroyed_routes) >= max_string_removals:
                break

            if customer in destroyed.unassigned:
                continue

            route = destroyed.find_route(customer)
            if route in destroyed_routes:
                continue

            customers = remove_string(route, customer, max_string_size, rng)
            destroyed.unassigned.extend(customers)
            destroyed_routes.append(route)

        return remove_empty_routes(destroyed)

    def remove_string(route, cust, max_string_size, rng):
        """
        Remove a string that constains the passed-in customer.
        """
        # Find consecutive indices to remove that contain the customer
        size = rng.integers(1, min(len(route), max_string_size) + 1)
        start = route.index(cust) - rng.integers(size)
        idcs = [idx % len(route) for idx in range(start, start + size)]

        # Remove indices in descending order
        removed_customers = []
        for idx in sorted(idcs, reverse=True):
            removed_customers.append(route.pop(idx))

        return removed_customers

    return string_removal


def create_random_route_removal_operator():
    def route_removal(state, rng):
        """
        Randomly select a route based on the inverse proportion of path length and remove all its customers.
        Shorter paths have higher probability of being selected.

        Parameters:
            state: Current solution state
            rng: Random number generator
        """
        destroyed = state.copy()

        if len(destroyed.routes) <= 1:
            return destroyed

        # Calculate the reciprocal of each route length as weight
        route_lengths = np.array([len(route) for route in destroyed.routes])
        weights = 1.0 / route_lengths
        # Normalize weights to get probabilities
        probs = weights / weights.sum()

        # Select route based on probabilities
        route_idx = rng.choice(len(destroyed.routes), p=probs)
        route = destroyed.routes[route_idx]

        # Remove all customers on this route
        destroyed.unassigned.extend(route)
        destroyed.routes.pop(route_idx)

        return remove_empty_routes(destroyed)

    return route_removal


def create_related_removal_operator(data):
    params = get_destroy_params(data["dimension"])

    def calculate_relatedness(i, j, routes, distances):
        """Calculate relatedness between customers"""
        # Normalize distance
        max_dist = np.max(distances)
        normalized_dist = distances[i][j] / max_dist

        # Check if in the same route
        same_route = 0
        for route in routes:
            if i in route and j in route:
                same_route = 1
                break

        return 1 / (
            normalized_dist + same_route + 0.1
        )  # Add 0.1 to avoid division by zero

    def related_removal(state, rng):
        """
        Removal based on relatedness
        """
        destroyed = state.copy()
        # Randomly select removal count
        num_removals = rng.integers(
            params["worst_removal"]["min_removals"],
            params["worst_removal"]["max_removals"] + 1,
        )
        # Randomly select the first customer
        initial_customer = rng.integers(1, data["dimension"])
        destroyed.unassigned.append(initial_customer)
        route = destroyed.find_route(initial_customer)
        route.remove(initial_customer)

        # Calculate and remove related customers
        remaining_customers = set(range(1, data["dimension"])) - set(
            destroyed.unassigned
        )

        while len(destroyed.unassigned) < num_removals and remaining_customers:
            # Calculate relatedness of all customers with removed customers
            relatedness = {}
            reference = rng.choice(destroyed.unassigned)

            for customer in remaining_customers:
                relatedness[customer] = calculate_relatedness(
                    reference, customer, destroyed.routes, data["edge_weight"]
                )

            # Select customer with highest relatedness
            most_related = max(relatedness.items(), key=lambda x: x[1])[0]
            destroyed.unassigned.append(most_related)
            route = destroyed.find_route(most_related)
            route.remove(most_related)
            remaining_customers.remove(most_related)

        return remove_empty_routes(destroyed)

    return related_removal


def create_worst_removal_operator(data):
    params = get_destroy_params(data["dimension"])

    def calculate_cost_change(route, customer, distances):
        """Calculate cost change after removing a customer"""
        if len(route) <= 1:
            return 0

        idx = route.index(customer)
        prev_idx = (idx - 1) % len(route)
        next_idx = (idx + 1) % len(route)

        # Current cost
        current_cost = (
            distances[route[prev_idx]][customer] + distances[customer][route[next_idx]]
        )

        # Cost after removal
        new_cost = distances[route[prev_idx]][route[next_idx]]

        return new_cost - current_cost

    def worst_removal(state, rng):
        """
        Removal based on importance (cost)

        Parameters:
            state: Current solution state
            rng: Random number generator
            num_removals: Number of customers to remove
            randomization_factor: Randomization factor for introducing randomness in worst choices
        """
        destroyed = state.copy()
        # Randomly select removal count
        num_removals = rng.integers(
            params["worst_removal"]["min_removals"],
            params["worst_removal"]["max_removals"] + 1,
        )
        randomization_factor = params["worst_removal"]["randomization_factor"]
        for _ in range(num_removals):
            if not any(route for route in destroyed.routes if len(route) > 0):
                break

            # Calculate removal cost for all customers
            cost_changes = {}
            for route in destroyed.routes:
                for customer in route:
                    if customer == 0:  # Skip depot
                        continue
                    cost_change = calculate_cost_change(
                        route, customer, data["edge_weight"]
                    )
                    cost_changes[customer] = cost_change

            if not cost_changes:
                break

            # Sort customers by cost
            sorted_customers = sorted(
                cost_changes.items(), key=lambda x: x[1], reverse=True
            )  # Cost in descending order

            # Select customer using randomization factor
            pos = min(
                int(rng.random() ** randomization_factor * len(sorted_customers)),
                len(sorted_customers) - 1,
            )
            customer_to_remove = sorted_customers[pos][0]

            # Remove selected customer
            destroyed.unassigned.append(customer_to_remove)
            route = destroyed.find_route(customer_to_remove)
            route.remove(customer_to_remove)

        return remove_empty_routes(destroyed)

    return worst_removal


def create_sequence_removal_operator(data):
    params = get_destroy_params(data["dimension"])

    def sequence_removal(state, rng):
        """
        Randomly select a continuous sequence from concatenated routes for removal

        Parameters:
            state: Current solution state
            rng: Random number generator
        """
        destroyed = state.copy()

        # Concatenate all routes into one large sequence (excluding depot)
        concatenated = []
        for route in destroyed.routes:
            concatenated.extend([c for c in route if c != 0])

        if not concatenated:
            return destroyed

        # Determine length of removal sequence
        seq_length = rng.integers(
            params["sequence_removal"]["min_seq_len"],
            params["sequence_removal"]["max_seq_len"],
        )

        # Randomly select starting position
        start_pos = rng.integers(0, len(concatenated))

        # Get customer sequence to remove
        customers_to_remove = set()
        for i in range(seq_length):
            idx = (start_pos + i) % len(concatenated)
            customers_to_remove.add(concatenated[idx])

        # Remove selected customers from routes
        for customer in customers_to_remove:
            route = destroyed.find_route(customer)
            route.remove(customer)
            destroyed.unassigned.append(customer)

        return remove_empty_routes(destroyed)

    return sequence_removal
