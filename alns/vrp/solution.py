import copy


class CvrpState:
    """
    Solution state for CVRP. It has two data members, routes and unassigned.
    Routes is a list of list of integers, where each inner list corresponds to
    a single route denoting the sequence of customers to be visited. A route
    does not contain the start and end depot. Unassigned is a list of integers,
    each integer representing an unassigned customer.
    """

    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        return CvrpState(copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        """
        Computes the total route costs.
        """
        return sum(route_cost(route) for route in self.routes)

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for route in self.routes:
            if customer in route:
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")


class CvrptwState:
    """
    Solution state for CVRPTW (VRP with Time Windows).
    Extends CVRP by adding time window constraints.
    Routes is a list of list of integers, where each inner list corresponds to
    a single route denoting the sequence of customers to be visited.
    """

    def __init__(self, routes, edge_weight, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []
        self.edge_weight = edge_weight

    def copy(self):
        return CvrptwState(
            copy.deepcopy(self.routes), self.edge_weight, self.unassigned.copy()
        )

    def objective(self):
        """
        Computes the total route costs including time window violations.
        """
        return sum(route_cost(self.edge_weight, route) for route in self.routes)

    def is_feasible(self, data):
        """
        Check if solution satisfies time window constraints
        """
        for route in self.routes:
            if not self._check_route_feasibility(data, route):
                return False
        return True

    def _check_route_feasibility(self, data, route):
        """
        Check if single route satisfies time window constraints
        """
        current_time = 0
        prev_node = 0  # Start from depot

        # Check if route satisfies capacity constraints
        # total_load = data["demand"][route].sum()
        total_load = sum(data["demand"][customer] for customer in route)

        if total_load > data["capacity"] + 1e-6:
            print(f"Route: {route}")
            print(f"Total load: {total_load}")
            print(f"Capacity: {data['capacity']}")
            return False

        for customer in route:
            # Add travel time
            travel_time = data["travel_times"][prev_node][customer]
            current_time += travel_time

            # Check if arrives within time window
            earliest = data["time_windows"][customer][0]
            latest = data["time_windows"][customer][1]

            # Wait if arrive early
            if current_time < earliest:
                current_time = earliest

            # Infeasible if arrive late
            if current_time > latest:
                # Print customers violating time window constraints
                print(f"Route: {route}")
                print(f"Customer {customer} violates time window at {current_time}")
                print(f"Time window: {earliest} - {latest}")
                return False

            # Add service time
            current_time += data["service_times"][customer]
            prev_node = customer

        # Check if can return to depot on time
        travel_time = data["travel_times"][prev_node][0]
        current_time += travel_time

        return current_time <= data["time_windows"][0][1]

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for route in self.routes:
            if customer in route:
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")


def route_cost(edge_weight, route):
    """
    Calculate route cost, including distance cost and time window violation penalty
    """
    # Basic distance cost
    distances = edge_weight
    tour = [0] + route + [0]
    distance_cost = sum(
        distances[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1)
    )
    return distance_cost
