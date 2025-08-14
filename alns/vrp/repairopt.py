import random
import numpy as np


def is_time_feasible(route, data):
    """Check time window feasibility of route"""
    current_time = 0
    prev_node = 0
    travel_times = data["travel_times"]
    time_windows = data["time_windows"]
    service_times = data["service_times"]

    for node in route:
        # Add travel time
        current_time += travel_times[prev_node][node]
        earliest, latest = time_windows[node]
        # Wait if arrive early
        if current_time < earliest:
            current_time = earliest
        elif current_time > latest:
            return False

        # Add service time
        current_time += service_times[node]
        prev_node = node

    # Check time constraint for returning to distribution center
    current_time += travel_times[prev_node][0]
    return current_time <= time_windows[0][1]


def best_insert(customer, state, data, epsilon=0):
    """Optimized best insertion search"""
    best_cost = float("inf")
    best_route = None
    best_idx = None

    customer_demand = data["demand"][customer]

    for route in state.routes:
        # Fast capacity check
        if (
            sum(data["demand"][n] for n in route) + customer_demand
            > data["capacity"] + 1e-6
        ):
            continue

        route_len = len(route)
        route_costs = []
        for idx in range(route_len + 1):
            pred = 0 if idx == 0 else route[idx - 1]
            succ = 0 if idx == route_len else route[idx]
            route_costs.append(
                (
                    (
                        data["edge_weight"][pred][customer]
                        + data["edge_weight"][customer][succ]
                        - data["edge_weight"][pred][succ]
                    ),
                    idx,
                )
            )
        # route_costs = [(
        #     data["edge_weight"][0 if idx == 0 else route[idx-1]][customer] +
        #     data["edge_weight"][customer][0 if idx == route_len else route[idx]] -
        #     data["edge_weight"][0 if idx == 0 else route[idx-1]][0 if idx == route_len else route[idx]],
        #     idx
        # ) for idx in range(route_len + 1)]
        # Sort by cost in ascending order
        route_costs.sort(key=lambda x: x[0])
        for cost, idx in route_costs:
            if cost >= best_cost:
                break
            temp_route = route.copy()
            temp_route.insert(idx, customer)
            if is_time_feasible(temp_route, data):
                best_cost = cost
                best_route = route
                best_idx = idx
                if random.random() < epsilon:
                    return best_route, best_idx

                break

    return best_route, best_idx


def create_greedy_repair_operator(data):
    def greedy_repair(state, rng):
        """
        Inserts the unassigned customers in the best route. If there are no
        feasible insertions, then a new route is created.
        """
        rng.shuffle(state.unassigned)

        while len(state.unassigned) != 0:
            customer = state.unassigned.pop()
            route, idx = best_insert(customer, state, data=data)

            if route is not None:
                route.insert(idx, customer)
            else:
                state.routes.append([customer])
        return state

    return greedy_repair


def normalize(x, min_val, max_val):
    """Normalization function"""
    if max_val == min_val:
        return 1.0
    return (x - min_val) / (max_val - min_val)


def create_criticality_repair_operator(data):
    min_demand = min(data["demand"])
    max_demand = max(data["demand"])
    min_tw = min(data["time_windows"][1] - data["time_windows"][0])
    max_tw = max(data["time_windows"][1] - data["time_windows"][0])
    min_dist = min(data["edge_weight"][0])
    max_dist = max(data["edge_weight"][0])

    width_tw = [(latest - earliest) for earliest, latest in data["time_windows"]][1:]
    depot_dist = data["edge_weight"][0][1:]

    def calculate_importance(customer):
        """Calculate customer importance (φi)"""
        # Get demand
        demand = data["demand"][customer]
        time_window = width_tw[customer - 1]
        distance_to_depot = depot_dist[customer - 1]

        # Normalize all indicators
        norm_demand = normalize(demand, min_demand, max_demand)
        norm_time = 1 / normalize(time_window, min_tw, max_tw)
        norm_dist = normalize(distance_to_depot, min_dist, max_dist)

        # Calculate importance score
        importance = norm_demand * norm_time + norm_dist
        return importance

    # Pre-calculate and cache importance of all customers
    importance_cache = {
        customer: calculate_importance(customer)
        for customer in range(1, data["dimension"])
    }

    def criticality_repair(state, rng):
        """Repair operation based on customer importance"""
        # Sort customers in descending order of importance
        sorted_customers = sorted(
            state.unassigned, key=lambda x: importance_cache[x], reverse=True
        )

        while sorted_customers:
            customer = sorted_customers.pop()
            state.unassigned.remove(customer)

            route, idx = best_insert(customer, state, data=data, epsilon=0.01)

            if route is not None:
                route.insert(idx, customer)
            else:
                state.routes.append([customer])

        return state

    return criticality_repair


def create_regret_repair_operator(data):
    # Pre-process data to avoid repeated access
    edge_weight = data["edge_weight"]
    demand = data["demand"]
    capacity = data["capacity"]

    dim = data["dimension"]
    if dim <= 21:
        k = 2
    elif dim <= 51:
        k = 3
    else:
        k = 4

    def calculate_insertion_costs(customer, state):
        """Calculate insertion costs for customer at all feasible positions"""
        insertion_costs = []
        customer_demand = demand[customer]

        for route in state.routes:
            # Fast capacity check
            if sum(demand[n] for n in route) + customer_demand > capacity + 1e-6:
                continue

            route_len = len(route)
            for idx in range(route_len + 1):
                pred = 0 if idx == 0 else route[idx - 1]
                succ = 0 if idx == route_len else route[idx]
                # Calculate insertion cost
                cost = (
                    edge_weight[pred][customer]
                    + edge_weight[customer][succ]
                    - edge_weight[pred][succ]
                )

                # Check time window constraints
                temp_route = route.copy()
                temp_route.insert(idx, customer)
                if is_time_feasible(temp_route, data):
                    insertion_costs.append((cost, route, idx))

        # Sort by cost
        insertion_costs.sort(key=lambda x: x[0])
        return insertion_costs

    def calculate_regret_value(insertions):
        """Improved regret value calculation"""
        if not insertions:
            return -float("inf"), None, None

        best_cost, best_route, best_idx = insertions[0]

        # Use numpy for vectorized calculation
        costs = np.array([cost for cost, _, _ in insertions])
        regret = np.sum(costs[1 : min(k + 1, len(costs))] - best_cost)

        return regret, best_route, best_idx

    def regret_repair(state, rng):
        """Improved repair operation"""
        # Use set to store unassigned customers for improved lookup efficiency
        unassigned = set(state.unassigned)

        while unassigned:
            best_regret_info = {
                "regret": 0,
                "customer": None,
                "route": None,
                "position": None,
            }

            # Calculate insertion cost and regret value for each customer
            for customer in unassigned:
                insertions = calculate_insertion_costs(customer, state)

                regret, route, position = calculate_regret_value(insertions)

                if regret > best_regret_info["regret"]:
                    best_regret_info.update(
                        {
                            "regret": regret,
                            "customer": customer,
                            "route": route,
                            "position": position,
                        }
                    )

            # Insert customer with maximum regret value
            if best_regret_info["customer"] is not None:
                customer = best_regret_info["customer"]
                unassigned.remove(customer)
                best_regret_info["route"].insert(best_regret_info["position"], customer)
            elif unassigned:
                # For remaining uninsertable nodes, take one and create new route
                customer = unassigned.pop()
                state.routes.append([customer])

        state.unassigned.clear()
        return state

    return regret_repair
