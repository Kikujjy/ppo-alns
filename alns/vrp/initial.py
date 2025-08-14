import numpy as np
from .solution import CvrpState, CvrptwState
from .repairopt import is_time_feasible


def neighbors(edge_weight, customer):
    """
    Return the nearest neighbors of the customer, excluding the depot.
    """
    locations = np.argsort(edge_weight[customer])
    return locations[locations != 0]


def nearest_neighbor(dimension, demand, capacity, edge_weight):
    """
    Build a solution by iteratively constructing routes, where the nearest
    customer is added until the route has met the vehicle capacity limit.
    """
    routes = []
    unvisited = set(range(1, dimension))

    while unvisited:
        route = [0]  # Start at the depot
        route_demands = 0

        while unvisited:
            # Add the nearest unvisited customer to the route till max capacity
            current = route[-1]
            nearest = [nb for nb in neighbors(edge_weight, current) if nb in unvisited][
                0
            ]

            if route_demands + demand[nearest] > capacity:
                break

            route.append(nearest)
            unvisited.remove(nearest)
            route_demands += demand[nearest]

        customers = route[1:]  # Remove the depot
        routes.append(customers)

    return CvrpState(routes)


def neighbors_tw(travel_times, time_windows, current_node, current_time):
    """
    Return feasible neighboring nodes of current node considering time window constraints

    Args:
        travel_times: Travel time matrix
        time_windows: Time window constraints
        current_node: Current node
        current_time: Current time
    """
    # Get all locations sorted by distance
    locations = np.argsort(travel_times[current_node])

    # Filter out depot and nodes that don't satisfy time window
    feasible = []
    for next_node in locations[locations != 0]:
        arrival_time = current_time + travel_times[current_node][next_node]
        earliest, latest = time_windows[next_node]
        # Wait if arrive early
        if arrival_time < earliest:
            arrival_time = earliest
        # If can arrive before the latest arrival time, it's a feasible node
        if arrival_time <= latest:
            feasible.append(next_node)

    return feasible


def nearest_neighbor_tw(data):
    """
    Nearest neighbor construction algorithm considering time window constraints
    """
    dimension = data["dimension"]
    demand = data["demand"]
    capacity = data["capacity"]
    travel_times = data["travel_times"]
    time_windows = data["time_windows"]
    service_times = data["service_times"]

    routes = []
    unvisited = set(range(1, dimension))

    while unvisited:
        route = [0]  # Start from depot
        route_demands = 0
        current_time = 0  # Current route time

        while unvisited:
            current = route[-1]

            # Get all feasible neighboring nodes that satisfy time window constraints
            feasible_neighbors = neighbors_tw(
                travel_times, time_windows, current, current_time
            )

            # Filter out visited nodes and nodes exceeding capacity
            feasible_neighbors = [
                nb
                for nb in feasible_neighbors
                if nb in unvisited and route_demands + demand[nb] <= capacity
            ]

            if not feasible_neighbors:
                break

            # Select the nearest feasible node
            nearest = feasible_neighbors[0]

            # Update arrival time
            travel_time = travel_times[current][nearest]
            arrival_time = current_time + travel_time

            # Wait if arrive early
            earliest = time_windows[nearest][0]
            if arrival_time < earliest:
                arrival_time = earliest

            # Update current time (add service time)
            current_time = arrival_time + service_times[nearest]

            # Check if can return to depot on time
            if current_time + travel_times[nearest][0] > time_windows[0][1]:
                # Exceeds maximum travel time, abandon current customer
                break

            # Update route
            route.append(nearest)
            unvisited.remove(nearest)
            route_demands += demand[nearest]

        # If route is not empty, add to routes
        if len(route) > 1:
            routes.append(route[1:])

    return CvrptwState(routes, travel_times)


def clarke_wright_tw(data):
    """
    VRPTW implementation of Clarke-Wright savings algorithm
    """
    dimension = data["dimension"]
    demand = data["demand"]
    capacity = data["capacity"]
    travel_times = data["travel_times"]

    # 1. Initialize: create independent route for each customer
    routes = [[i] for i in range(1, dimension)]

    # 2. Calculate savings values
    savings = []
    for i in range(1, dimension):
        for j in range(i + 1, dimension):
            # saving = cost(0,i) + cost(0,j) - cost(i,j)
            saving = travel_times[0][i] + travel_times[0][j] - travel_times[i][j]
            savings.append((saving, i, j))

    # Sort by savings values in descending order
    savings.sort(reverse=True)

    # 3. Merge routes based on savings values
    # Record route index for each customer
    customer_route = {i: idx for idx, route in enumerate(routes) for i in route}

    for saving, i, j in savings:
        if saving <= 0:
            break

        # Find routes containing i and j
        route_i = customer_route.get(i)
        route_j = customer_route.get(j)

        # If i and j are already in the same route or a point is not in any route
        if route_i is None or route_j is None or route_i == route_j:
            continue

        # Get two routes
        path_i = routes[route_i]
        path_j = routes[route_j]

        # Check if merged route satisfies capacity constraints
        total_demand = sum(demand[k] for k in path_i + path_j)
        if total_demand > capacity:
            continue

        # Try different merge methods and check time window constraints
        merged = None
        if (
            path_i[-1] == i and path_j[0] == j
        ):  # i at end of route i, j at start of route j
            merged = path_i + path_j
        elif (
            path_i[0] == i and path_j[-1] == j
        ):  # i at start of route i, j at end of route j
            merged = path_j + path_i

        if merged and is_time_feasible(merged, data):
            # Update route
            routes[route_i] = merged
            routes[route_j] = []
            # Update customer route indices
            for customer in merged:
                customer_route[customer] = route_i
            customer_route[j] = None

    # Remove empty routes and return results
    return CvrptwState([r for r in routes if r], travel_times)
