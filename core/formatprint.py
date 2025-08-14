from typing import List, Any


def format_vrptw_routes(
    routes: List[List[int]], depot: int = 0, separator: str = " -> "
) -> str:
    """Format VRPTW problem route results

    Args:
        routes: Route list, each sublist represents a route
        depot: Distribution center number, default is 0
        separator: Route separator, default is " -> "

    Returns:
        str: Formatted route string

    Example:
        >>> routes = [[1, 2, 3], [4, 5, 6]]
        >>> print(format_vrptw_routes(routes))
        Route1: 0 -> 1 -> 2 -> 3 -> 0
        Route2: 0 -> 4 -> 5 -> 6 -> 0
    """
    formatted_routes = []
    for i, route in enumerate(routes, 1):
        # Add starting depot
        full_route = [depot] + route + [depot]
        # Convert numbers to strings and join with separator
        route_str = separator.join(str(node) for node in full_route)
        # Add route number
        formatted_route = f"Route{i}: {route_str}"
        formatted_routes.append(formatted_route)

    # Return all route combinations, one route per line
    return "\n".join(formatted_routes)
