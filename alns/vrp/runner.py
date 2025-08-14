import time
from typing import Any, Dict
import numpy as np
from alns import ALNS
from alns.accept import RecordToRecordTravel, SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations
import alns.vrp as vrp
from alns.vrp.solution import CvrptwState
from core import LightLogger, format_vrptw_routes


def alns4vrptw(
    data: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    logger: LightLogger = None,
    init: CvrptwState = None,
    ac: str = "sa",
    al_init: str = "cw",
):
    """Solve VRPTW problem using ALNS"""
    assert data is not None, "data is None"
    assert params is not None, "params is None"

    start_time = time.time()
    # Instantiate the ALNS algorithm
    alns = ALNS(np.random.default_rng(params["seed"]))

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

    for i in range(params["num_destroy"]):
        alns.add_destroy_operator(d_opt_list[i])

    for i in range(params["num_repair"]):
        alns.add_repair_operator(r_opt_list[i])

    # Solve the VRP problem using ALNS
    if init is None:
        if al_init == "cw":
            init = vrp.initial.clarke_wright_tw(data=data)
        elif al_init == "greedy":
            init = vrp.initial.nearest_neighbor_tw(data=data)

    select = RouletteWheel(
        params["roulette_wheel_scores"],
        params["roulette_wheel_decay"],
        params["num_destroy"],
        params["num_repair"],
    )
    if ac == "rrt":
        accept = RecordToRecordTravel.autofit(
            init.objective(),
            params["autofit_start_gap"],
            params["autofit_end_gap"],
            params["num_iterations"],
        )
    elif ac == "sa":
        accept = SimulatedAnnealing.autofit(
            init.objective(), 0.05, 0.50, params["num_iterations"]
        )
    stop = MaxIterations(params["num_iterations"])
    result = alns.iterate(init, select, accept, stop)

    solution = result.best_state
    objective = solution.objective()
    routes = [route for route in solution.routes if route != []]
    end_time = time.time()
    if logger is not None:
        logger.format_params(
            params, title="ALNS Algorithm Parameter Configuration", style="table"
        )
        logger.info(f"Best heuristic objective is {objective}.")
        logger.info(f"NV(number of vehicles): {len(routes)}")
        logger.info("Routes: \n%s", format_vrptw_routes(routes))
        logger.info(f"Time used: {end_time - start_time:.2f} seconds")
        logger.info("Check the solution is feasible: %s", solution.is_feasible(data))

    return result
