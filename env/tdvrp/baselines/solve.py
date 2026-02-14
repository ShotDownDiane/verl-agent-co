
import os
from rl4co.envs.routing.tdvrp.baselines.evaluator import TDVRPEvaluator
from rl4co.envs.routing.tdvrp.baselines.heuristics import GreedyRandomized
from rl4co.envs.routing.tdvrp.baselines.sa import SimulatedAnnealing
from rl4co.envs.routing.tdvrp.baselines.aco import ACO
from rl4co.envs.routing.tdvrp.baselines.regret_k import RegretInsertion
from rl4co.envs.routing.tdvrp.baselines.ALNS import ALNS
from rl4co.envs.routing.tdvrp.baselines.grasp import TDVRPGRASPSolver
from rl4co.envs.routing.tdvrp.baselines.tabu_search import TabuSearch
from rl4co.envs.routing.tdvrp.baselines.mip_solver import MIPSolver

def solve_tdvrp_from_data(matrix, duration, time_windows, service_times, method="gr", iterations=None, ants=20, penalty_value=3.0, time_limit=None):
    evaluator = TDVRPEvaluator(
        matrix=matrix, 
        duration=duration, 
        time_windows=time_windows, 
        service_times=service_times,
        penalty_value=penalty_value
    )
    
    if method in ["gr", "greedy"]:
        iters = iterations if iterations else 100
        solver = GreedyRandomized(evaluator, k=5)
        # Note: GreedyRandomized and others might need update if they use evaluator differently
        return solver.solve(num_iterations=iters)
    
    elif method == "sa":
        iters = iterations if iterations else 10000
        # SA usually needs a good initial solution
        gr_init = GreedyRandomized(evaluator, k=3)
        gr_sol, _ = gr_init.solve(num_iterations=10)
        gr_perm = [node for route in gr_sol for node in route]
        
        solver = SimulatedAnnealing(evaluator, initial_solution=gr_perm, max_iterations=iters)
        return solver.solve()
    
    elif method == "aco":
        iters = iterations if iterations else 50
        solver = ACO(evaluator, num_ants=ants, num_iterations=iters)
        return solver.solve()
    
    elif method == "regret":
        solver = RegretInsertion(evaluator, k=3)
        return solver.solve()
    
    elif method == "alns":
        iters = iterations if iterations else 1000
        # Use GR as initial solution
        gr_init = GreedyRandomized(evaluator, k=3)
        gr_sol, _ = gr_init.solve(num_iterations=10)
        
        solver = ALNS(evaluator, max_iter=iters)
        return solver.solve(initial_solution=gr_sol)
        
    elif method == "grasp":
        iters = iterations if iterations else 100
        solver = TDVRPGRASPSolver(evaluator, max_iterations=iters, alpha=0.1, time_limit=time_limit)
        return solver.solve()

    elif method == "tabu":
        iters = iterations if iterations else 1000
        # Use GR as initial solution
        gr_init = GreedyRandomized(evaluator, k=3)
        gr_sol, _ = gr_init.solve(num_iterations=10)
        gr_perm = [node for route in gr_sol for node in route]
        
        solver = TabuSearch(evaluator, initial_solution=gr_perm, max_iterations=iters)
        return solver.solve()

    elif method == "mip":
        # Time limit defaults to 60s if not provided
        tl = time_limit if time_limit else 60
        solver = MIPSolver(evaluator, time_limit=tl, use_static_approx=True)
        return solver.solve()

    else:
        raise ValueError(f"Unknown method: {method}")
