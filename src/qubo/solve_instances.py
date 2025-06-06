#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import time
import os 
import glob 
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm


# Import solvers and heuristics for solving QUBOs
from src.qubo.solvers import solve_one_qubo__GUROBI_OPTIMODS, solve_one_qubo__GUROBI
from src.qubo.local_search import simulated_annealing, tabu_search
 
def safe_solve_one_qubo(i, Q, time_limit, npy_file, max_threads, solver_type, solver_args):
    """
    Wrapper around solve_one_qubo that catches exceptions.
    """
    try:
        if solver_type == "GUROBI_OPTIMODS":
            return solve_one_qubo__GUROBI_OPTIMODS(Q, time_limit)
        elif solver_type == "GUROBI":
            return solve_one_qubo__GUROBI(Q, time_limit, max_threads)
        elif solver_type == "TABU_SEARCH":
            return tabu_search(Q, time_limit=time_limit, **solver_args)
        elif solver_type == "SIMULATED_ANNEALING":
            return simulated_annealing(Q, time_limit=time_limit, **solver_args)
        else:
            raise ValueError(f"Invalid solver type: {solver_type}")
    except Exception as exc:
        logging.error(f"QUBO {i} in {npy_file} generated an exception: {exc}")
        return {"Q": Q, "objective": None, "solution": None, "time": None,"time_limit":time_limit}

def process_qubo_file(npy_file, time_limit, max_workers, gurobi_threads_per_solve, solver_type, solver_args):
    """
    Loads a .npy file of QUBOs, solves them in parallel using Joblib, and saves results.
    
    Parameters
    ----------
    npy_file : str
        Path to a .npy file containing QUBO matrices.
    time_limit : int
        Time limit (seconds) for each QUBO solve.
    max_workers : int
        Number of parallel jobs.
    gurobi_threads_per_solve : int
        Number of Gurobi threads allocated per solve.
    solver_type : str
        The type of solver to use for the QUBO instances.
    """
    logging.info(f"Loading QUBOs from {npy_file}")
    qubo_list = np.load(npy_file, allow_pickle=True)
    logging.info(f"Loaded {len(qubo_list)} QUBOs from {npy_file}")

    # Use Joblib's Parallel with a progress bar from tqdm
    results = Parallel(n_jobs=max_workers)(
        delayed(safe_solve_one_qubo)(i, Q, time_limit, npy_file, gurobi_threads_per_solve, solver_type, solver_args)
        for i, Q in enumerate(tqdm(qubo_list, desc="Solving QUBOs", dynamic_ncols=True))
    )

    # Save results in a file named <original_name>_results.npy
    base_name = os.path.basename(npy_file)  # e.g. QUBOs_chunk_3.npy
    prefix = base_name.replace(".npy", "")
    out_file = os.path.join(os.path.dirname(npy_file), f"{prefix}_results.npy")
    np.save(out_file, results)
    logging.info(f"Saved results for {npy_file} to {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npy_files", 
        type=str, 
        nargs="+", 
        required=True, 
        help="List of .npy files containing QUBO matrices."
    )
    parser.add_argument(
        "--time_limit", 
        type=int, 
        default=600, 
        help="Time limit for each instance (seconds)."
    )
    parser.add_argument(
        "--n_threads", 
        type=int, 
        default=24, 
        help="Total number of CPU cores available."
    )
    parser.add_argument(
        "--solver_type",
        type=str,
        default="GUROBI",
        choices=["GUROBI", "GUROBI_OPTIMODS", "TABU_SEARCH", "SIMULATED_ANNEALING"],
        help="Which solver to use."
    )
    parser.add_argument(
        "--gurobi_threads_per_solve",
        type=int,
        default=2,
        help="Number of Gurobi threads allocated per solve."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    solver_args = {}
    if args.solver_type == "TABU_SEARCH":
        # Add tabu search specific arguments here
        # For now, using placeholders
        solver_args['n_iter'] = 1000
        solver_args['tabu_tenure'] = 10
    elif args.solver_type == "SIMULATED_ANNEALING":
        # Add simulated annealing specific arguments here
        # For now, using placeholders
        solver_args['n_iter'] = 1000
        solver_args['start_temp'] = 1.0
        solver_args['end_temp'] = 0.01

    # Decide how many parallel jobs to run:
    # For instance, if each Gurobi solve uses 2 threads, run (n_threads / 2) solves in parallel.
    gurobi_threads_per_solve = args.gurobi_threads_per_solve
    max_workers = args.n_threads // gurobi_threads_per_solve
    logging.info(
        f"Will solve in parallel with {max_workers} workers, "
        f"each using {gurobi_threads_per_solve} Gurobi threads."
    )

    # Process each npy file in the list
    for npy_file in args.npy_files:
        process_qubo_file(
            npy_file=npy_file, 
            time_limit=args.time_limit, 
            max_workers=max_workers, 
            gurobi_threads_per_solve=gurobi_threads_per_solve,
            solver_type=args.solver_type,
            solver_args=solver_args
        )

if __name__ == "__main__": 

    main()
