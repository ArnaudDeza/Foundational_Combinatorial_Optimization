#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import time
import os 
import glob 
import pickle
from pathlib import Path
import sys

# Add the project root to the Python path
# This allows for absolute imports from the 'src' directory
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from joblib import Parallel, delayed
from tqdm import tqdm

# Import solvers and heuristics for solving QAPs
from src.qap.solvers import solve_one_qap



def safe_solve_one_qap(i: int, instance: dict, time_limit: int, pickle_file: str) -> dict:
    """
    Wrapper around solve_one_qap that catches exceptions.
    
    Args:
        i (int): The instance index.
        instance (dict): QAP instance.
        time_limit (int): Time limit for solve.
        pickle_file (str): Source file name (for logging).
    
    Returns:
        dict: The result dictionary (with error values if an exception occurred).
    """
    try:
        return solve_one_qap(instance, time_limit)
    except Exception as exc:
        logging.error(f"QAP instance {i} in {pickle_file} generated an exception: {exc}")
        return {
            "F": instance.get("F"),
            "positions": instance.get("positions"),
            "D": None,
            "objVal": None,
            "mipgap": None,
            "runtime": None,
            "solution": None,
            "time_limit": time_limit,
        }


def process_qap_file(pickle_file: str, time_limit: int, max_workers: int, gurobi_threads_per_solve: int):
    """
    Load a pickle file containing QAP instances, solve them in parallel,
    and save the results to a new pickle file.
    
    Args:
        pickle_file (str): Path to the pickle file with QAP instances.
        time_limit (int): Time limit per instance solve (seconds).
        max_workers (int): Number of parallel jobs.
        gurobi_threads_per_solve (int): Gurobi threads allocated per solve (for logging purposes).
    """
    logging.info(f"Loading QAP instances from {pickle_file}")
    with open(pickle_file, "rb") as f:
        qap_list = pickle.load(f)
    logging.info(f"Loaded {len(qap_list)} QAP instances from {pickle_file}")

    # Use Joblib's Parallel with a tqdm progress bar.
    results = Parallel(n_jobs=max_workers)(
        delayed(safe_solve_one_qap)(i, instance, time_limit, pickle_file)
        for i, instance in enumerate(tqdm(qap_list, desc="Solving QAPs", dynamic_ncols=True))
    )

    # Save results to a new pickle file.
    base_name = os.path.basename(pickle_file)
    prefix = base_name.replace(".pkl", "")
    out_file = os.path.join(os.path.dirname(pickle_file), f"{prefix}_results.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Saved results for {pickle_file} to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Parallel QAP Solver using Gurobi and Joblib")
    parser.add_argument(
        "--pickle_files", 
        type=str, 
        nargs="+", 
        required=True, 
        help="List of pickle files containing QAP instances."
    )
    parser.add_argument(
        "--time_limit", 
        type=int, 
        default=3600, 
        help="Time limit (in seconds) for each QAP instance solve."
    )
    parser.add_argument(
        "--n_threads", 
        type=int, 
        default=24, 
        help="Total number of CPU cores available."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    ) 
    # Decide how many parallel jobs to run.
    # For example, if each Gurobi solve uses 2 threads, run (n_threads / 2) solves in parallel.
    gurobi_threads_per_solve = 6
    max_workers = args.n_threads // gurobi_threads_per_solve
    logging.info(
        f"Solving in parallel with {max_workers} workers, "
        f"each using {gurobi_threads_per_solve} Gurobi threads."
    )

    # Process each pickle file.
    for pickle_file in args.pickle_files:
        process_qap_file(pickle_file, args.time_limit, max_workers, gurobi_threads_per_solve)


if __name__ == "__main__":
    main()