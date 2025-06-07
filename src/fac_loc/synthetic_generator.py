#!/usr/bin/env python3
"""
Unified synthetic generator for facility location problems.
Supports both Uncapacitated Facility Location Problem (UFLP) and P-median problems.
"""

import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import delayed
from tqdm.rich import tqdm, trange

try:
    from src.utils import ProgressParallel
    HAS_PROGRESS_PARALLEL = True
except ImportError:
    HAS_PROGRESS_PARALLEL = False
    print("Warning: ProgressParallel not available. Using sequential processing.")

# Default parameters
DEFAULT_NV = 100
DEFAULT_NC = 10 
DEFAULT_K = 5
DEFAULT_FAC_OPEN_COST = 1000


def make_facility_location_instance(config: Dict[str, Any] = None, seed: int = None) -> Dict[str, Any]:
    """
    Generate a random facility location problem instance.
    
    Parameters
    ----------
    config : Dict[str, Any], optional
        Configuration dictionary with generation parameters
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the generated instance data
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    
    if config is None:
        config = {}
    
    # Extract parameters
    problem_type = config.get("problem_type", "uflp")
    nv = config.get("nv", DEFAULT_NV)
    nc = config.get("nc", DEFAULT_NC) 
    k = config.get("k", DEFAULT_K)  # Only used for p-median
    penalty = config.get("penalty", 1000)
    fixed_cost = config.get("fixed_cost", DEFAULT_FAC_OPEN_COST)
    
    # Generate adjacency structure
    adj = np.zeros([nv * nc, 3])
    new_pairs = []
    
    for i in range(1, nv + 1):
        y = np.sort(rng.sample(range(1, nv + 1), nc))
        for j in y:
            new_pairs.append((i, j))
    
    new_pairs = np.array(new_pairs)
    adj[:, 0] = new_pairs[:, 0]
    adj[:, 1] = new_pairs[:, 1]

    # Generate costs based on method
    cost_method = config.get("cost_method", "weighted_range")
    
    if cost_method == "binary":
        # Randomly choose between 1 and 10 with weights
        adj[:, 2] = rng.choices([1, 10], weights=[3, 6], k=nc * nv)
    elif cost_method == "randint":
        # Use numpy's randint to choose a cost from 1 to 5
        adj[:, 2] = np.random.randint(low=1, high=6, size=nc * nv)
    elif cost_method == "binary100":
        # Randomly choose between 1 and 100
        adj[:, 2] = rng.choices([1, 100], k=nc * nv)
    elif cost_method == "uniform":
        # Uniform distribution between 1 and 20
        adj[:, 2] = np.random.uniform(1, 20, size=nc * nv)
    elif cost_method == "exponential":
        # Exponential distribution (scale=5, shifted to start at 1)
        adj[:, 2] = np.random.exponential(scale=5, size=nc * nv) + 1
    else:
        # Default method ("weighted_range"): weighted random choices for costs 1–10
        adj[:, 2] = rng.choices(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            weights=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            k=nc * nv,
        )

    # Make the first two columns 0-indexed
    adj[:, :2] -= 1

    # Create cost matrix
    cost_mtr = np.zeros([nv, nv]) + penalty
    for i in range(len(adj)):
        cost_mtr[int(adj[i, 1]), int(adj[i, 0])] = adj[i, 2]

    # Build instance data
    data = {
        "problem_type": problem_type,
        "fixed_costs": np.ones(nv) * fixed_cost,
        "transport_cost": cost_mtr, 
        "nv": nv,
        "nc": nc, 
        "seed": seed,     
        "cost_method": cost_method,
        "penalty": penalty,
    }
    
    # Add k parameter for p-median problems
    if problem_type in ["pmedian", "p-median"]:
        data["k"] = k
    
    return data


def generate_instance_batch(
    num_instances: int,
    config: Dict[str, Any],
    output_dir: str,
    sample_seed_start: int = 1,
    partition: Optional[int] = None,
    parallel: bool = True,
    parallel_kwargs: Optional[Dict[str, Any]] = None,
    save_format: str = "pickle"
) -> List[Dict[str, Any]]:
    """
    Generate and save a batch of facility location instances.

    Parameters
    ----------
    num_instances : int
        Number of instances to generate
    config : Dict[str, Any]
        Configuration dictionary containing generation settings
    output_dir : str
        Directory where the files will be saved
    sample_seed_start : int, optional
        Starting index for random seeds
    partition : int, optional
        Partition index (appended to filename if provided)
    parallel : bool, optional
        Whether to generate instances in parallel
    parallel_kwargs : Dict[str, Any], optional
        Additional arguments for parallelization
    save_format : str, optional
        Format to save files ("pickle" or "json")

    Returns
    -------
    List[Dict[str, Any]]
        List of generated instance dictionaries
    """ 
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate instances
    if not parallel or not HAS_PROGRESS_PARALLEL:
        instances = [ 
            make_facility_location_instance(config, seed=sample_seed_start + i) 
            for i in trange(num_instances, desc="Generating Instances")  
        ]
    else:
        if parallel_kwargs is None:
            parallel_kwargs = {}
        p = ProgressParallel(**parallel_kwargs)
        instances = p(
            (delayed(make_facility_location_instance)(config, seed=sample_seed_start + i) 
             for i in range(num_instances)),
            len=num_instances,
            desc=f"Generating {num_instances} {config.get('problem_type', 'facility location')} instances",
        )

    # Create filename
    problem_type = config.get("problem_type", "uflp")
    cost_method = config.get("cost_method", "weighted_range")
    nv = config.get("nv", DEFAULT_NV)
    nc = config.get("nc", DEFAULT_NC)
    seed = config.get("seed", 0)
    
    if problem_type in ["pmedian", "p-median"]:
        k = config.get("k", DEFAULT_K)
        data_name = f"method_{cost_method}_nv_{nv}_nc_{nc}_k_{k}_seed_{seed}_num_samples_{num_instances}"
    else:
        data_name = f"method_{cost_method}_nv_{nv}_nc_{nc}_seed_{seed}_num_samples_{num_instances}"
    
    if partition is not None:
        data_name += f"_part_{partition}"
    
    # Save instances
    if save_format.lower() == "json":
        import json
        output_file = f"{output_dir}/{data_name}.json"
        with open(output_file, 'w') as file:
            # Convert numpy arrays to lists for JSON serialization
            json_instances = []
            for instance in instances:
                json_instance = {}
                for key, value in instance.items():
                    if isinstance(value, np.ndarray):
                        json_instance[key] = value.tolist()
                    else:
                        json_instance[key] = value
                json_instances.append(json_instance)
            json.dump(json_instances, file, indent=2)
    else:
        output_file = f"{output_dir}/{data_name}.pkl"
        with open(output_file, 'wb') as file:
            pickle.dump(instances, file)
    
    print(f"Saved {num_instances} instances to {output_file}")
    return instances


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic facility location problem instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Problem configuration
    parser.add_argument(
        "--problem_type", 
        type=str, 
        choices=["uflp", "pmedian", "p-median"],
        default="uflp",
        help="Type of facility location problem to generate"
    )
    parser.add_argument(
        "--nv", 
        type=int, 
        default=DEFAULT_NV,
        help="Number of vertices/locations"
    )
    parser.add_argument(
        "--nc", 
        type=int, 
        default=DEFAULT_NC,
        help="Number of connections per vertex"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=DEFAULT_K,
        help="Number of facilities to open (P-median only)"
    )
    
    # Cost generation
    parser.add_argument(
        "--cost_method", 
        type=str,
        choices=["binary", "randint", "binary100", "weighted_range", "uniform", "exponential"],
        default="weighted_range",
        help="Method for generating transportation costs"
    )
    parser.add_argument(
        "--fixed_cost", 
        type=float, 
        default=DEFAULT_FAC_OPEN_COST,
        help="Fixed cost for opening facilities (UFLP only)"
    )
    parser.add_argument(
        "--penalty", 
        type=float, 
        default=1000,
        help="Penalty cost for unconnected pairs"
    )
    
    # Instance generation
    parser.add_argument(
        "--num_instances", 
        type=int, 
        default=100,
        help="Total number of instances to generate"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=69,
        help="Base random seed"
    )
    parser.add_argument(
        "--seed_start", 
        type=int, 
        default=1,
        help="Starting seed offset for instance generation"
    )
    
    # Partitioning
    parser.add_argument(
        "--partitions", 
        type=int, 
        default=1,
        help="Number of partitions to split instances into"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save generated instances"
    )
    parser.add_argument(
        "--save_format", 
        type=str,
        choices=["pickle", "json"],
        default="pickle",
        help="Format to save instances"
    )
    
    # Processing options
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Generate instances in parallel"
    )
    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=4,
        help="Number of parallel jobs (if parallel=True)"
    )
    
    # Batch configuration presets
    parser.add_argument(
        "--preset", 
        type=str,
        choices=["small", "medium", "large", "test"],
        help="Use predefined parameter presets"
    )
    
    return parser.parse_args()


def apply_preset(args, preset_name: str):
    """Apply predefined parameter presets."""
    presets = {
        "test": {
            "nv": 20, "nc": 3, "k": 3, "num_instances": 10, "partitions": 1
        },
        "small": {
            "nv": 50, "nc": 5, "k": 5, "num_instances": 300, "partitions": 10
        },
        "medium": {
            "nv": 100, "nc": 10, "k": 10, "num_instances": 100, "partitions": 4
        },
        "large": {
            "nv": 200, "nc": 20, "k": 20, "num_instances": 100, "partitions": 4
        }
    }
    
    if preset_name in presets:
        preset = presets[preset_name]
        for key, value in preset.items():
            setattr(args, key, value)
        print(f"Applied preset '{preset_name}': {preset}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Apply preset if specified
    if args.preset:
        apply_preset(args, args.preset)
    
    # Build configuration
    config = {
        "problem_type": args.problem_type,
        "nv": args.nv,
        "nc": args.nc,
        "k": args.k,
        "cost_method": args.cost_method,
        "fixed_cost": args.fixed_cost,
        "penalty": args.penalty,
        "seed": args.seed,
    }
    
    # Setup parallel processing
    parallel_kwargs = None
    if args.parallel and HAS_PROGRESS_PARALLEL:
        parallel_kwargs = {"n_jobs": args.n_jobs}
    
    print(f"Generating {args.num_instances} {args.problem_type} instances")
    print(f"Parameters: nv={args.nv}, nc={args.nc}", end="")
    if args.problem_type in ["pmedian", "p-median"]:
        print(f", k={args.k}", end="")
    print(f", cost_method={args.cost_method}")
    print(f"Output directory: {args.output_dir}")
    print(f"Partitions: {args.partitions}")
    print("-" * 50)
    
    # Generate instances
    instances_generated = 0
    
    for p in range(args.partitions):
        # Calculate instances for this partition
        count = args.num_instances // args.partitions
        if p < args.num_instances % args.partitions:
            count += 1
        
        if count == 0:
            continue
            
        current_seed_start = args.seed_start + instances_generated
        partition_num = p + 1 if args.partitions > 1 else None
        
        print(f"Generating partition {p+1}/{args.partitions} with {count} instances...")
        
        # Generate and save partition
        generate_instance_batch(
            num_instances=count,
            config=config,
            output_dir=args.output_dir,
            sample_seed_start=current_seed_start,
            partition=partition_num,
            parallel=args.parallel,
            parallel_kwargs=parallel_kwargs,
            save_format=args.save_format
        )
        
        instances_generated += count
    
    print(f"\nCompleted: Generated {instances_generated} total instances")


if __name__ == "__main__":
    main() 