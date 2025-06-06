#!/usr/bin/env python3
"""
Module for generating synthetic QUBO instances and saving them to disk.
"""

import os
import logging
import argparse
from typing import List, Optional, Dict
import numpy as np

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def generate_sparse_qubo(n: int, max_q_val: float, seed: int, num_samples: int,
                         density: Optional[float] = None) -> np.ndarray:
    """
    Generate a list of sparse, symmetric QUBO matrices.

    Args:
        n (int): Dimension of the QUBO matrix.
        max_q_val (float): Maximum absolute value for Q entries.
        seed (int): Random seed for reproducibility.
        num_samples (int): Number of QUBO instances to generate.
        density (Optional[float]): Probability of a nonzero entry. If None, no sparsity is enforced.

    Returns:
        np.ndarray: Array of generated QUBO matrices with shape (num_samples, n, n).
    """
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(num_samples):
        # Create a random n x n matrix with values in [-max_q_val, max_q_val]
        Q = rng.uniform(-max_q_val, max_q_val, size=(n, n))
        if density is not None:
            # Enforce sparsity via a binary mask
            mask = rng.random((n, n)) < density
            Q *= mask.astype(float)
        # Ensure the matrix is symmetric
        Q = (Q + Q.T) / 2.0
        instances.append(Q)
    return np.array(instances)


def save_qubo_instances_for_setting(n: int,
                                    num_instances: int,
                                    max_q_val: float,
                                    seed: int,
                                    output_folder: str,
                                    density: Optional[float] = None,
                                    partitions: Optional[int] = None) -> None:
    """
    Generate and save synthetic QUBO instances for a single configuration.

    Args:
        n (int): Dimension of the QUBO matrix.
        num_instances (int): Number of instances to generate.
        max_q_val (float): Maximum absolute value for Q entries.
        seed (int): Random seed for reproducibility.
        output_folder (str): Directory to save the instance files.
        density (Optional[float]): Sparsity of the QUBO matrix. If None, dense matrices are created.
        partitions (Optional[int]): Number of partitions to split the instances into.
    """
    if density is not None:
        folder_name = f"data_n_{n}_density_{density:.3f}"
        logging.info("Generating QUBO instances for n=%d, density=%.3f", n, density)
    else:
        folder_name = f"data_n_{n}"
        logging.info("Generating QUBO instances for n=%d (fully dense)", n)

    instance_folder = os.path.join(output_folder, folder_name)
    os.makedirs(instance_folder, exist_ok=True)

    Q_list = generate_sparse_qubo(n, max_q_val, seed, num_instances, density)

    if partitions:
        partitions_list = np.array_split(Q_list, partitions)
        for idx, partition_data in enumerate(partitions_list, start=1):
            filename = os.path.join(instance_folder, f"Q_part_{idx}.npy")
            np.save(filename, partition_data)
            logging.info("Saved partition %d for n=%d to %s", idx, n, filename)
    else:
        filename = os.path.join(instance_folder, "Q.npy")
        np.save(filename, Q_list)
        logging.info("Saved QUBO instances for n=%d to %s", n, filename)


def main() -> None:
    """
    Main function to parse arguments and run QUBO instance generation.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic QUBO instances.")
    parser.add_argument("--n_values", type=int, nargs="+", required=True, help="List of QUBO dimensions.")
    parser.add_argument("--density_values", type=float, nargs="+", default=None, help="List of sparsity levels. If not provided, dense matrices are generated.")
    parser.add_argument("--num_instances", type=int, required=True, help="Number of instances per configuration.")
    parser.add_argument("--max_q_val", type=float, default=100.0, help="Maximum absolute value for Q entries.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--output_folder", type=str, required=True, help="Root folder to store generated instances.")
    parser.add_argument("--partitions", type=str, nargs="*", help="Partitioning settings for large instances, e.g., '100:10' for n=100 into 10 partitions.")

    args = parser.parse_args()

    try:
        partitions_dict = {}
        if args.partitions:
            for p in args.partitions:
                try:
                    n_str, part_str = p.split(':')
                    partitions_dict[int(n_str)] = int(part_str)
                except ValueError:
                    logging.error(f"Invalid partition format: {p}. Use 'n:partitions'.")
                    return

        if args.density_values:
            output_base_folder = os.path.join(args.output_folder, 'varying_density')
            densities = args.density_values
        else:
            output_base_folder = os.path.join(args.output_folder, 'not_varying_density')
            densities = [None]
        
        os.makedirs(output_base_folder, exist_ok=True)

        total_configs = len(args.n_values) * len(densities)
        total_instances = total_configs * args.num_instances
        logging.info("Generating %d configurations with %d instances each. Total instances: %d",
                     total_configs, args.num_instances, total_instances)

        for n in args.n_values:
            for density in densities:
                save_qubo_instances_for_setting(
                    n=n,
                    density=density,
                    num_instances=args.num_instances,
                    max_q_val=args.max_q_val,
                    seed=args.seed,
                    partitions=partitions_dict.get(n),
                    output_folder=output_base_folder
                )
        
        logging.info("QUBO instance generation completed successfully.")
    except Exception as e:
        logging.exception("An error occurred during QUBO instance generation: %s", str(e))


if __name__ == '__main__':
    main()
