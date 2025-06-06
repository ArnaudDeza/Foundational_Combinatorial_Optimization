#!/usr/bin/env python3
"""
Synthetic QAP Data Generator

This script generates synthetic instances for the Quadratic Assignment Problem (QAP)
using various random graph models (Erdos-Renyi, Barabasi-Albert) and different weight scales.
Instances are saved as pickle files in a common format.

The generated dataset is a list of dictionaries. For example, an instance generated using the
"0-1" method is stored as:
    {"F": <weight matrix>, "positions": <node positions>}

Usage:
    python qap_data_generator.py --n 20 30 --p 0.7 0.8 --dataset_type 0-1 --num_instances 100 --output_dir ./synthetic_data


This appears in 'Learning Solution-Aware Transformers for Efficiently Solving Quadratic
Assignment Problem'.  For all experiments, we use a training set of up to 5120 instances and evaluate
results on a test set of 256 different instances from the same distribution. We set p = 0.7 for all tasks
"""

import os
import argparse
import pickle
from pathlib import Path
import math
import networkx as nx
import numpy as np


def generate_erdos_qap_instances_0_1(N: int, p: float):
    """
    Generate a QAP instance using an Erdos-Renyi graph.
    Weights are drawn from a uniform distribution in [0,1] (rounded to 2 decimals),
    and node positions are 2D coordinates in [0,1].

    Returns:
        weight_F_final: numpy.ndarray, shape (N,N)
        location_D: numpy.ndarray, shape (N,2)
    """
    weight_F = np.zeros((N, N), dtype=float)
    location_D = np.zeros((N, 2), dtype=float)

    # Generate symmetric weight matrix with random weights in [0,1]
    for i in range(N - 1):
        for j in range(i + 1, N):
            weight = np.round(np.random.uniform(0, 1), 2)
            weight_F[i, j] = weight
            weight_F[j, i] = weight

    # Generate random positions in [0,1]
    for i in range(N):
        location_D[i, 0] = np.round(np.random.uniform(0, 1), 2)
        location_D[i, 1] = np.round(np.random.uniform(0, 1), 2)

    # Build graph and assign edge weights
    G_F = nx.erdos_renyi_graph(N, p)
    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]

    # Construct final weight matrix from graph edges (ensuring symmetry)
    weight_F_final = np.zeros((N, N), dtype=float)
    for i, j in G_F.edges():
        weight_F_final[i, j] = G_F[i][j]['weight']
        weight_F_final[j, i] = weight_F_final[i, j]

    return weight_F_final, location_D


def generate_erdos_qap_instances_int(N: int, p: float):
    """
    Generate a QAP instance using an Erdos-Renyi graph.
    Weights and positions are drawn from a uniform distribution in [0,50] (rounded to nearest integer).

    Returns:
        weight_F_final: numpy.ndarray, shape (N,N)
        location_D: numpy.ndarray, shape (N,2)
    """
    weight_F = np.zeros((N, N), dtype=float)
    location_D = np.zeros((N, 2), dtype=float)

    for i in range(N - 1):
        for j in range(i + 1, N):
            weight = np.round(np.random.uniform(0, 50))
            weight_F[i, j] = weight
            weight_F[j, i] = weight

    for i in range(N):
        location_D[i, 0] = np.round(np.random.uniform(0, 50))
        location_D[i, 1] = np.round(np.random.uniform(0, 50))

    G_F = nx.erdos_renyi_graph(N, p)
    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]

    weight_F_final = np.zeros((N, N), dtype=float)
    for i, j in G_F.edges():
        weight_F_final[i, j] = G_F[i][j]['weight']
        weight_F_final[j, i] = weight_F_final[i, j]

    return weight_F_final, location_D


def generate_erdos_qap_instances(N: int, p: float, F_weight: tuple = (0, 50), D_weight: tuple = (0, 50)):
    """
    Generate a QAP instance using two Erdos-Renyi graphs: one for the flow matrix (F) and one for the
    distance matrix (D) with integer weights.

    Returns:
        weight_F_final: numpy.ndarray, shape (N,N)
        weight_D_final: numpy.ndarray, shape (N,N)
    """
    weight_F = np.zeros((N, N), dtype=int)
    weight_D = np.zeros((N, N), dtype=int)
    F_lower, F_upper = F_weight
    D_lower, D_upper = D_weight

    for i in range(N - 1):
        for j in range(i + 1, N):
            f_val = np.random.randint(F_lower, F_upper)
            d_val = np.random.randint(D_lower, D_upper)
            weight_F[i, j] = f_val
            weight_F[j, i] = f_val
            weight_D[i, j] = d_val
            weight_D[j, i] = d_val

    G_F = nx.erdos_renyi_graph(N, p)
    G_D = nx.erdos_renyi_graph(N, p)

    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]
    for i, j in G_D.edges():
        G_D[i][j]['weight'] = weight_D[i, j]

    weight_F_final = np.zeros((N, N), dtype=int)
    weight_D_final = np.zeros((N, N), dtype=int)

    for i, j in G_F.edges():
        weight_F_final[i, j] = G_F[i][j]['weight']
        weight_F_final[j, i] = weight_F_final[i, j]
    for i, j in G_D.edges():
        weight_D_final[i, j] = G_D[i][j]['weight']
        weight_D_final[j, i] = weight_D_final[i, j]

    return weight_F_final, weight_D_final


def generate_barabasi_qap_instances(N: int, m: int, F_weight: tuple = (0, 50), D_weight: tuple = (0, 50)):
    """
    Generate a QAP instance using Barabasi-Albert graphs for the flow (F) and distance (D) matrices.
    Weights are generated as random integers.
    
    Returns:
        weight_F_final: numpy.ndarray, shape (N,N)
        weight_D_final: numpy.ndarray, shape (N,N)
    """
    weight_F = np.zeros((N, N), dtype=int)
    weight_D = np.zeros((N, N), dtype=int)
    F_lower, F_upper = F_weight
    D_lower, D_upper = D_weight

    for i in range(N - 1):
        for j in range(i + 1, N):
            f_val = np.random.randint(F_lower, F_upper)
            d_val = np.random.randint(D_lower, D_upper)
            weight_F[i, j] = f_val
            weight_F[j, i] = f_val
            weight_D[i, j] = d_val
            weight_D[j, i] = d_val

    G_F = nx.barabasi_albert_graph(N, m)
    G_D = nx.barabasi_albert_graph(N, m)

    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]
    for i, j in G_D.edges():
        G_D[i][j]['weight'] = weight_D[i, j]

    weight_F_final = np.zeros((N, N), dtype=int)
    weight_D_final = np.zeros((N, N), dtype=int)

    for i, j in G_F.edges():
        weight_F_final[i, j] = G_F[i][j]['weight']
        weight_F_final[j, i] = weight_F_final[i, j]
    for i, j in G_D.edges():
        weight_D_final[i, j] = G_D[i][j]['weight']
        weight_D_final[j, i] = weight_D_final[i, j]

    return weight_F_final, weight_D_final


import argparse
import os
import pickle
import math
 
def save_dataset(data, output_path: str):
    """
    Save the dataset to a file using pickle.
    """
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic QAP datasets using various random graph models."
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[10],
        help="List of N values (number of nodes). E.g., --n 20 30",
    )
    parser.add_argument(
        "--p",
        nargs="+",
        type=float,
        default=[0.7],
        help="List of p values (edge probability for Erdos-Renyi). E.g., --p 0.7 0.8",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["erdos", "barabasi", "0_1", "erdos_int"],
        default="0_1",
        help="Type of dataset to generate.",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=25,
        help="Number of instances per configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/adeza3/Documents/Nuni__PhD/Year_1_2024_2025/foundationalCO/data/synthetic/qap",
        help="Directory to save datasets.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=15,
        help="Parameter m for Barabasi-Albert graph (used only if dataset_type is 'barabasi').",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        default=100,
        help="Number of parts (pickle files) to split the dataset into.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Loop over every combination of N and p
    for N in args.n:
        for p in args.p:
            # Generate the dataset in-memory first
            dataset = []
            if args.dataset_type == "erdos":
                for _ in range(args.num_instances):
                    F, D = generate_erdos_qap_instances(N, p)
                    dataset.append({"F": F, "D": D})
                subfolder_name = f"erdos_N{N}_p{p}"
            elif args.dataset_type == "barabasi":
                for _ in range(args.num_instances):
                    F, D = generate_barabasi_qap_instances(N, args.m)
                    dataset.append({"F": F, "D": D})
                subfolder_name = f"barabasi_N{N}_m{args.m}"
            elif args.dataset_type == "0_1":
                for _ in range(args.num_instances):
                    F, positions = generate_erdos_qap_instances_0_1(N, p)
                    dataset.append({"F": F, "positions": positions})
                subfolder_name = f"0_1_N{N}_p{p}"
            elif args.dataset_type == "erdos_int":
                for _ in range(args.num_instances):
                    F, positions = generate_erdos_qap_instances_int(N, p)
                    dataset.append({"F": F, "positions": positions})
                subfolder_name = f"erdos_int_N{N}_p{p}"

            # Create a subfolder for this particular setting
            subfolder_path = os.path.join(args.output_dir, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)

            # Split dataset into num_parts
            if args.num_parts < 1:
                raise ValueError("--num_parts must be at least 1.")

            part_size = math.ceil(args.num_instances / args.num_parts)

            for part_idx in range(args.num_parts):
                start_idx = part_idx * part_size
                end_idx = min(start_idx + part_size, len(dataset))

                # It's possible that num_parts > num_instances. Skip empty splits
                if start_idx >= len(dataset):
                    break

                dataset_part = dataset[start_idx:end_idx]

                # Construct filename for this part
                filename = f"qap_part{part_idx+1}.pkl"
                output_path = os.path.join(subfolder_path, filename)

                # Save this part
                save_dataset(dataset_part, output_path)
                print(f"Saved part {part_idx+1}/{args.num_parts} for {subfolder_name} to {output_path}")