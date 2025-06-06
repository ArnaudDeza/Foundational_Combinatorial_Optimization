# QUBO Data Generation and Solving

This document provides instructions and examples for generating synthetic QUBO datasets and solving them using various optimization methods.

## Generating QUBO Datasets

The `synthetic_generator.py` script allows for the creation of custom QUBO instances with varying sizes, densities, and other parameters.

### Basic Usage

To generate QUBO instances, you need to specify the dimensions (`--n_values`), the number of instances per configuration (`--num_instances`), and an output folder (`--output_folder`).

**Example: Generating Dense QUBOs**

This command generates 1000 instances for QUBOs of size 50x50 and 100x100. Since `--density_values` is not provided, the matrices will be dense.

```bash
python src/qubo/synthetic_generator.py \
    --n_values 10 2 \
    --num_instances 1000 \
    --output_folder ./data
```

### Advanced Usage

**Example: Generating Sparse QUBOs with Varying Densities**

This command generates 500 instances for each combination of size (50, 100) and density (0.1, 0.5, 0.8).

```bash
python src/qubo/synthetic_generator.py \
    --n_values 50 100 \
    --density_values 0.1 0.5 0.8 \
    --num_instances 500 \
    --output_folder ./data
```

**Example: Partitioning Large Datasets**

For large `n`, you can split the generated instances into smaller files using the `--partitions` argument. The format is `n:number_of_partitions`.

```bash
python src/qubo/synthetic_generator.py \
    --n_values 20 \
    --num_instances 5000 \
    --output_folder /Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization/data/qubo/synthetic \
    --partitions '20:10'
```
This will create 10 separate `.npy` files for the 10,000 instances of size 200x200.

## Solving QUBO Instances

The `solve_instances.py` script is used to solve the generated QUBO problems. It supports exact solvers like Gurobi and heuristic methods like Tabu Search and Simulated Annealing.

### Basic Usage

To solve a set of QUBO instances, you need to provide the path to the `.npy` files.

**Example: Solving with Gurobi (Default Solver)**

This command will find all `.npy` files in the specified directory and solve them using the default Gurobi solver.

```bash
python src/qubo/solve_instances.py \
    --npy_files ./data/not_varying_density/data_n_50/*.npy \
    --time_limit 600
```

### Solver Options

You can specify the solver using the `--solver_type` argument.

**Example: Solving with GUROBI_OPTIMODS**

```bash
python src/qubo/solve_instances.py \
    --npy_files ./data/not_varying_density/data_n_50/*.npy \
    --solver_type GUROBI_OPTIMODS \
    --time_limit 600
```

**Example: Solving with Tabu Search**

```bash
python src/qubo/solve_instances.py \
    --npy_files ./data/not_varying_density/data_n_50/*.npy \
    --solver_type TABU_SEARCH \
    --time_limit 600
```

**Example: Solving with Simulated Annealing**

```bash
python src/qubo/solve_instances.py \
    --npy_files ./data/not_varying_density/data_n_50/*.npy \
    --solver_type SIMULATED_ANNEALING \
    --time_limit 600
```

### Parallel Execution

The script automatically manages parallel execution. You can control the total number of CPU cores and the threads per Gurobi solve.

```bash
python src/qubo/solve_instances.py \
    --npy_files ./data/not_varying_density/data_n_100/*.npy \
    --n_threads 32 \
    --gurobi_threads_per_solve 4
```

This will run `32 / 4 = 8` Gurobi solves in parallel, each utilizing 4 threads. 