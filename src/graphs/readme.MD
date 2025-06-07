# Graph Combinatorial Optimization Solvers

This directory contains implementations for solving various graph combinatorial optimization problems using Gurobi MIP formulations.

## Available Problem Types

- **max_cut**: Maximum Cut Problem
- **mis** / **maximum_independent_set**: Maximum Independent Set
- **mvc** / **minimum_vertex_cover**: Minimum Vertex Cover  
- **max_clique** / **maximum_clique**: Maximum Clique
- **graph_coloring** / **coloring**: Graph Coloring
- **min_dominating_set** / **dominating_set**: Minimum Dominating Set
- **total_dominating_set**: Total Dominating Set

## Files

- `solve_instances.py`: Main script for solving multiple graph instances in parallel
- `solvers.py`: Core solver implementations and direct solver interface
- `synthetic_generator.py`: Graph generation utilities

## Prerequisites

```bash
# Activate conda environment with Gurobi
conda activate jssp

# Ensure you have the required packages
pip install numpy networkx joblib tqdm
```

## Basic Usage Examples

### Single Instance Solving

Solve max cut on a single graph:
```bash
python solve_instances.py \
    --graph_files /Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization/data/synthetic_graphs/dai_paper/ba_ngraphs_1000_nmin_16_nmax_20/BA_16_20_4_0.gpickle \
    --problem_type max_cut \
    --time_limit 10
```

Solve maximum independent set with quadratic formulation:
```bash
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_233.gpickle \
    --problem_type mis \
    --time_limit 5 \
    --quadratic
```

### Multiple Instance Solving

Solve max cut on multiple specific files:
```bash
python solve_instances.py \
    --graph_files \
        ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_466.gpickle \
        ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_476.gpickle \
        ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_233.gpickle \
    --problem_type max_cut \
    --time_limit 10 \
    --nparallel 3
```

### Directory-Based Solving

Solve all graphs in a directory:
```bash
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ \
    --problem_type max_cut \
    --time_limit 5 \
    --nparallel 4 \
    --output_summary max_cut_results.json
```

### Different Problem Types

Minimum vertex cover:
```bash
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_223.gpickle \
    --problem_type mvc \
    --time_limit 10
```

Maximum clique:
```bash
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_155.gpickle \
    --problem_type max_clique \
    --time_limit 15
```

Graph coloring (with color limit):
```bash
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_98.gpickle \
    --problem_type graph_coloring \
    --time_limit 10 \
    --max_colors 30
```

Minimum dominating set:
```bash
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_145.gpickle \
    --problem_type min_dominating_set \
    --time_limit 8
```

## Advanced Usage

### Parallel Processing with Custom Results Folder

```bash
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ \
    --problem_type mis \
    --time_limit 5 \
    --nparallel 6 \
    --max_threads 2 \
    --quadratic \
    --results_folder ./mis_results \
    --output_summary ./mis_summary.json \
    --overwrite
```

### Batch Processing Different Problem Types

```bash
# Max Cut
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_46*.gpickle \
    --problem_type max_cut \
    --time_limit 10 \
    --nparallel 4 \
    --output_summary maxcut_batch_results.json

# Maximum Independent Set  
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_46*.gpickle \
    --problem_type mis \
    --time_limit 10 \
    --nparallel 4 \
    --quadratic \
    --output_summary mis_batch_results.json

# Minimum Vertex Cover
python solve_instances.py \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_46*.gpickle \
    --problem_type mvc \
    --time_limit 10 \
    --nparallel 4 \
    --output_summary mvc_batch_results.json
```

### Direct Solver Interface

Using the solvers.py script directly:
```bash
python solvers.py \
    --problem_type max_cut \
    --graph_files ../../data/synthetic_graphs/dai_paper/er_ngraphs_1000_nmin_512_nmax_600/ER_512_600_0.15_466.gpickle \
    --time_limit 10 \
    --max_threads 1
```

## Output and Results

### Individual Results
Each solved instance produces a `.npy` file containing:
- `objective_value`: Optimal/best found objective value
- `node_solution`: Binary array indicating selected nodes
- `solver_runtime`: Time taken by Gurobi
- `nodes_branched`: Number of branch-and-bound nodes explored
- `mipgap`: Final MIP gap
- `solution_valid`: Whether solution passes validation
- Graph metadata (nodes, edges, density)

### Summary Results
When using `--output_summary`, produces a JSON file with:
- Success/failure statistics
- Objective value statistics (mean, std, min, max)
- Runtime statistics
- Graph characteristics
- Validation results

Example output:
```
==================================================
SOLVING SUMMARY
==================================================
Problem type: max_cut
Total instances: 3
Successful solves: 3
Failed solves: 0
Success rate: 100.0%

Objective Statistics:
  Average: 13520.00
  Std Dev: 1382.00
  Min: 12138.00
  Max: 14902.00

Runtime Statistics:
  Average: 8.50s
  Std Dev: 1.20s
  Min: 7.23s
  Max: 10.01s
  Total: 25.50s

Validation:
  Valid solutions: 3
  Validation rate: 100.0%
```

## Performance Tips

1. **Parallel Processing**: Use `--nparallel` to solve multiple instances simultaneously
2. **Time Limits**: Set appropriate `--time_limit` based on graph size and problem difficulty
3. **Gurobi Threads**: Adjust `--max_threads` based on available CPU cores
4. **Quadratic Formulations**: Use `--quadratic` for MIS when memory allows
5. **Results Management**: Use `--results_folder` to organize outputs

## Troubleshooting

- Ensure Gurobi license is valid and accessible
- For large graphs, increase time limits or reduce parallel jobs
- Graph coloring may need higher `--max_colors` for dense graphs
- Use `--log_level DEBUG` for detailed solver information