# Quadratic Assignment Problem (QAP) Module

This module provides a complete framework for working with Quadratic Assignment Problems, including data generation, solving, and machine learning training components.

## Overview

The QAP module consists of several components:

- **Data Generation**: `synthetic_generator.py` - Generate synthetic QAP instances
- **Solving**: `solve_instances.py` and `solvers.py` - Solve QAP instances using Gurobi
- **Training**: `datamodule.py` and `train_module.py` - Machine learning components for QAP
- **Job Submission**: `../slurm/solve_instances/submit_qap_synthetic.py` - SLURM job management

## Data Generation

### Generate Synthetic QAP Instances

```bash
# Generate 0-1 weighted instances (positions and flow matrix)
python src/qap/synthetic_generator.py \
    --n 20 30 \
    --p 0.7 0.8 \
    --dataset_type 0_1 \
    --num_instances 100 \
    --output_dir ./data/synthetic/qap

# Generate Erdos-Renyi instances (flow and distance matrices)
python src/qap/synthetic_generator.py \
    --n 15 25 \
    --p 0.5 0.7 \
    --dataset_type erdos \
    --num_instances 50 \
    --output_dir ./data/synthetic/qap

# Generate Barabasi-Albert instances
python src/qap/synthetic_generator.py \
    --n 20 \
    --dataset_type barabasi \
    --m 3 \
    --num_instances 75 \
    --output_dir ./data/synthetic/qap
```

### Data Format

Generated instances are saved as pickle files containing lists of dictionaries:

**For `0_1` and `erdos_int` types:**
```python
{
    "F": numpy.ndarray,        # Flow matrix (n x n)
    "positions": numpy.ndarray # Node positions (n x 2)
}
```

**For `erdos` and `barabasi` types:**
```python
{
    "F": numpy.ndarray,  # Flow matrix (n x n)
    "D": numpy.ndarray   # Distance matrix (n x n)
}
```

## Solving QAP Instances

### Solve Individual Files

```bash
# Solve specific pickle files
python src/qap/solve_instances.py \
    --pickle_files data/synthetic/qap/0_1_N20_p0.7/qap_part1.pkl \
    --time_limit 3600 \
    --n_threads 24
```

### Batch Solving with SLURM

```bash
# Submit jobs for solving synthetic QAP instances
python slurm/solve_instances/submit_qap_synthetic.py \
    --dataset_folder data/synthetic/qap \
    --problem_type 0_1 \
    --N 20 30 \
    --p 0.7 0.8 \
    --num_jobs_to_submit 10 \
    --time_limit 7200
```

### Solver Output Format

Solved instances are saved with `_results` suffix and contain:

```python
{
    "F": numpy.ndarray,           # Original flow matrix
    "positions": numpy.ndarray,   # Original positions (if available)
    "D": numpy.ndarray,           # Distance matrix (computed or provided)
    "objVal": float,              # Optimal objective value
    "mipgap": float,              # MIP gap achieved
    "runtime": float,             # Solver runtime in seconds
    "solution": numpy.ndarray,    # Assignment matrix (n x n)
    "time_limit": int             # Time limit used
}
```

## Machine Learning Training

### Data Module

The `QAPDataModule` handles:
- Loading QAP instances from pickle files
- Computing distance matrices from positions when needed
- Padding and batching for neural network training
- Managing cache for iterative improvement algorithms

### Training Module

The `QAPModule` supports different learning paradigms:
- **Supervised learning**: Learn from optimal solutions
- **Unsupervised objective**: Learn to predict objective values
- **Unsupervised GST**: Guided Search Training with iterative improvement

### Usage with Trainer

```python
# In your hyperparameters configuration
hparams = {
    "dataset": {
        "problem_type": "qap",
        "folder": ["data/synthetic/qap/0_1_N20_p0.7"],
        "max_num_instances": 1000,
        "max_num_instances_per_size": 100,
        "completely_unsupervised": False
    },
    # ... other configurations
}

# The trainer will automatically use QAPDataModule and QAPModule
trainer = Trainer(hparams)
trainer.fit()
```

## File Structure

```
src/qap/
├── README.md                 # This documentation
├── synthetic_generator.py    # Generate synthetic QAP instances
├── solve_instances.py        # Solve QAP instances in parallel
├── solvers.py               # Gurobi solver implementation
├── datamodule.py            # PyTorch Lightning data module
├── train_module.py          # PyTorch Lightning training module
├── benchmark_downloader.py  # Download benchmark instances (placeholder)
└── config.json              # Configuration file (placeholder)

slurm/solve_instances/
└── submit_qap_synthetic.py  # SLURM job submission script
```

## Loading QAPLIB Instances

The module includes a robust loader for QAPLIB-format files:

```python
from src.qap.loader import load_qap_instance, load_qap_as_symmetric

# Load a QAP instance from QAPLIB format
F, D = load_qap_instance("path/to/instance.dat")

# Load and ensure matrices are symmetric
F_sym, D_sym = load_qap_as_symmetric("path/to/instance.dat")
```

### QAPLIB Format

The loader supports the standard QAPLIB format:
- First line: problem size `n`
- Next `n×n` numbers: flow matrix F
- Next `n×n` numbers: distance matrix D
- Supports comments (lines starting with `#`, `//`, or `!`)
- Handles matrices split across multiple lines

### Download QAPLIB Benchmarks

```bash
# Download and convert QAPLIB instances
python src/qap/benchmark_downloader.py --output_dir data/qap/benchmarks

# Convert existing data only
python src/qap/benchmark_downloader.py --convert_only --output_dir data/qap/benchmarks
```

## Dependencies

- **Core**: `numpy`, `torch`, `pytorch-lightning`
- **Optimization**: `gurobipy`
- **Graph Generation**: `networkx`
- **Distance Computation**: `scipy`
- **Parallel Processing**: `joblib`
- **Progress Tracking**: `tqdm`
- **Data Loading**: `requests` (for benchmark download)

## Examples

### Complete Workflow

1. **Generate Data**:
```bash
python src/qap/synthetic_generator.py --n 15 20 --p 0.7 --dataset_type 0_1 --num_instances 200
```

2. **Solve Instances**:
```bash
python slurm/solve_instances/submit_qap_synthetic.py --N 15 20 --p 0.7 --problem_type 0_1
```

3. **Train Model**:
```bash
python train.py --hp config/qap_config.json
```

### Integration with QUBO Framework

The QAP module integrates seamlessly with the existing combinatorial optimization framework:

- Uses the same `DOP_Module` base class
- Follows the same registry pattern in `train.py`
- Supports the same learning paradigms
- Compatible with existing logging and monitoring infrastructure

The modular design allows easy extension to additional combinatorial optimization problems by following the same pattern. 