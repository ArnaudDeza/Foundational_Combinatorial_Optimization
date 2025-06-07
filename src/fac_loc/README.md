# Facility Location Problem Generator

This directory contains a unified synthetic generator for facility location problems, supporting both **Uncapacitated Facility Location Problem (UFLP)** and **P-median** problems.

## Files

- `synthetic_generator.py`: Main consolidated generator script
- `synthetic_generator_uflp.py`: Original UFLP generator (deprecated)
- `synthetic_generator_pmedian.py`: Original P-median generator (deprecated)
- `README.md`: This documentation file

## Features

- **Dual Problem Support**: Generate both UFLP and P-median instances
- **Multiple Cost Methods**: 6 different transportation cost generation methods
- **Flexible Parameters**: Configurable problem size and structure
- **Preset Configurations**: Quick-start templates for common scenarios
- **Parallel Processing**: Optional parallel instance generation
- **Multiple Output Formats**: Save as pickle or JSON
- **Partitioned Output**: Split large datasets into manageable chunks

## Quick Start

### Basic UFLP Generation
```bash
python synthetic_generator.py \
    --problem_type uflp \
    --output_dir ./data/uflp \
    --preset small
```

### Basic P-median Generation
```bash
python synthetic_generator.py \
    --problem_type pmedian \
    --k 10 \
    --output_dir ./data/pmedian \
    --preset medium
```

### Test Generation
```bash
python synthetic_generator.py \
    --problem_type uflp \
    --output_dir ./test_data \
    --preset test
```

## Command Line Interface

### Problem Configuration
```bash
python synthetic_generator.py \
    --problem_type uflp \              # or pmedian/p-median
    --nv 100 \                         # number of locations
    --nc 10 \                          # connections per location
    --k 5 \                            # facilities to open (P-median only)
    --output_dir ./data
```

### Cost Generation Methods
```bash
# Weighted random (default)
python synthetic_generator.py --cost_method weighted_range --output_dir ./data

# Binary costs (1 or 10)
python synthetic_generator.py --cost_method binary --output_dir ./data

# Random integers (1-5)
python synthetic_generator.py --cost_method randint --output_dir ./data

# Binary extreme (1 or 100)
python synthetic_generator.py --cost_method binary100 --output_dir ./data

# Uniform distribution (1-20)
python synthetic_generator.py --cost_method uniform --output_dir ./data

# Exponential distribution
python synthetic_generator.py --cost_method exponential --output_dir ./data
```

### Advanced Options
```bash
python synthetic_generator.py \
    --problem_type uflp \
    --nv 200 \
    --nc 20 \
    --num_instances 500 \
    --partitions 5 \
    --fixed_cost 1500 \
    --penalty 2000 \
    --seed 42 \
    --parallel \
    --n_jobs 8 \
    --save_format json \
    --output_dir ./large_dataset
```

## Presets

Use `--preset` for quick configuration:

### Test Preset
```bash
--preset test
# nv=20, nc=3, k=3, num_instances=10, partitions=1
```

### Small Preset  
```bash
--preset small
# nv=50, nc=5, k=5, num_instances=300, partitions=10
```

### Medium Preset
```bash
--preset medium  
# nv=100, nc=10, k=10, num_instances=100, partitions=4
```

### Large Preset
```bash
--preset large
# nv=200, nc=20, k=20, num_instances=100, partitions=4
```

## Complete Parameter Reference

### Required Parameters
- `--output_dir`: Directory to save generated instances

### Problem Parameters
- `--problem_type`: Problem type (`uflp`, `pmedian`, `p-median`) [default: uflp]
- `--nv`: Number of vertices/locations [default: 100]
- `--nc`: Number of connections per vertex [default: 10]
- `--k`: Number of facilities to open (P-median only) [default: 5]

### Cost Parameters
- `--cost_method`: Cost generation method [default: weighted_range]
  - `weighted_range`: Weighted random costs 1-10
  - `binary`: Random choice between 1 and 10 (weights 3:6)
  - `randint`: Random integers 1-5
  - `binary100`: Random choice between 1 and 100
  - `uniform`: Uniform distribution 1-20
  - `exponential`: Exponential distribution (scale=5) + 1
- `--fixed_cost`: Fixed facility opening cost (UFLP only) [default: 1000]
- `--penalty`: Penalty for unconnected pairs [default: 1000]

### Generation Parameters
- `--num_instances`: Total number of instances [default: 100]
- `--seed`: Base random seed [default: 69]
- `--seed_start`: Starting seed offset [default: 1]
- `--partitions`: Number of output file partitions [default: 1]

### Output Parameters
- `--save_format`: Output format (`pickle`, `json`) [default: pickle]

### Processing Parameters
- `--parallel`: Enable parallel generation
- `--n_jobs`: Number of parallel jobs [default: 4]

### Quick Configuration
- `--preset`: Use predefined settings (`test`, `small`, `medium`, `large`)

## Output Structure

### File Naming Convention

**UFLP instances:**
```
method_{cost_method}_nv_{nv}_nc_{nc}_seed_{seed}_num_samples_{num_instances}.pkl
```

**P-median instances:**
```
method_{cost_method}_nv_{nv}_nc_{nc}_k_{k}_seed_{seed}_num_samples_{num_instances}.pkl
```

**With partitions:**
```
method_{cost_method}_nv_{nv}_nc_{nc}_k_{k}_seed_{seed}_num_samples_{num_instances}_part_{partition}.pkl
```

### Instance Data Structure

Each generated instance contains:
```python
{
    "problem_type": "uflp" or "pmedian",
    "fixed_costs": np.array,      # Fixed opening costs per facility
    "transport_cost": np.array,   # Transportation cost matrix [nv x nv]
    "nv": int,                    # Number of locations
    "nc": int,                    # Connections per vertex
    "k": int,                     # Facilities to open (P-median only)
    "seed": int,                  # Random seed used
    "cost_method": str,           # Cost generation method
    "penalty": float,             # Penalty for unconnected pairs
}
```

## Example Usage Scenarios

### Generate Small Test Dataset
```bash
python synthetic_generator.py \
    --problem_type uflp \
    --preset test \
    --output_dir ./test_instances
```

### Generate Large UFLP Dataset for Research
```bash
python synthetic_generator.py \
    --problem_type uflp \
    --nv 500 \
    --nc 50 \
    --num_instances 1000 \
    --partitions 20 \
    --cost_method weighted_range \
    --parallel \
    --n_jobs 8 \
    --output_dir ./research_data/uflp_large
```

### Generate P-median with Different Cost Methods
```bash
# Binary costs
python synthetic_generator.py \
    --problem_type pmedian \
    --k 15 \
    --cost_method binary \
    --num_instances 200 \
    --output_dir ./pmedian_binary

# Exponential costs  
python synthetic_generator.py \
    --problem_type pmedian \
    --k 15 \
    --cost_method exponential \
    --num_instances 200 \
    --output_dir ./pmedian_exponential
```

### Generate Multiple Problem Sizes
```bash
# Small problems
python synthetic_generator.py --preset small --problem_type uflp --output_dir ./data/small_uflp
python synthetic_generator.py --preset small --problem_type pmedian --output_dir ./data/small_pmedian

# Medium problems
python synthetic_generator.py --preset medium --problem_type uflp --output_dir ./data/medium_uflp
python synthetic_generator.py --preset medium --problem_type pmedian --output_dir ./data/medium_pmedian

# Large problems
python synthetic_generator.py --preset large --problem_type uflp --output_dir ./data/large_uflp
python synthetic_generator.py --preset large --problem_type pmedian --output_dir ./data/large_pmedian
```

## Loading Generated Instances

### Python (Pickle Format)
```python
import pickle

# Load instances
with open('method_weighted_range_nv_50_nc_5_seed_69_num_samples_30_part_1.pkl', 'rb') as f:
    instances = pickle.load(f)

# Access first instance
instance = instances[0]
print(f"Problem type: {instance['problem_type']}")
print(f"Locations: {instance['nv']}")
print(f"Transport costs shape: {instance['transport_cost'].shape}")
```

### Python (JSON Format)
```python
import json
import numpy as np

# Load instances
with open('instances.json', 'r') as f:
    instances = json.load(f)

# Convert lists back to numpy arrays
for instance in instances:
    instance['fixed_costs'] = np.array(instance['fixed_costs'])
    instance['transport_cost'] = np.array(instance['transport_cost'])
```

## Performance Tips

1. **Use parallel processing** for large datasets: `--parallel --n_jobs 8`
2. **Partition large datasets** for memory efficiency: `--partitions 10`
3. **Use pickle format** for faster I/O vs JSON
4. **Start with test preset** to verify parameters before large runs
5. **Use appropriate cost methods** based on your research needs

## Troubleshooting

### Common Issues

1. **Import Error for ProgressParallel**: Sequential processing will be used automatically
2. **Memory issues with large instances**: Increase partitions or reduce instance count
3. **Slow generation**: Enable parallel processing with `--parallel`

### Validation

Check generated instances:
```python
import pickle
with open('your_file.pkl', 'rb') as f:
    instances = pickle.load(f)

instance = instances[0]
print(f"Instance keys: {instance.keys()}")
print(f"Transport cost matrix shape: {instance['transport_cost'].shape}")
print(f"Fixed costs shape: {instance['fixed_costs'].shape}")
print(f"Valid cost matrix: {np.all(instance['transport_cost'] >= 0)}")
``` 