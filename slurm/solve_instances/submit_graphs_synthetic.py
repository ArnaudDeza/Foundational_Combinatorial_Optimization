import os
import logging
import numpy as np
import argparse
import logging
import subprocess
import sys
from pathlib import Path 

# Configure logger for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_pickle_files(dataset_folder: str, problem_type: str):
    """Get list of .gpickle files that still need to be solved."""
    # Step 0: We need to get the graph files
    gpickle_files = [entry.path for entry in os.scandir(dataset_folder) if entry.is_file() and entry.name.endswith('.gpickle')]
    gpickle_files = sorted(gpickle_files, reverse=False)  # Sort the files in reverse order
    print("Found %d instance files in %s" % (len(gpickle_files), dataset_folder))
    
    # Step 1: Do we need to filter out any instances already done?
    # Check for individual result files in the same directory as the graphs
    instances_to_do = []
    for instance_file in gpickle_files: 
        instance_id = str(instance_file).split('/')[-1].split('.gpickle')[0]  # Extract the instance ID from the file name
        # Check for result file in same directory as graph file
        instance_dir = os.path.dirname(instance_file)
        results_file = os.path.join(instance_dir, f"{instance_id}_{problem_type}_results.npy")
        if not os.path.exists(results_file):
            instances_to_do.append(instance_file)
        else:
            logger.debug(f"Results already exist for {instance_id}")
            
    print(f"Found {len(instances_to_do)} unsolved instances out of {len(gpickle_files)} total")
    return instances_to_do


def create_sbatch_script(
    pkl_files: str,
    scratch_dir: Path,
    job_time: str,
    job_cpus: int,
    job_mem: str,
    account: str,
    qos: str,
    job_name_prefix: str,
    solver_script: Path,
    time_limit: int,
    problem_type: str,
    max_threads: int,
    nparallel: int,
    quadratic: bool,
    max_colors: int,
    k: int
) -> str:
    """Create SBATCH script content for SLURM job submission."""
    # Create a logs folder within the scratch directory.
    logs_folder = scratch_dir / "logs"
    logs_folder.mkdir(exist_ok=True, parents=True)
    
    instance_id = problem_type
    job_name = f"{job_name_prefix}_{instance_id}"
    out_log = logs_folder / f"{job_name}_{k}.out"
    err_log = logs_folder / f"{job_name}_{k}.err"

    # Build the command arguments for solve_instances.py
    cmd_args = [
        f"--graph_files {pkl_files}",  # Updated to match solve_instances.py interface
        f"--problem_type {problem_type}",
        f"--time_limit {time_limit}",
        f"--nparallel {nparallel}",
        f"--max_threads {max_threads}",
        "--save_individual"  # Always save individual results
    ]
    
    # Add optional arguments
    if quadratic:
        cmd_args.append("--quadratic")
    
    if max_colors is not None and max_colors > 0:
        cmd_args.append(f"--max_colors {max_colors}")
    
    # Join all arguments
    solver_args = " ".join(cmd_args)

    # Build the SBATCH script content.
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --output={out_log}
#SBATCH --error={err_log}
#SBATCH --time={job_time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={job_cpus}
#SBATCH --qos={qos}
#SBATCH --mem={job_mem}

# Load required modules
module load gurobi mamba

# Activate conda environment
mamba activate jssp

# Set working directory to the solver script location
cd {solver_script.parent}

# Run the solver
python {solver_script.name} {solver_args}

echo "Job completed successfully"
"""
    return script


def submit_jobs_graphs_syn_data_collection(args):
    """Submit SLURM jobs for solving graph instances."""
    
    instance_dir = Path(args.dataset_folder).resolve()
    scratch_dir = instance_dir / "scratch_{}".format(args.problem_type)
    solver_script = Path(args.solver_script).resolve()

    # Ensure the scratch directory exists.
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate solver script exists
    if not solver_script.exists():
        raise FileNotFoundError(f"Solver script not found: {solver_script}")
    
    # Get list of unsolved pickle files
    gpickle_files = get_pickle_files(args.dataset_folder, args.problem_type)
    
    if len(gpickle_files) == 0:
        print("No unsolved instances found. All instances may already be solved.")
        return
    
    print(f"\t\t Found {len(gpickle_files)} unsolved instance files")
    
    # Sort files for consistent processing
    gpickle_files = sorted(gpickle_files, reverse=False)  
    
    # Split files into chunks for parallel processing
    parts = np.array_split(np.arange(len(gpickle_files)), args.num_jobs_to_submit)

    jobs_submitted = 0
    for k, part in enumerate(parts):
        current_files_submit = [gpickle_files[i] for i in part]
        current_files_submit = sorted(current_files_submit, reverse=False)

        if len(current_files_submit) == 0:
            print(f"No files to submit in part {k+1}")
            continue
            
        print(f"Submitting job {k+1}/{len(parts)} with {len(current_files_submit)} instances")

        # Create space-separated string of file paths
        pkl_files = " ".join(str(f) for f in current_files_submit)
        
        # Create SBATCH script
        temp_script_path = scratch_dir / f"sbatch_{k}.sh"
        sbatch_content = create_sbatch_script(
            pkl_files=pkl_files,
            scratch_dir=scratch_dir,
            job_time=args.job_time,
            job_cpus=args.cpus,
            job_mem=args.mem,
            account=args.account,
            qos=args.qos,
            job_name_prefix=args.job_name_prefix,
            solver_script=solver_script,
            time_limit=args.time_limit,
            problem_type=str(args.problem_type),
            max_threads=args.max_threads,
            nparallel=args.nparallel_per_job,
            quadratic=args.quadratic,
            max_colors=args.max_colors,
            k=k
        )

        # Write the SBATCH script to a temporary file.
        with open(temp_script_path, "w") as f:
            f.write(sbatch_content)

        # Submit the job using SLURM's sbatch command.
        if not args.dry_run:
            submit_cmd = ["sbatch", str(temp_script_path)]
            try:
                result = subprocess.run(submit_cmd, check=True, capture_output=True, text=True)
                job_id = result.stdout.strip().split()[-1]
                logger.info(f"Job {k+1} submitted successfully with ID: {job_id}")
                jobs_submitted += 1
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to submit job for part {k+1}: {e}")
                logger.error(f"Error output: {e.stderr}")
        else:
            print(f"DRY RUN: Would submit job {k+1} with script:")
            print(f"  Script path: {temp_script_path}")
            print(f"  Files: {len(current_files_submit)} instances")
            jobs_submitted += 1
    
    print(f"\nSummary: {jobs_submitted} jobs submitted for {len(gpickle_files)} instances")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SLURM job submitter for graph combinatorial optimization instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and problem configuration
    parser.add_argument(
        "--dataset_folder", 
        type=str,
        required=True,
        help="Directory containing instance (.gpickle) files."
    )
    parser.add_argument(
        "--problem_type", 
        type=str, 
        required=True,
        choices=['max_cut', 'mis', 'maximum_independent_set', 'mvc', 'minimum_vertex_cover',
                'max_clique', 'maximum_clique', 'graph_coloring', 'coloring',
                'min_dominating_set', 'dominating_set', 'total_dominating_set'],
        help="Type of graph problem to solve"
    )
    
    # Solver configuration
    parser.add_argument(
        "--solver_script", 
        type=str,
        default='../../src/graphs/solve_instances.py',
        help="Path to the solve_instances.py script."
    )
    parser.add_argument(
        "--time_limit", 
        type=int, 
        default=3600, 
        help="Time limit for each instance solve (in seconds)."
    )
    parser.add_argument(
        "--max_threads", 
        type=int, 
        default=4, 
        help="Maximum threads per Gurobi solver."
    )
    parser.add_argument(
        "--nparallel_per_job", 
        type=int, 
        default=6, 
        help="Number of parallel solves per SLURM job."
    )
    parser.add_argument(
        "--quadratic", 
        action="store_true",
        help="Use quadratic formulation where available (e.g., MIS)."
    )
    parser.add_argument(
        "--max_colors", 
        type=int, 
        default=None,
        help="Maximum colors for graph coloring problem."
    )
    
    # SLURM job configuration
    parser.add_argument(
        "--num_jobs_to_submit", 
        type=int, 
        default=10, 
        help="Number of SLURM jobs to submit"
    )
    parser.add_argument(
        "--job_time", 
        type=str, 
        default="04:00:00", 
        help="Wall time for each job."
    )
    parser.add_argument(
        "--cpus", 
        type=int, 
        default=24, 
        help="Number of CPUs per job."
    )
    parser.add_argument(
        "--mem", 
        type=str, 
        default="24G", 
        help="Memory per job."
    )
    parser.add_argument(
        "--account", 
        type=str, 
        default="gts-phentenryck3-coda20", 
        help="SLURM account to charge."
    )
    parser.add_argument(
        "--qos", 
        type=str, 
        default="embers", 
        help="SLURM QoS level."
    )
    parser.add_argument(
        "--job_name_prefix",
        type=str,
        default="graph_solve", 
        help="Prefix for SLURM job names."
    )
    
    # Utility options
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Print commands without actually submitting jobs."
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate inputs
    dataset_path = Path(args.dataset_folder)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    
    solver_path = Path(args.solver_script)
    if not solver_path.exists():
        raise FileNotFoundError(f"Solver script not found: {solver_path}")
    
    print(f"Dataset folder: {dataset_path}")
    print(f"Problem type: {args.problem_type}")
    print(f"Solver script: {solver_path}")
    print(f"Time limit: {args.time_limit}s")
    print(f"Jobs to submit: {args.num_jobs_to_submit}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 50)
    
    # Submit jobs
    submit_jobs_graphs_syn_data_collection(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()