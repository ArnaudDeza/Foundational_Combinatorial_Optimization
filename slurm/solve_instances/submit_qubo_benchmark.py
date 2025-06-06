import os
import logging
import numpy as np
import argparse
import subprocess
import sys
from pathlib import Path

# Configure logger for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_npy_files(dataset_folder: str):
    """
    Returns all '*.npy' files in `dataset_folder` for which
    there is no corresponding '<basename>_results.npy' file.
    """
    instance_files = []
    result_basenames = set()

    # Scan for instance and result files
    for entry in os.scandir(dataset_folder):
        if not entry.is_file() or not entry.name.endswith(".npy"):
            continue

        if entry.name.endswith("_results.npy"):
            base = entry.name[:-len("_results.npy")]
            result_basenames.add(base)
        else:
            instance_files.append(entry.path)

    # Filter out instances that already have results
    unsolved_files = [
        fp for fp in instance_files if Path(fp).stem not in result_basenames
    ]
    
    logger.info(f"Directory: {dataset_folder}")
    logger.info(f"  Found {len(instance_files)} total instances.")
    logger.info(f"  Found {len(unsolved_files)} unsolved instances.")
    
    return sorted(unsolved_files)


def get_all_benchmark_folders(root_dir: str):
    """
    Recursively finds all directories containing .npy files.
    """
    benchmark_folders = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(name.endswith('.npy') and not name.endswith('_results.npy') for name in filenames):
            benchmark_folders.append(dirpath)
    return benchmark_folders


def create_sbatch_script(
    npy_files_str: str,
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
    k: int,
    solver_type: str,
    gurobi_threads_per_solve: int,
) -> str:
    logs_folder = scratch_dir / "logs"
    logs_folder.mkdir(exist_ok=True, parents=True)
    
    job_name = f"{job_name_prefix}_{problem_type}_{solver_type}"
    out_log = logs_folder / f"{job_name}_{k}.out"
    err_log = logs_folder / f"{job_name}_{k}.err"

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

module load gurobi mamba
mamba activate foundationalML

python {solver_script.resolve()} \\
    --npy_files {npy_files_str} \\
    --time_limit {time_limit} \\
    --n_threads {job_cpus} \\
    --solver_type {solver_type} \\
    --gurobi_threads_per_solve {gurobi_threads_per_solve}
"""
    return script


def submit_jobs_for_folder(dataset_folder_path: str, args: argparse.Namespace):
    """
    Submits SLURM jobs for all unsolved .npy files in a given folder.
    """
    instance_dir = Path(dataset_folder_path).resolve()
    problem_type = f"{instance_dir.parent.name}_{instance_dir.name}"
    
    scratch_dir = instance_dir / f"scratch_solve_{args.solver_type}"
    solver_script = Path(args.solver_script).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
 
    npy_files = get_npy_files(dataset_folder_path)
    if not npy_files:
        logger.info(f"No unsolved instances in {dataset_folder_path}. Skipping.")
        return

    parts = np.array_split(npy_files, args.num_jobs_to_submit)

    for k, part_files in enumerate(parts):
        if not part_files.size > 0:
            logger.info(f"Part {k+1} is empty, skipping submission.")
            continue

        logger.info(f"Submitting {len(part_files)} instances in job part {k+1}/{len(parts)}.")
        
        npy_files_str = " ".join(map(str, part_files))
        temp_script_path = scratch_dir / f"sbatch_{k}.sh"
        
        sbatch_content = create_sbatch_script(
            npy_files_str=npy_files_str,
            scratch_dir=scratch_dir,
            job_time=args.job_time,
            job_cpus=args.cpus,
            job_mem=args.mem,
            account=args.account,
            qos=args.qos,
            job_name_prefix=args.job_name_prefix,
            solver_script=solver_script,
            time_limit=args.time_limit,
            problem_type=problem_type,
            k=k,
            solver_type=args.solver_type,
            gurobi_threads_per_solve=args.gurobi_threads_per_solve,
        )

        with open(temp_script_path, "w") as f:
            f.write(sbatch_content)
 
        '''submit_cmd = ["sbatch", str(temp_script_path)]
        try:
            subprocess.run(submit_cmd, check=True)
            logger.info(f"Job submitted for part {k+1} from {instance_dir.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job for part {k+1}: {e}")'''

 
def parse_arguments():
    project_root = Path(__file__).resolve().parents[2]
    default_solver_script = project_root / 'src' / 'qubo' / 'solve_instances.py'
    default_benchmark_dir = project_root / 'data' / 'qubo' / 'benchmarks'

    parser = argparse.ArgumentParser(
        description="Master job submitter for QUBO benchmark instances."
    )

    path_group = parser.add_argument_group('Path Configuration')
    path_group.add_argument(
        "--benchmark_root_dir",
        type=str,
        default=str(default_benchmark_dir),
        help="Root directory containing the QUBO benchmark sub-folders."
    )
    path_group.add_argument(
        "--solver_script",
        type=str,
        default=str(default_solver_script),
        help="Path to the Python script that solves QUBO instances."
    )

    submission_group = parser.add_argument_group('Job Submission Configuration')
    submission_group.add_argument(
        "--num_jobs_to_submit",
        type=int,
        default=1,
        help="Number of SLURM jobs to submit PER BENCHMARK SUB-FOLDER."
    )

    slurm_group = parser.add_argument_group('SLURM Configuration')
    slurm_group.add_argument("--job_name_prefix", type=str, default="QUBO_b", help="Prefix for SLURM job names.")
    slurm_group.add_argument("--job_time", type=str, default="02:00:00", help="Wall time limit for each SLURM job (e.g., 'HH:MM:SS').")
    slurm_group.add_argument("--cpus", type=int, default=24, help="Number of CPUs to request per SLURM job.")
    slurm_group.add_argument("--mem", type=str, default="60G", help="Memory to request per SLURM job (e.g., '60G').")
    slurm_group.add_argument("--account", type=str, default="gts-phentenryck3-coda20", required=False, help="SLURM account to charge.")
    slurm_group.add_argument("--qos", type=str, default="embers", help="SLURM Quality of Service (QoS) level.")

    solver_group = parser.add_argument_group('Solver Configuration')
    solver_group.add_argument("--time_limit", type=int, default=600, help="Time limit/solve in seconds.")
    solver_group.add_argument(
        "--solver_type",
        type=str,
        default="GUROBI",
        choices=["GUROBI", "GUROBI_OPTIMODS", "TABU_SEARCH", "SIMULATED_ANNEALING"],
        help="The solver to use for the QUBO instances."
    )
    solver_group.add_argument(
        "--gurobi_threads_per_solve",
        type=int,
        default=2,
        help="Number of threads for Gurobi to use for each solve."
    )
    return parser.parse_args()

 
def main():
    args = parse_arguments()
    benchmark_folders = get_all_benchmark_folders(args.benchmark_root_dir)

    if not benchmark_folders:
        logging.warning("No benchmark folders with .npy files found. Exiting.")
        return

    logging.info(f"Found {len(benchmark_folders)} benchmark datasets to process.")
    for folder in benchmark_folders:
        submit_jobs_for_folder(folder, args)

if __name__ == "__main__":
    main()
