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
import os

def get_npy_files(dataset_folder: str):
    """
    Returns all '*.npy' files in `dataset_folder` for which
    there is no corresponding '<basename>_results.npy' file.
    """

    # Step 0: Scan once and separate instance files vs. result files
    instance_files = []
    result_basenames = set()

    for entry in os.scandir(dataset_folder):
        if not entry.is_file():
            continue

        name = entry.name
        if not name.endswith(".npy"):
            continue

        if name.endswith("_results.npy"):
            # record base name (strip off "_results.npy")
            base = name[: -len("_results.npy")]
            result_basenames.add(base)
        else:
            # raw instance, record full path but keep basename for lookup
            instance_files.append(entry.path)

    instance_files.sort()  # sort in ascending order

    print(f"Found {len(instance_files)} instance files in {dataset_folder}")

    # Step 1: Filter out those that already have a '<basename>_results.npy'
    to_do = []
    for fullpath in instance_files:
        # get the filename without its ".npy" extension
        fname = os.path.basename(fullpath)
        basename = fname[: -len(".npy")]

        if basename not in result_basenames:
            to_do.append(fullpath)

    print(f"\t\tFound {len(to_do)} unsolved instance files")
    return to_do

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
    # Create a logs folder within the scratch directory.
    logs_folder = scratch_dir / "logs"
    logs_folder.mkdir(exist_ok=True, parents=True)
    
    instance_id = problem_type
    job_name = f"{job_name_prefix}_{instance_id}_{solver_type}"
    out_log = logs_folder / f"{job_name}_{k}.out"
    err_log = logs_folder / f"{job_name}_{k}.err"

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


def submit_jobs_qubo_syn_data_collection(args):
    dataset_folder = args.dataset_folder  

    problem_type_QUBO = dataset_folder.split('/')[-1]  # Extract the problem type from the folder name

    instance_dir = Path(dataset_folder).resolve()
    scratch_dir = instance_dir / "scratch_solve"
    solver_script = Path(args.solver_script).resolve()

    # Ensure the scratch directory exists.
    scratch_dir.mkdir(parents=True, exist_ok=True)
 
    npy_files = get_npy_files(dataset_folder)
  
    print("\t\t Found %d unsolved instance files" % (len(npy_files)))

    npy_files = sorted(npy_files,reverse=False)  
    parts = np.array_split(np.arange(len(npy_files)), args.num_jobs_to_submit)

    for k,part in enumerate(parts):
        print("Attempting to submit part %d of %d" % (k+1, len(parts)))
        current_files_submit = [npy_files[i] for i in part]
        current_files_submit = sorted(current_files_submit,reverse=True)

        if len(current_files_submit) == 0:
            print("No files to submit in this part")
            continue
        if len(current_files_submit) > 0:
            print("Submitting %d jobs for part %d" % (len(current_files_submit), k+1))

            npy_files_str = " ".join(map(str, current_files_submit))
            
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
                problem_type = problem_type_QUBO,
                k = k,
                solver_type=args.solver_type,
                gurobi_threads_per_solve=args.gurobi_threads_per_solve,
            )
 
            # Write the SBATCH script to a temporary file.
            with open(temp_script_path, "w") as f:
                f.write(sbatch_content)
 
            '''# Submit the job using SLURM's sbatch command.
            submit_cmd = ["sbatch", str(temp_script_path)]
            try:
                subprocess.run(submit_cmd, check=True)
                #logging.info("\n\n\t\tJob submitted for part = %d", k)
            except subprocess.CalledProcessError as e:
                logging.error("Failed to submit job for part = %d: %s", k, e)'''
 
def parse_arguments():
    # Determine the project root, which is 3 levels up from the current script.
    project_root = Path(__file__).resolve().parents[2]
    default_solver_script = project_root / 'src' / 'qubo' / 'solve_instances.py'
    default_dataset_folder = project_root / 'data' / 'qubo' / 'synthetic' / 'not_varying_density' / 'data_n_20'

    parser = argparse.ArgumentParser(
        description="Master job submitter for QUBO instances. This script generates and submits SLURM jobs."
    )

    path_group = parser.add_argument_group('Path Configuration')
    path_group.add_argument(
        "--dataset_folder",
        type=str,
        default=str(default_dataset_folder),
        help="Directory containing QUBO instance (.npy) files."
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
        help="Total number of SLURM jobs to submit."
    )

    slurm_group = parser.add_argument_group('SLURM Configuration')
    slurm_group.add_argument("--job_name_prefix", type=str, default="QUBO", help="Prefix for SLURM job names.")
    slurm_group.add_argument("--job_time", type=str, default="08:00:00", help="Wall time limit for each SLURM job (e.g., 'HH:MM:SS').")
    slurm_group.add_argument("--cpus", type=int, default=24, help="Number of CPUs to request per SLURM job.")
    slurm_group.add_argument("--mem", type=str, default="60G", help="Memory to request per SLURM job (e.g., '60G').")
    slurm_group.add_argument("--account", type=str, default="gts-phentenryck3-coda20", required=False, help="SLURM account to charge for the jobs.")
    slurm_group.add_argument("--qos", type=str, default="embers", help="SLURM Quality of Service (QoS) level.")

    solver_group = parser.add_argument_group('Solver Configuration')
    solver_group.add_argument(
        "--time_limit",
        type=int,
        default=3600 * 4,
        help="Time limit for each individual QUBO solve in seconds (default: 14400)."
    )
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

 
if __name__ == "__main__":
    logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_arguments()
    submit_jobs_qubo_syn_data_collection(args)