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

def get_pickle_files(dataset_folder: str):

    # Step 0: We need to get the pkl files
    pkl_files = [entry.path for entry in os.scandir(dataset_folder) if entry.is_file() and entry.name.endswith('.pkl')]
    pkl_files = [p for p in pkl_files if "results.pkl" not in p]  # Filter out the results files
    pkl_files = sorted(pkl_files, reverse=False)  # Sort the files in reverse order
    print("Found %d instance files in %s" % (len(pkl_files), dataset_folder))

    # Step 1: Filter out instances done
    instances_to_do = []
    for instance_file in pkl_files: 
        res_file = "{}/{}_results.pkl".format(dataset_folder, instance_file.split('/')[-1].split('.pkl')[0])
        #print(res_file)
        if not os.path.exists(res_file):
            instances_to_do.append(instance_file)
    print("\t\t Found %d unsolved instance files" % (len(instances_to_do)))

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
    k: int
) -> str:
    # Create a logs folder within the scratch directory.
    logs_folder = scratch_dir / "logs"
    logs_folder.mkdir(exist_ok=True, parents=True)
    
    instance_id = problem_type
    job_name = f"{job_name_prefix}_{instance_id}"
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

python {solver_script.resolve()} --pickle_files {pkl_files} --time_limit {time_limit} --n_threads 24
"""
    return script


def submit_jobs_graphs_syn_data_collection(args, N, p):

    # decipher problem_type
    problem_type_QAP = "{}_N{}_p{}".format(args.problem_type, N, p)
    dataset_folder = args.dataset_folder + "/{}".format(problem_type_QAP)

    instance_dir = Path(dataset_folder).resolve()
    scratch_dir = instance_dir / "scratch_{}".format(problem_type_QAP)
    solver_script = Path(args.solver_script).resolve()

    # Ensure the scratch directory exists.
    scratch_dir.mkdir(parents=True, exist_ok=True)
 
    gpickle_files = get_pickle_files(dataset_folder)
  
    print("\t\t Found %d unsolved instance files" % (len(gpickle_files)))


    gpickle_files = sorted(gpickle_files,reverse=False)  
    parts = np.array_split(np.arange(len(gpickle_files)), args.num_jobs_to_submit)

    for k,part in enumerate(parts):
        print("Submitting part %d of %d" % (k+1, len(parts)))
        current_files_submit = [gpickle_files[i] for i in part]
        current_files_submit = sorted(current_files_submit,reverse=True)

        if len(current_files_submit) == 0:
            print("No files to submit in this part")
            continue
        if len(current_files_submit) > 0:
            print("Submitting %d jobs for part %d" % (len(current_files_submit), k+1))

            pkl_files = ""
            for instance_file in current_files_submit:
                pkl_files += str(instance_file) + " "

            # get rid of last character (space) in pkl_files
            pkl_files = pkl_files.rstrip()
            
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
                problem_type = problem_type_QAP,
                k = k
            )
 
            # Write the SBATCH script to a temporary file.
            with open(temp_script_path, "w") as f:
                f.write(sbatch_content)
 
            # Submit the job using SLURM's sbatch command.
            submit_cmd = ["sbatch", str(temp_script_path)]
            try:
                subprocess.run(submit_cmd, check=True)
                #logging.info("\n\n\t\tJob submitted for part = %d", k)
            except subprocess.CalledProcessError as e:
                logging.error("Failed to submit job for part = %d: %s", k, e)
 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Master job submitter for Graph instances."  )
    parser.add_argument( "--dataset_folder", type=str, default='data/synthetic/qap',help="Directory containing instance (.pkl) files." )
    parser.add_argument( "--problem_type", type=str, default="0_1", help="") 
    parser.add_argument( "--N", type=int, nargs="+", default=[12], help="One or more problem sizes N (e.g. --N 5 10 20).") 
    parser.add_argument( "--p", type=float, nargs="+", default=[0.8], help="One or more density values p (e.g. --p 0.25 0.5 0.75).")
    parser.add_argument("--num_jobs_to_submit", type=int, default=50, help="Number of jobs to submit")
    parser.add_argument( "--solver_script", type=str, default = 'src/qap/solve_instances.py', help="Path to the qap_solver script." )
    parser.add_argument("--job_time", type=str, default="08:00:00", help="Wall time for each job (default: 01:00:00).")
    parser.add_argument("--cpus", type=int, default=24, help="Number of CPUs per job (default: 1).")
    parser.add_argument("--mem",  type=str, default="60G", help="Memory per job (default: 4G)." )
    parser.add_argument("--account", type=str, default= "gts-phentenryck3-coda20", required=False,  help="SLURM account to charge." )
    parser.add_argument("--qos", type=str, default="embers", help="SLURM QoS level (default: normal).")
    parser.add_argument("--job_name_prefix",type=str,default="nuni", help="Prefix for SLURM job names (default: qubo)." )
    parser.add_argument("--time_limit", type=int, default=3600*4, help="Time limit for the MC solver (in seconds, default: 3600).")
    return parser.parse_args()

 
if __name__ == "__main__":
    logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_arguments()
    for N in args.N:
        for p in args.p:
            logging.info(f"Submitting jobs for N={N}, p={p}")
            submit_jobs_graphs_syn_data_collection(args, N = N, p = p)