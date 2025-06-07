import os
import sys
import pickle
import random
import numpy as np
import shutil
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import multiprocessing
import networkx as nx

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.dataset_generators.utils__rb import get_random_instance as get_random_instance_RB
except ImportError:
    # Fallback to direct import if src module path doesn't work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from utils__rb import get_random_instance as get_random_instance_RB


DAI_SIZE_CONFIG = { 
    "er": {  1: (16,20), 2: (32,40), 3: (65,75), 4: (128,150), 5: (256,300), 6: (512,600), 7: (1024,1100)},
    "ba": {  1: (16,20), 2: (32,40), 3: (65,75), 4: (128,150), 5: (256,300), 6: (512,600), 7: (1024,1100)},
}

ER_DATASET_SIZE_CONFIG = {
    "small": (200,300),
    "large": (800,1200)
}

RB_DATASET_SIZE_CONFIG = {
    "small": (200,300),
    "large": (800,1200)
}

def gen_dai_paper_datasets(base_folder, num_graphs_to_generate = 10000):

    base_folder = base_folder + '/dai_paper'
    # Step 1: Make sure the base folder exists
    os.makedirs(base_folder, exist_ok=True)

    # Step 2: Loop through the size configurations and generate the datasets
    for mode in ["er", "ba"]:
        for i in [1, 2, 3, 4, 5, 6, 7]:
            min_n, max_n = DAI_SIZE_CONFIG[mode][i]
            print(f"Generating {num_graphs_to_generate} graphs for mode {mode} i = {i} with size config {DAI_SIZE_CONFIG[mode][i]}")
            folder = os.path.join(base_folder, f"{mode}_ngraphs_{num_graphs_to_generate}_nmin_{min_n}_nmax_{max_n}") 
            # create folder if it does not exist
            os.makedirs(folder, exist_ok=True)

            if mode == "er":
                python_cmd = f"python /Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization/other_git_repos/mis-benchmark-framework-master/main.py gendata random dummy {folder} --model er --min_n {min_n} --max_n {max_n} --num_graphs {num_graphs_to_generate} --er_p 0.15"
            else:
                python_cmd = f"python /Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization/other_git_repos/mis-benchmark-framework-master/main.py gendata random dummy {folder} --model ba --min_n {min_n} --max_n {max_n} --num_graphs {num_graphs_to_generate} --ba_m 4"
            
            # submit the python command using os
            os.system(python_cmd)


def gen_ER_dataset(base_folder, num_graphs_to_generate = 15000):
    # Step 1: Make sure the base folder exists
    os.makedirs(base_folder, exist_ok=True)

    mode = "er"
    for i in ["small", "large"]:
        min_n, max_n = ER_DATASET_SIZE_CONFIG[i]
        print(f"Generating {num_graphs_to_generate} graphs for mode {mode} i = {i} with size config {ER_DATASET_SIZE_CONFIG[i]}")
        folder = os.path.join(base_folder, f"{mode}_{i}_{num_graphs_to_generate}_0") 
        # create folder if it does not exist
        os.makedirs(folder, exist_ok=True)

        python_cmd = f"python /Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization/other_git_repos/mis-benchmark-framework-master/main.py gendata random dummy {folder} --model er --min_n {min_n} --max_n {max_n} --num_graphs {num_graphs_to_generate} --er_p 0.15"
        
        # submit the python command using os
        os.system(python_cmd)



def gen_RB_dataset(base_folder, num_graphs_to_generate = 15000):
    # Step 1: Make sure the base folder exists
    os.makedirs(base_folder, exist_ok=True)


    seed = 0 

    np.random.seed(seed=seed)

    for mode in ["small", "large"]:

        new_folder = os.path.join(base_folder, f"rb_{mode}_{num_graphs_to_generate}_{seed}")
        # create folder if it does not exist
        os.makedirs(new_folder, exist_ok=True)

        print("Final Output: {}".format(new_folder))
        print("Generating graphs...")

        if mode == "small":
            min_n, max_n = 200, 300
        elif mode == "large":
            min_n, max_n = 800, 1200

        for num_g in tqdm(range(num_graphs_to_generate)):
            path = Path(f'{new_folder}')
            stub = f"GR_{min_n}_{max_n}_{num_g}"
            while True:
                g, _ = get_random_instance_RB(mode)
                g.remove_nodes_from(list(nx.isolates(g)))
                if min_n <= g.number_of_nodes() <= max_n:
                    break
            output_file = path / (f"{stub}.gpickle")

            with open(output_file, 'wb') as f:
                pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
            print(f"Generated graph {path}")
    
    


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic graph datasets for combinatorial optimization.')
    
    parser.add_argument('--dataset_types', 
                        nargs='+', 
                        choices=['dai', 'er', 'rb'], 
                        default=['dai', 'er', 'rb'],
                        help='Dataset types to generate (default: all)')
    
    parser.add_argument('--num_graphs', 
                        type=int, 
                        default=12000,
                        help='Number of graphs to generate (default: 12000)')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='/Users/adeza3/Summer_2025/Research/Foundational_Combinatorial_Optimization/data/synthetic_graphs',
                        help='Output directory for generated graphs')
    
    parser.add_argument('--force', 
                        action='store_true',
                        help='Force delete existing output directory without prompting')
    
    args = parser.parse_args()
    
    base_folder = args.output_dir
    num_graphs = args.num_graphs
    
    # Handle existing folder
    if os.path.exists(base_folder):
        if args.force:
            shutil.rmtree(base_folder)
            print(f"Folder {base_folder} deleted (forced).")
        else:
            delete = input(f"Folder {base_folder} already exists. Do you want to delete it? (y/n): ")
            if delete.lower() == "y":
                shutil.rmtree(base_folder)
                print(f"Folder {base_folder} deleted.")
            else:
                print(f"Folder {base_folder} not deleted. Exiting.")
                return
    
    # Generate requested datasets
    if 'dai' in args.dataset_types:
        print(f"Generating DAI paper datasets with {num_graphs} graphs...")
        gen_dai_paper_datasets(base_folder, num_graphs)
        print("DAI paper datasets generated.")
    
    if 'er' in args.dataset_types:
        print(f"Generating ER datasets with {num_graphs} graphs...")
        gen_ER_dataset(base_folder, num_graphs)
        print("ER datasets generated.")
    
    if 'rb' in args.dataset_types:
        print(f"Generating RB datasets with {num_graphs} graphs...")
        gen_RB_dataset(base_folder, num_graphs)
        print("RB datasets generated.")
    
    print("All requested datasets generated successfully!")


if __name__ == "__main__":
    main()
