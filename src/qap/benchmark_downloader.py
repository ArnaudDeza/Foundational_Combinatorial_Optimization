import os
import requests
import tarfile
import zipfile
import argparse
import pickle
from pathlib import Path
import numpy as np
from .loader import load_qap_instance, load_qap_solution

'''
QAP Benchmark Downloader and Converter

Modified from https://github.com/alpyurtsever/NonconvexTOS/blob/main/Download_QAPLIB_data.m

Downloads QAPLIB instances and converts them to the pickle format used by our training framework.
'''

def download_file(url, dest_path):
    """Download file from URL and save it to dest_path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error on bad status
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_tar(file_path, extract_path):
    """Extract a tar.gz archive to the specified directory."""
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

def extract_zip(file_path, extract_path):
    """Extract a zip archive to the specified directory."""
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(path=extract_path)

def process_qaplib_instances(data_dir: str, output_dir: str):
    """
    Process downloaded QAPLIB instances and convert them to pickle format.
    
    Args:
        data_dir: Directory containing extracted QAPLIB data
        output_dir: Directory to save converted pickle files
    """
    qapdata_dir = os.path.join(data_dir, 'qapdata')
    qapsoln_dir = os.path.join(data_dir, 'qapsoln')
    
    if not os.path.exists(qapdata_dir):
        print(f"Warning: {qapdata_dir} not found. Skipping instance processing.")
        return
        
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .dat files (QAP instances)
    dat_files = []
    for root, dirs, files in os.walk(qapdata_dir):
        for file in files:
            if file.endswith('.dat'):
                dat_files.append(os.path.join(root, file))
    
    print(f"Found {len(dat_files)} QAP instance files")
    
    # Process instances by size
    instances_by_size = {}
    
    for dat_file in dat_files:
        try:
            # Load the QAP instance
            F, D = load_qap_instance(dat_file)
            n = F.shape[0]
            
            # Get instance name
            instance_name = os.path.splitext(os.path.basename(dat_file))[0]
            
            # Load solution if available
            solution_file = os.path.join(qapsoln_dir, f"{instance_name}.sln")
            solution = None
            objective = None
            
            if os.path.exists(solution_file):
                try:
                    # Use our robust solution loader
                    n_sol, objective, perm = load_qap_solution(solution_file)
                    
                    if n_sol != n:
                        print(f"Warning: Size mismatch in {solution_file}: expected {n}, got {n_sol}")
                    else:
                        # Convert permutation to assignment matrix
                        solution = np.zeros((n, n), dtype=float)
                        for i, j in enumerate(perm):
                            solution[i, j] = 1.0
                            
                except Exception as e:
                    print(f"Warning: Could not load solution for {instance_name}: {e}")
            
            # Create instance dictionary
            instance_dict = {
                "F": F,
                "D": D,
                "name": instance_name
            }
            
            if solution is not None:
                instance_dict["solution"] = solution
            if objective is not None:
                instance_dict["objVal"] = objective
            
            # Group by size
            if n not in instances_by_size:
                instances_by_size[n] = []
            instances_by_size[n].append(instance_dict)
            
        except Exception as e:
            print(f"Error processing {dat_file}: {e}")
            continue
    
    # Save instances grouped by size
    for size, instances in instances_by_size.items():
        size_dir = os.path.join(output_dir, f"qaplib_n{size}")
        os.makedirs(size_dir, exist_ok=True)
        
        # Save as pickle file
        pickle_file = os.path.join(size_dir, f"qaplib_n{size}_instances.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(instances, f)
        
        print(f"Saved {len(instances)} instances of size {size} to {pickle_file}")


def main():
    parser = argparse.ArgumentParser(description="Download and process QAPLIB benchmark instances")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/qap/benchmarks",
        help="Directory to save downloaded and processed data"
    )
    parser.add_argument(
        "--skip_download", 
        action="store_true",
        help="Skip download and only process existing data"
    )
    parser.add_argument(
        "--convert_only", 
        action="store_true",
        help="Only convert existing raw data to pickle format"
    )
    args = parser.parse_args()
    
    # Directory to store downloaded data
    data_dir = args.output_dir
    os.makedirs(data_dir, exist_ok=True)
    
    if not args.skip_download and not args.convert_only:
        # URLs for QAPLIB data
        qapdata_url = 'http://coral.ise.lehigh.edu/wp-content/uploads/2014/07/qapdata.tar.gz'
        qapsoln_url = 'http://coral.ise.lehigh.edu/wp-content/uploads/2014/07/qapsoln.tar.gz'
        
        # Paths to save the downloaded files
        qapdata_tar = os.path.join(data_dir, 'qapdata.tar.gz')
        qapsoln_tar = os.path.join(data_dir, 'qapsoln.tar.gz')
        
        # Download files
        print("Downloading qapdata...")
        download_file(qapdata_url, qapdata_tar)
        print("Downloading qapsoln...")
        download_file(qapsoln_url, qapsoln_tar)
        
        # Extract tar.gz archives
        print("Extracting qapdata.tar.gz...")
        extract_tar(qapdata_tar, data_dir)
        print("Extracting qapsoln.tar.gz...")
        extract_tar(qapsoln_tar, data_dir)
        
        # Update: Unzip the update file if it exists
        update_zip_path = os.path.join(data_dir, 'qapsoln_update.zip')
        if os.path.exists(update_zip_path):
            print("Extracting qapsoln_update.zip...")
            extract_zip(update_zip_path, data_dir)
        else:
            print("No update file found, skipping update.")
        
        print("QAPLIB data downloaded and extracted")
    
    # Process and convert instances to pickle format
    print("Converting QAPLIB instances to pickle format...")
    converted_dir = os.path.join(data_dir, "converted")
    process_qaplib_instances(data_dir, converted_dir)
    
    print("QAPLIB benchmark processing completed!")
    print(f"Converted instances available in: {converted_dir}")

if __name__ == '__main__':
    main()
