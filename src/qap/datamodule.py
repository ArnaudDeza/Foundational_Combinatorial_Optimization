from src.lit import ML_4_Combinatorial_Optimization_DataModule
from torch.utils.data import TensorDataset, Dataset
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
import pickle


class QAPDataModule(ML_4_Combinatorial_Optimization_DataModule):
    """
    Data module for QAP problems.
    Loads QAP datasets, builds datasets with problem data, and populates the cache with initial solutions.
    """ 
    def load_dataset_synthetic(self, dataset_folder, max_num_instances, max_num_instances_per_size):
        """
        Load up to `max_num_instances` QAP instances from the list of folders.
        From each folder, load at most `max_num_instances_per_size` instances
        that have the required data.

        Each .pkl file is assumed to contain a list of dicts, where each dict has:
        - "F"        : numpy array, shape (n_i, n_i) - flow matrix
        - "positions": numpy array, shape (n_i, 2) - facility positions 
        - "D"        : numpy array, shape (n_i, n_i) - distance matrix (for solved instances)
        - "solution" : numpy array, shape (n_i, n_i) - assignment matrix (for solved instances)
        - "objVal"   : float - objective value (for solved instances)

        Returns:
            data: a list of tuples (F_t, D_t, sol_t, obj_val, n_i, hash_id), where
                F_t     : torch.FloatTensor of shape (n_i, n_i) - flow matrix
                D_t     : torch.FloatTensor of shape (n_i, n_i) - distance matrix  
                sol_t   : torch.FloatTensor of shape (n_i, n_i) - assignment matrix
                obj_val : float
                n_i     : int
                hash_id : int (unique index in [0 .. N-1])
        """
        data = []
        total_count = 0

        for folder in dataset_folder:
            print("\t Looking at data in ", folder)
            if total_count >= max_num_instances:
                break

            per_size_count = 0
            if self.completely_unsupervised == False:
                pkl_paths = list(Path(folder).rglob("*results*.pkl"))
            else:
                all_pkl_files = Path(folder).rglob("*.pkl")
                pkl_paths = [p for p in all_pkl_files if "results" not in p.name]

            for pkl_path in pkl_paths:
                if total_count >= max_num_instances or per_size_count >= max_num_instances_per_size:
                    break

                try:
                    with open(pkl_path, "rb") as f:
                        instances = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")
                    continue

                for inst in instances:
                    if total_count >= max_num_instances or per_size_count >= max_num_instances_per_size:
                        break

                    hash_id = total_count 
                    if self.completely_unsupervised == False:
                        F_np = inst["F"]
                        F_t = torch.from_numpy(F_np).float()       # (n_i, n_i)
                        n_i = F_t.shape[0]
                        
                        # Handle distance matrix
                        if "D" in inst and inst["D"] is not None:
                            D_np = inst["D"] 
                            D_t = torch.from_numpy(D_np).float()
                        else:
                            # Compute distance matrix from positions if available
                            if "positions" in inst:
                                from scipy.spatial import distance_matrix
                                positions = inst["positions"]
                                D_np = distance_matrix(positions, positions)
                                D_t = torch.from_numpy(D_np).float()
                            else:
                                continue  # Skip if no distance info available
                        
                        sol_np = inst.get("solution", None)
                        if sol_np is None:
                            continue
                        obj_val = inst.get("objVal", 0.0)
                        sol_t = torch.from_numpy(sol_np).float()   # (n_i, n_i)
                        obj_f = float(obj_val)
                        data.append((F_t, D_t, sol_t, obj_f, n_i, hash_id))
                    else:
                        F_np = inst["F"]
                        F_t = torch.from_numpy(F_np).float()       # (n_i, n_i)
                        n_i = F_t.shape[0]
                        
                        # For unsupervised, we need distance matrix
                        if "positions" in inst:
                            from scipy.spatial import distance_matrix
                            positions = inst["positions"]
                            D_np = distance_matrix(positions, positions)
                            D_t = torch.from_numpy(D_np).float()
                        elif "D" in inst and inst["D"] is not None:
                            D_np = inst["D"]
                            D_t = torch.from_numpy(D_np).float()
                        else:
                            continue  # Skip if no distance info available
                            
                        data.append((F_t, D_t, n_i, hash_id))
                    
                    total_count += 1
                    per_size_count += 1

            print(f"\t--> Loaded {per_size_count} instances from '{Path(folder).name}'")

        print(f"\t--> Total loaded: {total_count} instances.")
        return data
    
    def build_dense_padded_tensordataset(
            self, raw_data, mode
        ) -> TensorDataset:
            """
            Given raw_data = List of (F_t, D_t, sol_t, obj_val, n_i, hash_id) or (F_t, D_t, n_i, hash_id),
            builds a TensorDataset of dense‐padded F, D, sol, mask, hash_id, and obj_val.

            Returns:
                TensorDataset with tensors:
                - F_tensor   : FloatTensor of shape (N, max_n, max_n) - flow matrices
                - D_tensor   : FloatTensor of shape (N, max_n, max_n) - distance matrices  
                - sol_tensor : FloatTensor of shape (N, max_n, max_n) - assignment matrices (if supervised)
                - mask_tensor: FloatTensor of shape (N, max_n) - validity mask
                - hash_ids   : LongTensor  of shape (N,)
                - obj_vals   : FloatTensor of shape (N,) (if supervised)
            """
            # Sort by hash_id to ensure ordering
            if (self.completely_unsupervised == False) or (self.completely_unsupervised and mode == "rw"): 
                raw_sorted = sorted(raw_data, key=lambda x: x[5])  # hash_id is index 5
                N = len(raw_sorted)
                max_n = max(item[4] for item in raw_sorted)  # n_i is index 4
            else:
                raw_sorted = sorted(raw_data, key=lambda x: x[3])  # hash_id is index 3 for unsupervised
                N = len(raw_sorted)
                max_n = max(item[2] for item in raw_sorted)  # n_i is index 2 for unsupervised

            # Preallocate
            F_tensor = torch.zeros((N, max_n, max_n), dtype=torch.float32)
            D_tensor = torch.zeros((N, max_n, max_n), dtype=torch.float32)
            mask_tensor = torch.zeros((N, max_n), dtype=torch.float32)
            hash_ids = torch.zeros((N,), dtype=torch.long)

            if (self.completely_unsupervised == False) or (self.completely_unsupervised and mode == "rw"):
                sol_tensor = torch.zeros((N, max_n, max_n), dtype=torch.float32)
                obj_vals = torch.zeros((N,), dtype=torch.float32)

            for idx, inst in enumerate(raw_sorted):

                if (self.completely_unsupervised == False) or (self.completely_unsupervised and mode == "rw"):
                    F_t, D_t, sol_t, obj_f, n_i, h_id = inst
                    pad_amt = max_n - n_i
                    # Pad F_t to (max_n, max_n): 
                    F_padded = F.pad(F_t, (0, pad_amt, 0, pad_amt))  # shape (max_n, max_n)
                    D_padded = F.pad(D_t, (0, pad_amt, 0, pad_amt))  # shape (max_n, max_n)
                    # Pad sol_t to (max_n, max_n)
                    sol_padded = F.pad(sol_t, (0, pad_amt, 0, pad_amt))  # shape (max_n, max_n)
                    # Build mask: 1.0 for first n_i entries, 0.0 otherwise
                    mask = torch.zeros((max_n,), dtype=torch.float32)
                    mask[:n_i] = 1.0
                    sol_tensor[idx] = sol_padded
                    obj_vals[idx] = obj_f
                else:
                    F_t, D_t, n_i, h_id = inst
                    pad_amt = max_n - n_i
                    # Pad F_t to (max_n, max_n): 
                    F_padded = F.pad(F_t, (0, pad_amt, 0, pad_amt))  # shape (max_n, max_n)
                    D_padded = F.pad(D_t, (0, pad_amt, 0, pad_amt))  # shape (max_n, max_n)
                    # Build mask: 1.0 for first n_i entries, 0.0 otherwise
                    mask = torch.zeros((max_n,), dtype=torch.float32)
                    mask[:n_i] = 1.0

                F_tensor[idx] = F_padded
                D_tensor[idx] = D_padded
                mask_tensor[idx] = mask
                hash_ids[idx] = h_id

                #############################################
                ############# do improve cache  #############
                # Get initial random assignment, compute obj and pad
                sol_init = torch.eye(n_i, dtype=torch.float32)  # Identity as initial assignment
                obj_init = (F_t * (sol_init @ D_t @ sol_init.T)).sum().item()
                sol_init_padded = F.pad(sol_init, (0, pad_amt, 0, pad_amt))
                if mode == "synthetic":
                    self.IMPROVE_CACHE[h_id] = [sol_init_padded, obj_init]
                elif mode == "rw":
                    self.IMPROVE_CACHE_rw[h_id] = [sol_init_padded, obj_init]
                #############################################
                #############################################
            
            # Create TensorDataset
            if (self.completely_unsupervised == False) or (self.completely_unsupervised and mode == "rw"):
                dataset = TensorDataset(F_tensor, D_tensor, sol_tensor, mask_tensor, hash_ids, obj_vals)
            else:
                dataset = TensorDataset(F_tensor, D_tensor, mask_tensor, hash_ids)
            return dataset
    
    def prepare_dataset_synthetic(self) -> Dataset:  
        raw = self.load_dataset_synthetic(
            self.hparams["dataset"]["folder"], 
            self.hparams["dataset"]["max_num_instances"],
            self.hparams["dataset"]["max_num_instances_per_size"]
        )
        ds = self.build_dense_padded_tensordataset(raw, mode="synthetic")
        return ds
        
    def prepare_dataset_real_world(self, real_world_folders) -> Dataset:
        benchmark_ds = []
        rw_hash_id = 0
        for real_world_folder in real_world_folders:
            curr_data = []
            print(real_world_folder)
            # Find all results files
            all_results_files = Path(real_world_folder).rglob("*results*.pkl") 
            
            for pkl_path in all_results_files:
                try:
                    with open(pkl_path, "rb") as f:
                        results = pickle.load(f)
                    
                    for res in results:
                        if res.get("objVal") is None or res.get("solution") is None:
                            continue
                            
                        F = torch.from_numpy(res['F']).float()
                        if "D" in res and res["D"] is not None:
                            D = torch.from_numpy(res['D']).float()
                        else:
                            # Compute D from positions if available
                            if "positions" in res:
                                from scipy.spatial import distance_matrix
                                positions = res["positions"]
                                D_np = distance_matrix(positions, positions)
                                D = torch.from_numpy(D_np).float()
                            else:
                                continue
                                
                        solution = torch.from_numpy(res['solution']).float()
                        objective = float(res['objVal'])
                        data_tuple = (F, D, solution, objective, F.shape[0], rw_hash_id)
                        curr_data.append(data_tuple)
                        rw_hash_id += 1
                        
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")
                    continue
            
            if curr_data:
                ds = self.build_dense_padded_tensordataset(curr_data, mode="rw")
                benchmark_ds.append(ds)
        return benchmark_ds
