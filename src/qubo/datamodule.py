from src.lit import ML_4_Combinatorial_Optimization_DataModule
from torch.utils.data import TensorDataset, Dataset
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F


class QUBODataModule(ML_4_Combinatorial_Optimization_DataModule):
    """
    Data module for QUBO problems.
    Loads the QUBO dataset, builds the dataset with sparse problem data, and populates the cache with the initial historical best solutions.
    """ 
    def load_dataset_synthetic(self, dataset_folder, max_num_instances, max_num_instances_per_size):
        """
        Load up to `max_num_instances` QUBO instances from the list of folders.
        From each folder, load at most `max_num_instances_per_size` instances
        that have a non‐None 'solution'.

        Each .npy file is assumed to contain an array (or list) of dicts,
        where each dict has at least:
        - "Q"        : numpy array, shape (n_i, n_i)
        - "solution" : numpy array, shape (n_i,), with 0/1 entries
        - "objective": float

        Returns:
            data: a list of tuples (Q_t, sol_t, obj_val, n_i, hash_id), where
                Q_t    : torch.FloatTensor of shape (n_i, n_i)
                sol_t  : torch.FloatTensor of shape (n_i,)
                obj_val: float
                n_i    : int
                hash_id: int (unique index in [0 .. N-1])
        """
        data = []
        total_count = 0

        for folder in dataset_folder:
            print("\t Looking at data in ",folder)
            if total_count >= max_num_instances:
                break

            per_size_count = 0
            if self.completely_unsupervised == False:
                npy_paths = list(Path(folder).rglob("*results*.npy"))
            else:
                all_npy_files = Path(folder).rglob("*.npy")
                npy_paths = [p for p in all_npy_files if "results" not in p.name]

            for npy_path in npy_paths:
                if total_count >= max_num_instances or per_size_count >= max_num_instances_per_size:
                    break

                try:
                    instances = np.load(npy_path, allow_pickle=True)
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    continue

                for inst in instances:
                    if total_count >= max_num_instances or per_size_count >= max_num_instances_per_size:
                        break
 

                    hash_id = total_count 
                    if self.completely_unsupervised == False:
                        Q_np = inst["Q"]
                        Q_t = torch.from_numpy(Q_np).float()       # (n_i, n_i)
                        n_i = Q_t.shape[0]
                        sol_np = inst.get("solution", None)
                        if sol_np is None:
                            continue
                        obj_val = inst["objective"]
                        sol_t = torch.tensor(sol_np, dtype=torch.float32)   # (n_i,)
                        obj_f = float(obj_val)
                        data.append((Q_t, sol_t, obj_f, n_i, hash_id))
                    else:
                        Q_np = inst
                        Q_t = torch.from_numpy(Q_np).float()       # (n_i, n_i)
                        n_i = Q_t.shape[0]
                        data.append((Q_t, n_i, hash_id))
                    
                    total_count += 1
                    per_size_count += 1

            print(f"\t--> Loaded {per_size_count} instances from '{Path(folder).name}'")

        print(f"\t--> Total loaded: {total_count} instances.")
        return data
    
    def build_dense_padded_tensordataset(
            self, raw_data, mode
        ) -> TensorDataset:
            """
            Given raw_data = List of (Q_t, sol_t, obj_val, n_i, hash_id),
            builds a TensorDataset of dense‐padded Q, sol, mask, hash_id, and obj_val.

            Returns:
                TensorDataset with five tensors:
                - Q_tensor   : FloatTensor of shape (N, max_n, max_n)
                - sol_tensor : FloatTensor of shape (N, max_n)
                - mask_tensor: FloatTensor of shape (N, max_n)
                - hash_ids   : LongTensor  of shape (N,)
                - obj_vals   : FloatTensor of shape (N,)
            """
            # Sort by hash_id to ensure ordering
            if (self.completely_unsupervised == False) or (self.completely_unsupervised and mode == "rw"): 
                raw_sorted = sorted(raw_data, key=lambda x: x[4])
                N = len(raw_sorted)
                max_n = max(item[3] for item in raw_sorted)  # maximum n_i across all instances
            else:
                raw_sorted = sorted(raw_data, key=lambda x: x[2])
                N = len(raw_sorted)
                max_n = max(item[1] for item in raw_sorted)  # maximum n_i across all instances

            # Preallocate
            Q_tensor = torch.zeros((N, max_n, max_n), dtype=torch.float32)
            mask_tensor = torch.zeros((N, max_n), dtype=torch.float32)
            hash_ids = torch.zeros((N,), dtype=torch.long)

            if (self.completely_unsupervised == False) or (self.completely_unsupervised and mode == "rw"):
                sol_tensor = torch.zeros((N, max_n), dtype=torch.float32)
                obj_vals = torch.zeros((N,), dtype=torch.float32)

            for idx, inst in enumerate(raw_sorted):

                if (self.completely_unsupervised == False) or (self.completely_unsupervised and mode == "rw"):
                    Q_t, sol_t, obj_f, n_i, h_id =  inst
                    pad_amt = max_n - n_i
                    # Pad Q_t to (max_n, max_n): 
                    Q_padded = F.pad(Q_t, (0, pad_amt, 0, pad_amt))  # shape (max_n, max_n)
                    # Pad sol_t to length max_n
                    sol_padded = F.pad(sol_t, (0, pad_amt))          # shape (max_n,)
                    # Build mask: 1.0 for first n_i entries, 0.0 otherwise
                    mask = torch.zeros((max_n,), dtype=torch.float32)
                    mask[:n_i] = 1.0
                    sol_tensor[idx] = sol_padded
                    obj_vals[idx] = obj_f
                else:
                    Q_t, n_i, h_id = inst
                    pad_amt = max_n - n_i
                    # Pad Q_t to (max_n, max_n): 
                    Q_padded = F.pad(Q_t, (0, pad_amt, 0, pad_amt))  # shape (max_n, max_n)
                    # Build mask: 1.0 for first n_i entries, 0.0 otherwise
                    mask = torch.zeros((max_n,), dtype=torch.float32)
                    mask[:n_i] = 1.0

                Q_tensor[idx] = Q_padded
                mask_tensor[idx] = mask
                hash_ids[idx] = h_id

                #############################################
                ############# do improve cache  #############
                # Get init sol of all ones, compute obj and pad
                sol_init = torch.ones((n_i), dtype=torch.float32)
                obj_init = (sol_init @ Q_t @ sol_init).item()
                sol_init_padded = F.pad(sol_init, (0, pad_amt))
                if mode == "synthetic":
                    self.IMPROVE_CACHE[h_id] = [sol_init_padded, obj_init]
                elif mode == "rw":
                    self.IMPROVE_CACHE_rw[h_id] = [sol_init_padded, obj_init]
                #############################################
                #############################################
            # Create TensorDataset
            if (self.completely_unsupervised == False) or (self.completely_unsupervised and mode == "rw"):
                dataset = TensorDataset(Q_tensor, sol_tensor, mask_tensor, hash_ids, obj_vals)
            else:
                dataset = TensorDataset(Q_tensor, mask_tensor, hash_ids)
            return dataset
    
    def prepare_dataset_synthetic(self) -> Dataset:  
        raw = self.load_dataset_synthetic(self.hparams["dataset"]["folder"], self.hparams["dataset"]["max_num_instances"],self.hparams["dataset"]["max_num_instances_per_size"])
        ds = self.build_dense_padded_tensordataset(raw, mode = "synthetic")
        return ds
    def prepare_dataset_real_world(self, real_world_folders) -> Dataset:
        benchmark_ds = []
        rw_hash_id = 0
        for real_world_folder in real_world_folders:
            curr_data = []
            print(real_world_folder)
            # Corrected and optimized file searching
            all_results_files = Path(real_world_folder).rglob("*results*.npy") 
            
            for npy_path in all_results_files: # npy_path is a Path object
                res = np.load(npy_path, allow_pickle=True).item() # Use Path object directly
                Q = torch.from_numpy(res['Q']).float()       # (n_i, n_i)
                solution = torch.FloatTensor(res['solution'])      # (n_i, n_i)
                objective = float(res['objective'])
                data_tuple = (Q, solution, objective, Q.shape[0], rw_hash_id)
                curr_data.append(data_tuple)

                rw_hash_id +=1
            
            ds = self.build_dense_padded_tensordataset(curr_data, mode = "rw")
            benchmark_ds.append(ds)
        return benchmark_ds
