from copy import deepcopy
from abc import abstractmethod 
import os, glob, torch 
from torch.utils.data import Dataset, TensorDataset
import numpy as np 
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Dataset 
import lightning.pytorch as pl  
import numpy as np
import torch.nn.functional as F
from abc import ABC, abstractmethod  
from typing import List, Tuple

class CSEQModel(torch.nn.Module, ABC):
    """Abstract class for CSEQ models. Just has to implement `batch_loss`."""

    def __init__(self, hparams ):
        # hparams and dm should be used in subclasses
        super().__init__()


class ML_4_Combinatorial_Optimization_DataModule(pl.LightningDataModule):
    """
    Base data module that implements common functionality.
    Subclasses need to implement `prepare_dataset()` for their specific problem type.
    """
    def __init__(self, hparams: dict):
        super().__init__()
        # Deepcopy and save hyperparameters for reproducibility. 
        self.save_hyperparameters(deepcopy(hparams))
        # This cache will store additional data for the unsupervised training with a historical best solution
        self.IMPROVE_CACHE = {}
        self.IMPROVE_CACHE_rw = {}
        self.completely_unsupervised = self.hparams["dataset"]["completely_unsupervised"]
        # Load and prepare the dataset via the subclass
        print("\n\n \t\t\t Prepare the synthetic train dataset \n\n")
        self.ds = self.prepare_dataset_synthetic()

        ###############################################################
        if self.hparams["dataset"]["validate_on_real_world"]:
            self.benchmark_ds = self.prepare_dataset_real_world(self.hparams["dataset"]["real_world_folders"])
        ###############################################################

        # Adjust the dataset splits if they are provided as ratios
        N = len(self.ds)
        splits = self.hparams["dataset"]["splits"]
        if 0.95 < sum(splits) < 1.05:
            n_train = int(splits[0] * N)
            n_val = int(splits[1] * N)
            n_test = N - n_train - n_val
            self.hparams["dataset"]["splits"] = (n_train, n_val, n_test)
        # Ensure that the splits are valid
        assert sum(self.hparams["dataset"]["splits"]) == len(self.ds), "`splits` must sum to the length of the dataset."
        assert len(self.hparams["dataset"]["splits"]) == 3, "`splits` must be a tuple of length 3: (n_train, n_val, n_test)."
    @abstractmethod
    def prepare_dataset_synthetic(self) -> Dataset:
        """
        Load and process the dataset.
        Must be implemented by subclasses.
        """
        pass
    @abstractmethod
    def prepare_dataset_real_world(self) -> Dataset:
        """
        Load and process the dataset.
        Must be implemented by subclasses.
        """
        pass
    @abstractmethod
    def load_files_synthetic_dataset(self) -> Dataset:
        """
        Load the files of the synthetic dataset.
        """
        pass
    @abstractmethod
    def load_files_real_world_dataset(self) -> Dataset:
        """
        Load the files of the real-world/benchmark dataset.
        """
        pass
    def setup(self, stage=None):
        """Split the dataset into train, validation, and test sets."""
        self.train, self.val, self.test = random_split( self.ds, self.hparams["dataset"]["splits"], generator=torch.Generator().manual_seed(self.hparams["seeds"]["split"]))
    def train_dataloader(self):
        ''' train dataloader configs det by self.hparams'''
        return DataLoader(self.train, generator=torch.Generator().manual_seed(self.hparams["seeds"]["train_dataloader"]), **self.hparams["dataloader"]["train"])
    def val_dataloader(self):
        ''' validation dataloader configs det by self.hparams'''
        if self.hparams["dataset"]["validate_on_real_world"]:
            datasets = [DataLoader(self.val, generator=torch.Generator().manual_seed(self.hparams["seeds"]["val_dataloader"]), **self.hparams["dataloader"]["val"])]
            for rw_ds in self.benchmark_ds: 
                datasets.append(DataLoader(rw_ds, generator=torch.Generator().manual_seed(self.hparams["seeds"]["val_dataloader"]), batch_size = 256,shuffle = False, num_workers = 2, persistent_workers = True, pin_memory = True))
            return datasets
        else:
            return DataLoader( self.val, generator=torch.Generator().manual_seed(self.hparams["seeds"]["val_dataloader"]), **self.hparams["dataloader"]["val"])
    def test_dataloader(self):
        ''' test dataloader configs det by self.hparams'''
        if self.hparams["dataset"]["validate_on_real_world"]:
            datasets = [DataLoader(self.test, generator=torch.Generator().manual_seed(self.hparams["seeds"]["test_dataloader"]), **self.hparams["dataloader"]["test"])]
            for rw_ds in self.benchmark_ds:
                datasets.append(DataLoader(rw_ds, generator=torch.Generator().manual_seed(self.hparams["seeds"]["test_dataloader"]),batch_size = 1,shuffle = False, num_workers = 2, persistent_workers = True, pin_memory = True))
            return datasets
        else:
            return DataLoader(self.test, generator=torch.Generator().manual_seed(self.hparams["seeds"]["test_dataloader"]), **self.hparams["dataloader"]["test"])
 
