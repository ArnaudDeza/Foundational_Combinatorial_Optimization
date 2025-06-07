import argparse, json, os
from pathlib import Path  
import lightning.pytorch as pl
from copy import deepcopy
from src.utils import init_from_config 
import torch; torch.set_float32_matmul_precision("high") 
from lightning.pytorch import seed_everything; seed_everything(42)
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.plugins.environments import SLURMEnvironment; SLURMEnvironment.detect = lambda: False
import numpy as np
from lightning.pytorch.loggers import WandbLogger 
from abc import ABC, abstractmethod

from src import get_init_model
from src.utils import load_hyperparameters 
from src.lit import CSEQModel
from src.qubo.datamodule import QUBODataModule
from src.qubo.train_module import QUBOModule
from src.qap.datamodule import QAPDataModule
from src.qap.train_module import QAPModule
 
class DOP_Module(pl.LightningModule, ABC):
    """
    DOP_Module: Abstract base class for Discrete Optimization Proxy Modules
    Contains common functionality shared across all CO problems.
    """
    def __init__(self, hparams: dict, model: CSEQModel):
        super().__init__()
        self.save_hyperparameters(deepcopy(hparams))
        self.model = model
        self.problem_type = hparams["dataset"]["problem_type"] 
        self.completely_unsupervised = hparams["dataset"]["completely_unsupervised"]
        self.learning = hparams["model"]["learning"]

    def training_step(self, batch, _): 
        return self._step(batch, "train")
    def validation_step(self, batch, _, dataloader_idx): 
        return self._step(batch, "val", dataloader_idx)
    def test_step(self, batch, stage, dataloader_idx):       
        return self._step(batch, "test", dataloader_idx)
    
    def configure_optimizers(self):
        optimizer_config = self.hparams["optimizer"] 
        self.optimizer = init_from_config(optimizer_config, torch.optim, params=self.model.parameters())
        d = {"optimizer": self.optimizer}
        if self.hparams.get("lr_scheduler"):
            self.lr_scheduler = init_from_config( self.hparams["lr_scheduler"], torch.optim.lr_scheduler, optimizer=self.optimizer )
            d["lr_scheduler"] = self.lr_scheduler
        return d 
     
    def _move_cache_to_device(self, device):
        # Helper to move cache tensors to the specified device
        for cache_name in ["IMPROVE_CACHE", "IMPROVE_CACHE_rw"]:
            cache = getattr(self, cache_name, None)
            if cache: # Ensure cache exists and is not empty
                for key in cache:
                    # Cache items are [tensor, float]
                    if isinstance(cache[key], list) and len(cache[key]) > 0 and isinstance(cache[key][0], torch.Tensor):
                        if cache[key][0].device != device:
                            cache[key][0] = cache[key][0].to(device)

    def on_train_start(self):
        super().on_train_start() 
        if hasattr(self, 'IMPROVE_CACHE') and hasattr(self, 'IMPROVE_CACHE_rw'):
             self._move_cache_to_device(self.device)

    def on_validation_start(self):
        super().on_validation_start()
        if hasattr(self, 'IMPROVE_CACHE') and hasattr(self, 'IMPROVE_CACHE_rw'):
            self._move_cache_to_device(self.device)
    
    def on_test_start(self):
        super().on_test_start()
        if hasattr(self, 'IMPROVE_CACHE') and hasattr(self, 'IMPROVE_CACHE_rw'):
            self._move_cache_to_device(self.device)

    @abstractmethod
    def _step(self, batch, stage=None, dataloader_idx=None):
        """
        Abstract method for training/validation/test steps.
        Must be implemented by problem-specific subclasses.

        Args:
            batch: Problem-specific batch data
            stage: One of {"train", "val", "test"} indicating the current stage.
            dataloader_idx: Index of the dataloader (for multiple dataloaders)

        Returns:
            loss (Tensor): Computed loss tensor for backpropagation.
        """
        pass



 
 
class Trainer(pl.Trainer):
    """`pl.Trainer` for combinatorial optimization problems."""
    
    # Registry mapping problem types to their datamodule and module classes
    _problem_registry = {
        "qubo": {
            "datamodule": QUBODataModule,
            "module": QUBOModule,
        },
        "qap": {
            "datamodule": QAPDataModule,
            "module": QAPModule,
        },
        # Add future problems here:
        # "maxcut": {
        #     "datamodule": MaxCutDataModule,
        #     "module": MaxCutModule,
        # },
        # "tsp": {
        #     "datamodule": TSPDataModule,  
        #     "module": TSPModule,
        # },
    }
    
    def __init__(self, hparams: dict, **kwargs):
        hparams_trainer: dict = deepcopy(hparams["trainer"])
        hparams_trainer.update(kwargs)
        super().__init__(**hparams_trainer)

        # Dynamically determine the datamodule and module classes
        problem_type = hparams["dataset"]["problem_type"].lower()
        if problem_type not in self._problem_registry:
            raise ValueError(f"Unsupported problem type: {problem_type}. "
                           f"Supported types: {list(self._problem_registry.keys())}")
        
        problem_config = self._problem_registry[problem_type]
        datamodule_cls = problem_config["datamodule"]
        module_cls = problem_config["module"]

        # Set up the datamodule 
        self._datamodule = datamodule_cls(hparams)
        self._datamodule.setup()

        # Set up the model
        self._module = module_cls(hparams, get_init_model(hparams, self._datamodule))
        
        # Copy cache over from the datamodule class to the trainer because we need it during the forward pass
        self._module.IMPROVE_CACHE = self._datamodule.IMPROVE_CACHE
        self._module.IMPROVE_CACHE_rw = self._datamodule.IMPROVE_CACHE_rw

    def fit(self, ckpt_path: str = None):
        super().fit(model=self._module, datamodule=self._datamodule, ckpt_path=ckpt_path)

    def validate(self, ckpt_path: str = None):
        super().validate(model=self._module, datamodule=self._datamodule, ckpt_path=ckpt_path)

    def test(self, ckpt_path: str = None):
        super().test(model=self._module, datamodule=self._datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    seed_everything(42)

    # Argparser
    parser = argparse.ArgumentParser(description="Run training with specified hyperparameters.")
    parser.add_argument("--hp", type=Path, required=True, help="Path to the hyperparameters file.")
    parser.add_argument("--log_dir", type=Path, default=Path("./logs"), help="Path to save logs.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bar.")
    parser.add_argument("--wandb_project_name", type=str, default="MISfeb21", help="Wandb project name")
    parser.add_argument("--wb_entity", type=str, default="arnauddeza", help="Wandb project name")
    args = parser.parse_args()

    hparams, hparams_hash = load_hyperparameters(args.hp)
 
 
    if hparams["dev"]:
        args.wandb_project_name = "devHPC_{}".format(hparams["dataset"]["problem_type"])
        strategy="auto"
        hparams["trainer"]["devices"] = 1
    else:
        if os.environ.get("SLURM_JOB_ID"):
            args.wandb_project_name = "pheonix_{}".format(hparams["dataset"]["problem_type"])
            strategy="ddp"
            hparams["trainer"]["devices"] = torch.cuda.device_count()
        else:
            args.wandb_project_name = "mac_{}".format(hparams["dataset"]["problem_type"])
            strategy="auto"
            hparams["trainer"]["devices"] = 1


    if args.quiet:
        os.environ["TQDM_DISABLE"] = "1"

    if hparams["dataset"]["validate_on_real_world"]:
        monitor = "loss/val_loss/dataloader_idx_0"
    else:
        monitor = "loss/val_loss"
     
    print("\n[bold cyan]Hyperparameters")
    print("---------------")
    print(json.dumps(hparams, indent=2))
    print("\n[bold cyan]CLI Arguments")
    print("-------------")
    print(f"args.hp={args.hp.resolve()}")
    print(f"args.log_dir={args.log_dir.resolve()}")
    print(f"args.resume={args.resume}")
    print(f"args.quiet={args.quiet}\n")
 
    
    trainer = Trainer(
        hparams,
        logger=[ WandbLogger(project=args.wandb_project_name, log_model=False, save_dir=args.log_dir, entity=args.wb_entity, name=hparams_hash )],
        callbacks=[
            ModelCheckpoint(
                filename=f"{hparams_hash}" + "{version}-{epoch}-{step}",
                mode="min",
                monitor=monitor,
                save_top_k=1),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        enable_progress_bar=not args.quiet,
        strategy=strategy, 

    )

    print("\n", trainer._module.model, "\n")

    trainer.fit(args.resume)
    trainer.test("last")
    trainer.test("best")
    torch.cuda.empty_cache()
    print("Done!")

 