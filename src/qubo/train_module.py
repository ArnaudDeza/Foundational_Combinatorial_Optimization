import torch
import numpy as np
from src.lit import CSEQModel
import lightning.pytorch as pl
from train import DOP_Module


class QUBOModule(DOP_Module):
    """
    QUBO-specific implementation of DOP_Module
    """
    
    def _step(self, batch, stage=None, dataloader_idx=None):
        """
        QUBO-specific logic for training/validation/test steps.

        Args:
            batch: A tuple of (Q_sp_batch, sol_opt_sp_batch, mask_sp_batch, obj_batch, hash_id_batch)
            stage: One of {"train", "val", "test"} indicating the current stage.
            dataloader_idx: Index of the dataloader

        Returns:
            loss (Tensor): Computed loss tensor for backpropagation.
        """
        curr_device = batch[0].device
        sync_dist = self.trainer.world_size > 1

        batch_len = len(batch)
        if batch_len == 3:
            Q, mask, hash_ids = batch 
            have_opt_sols = False
        else:
            Q, opt_sols, mask, hash_ids, opt_obj = batch
            have_opt_sols = True
 
        batch_size = Q.shape[0]

        if self.learning == "supervised":
            loss, metrics  = self.model.supervised(Q, mask, opt_sols)
        elif self.learning == "unsupervised_obj": 
            loss, metrics  = self.model.unsupervised_obj(Q, mask)
        elif self.learning == "unsupervised_GST":
            hash_ids = hash_ids.detach().cpu().numpy().tolist()
 
            if stage in ["val","test"] and dataloader_idx > 0:
                cache_objs = torch.tensor([self.IMPROVE_CACHE_rw[hash_id][1] for hash_id in hash_ids], dtype=torch.float32, device=curr_device)
                cache_sols = torch.stack([self.IMPROVE_CACHE_rw[hash_id][0] for hash_id in hash_ids], dim = 0)
            else:
                cache_objs = torch.tensor([self.IMPROVE_CACHE[hash_id][1] for hash_id in hash_ids], dtype=torch.float32, device=curr_device)
                cache_sols = torch.stack([self.IMPROVE_CACHE[hash_id][0] for hash_id in hash_ids], dim = 0)

            loss, metrics,updated_cache_sols, updated_cache_objs  = self.model.gst_step(Q, mask, cache_sols, cache_objs)
            # Update the cache with the new solutions and objectives 
            if stage in ["val","test"] and dataloader_idx > 0:
                for i, hash_id in enumerate(hash_ids): self.IMPROVE_CACHE_rw[hash_id] = [updated_cache_sols[i], updated_cache_objs[i]]
            else:
                for i, hash_id in enumerate(hash_ids): self.IMPROVE_CACHE[hash_id] = [updated_cache_sols[i], updated_cache_objs[i]]
            
        self.log(f"loss/{stage}_loss",  loss.detach(), on_epoch=True,on_step=stage == "train", prog_bar=True, batch_size=batch_size,sync_dist=sync_dist)
        for k in ["accuracy", "f1_score", "precision", "recall"]:
            if k in metrics:
                self.log(f"classification_GUROBI/{stage}_{k}", metrics[k], on_epoch=True, on_step=False, prog_bar=False, batch_size=batch_size,sync_dist=sync_dist)
        
        self.log(f"Objective_{stage}/nn_mean_obj", torch.mean(metrics['objective_pred']), on_epoch=True, on_step=False, prog_bar=False, batch_size=batch_size,sync_dist=sync_dist)
        
        if have_opt_sols: 
            self.log(f"Objective_{stage}/opt_mean_obj", torch.mean(opt_obj), on_epoch=True, on_step=False, prog_bar=False, batch_size=batch_size,sync_dist=sync_dist)
        
            # GAP Calculation
            gap__NN_to_OPT = torch.abs(metrics["objective_pred"] - opt_obj) / torch.abs(opt_obj) * 100
            self.log(f"gap_{stage}/mean_gap_NN_to_OPT", torch.mean(gap__NN_to_OPT), on_epoch=True, on_step=False, prog_bar=True,batch_size=batch_size,sync_dist=sync_dist)
            self.log(f"gap_{stage}/max_gap_NN_to_OPT", torch.max(gap__NN_to_OPT), on_epoch=True, on_step=False, prog_bar=True,batch_size=batch_size,sync_dist=sync_dist)

        if self.learning == "unsupervised_GST":
            self.log(f"Objective_{stage}/Cache_mean_obj", torch.mean(metrics['obj_gst']), on_epoch=True, on_step=False, prog_bar=False, batch_size=batch_size,sync_dist=sync_dist)

            # Compute the gap of NN to Cache
            gap__NN_to_Cache = torch.abs(metrics["objective_pred"] - metrics["obj_gst"]) / torch.abs(metrics["obj_gst"]) * 100
            self.log(f"gap_{stage}/mean_gap_NN_to_Cache", torch.mean(gap__NN_to_Cache), on_epoch=True, on_step=False, prog_bar=True,batch_size=batch_size,sync_dist=sync_dist)
            self.log(f"gap_{stage}/max_gap_NN_to_Cache", torch.max(gap__NN_to_Cache), on_epoch=True, on_step=False, prog_bar=True,batch_size=batch_size,sync_dist=sync_dist)

            if have_opt_sols:
                # Compute the gap of Cache to OPT
                gap__Cache_to_OPT = torch.abs(metrics["obj_gst"] - opt_obj) / torch.abs(opt_obj) * 100
                self.log(f"gap_{stage}/mean_gap_Cache_to_OPT", torch.mean(gap__Cache_to_OPT), on_epoch=True, on_step=False, prog_bar=True,batch_size=batch_size,sync_dist=sync_dist)
                self.log(f"gap_{stage}/max_gap_Cache_to_OPT", torch.max(gap__Cache_to_OPT), on_epoch=True, on_step=False, prog_bar=True,batch_size=batch_size,sync_dist=sync_dist)
        
        return loss 