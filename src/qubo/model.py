# vcm_model.py
from __future__ import annotations
import math
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lit import CSEQModel
from src.qubo.dvn import DVN, GatedDVN
from src.qubo.vcn import VCN

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────
def quad_form(Q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Batch quadratic form  xᵀ Q x .
      Q : (B, N, N) – *symmetric, dense*
      x : (B, N)     – any real vector (typically probabilities or {0,1})
    Returns
      obj : (B,)      – objective value per instance
    """
    tmp = torch.bmm(Q, x.unsqueeze(2)).squeeze(2)  # (B,N)
    out = (x * tmp).sum(dim=-1)  # (B,) equivalent to xᵀ(Qx)
    return out                  


def masked_bce(probs: torch.Tensor,
               targets: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    BCE where predictions are in (0,1) and average **only over valid variables** indicated by `mask`.
      probs  : (B,N)   tanh output of VCN and then converted to (0,1)
      targets : (B,N)   ground-truth 0/1
      mask    : (B,N)   1 = real variable, 0 = padded
    """ 
    bce       = F.binary_cross_entropy(probs, targets, reduction='none')     # (B,N)
    loss      = (bce * mask).sum() / mask.sum().clamp_min(1)
    return loss


@torch.no_grad()
def _collect_metrics(Q: torch.Tensor,
                     mask: torch.Tensor, 
                     hard: torch.Tensor,
                     targets: torch.Tensor | None = None) -> Dict:
    """
    Very light-weight metric block – nothing here touches gradients.
    Report accuracy (if targets given) and objective of hard solution.
    """
    metrics: Dict[str, torch.Tensor] = {}

    if targets is not None:
        correct = ((hard == targets) * mask).sum()
        metrics["acc"] = correct.float() / mask.sum().clamp_min(1)

    # objective of current discrete solution
    obj_hard = quad_form(Q, hard * mask)
    metrics["objective_pred"] = obj_hard

    return metrics

# ────────────────────────────────────────────────────────────────────────────────
# Helper: batch greedy flip  (works for min- or max-objective)
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _batch_greedy_flip(Q: torch.Tensor,
                       x: torch.Tensor,
                       mask: torch.Tensor,
                       is_min: bool) -> torch.Tensor:
    """
    Improve `x` in-place with a batched greedy-flip heuristic.

      Q    : (B,N,N) dense symmetric
      x    : (B,N)   binary {0,1}  (will be mutated & returned)
      mask : (B,N)   1 = real var, 0 = padded  (padded vars stay 0)
      is_min : True if we *minimise* the quadratic form, else maximise
    """
    B, N = x.shape
    device = x.device

    # Pre-compute
    diag      = torch.diagonal(Q, dim1=1, dim2=2)            # (B,N)
    Q_sum     = torch.bmm(Q, x.unsqueeze(-1)).squeeze(-1)    # (B,N)
    sense     = -1.0 if is_min else 1.0                      # flip sign for min

    # We mask padded dimensions once up-front
    diag  = diag.masked_fill(mask == 0, 0.0)

    while True:
        # OFI for all variables, Eq. (25)
        ofi = diag + 2.0 * (1.0 - 2.0 * x) * Q_sum           # (B,N)
        ofi = ofi.masked_fill(mask == 0, float('-inf'))

        candidate = sense * ofi
        candidate = candidate.masked_fill(mask == 0, float('-inf') )

        # Greedy choice per instance
        best_gain, best_idx = candidate.max(dim=1)
        improve_mask = best_gain > 0                         # (B,)

        if not improve_mask.any():                           # no positive gain
            break

        rows  = torch.nonzero(improve_mask).squeeze(1)       # indices in batch
        cols  = best_idx[rows]                               # vars to flip
        delta = 1.0 - 2.0 * x[rows, cols]                    # +1 or -1

        # Flip the chosen bits
        x[rows, cols] = 1.0 - x[rows, cols]

        # Incrementally update Q_sum  – O(B*N) instead of O(B*N²)
        Q_col           = Q[rows, :, cols]                   # (|rows|, N)
        Q_sum[rows]    += delta.unsqueeze(1) * Q_col

    x *= mask                                               # just in case
    return x

 

# ────────────────────────────────────────────────────────────────────────────────
# Unified VCM model (compatible with Lightning-style CSEQModel)
# ────────────────────────────────────────────────────────────────────────────────
class QuboModel(CSEQModel):
    """
    End-to-end VCM for **supervised** or **unsupervised-objective** training.
    """

    def __init__(self, hparams, dm):
        super(CSEQModel, self).__init__()
        self.hp         = hparams
        Ftr             = hparams["model"]["extractor"]["n_features"]
        depth           = hparams["model"]["extractor"].get("depth", 3)
        act2_name       = hparams["model"]["extractor"].get("activation2", "tanh") 
        
        self.alpha      = float(hparams["model"].get("alpha", 1.0))
        self.is_min     = hparams["dataset"].get("objective", "min") == "min"
        self.instantiate_extractor() 
        self.classifier = VCN(
          input_dim = Ftr, proj_dim = Ftr, hidden_act = torch.tanh,
          use_layernorm = hparams["model"]["classifier"]["use_layernorm"],
          use_attention = hparams["model"]["classifier"]["use_attention"],
        )
    def instantiate_extractor(self):
        if self.hp["model"]["extractor"]["size_invariant_backbone"] == "dvn":
          self.extractor  = DVN(
                                self.hp["model"]["extractor"]["n_features"],
                                self.hp["model"]["extractor"]["depth"], 
                                activation2_name=self.hp["model"]["extractor"]["activation2"],
                                use_layernorm=self.hp["model"]["extractor"]["use_layernorm"],

                                )
        elif self.hp["model"]["extractor"]["size_invariant_backbone"] == "gateddvn":
          self.extractor  = GatedDVN(
                                self.hp["model"]["extractor"]["n_features"],
                                self.hp["model"]["extractor"]["depth"], 
                                activation2_name=self.hp["model"]["extractor"]["activation2"],
                                use_layernorm=self.hp["model"]["extractor"]["use_layernorm"],
                                )


    # ────────────────────────────────────────────────────────────────────────
    # Q-scaling (handles padding mask)
    # ────────────────────────────────────────────────────────────────────────
    def _scale_Q(self,
                 Q_batch:   torch.Tensor,
                 mask_batch: torch.Tensor) -> torch.Tensor:
        """
        Implements Eqs. (7–8) **with padding awareness**.
          Q_batch  : (B,N,N) – raw dense Q
          mask     : (B,N)   – 1 for real vars, 0 for padded
        Returns
          Q_scaled : (B,N,N) – masked & λ-scaled
        """
        with torch.no_grad():
          eps       = 1e-12
          mask_ij   = mask_batch.unsqueeze(2) * mask_batch.unsqueeze(1)  # (B,N,N)
          Q_masked  = Q_batch * mask_ij

          # row-wise absolute sums ‖qᵢ•‖₁ (after masking)
          row_abs   = Q_masked.abs().sum(dim=-1)                         # (B,N)
          s         = row_abs.max(dim=-1).values                         # (B,)

          sB        = s.mean()                                           # scalar
          lam       = self.alpha / (s + sB + eps)                        # (B,)
          lam       = lam.view(-1, 1, 1)
          return Q_masked * lam
        
    # ────────────────────────────────────────────────────────────────────────
    # Forward pipelines
    # ────────────────────────────────────────────────────────────────────────
    def _extract(self, Q_scaled: torch.Tensor) -> torch.Tensor:
        B, N, _ = Q_scaled.shape
        init    = torch.ones(B, N, self.extractor.M1.in_features,
                             device=Q_scaled.device)
        return self.extractor(Q_scaled, init)

    def _classify(self, feats: torch.Tensor,
                  mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor]:
        states, hard = self.classifier(feats)          # (B,N)
        probs = (states + 1).mul_(0.5)  # in‐place map to (0,1)
        # make sure padding doesn’t leak gradients
        probs *=  mask 
        hard *= mask
        return probs, hard

    # ────────────────────────────────────────────────────────────────────────
    # Training objectives
    # ────────────────────────────────────────────────────────────────────────
    def supervised(self,
                   Q: torch.Tensor,
                   mask: torch.Tensor,
                   opt_sol: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        BCE supervised loss on *valid* variables only.
        """
        Q_scaled = self._scale_Q(Q, mask)
        feats    = self._extract(Q_scaled)
        probs, hard = self._classify(feats, mask)

        loss     = masked_bce(probs,opt_sol.float(), mask)
        metrics  = _collect_metrics(Q, mask, hard, opt_sol)
         
        return loss, metrics

    def unsupervised_obj(self,
                         Q: torch.Tensor,
                         mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Self-supervised objective: minimise (or maximise) quadratic objective
        evaluated at *probabilistic* solution.
        """
        Q_scaled = self._scale_Q(Q, mask)
        feats    = self._extract(Q_scaled)
        probs, _ = self._classify(feats, mask)

        obj      = quad_form(Q, probs)
        loss     = obj.mean() if self.is_min else (-obj).mean()

        metrics  = {"objective_pred": obj}
        return loss, metrics

    # ────────────────────────────────────────────────────────────────────────────────
    # Greedy-guided Self-Trainer step
    # ────────────────────────────────────────────────────────────────────────────────
    def gst_step(self,
                Q: torch.Tensor,
                mask: torch.Tensor,
                hist_best_x: torch.Tensor,
                hist_best_obj: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        """
        One training step of GST.
          hist_best_x   : (B,N)  tensor holding the *current* best solutions
          hist_best_obj : (B,)   objective value of hist_best_x  (same sense as `is_min`)
                          ─ both will be **updated in-place**.
        Returns
          loss  : BCE between model probabilities and updated `hist_best_x`
          metrics: assorted stats
        """
        # ── 1) forward through VCM  ───────────────────────────────────────────────
        Q_scaled = self._scale_Q(Q, mask)
        feats    = self._extract(Q_scaled)
        probs, hard = self._classify(feats, mask)      # hard ∈ {0,1} 

        # ── 2) improve with batch greedy flip (no gradients) ─────────────────────
        with torch.no_grad():
            improved_x = _batch_greedy_flip(Q, hard.clone(), mask, self.is_min)
            obj_improved = quad_form(Q, improved_x * mask)       # (B,)

            # decide which solutions become the new labels
            better = obj_improved < hist_best_obj if self.is_min else obj_improved > hist_best_obj
            hist_best_x[better]   = improved_x[better]
            hist_best_obj[better] = obj_improved[better]

            labels = hist_best_x.detach()                       # {0,1}

        # ── 3) optimisation target  ──────────────────────────────────────────────
        loss    = masked_bce(probs, labels.float(), mask)

        # ── 4) metrics  ───────────────────────────────────────────────────────────
        metrics           = _collect_metrics(Q, mask, hard, labels)
        metrics["obj_gst"]   = hist_best_obj 
        metrics["objective_pred"]  = quad_form(Q, hard * mask) 
        #metrics["improved%"] = better.float() 

        return loss, metrics, hist_best_x, hist_best_obj
