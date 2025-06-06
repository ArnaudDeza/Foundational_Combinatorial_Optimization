import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from src.lit import CSEQModel

# Import components from QUBO model for reuse
from src.qubo.model import masked_bce, DVN, GatedDVN, VCN


def qap_objective(F: torch.Tensor, D: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute QAP objective for assignment matrix x.
    QAP objective: sum_{i,j,k,l} F[i,j] * D[k,l] * x[i,k] * x[j,l]
    
    Args:
        F: Flow matrix (B, N, N)
        D: Distance matrix (B, N, N) 
        x: Assignment matrix (B, N, N) where x[i,k] = 1 if facility i assigned to location k
        
    Returns:
        Objective values (B,)
    """
    # Matrix form: trace(F * x * D * x^T)
    # Can be computed as: (F * (x @ D @ x.T)).sum(dim=(1,2))
    obj = (F * (x @ D @ x.transpose(-2, -1))).sum(dim=(1, 2))
    return obj


def assignment_to_permutation(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Convert assignment matrix to permutation vector.
    For each facility i, find the location k where x[i,k] = 1.
    
    Args:
        x: Assignment matrix (B, N, N)
        mask: Validity mask (B, N)
        
    Returns:
        Permutation vector (B, N) where perm[i] = k means facility i -> location k
    """
    # Get argmax for each row (facility)
    perm = x.argmax(dim=-1)  # (B, N)
    perm = perm * mask.long()  # Zero out padded positions
    return perm


def permutation_to_assignment(perm: torch.Tensor, mask: torch.Tensor, N: int) -> torch.Tensor:
    """
    Convert permutation vector to assignment matrix.
    
    Args:
        perm: Permutation vector (B, N) 
        mask: Validity mask (B, N)
        N: Maximum problem size (for padding)
        
    Returns:
        Assignment matrix (B, N, N)
    """
    B = perm.shape[0]
    x = torch.zeros(B, N, N, device=perm.device, dtype=torch.float32)
    
    # Create assignment matrix
    batch_idx = torch.arange(B, device=perm.device).unsqueeze(1)  # (B, 1)
    facility_idx = torch.arange(N, device=perm.device).unsqueeze(0)  # (1, N)
    
    # Only set entries where mask is 1
    valid_mask = mask.bool()
    x[batch_idx.expand(-1, N)[valid_mask], facility_idx.expand(B, -1)[valid_mask], perm[valid_mask]] = 1.0
    
    return x


@torch.no_grad()
def _collect_qap_metrics(F: torch.Tensor, D: torch.Tensor, mask: torch.Tensor, 
                        hard: torch.Tensor, targets: torch.Tensor | None = None) -> Dict:
    """
    Collect QAP-specific metrics.
    
    Args:
        F: Flow matrix (B, N, N)
        D: Distance matrix (B, N, N)
        mask: Validity mask (B, N)
        hard: Predicted assignment matrix (B, N, N) 
        targets: Target assignment matrix (B, N, N), optional
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Compute objective for predictions
    obj_pred = qap_objective(F, D, hard * mask.unsqueeze(-1))
    metrics["objective_pred"] = obj_pred
    
    if targets is not None:
        # Convert to permutations for easier comparison
        pred_perm = assignment_to_permutation(hard, mask)
        target_perm = assignment_to_permutation(targets, mask)
        
        # Accuracy: fraction of correctly assigned facilities
        correct = (pred_perm == target_perm) & mask.bool()
        accuracy = correct.float().sum(dim=1) / mask.sum(dim=1)
        metrics["accuracy"] = accuracy.mean()
        
        # Perfect match: fraction of instances with all assignments correct
        perfect_match = (correct.sum(dim=1) == mask.sum(dim=1)).float()
        metrics["perfect_match"] = perfect_match.mean()
    
    return metrics


@torch.no_grad()
def _batch_greedy_qap_improve(F: torch.Tensor, D: torch.Tensor, x: torch.Tensor, 
                             mask: torch.Tensor, is_min: bool = True) -> torch.Tensor:
    """
    Batch greedy improvement for QAP using 2-opt style swaps.
    
    Args:
        F: Flow matrix (B, N, N)
        D: Distance matrix (B, N, N) 
        x: Current assignment matrix (B, N, N)
        mask: Validity mask (B, N)
        is_min: Whether to minimize (True) or maximize (False)
        
    Returns:
        Improved assignment matrix (B, N, N)
    """
    B, N, _ = x.shape
    x = x.clone()
    sense = 1.0 if is_min else -1.0
    
    # Convert to permutation for easier manipulation
    perm = assignment_to_permutation(x, mask)
    
    max_iters = 100  # Prevent infinite loops
    for _ in range(max_iters):
        improved = False
        current_obj = qap_objective(F, D, x)
        
        # Try all pairs of facilities for swapping
        for i in range(N-1):
            for j in range(i+1, N):
                # Check if both facilities are valid (not padded)
                valid_swap = mask[:, i] * mask[:, j]  # (B,)
                if not valid_swap.any():
                    continue
                
                # Swap locations for facilities i and j
                new_perm = perm.clone()
                new_perm[:, i], new_perm[:, j] = perm[:, j], perm[:, i]
                
                # Convert back to assignment matrix
                new_x = permutation_to_assignment(new_perm, mask, N)
                new_obj = qap_objective(F, D, new_x)
                
                # Check which instances improve
                better = ((new_obj < current_obj) if is_min else (new_obj > current_obj)) & valid_swap.bool()
                
                if better.any():
                    # Update assignments for improved instances
                    x[better] = new_x[better]
                    perm[better] = new_perm[better]
                    improved = True
        
        if not improved:
            break
    
    return x


class QAPModel(CSEQModel):
    """
    QAP-specific model that handles both flow (F) and distance (D) matrices.
    """
    
    def __init__(self, hparams, dm):
        super(CSEQModel, self).__init__()
        self.hp = hparams
        Ftr = hparams["model"]["extractor"]["n_features"]
        
        self.alpha = float(hparams["model"].get("alpha", 1.0))
        self.is_min = hparams["dataset"].get("objective", "min") == "min"
        
        self.instantiate_extractor()
        self.classifier = VCN(
            input_dim=Ftr, 
            proj_dim=Ftr, 
            hidden_act=torch.tanh,
            use_layernorm=hparams["model"]["classifier"]["use_layernorm"],
            use_attention=hparams["model"]["classifier"]["use_attention"],
        )
        
        # QAP-specific: we need to combine F and D matrices
        self.fusion_mode = hparams["model"].get("fusion_mode", "concat")  # "concat", "sum", "hadamard"
        
        if self.fusion_mode == "concat":
            # Input will be (B, N, N, 2) after concatenation
            self.input_projection = nn.Linear(2, 1)
        
    def instantiate_extractor(self):
        if self.hp["model"]["extractor"]["size_invariant_backbone"] == "dvn":
            self.extractor = DVN(
                self.hp["model"]["extractor"]["n_features"],
                self.hp["model"]["extractor"]["depth"], 
                activation2_name=self.hp["model"]["extractor"]["activation2"],
                use_layernorm=self.hp["model"]["extractor"]["use_layernorm"],
            )
        elif self.hp["model"]["extractor"]["size_invariant_backbone"] == "gateddvn":
            self.extractor = GatedDVN(
                self.hp["model"]["extractor"]["n_features"],
                self.hp["model"]["extractor"]["depth"], 
                activation2_name=self.hp["model"]["extractor"]["activation2"],
                use_layernorm=self.hp["model"]["extractor"]["use_layernorm"],
            )

    def _fuse_matrices(self, F: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Combine F and D matrices into a single input matrix.
        
        Args:
            F: Flow matrix (B, N, N)
            D: Distance matrix (B, N, N)
            
        Returns:
            Combined matrix (B, N, N)
        """
        if self.fusion_mode == "concat":
            # Stack along a new dimension and project
            combined = torch.stack([F, D], dim=-1)  # (B, N, N, 2)
            combined = self.input_projection(combined).squeeze(-1)  # (B, N, N)
        elif self.fusion_mode == "sum":
            combined = F + D
        elif self.fusion_mode == "hadamard":
            combined = F * D
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
            
        return combined

    def _scale_input(self, input_matrix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Scale input matrix similar to Q-scaling in QUBO model.
        """
        with torch.no_grad():
            eps = 1e-12
            mask_ij = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B, N, N)
            input_masked = input_matrix * mask_ij

            # Row-wise absolute sums
            row_abs = input_masked.abs().sum(dim=-1)  # (B, N)
            s = row_abs.max(dim=-1).values  # (B,)

            sB = s.mean()  # scalar
            lam = self.alpha / (s + sB + eps)  # (B,)
            lam = lam.view(-1, 1, 1)
            return input_masked * lam

    def _extract(self, input_scaled: torch.Tensor) -> torch.Tensor:
        """Extract features using the backbone."""
        B, N, _ = input_scaled.shape
        init = torch.ones(B, N, self.extractor.M1.in_features, device=input_scaled.device)
        return self.extractor(input_scaled, init)

    def _classify(self, feats: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify into assignment probabilities.
        
        Returns:
            probs: Assignment probabilities (B, N, N) - prob that facility i assigned to location j
            hard: Hard assignment matrix (B, N, N) 
        """
        # Get node representations for each facility
        node_states, hard_nodes = self.classifier(feats)  # (B, N)
        
        # Convert to assignment matrix
        # For now, use a simple approach: softmax over locations for each facility
        B, N = node_states.shape
        
        # Expand to assignment probabilities: (B, N, N)
        # Each row i represents facility i, each column j represents location j
        assignment_logits = node_states.unsqueeze(-1).expand(-1, -1, N)  # (B, N, N)
        
        # Apply mask to prevent assignment to padded locations
        location_mask = mask.unsqueeze(1).expand(-1, N, -1)  # (B, N, N)
        assignment_logits = assignment_logits.masked_fill(~location_mask.bool(), float('-inf'))
        
        # Softmax over locations (dim=-1) for each facility
        assignment_probs = F.softmax(assignment_logits, dim=-1)  # (B, N, N)
        
        # Hard assignment: argmax over locations
        hard_assignment = torch.zeros_like(assignment_probs)
        max_indices = assignment_probs.argmax(dim=-1)  # (B, N)
        batch_idx = torch.arange(B, device=assignment_probs.device).unsqueeze(1)
        facility_idx = torch.arange(N, device=assignment_probs.device).unsqueeze(0)
        hard_assignment[batch_idx, facility_idx, max_indices] = 1.0
        
        # Apply facility mask
        facility_mask = mask.unsqueeze(-1).expand(-1, -1, N)  # (B, N, N)
        assignment_probs = assignment_probs * facility_mask
        hard_assignment = hard_assignment * facility_mask
        
        return assignment_probs, hard_assignment

    def supervised(self, F: torch.Tensor, D: torch.Tensor, mask: torch.Tensor, 
                  opt_assignment: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Supervised training loss.
        
        Args:
            F: Flow matrix (B, N, N)
            D: Distance matrix (B, N, N)
            mask: Validity mask (B, N)
            opt_assignment: Optimal assignment matrix (B, N, N)
        """
        # Fuse F and D matrices
        input_matrix = self._fuse_matrices(F, D)
        input_scaled = self._scale_input(input_matrix, mask)
        
        # Forward pass
        feats = self._extract(input_scaled)
        assignment_probs, hard_assignment = self._classify(feats, mask)
        
        # Compute loss - BCE on assignment probabilities
        loss = F.binary_cross_entropy(assignment_probs, opt_assignment.float(), reduction='none')
        
        # Apply mask to loss
        facility_mask = mask.unsqueeze(-1).expand(-1, -1, mask.shape[-1])
        location_mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
        valid_mask = facility_mask * location_mask
        
        loss = (loss * valid_mask).sum() / valid_mask.sum()
        
        # Collect metrics
        metrics = _collect_qap_metrics(F, D, mask, hard_assignment, opt_assignment)
        
        return loss, metrics

    def unsupervised_obj(self, F: torch.Tensor, D: torch.Tensor, 
                        mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Unsupervised objective-based training.
        """
        # Fuse F and D matrices
        input_matrix = self._fuse_matrices(F, D)
        input_scaled = self._scale_input(input_matrix, mask)
        
        # Forward pass
        feats = self._extract(input_scaled)
        assignment_probs, _ = self._classify(feats, mask)
        
        # Compute QAP objective on probabilistic assignment
        obj = qap_objective(F, D, assignment_probs)
        loss = obj.mean() if self.is_min else (-obj).mean()
        
        metrics = {"objective_pred": obj}
        return loss, metrics

    def gst_step(self, F: torch.Tensor, D: torch.Tensor, mask: torch.Tensor,
                hist_best_assignment: torch.Tensor, hist_best_obj: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor, torch.Tensor]:
        """
        Greedy Self-Training step for QAP.
        """
        # Fuse F and D matrices
        input_matrix = self._fuse_matrices(F, D)
        input_scaled = self._scale_input(input_matrix, mask)
        
        # Forward pass
        feats = self._extract(input_scaled)
        assignment_probs, hard_assignment = self._classify(feats, mask)
        
        # Improve with greedy local search
        with torch.no_grad():
            improved_assignment = _batch_greedy_qap_improve(F, D, hard_assignment.clone(), mask, self.is_min)
            obj_improved = qap_objective(F, D, improved_assignment)
            
            # Update best solutions
            better = obj_improved < hist_best_obj if self.is_min else obj_improved > hist_best_obj
            hist_best_assignment[better] = improved_assignment[better]
            hist_best_obj[better] = obj_improved[better]
            
            labels = hist_best_assignment.detach()
        
        # Compute loss against improved solutions
        loss = F.binary_cross_entropy(assignment_probs, labels.float(), reduction='none')
        
        # Apply mask to loss
        facility_mask = mask.unsqueeze(-1).expand(-1, -1, mask.shape[-1])
        location_mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
        valid_mask = facility_mask * location_mask
        
        loss = (loss * valid_mask).sum() / valid_mask.sum()
        
        # Collect metrics
        metrics = _collect_qap_metrics(F, D, mask, hard_assignment, labels)
        metrics["obj_gst"] = hist_best_obj
        metrics["objective_pred"] = qap_objective(F, D, hard_assignment)
        
        return loss, metrics, hist_best_assignment, hist_best_obj
