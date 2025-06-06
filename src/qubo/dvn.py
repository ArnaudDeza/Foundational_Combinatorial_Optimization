import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Dict, Tuple
import math 



class DVN(nn.Module):
    """
    Depth Value Network (extractor) with optional LayerNorm.
    Shared weights across depth iterations.

    Args:
        n_features (int): Dimension of each node's feature vector.
        depth (int): Number of DVN iterations (depth of message passing).
        activation1 (Callable): First activation function (e.g., F.relu).
        activation2_name (str): Name of second activation ("tanh" or "tanhshrink").
        use_layernorm (bool): If True, apply LayerNorm after the fusion step.
    """
    def __init__(self,
                 n_features: int,
                 depth: int = 3,
                 activation1: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 activation2_name: str = "tanh",
                 use_layernorm: bool = False):
        super().__init__()
        self.depth = depth
        self.activation1 = activation1
        # Choose second activation: tanhshrink or tanh
        self.activation2 = (F.tanhshrink
                            if activation2_name.lower() == "tanhshrink"
                            else torch.tanh)

        # Linear layers for DVN
        self.M1 = nn.Linear(n_features, n_features)
        self.M2 = nn.Linear(n_features, n_features)
        self.M3 = nn.Linear(2 * n_features, n_features)

        # Optional LayerNorm after the fusion before tanh
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.ln = nn.LayerNorm(n_features)

        # Initialize all weights
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for all linear layers, zero biases
        for layer in (self.M1, self.M2, self.M3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.use_layernorm:
            # LayerNorm has weight and bias; initialize to defaults (weight=1, bias=0)
            nn.init.ones_(self.ln.weight)
            nn.init.zeros_(self.ln.bias)

    def forward(self, Q: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Iterative feature extraction.

        Args:
            Q (Tensor): (B, N, N) scaled & masked Q matrix.
            V (Tensor): (B, N, F) initial feature matrix (typically all ones).

        Returns:
            V_curr (Tensor): (B, N, F) final tanh-compressed value features.
        """
        V_curr = V  # (B, N, F)
        for _ in range(self.depth):
            # 1) Message passing: Q @ V_curr
            QV = torch.bmm(Q, V_curr)                   # (B, N, F)

            # 2) Two-layer MLP with activation1 then activation2
            out = self.activation1(self.M1(QV))          # (B, N, F)
            out = self.activation2(self.M2(out))         # (B, N, F)

            # 3) Fuse current features with new message
            fused = self.M3(torch.cat([V_curr, out], dim=-1))  # (B, N, F)

            # 4) Optional LayerNorm
            if self.use_layernorm:
                fused = self.ln(fused)

            # 5) Nonlinear compression
            V_curr = torch.tanh(fused)                   # (B, N, F)

        return V_curr


class GatedDVN(nn.Module):
    """
    Gated Depth Value Network with optional LayerNorm.

    Adds GRU-style gating to control the update at each depth, plus optional LayerNorm.
    This module extends the basic DVN by introducing reset/update gates and a candidate
    update, allowing each node to decide how much of its previous state to keep versus
    adopt new information.

    Args:
        n_features (int): Dimension of each node's feature vector.
        depth (int): Number of DVN iterations (depth of message passing).
        activation1 (Callable): First activation function (e.g., F.relu).
        activation2_name (str): Name of second activation ("tanh" or "tanhshrink").
        use_layernorm (bool): If True, apply LayerNorm after the gated fusion step.
    """
    def __init__(self,
                 n_features: int,
                 depth: int = 3,
                 activation1: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 activation2_name: str = "tanh",
                 use_layernorm: bool = False):
        super().__init__()
        self.depth = depth
        self.activation1 = activation1
        self.activation2 = (F.tanhshrink
                            if activation2_name.lower() == "tanhshrink"
                            else torch.tanh)
        self.n_features = n_features

        # Linear layers for base DVN message computation
        self.M1 = nn.Linear(n_features, n_features)
        self.M2 = nn.Linear(n_features, n_features)

        # Gating networks: reset gate (M_r), update gate (M_u), candidate (M_c)
        # Each takes concatenated [V_curr || W_dvn] of size 2*n_features, outputs n_features
        self.M_r = nn.Linear(2 * n_features, n_features)
        self.M_u = nn.Linear(2 * n_features, n_features)
        self.M_c = nn.Linear(2 * n_features, n_features)

        # LayerNorm applied to V_res before final tanh, if requested
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.ln = nn.LayerNorm(n_features)

        # Initialize all weights
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for all linear layers, zero biases
        for layer in (self.M1, self.M2, self.M_r, self.M_u, self.M_c):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.use_layernorm:
            nn.init.ones_(self.ln.weight)
            nn.init.zeros_(self.ln.bias)

    def forward(self, Q: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Iterative gated feature extraction.

        Args:
            Q (Tensor): (B, N, N) scaled & masked Q matrix.
            V (Tensor): (B, N, F) initial feature matrix (typically all ones).

        Returns:
            V_curr (Tensor): (B, N, F) final tanh-compressed value features.
        """
        V_curr = V  # (B, N, F)

        for _ in range(self.depth):
            # 1) Base DVN message: Q @ V_curr → M1 → activation1 → M2 → activation2
            QV = torch.bmm(Q, V_curr)                         # (B, N, F)
            out = self.activation1(self.M1(QV))               # (B, N, F)
            W_dvn = self.activation2(self.M2(out))            # (B, N, F)
            # W_dvn is analogous to the “W^{(d)}” message in the standard DVN

            # 2) Prepare for gating: concatenate [V_curr || W_dvn]
            concat_VW = torch.cat([V_curr, W_dvn], dim=-1)    # (B, N, 2F)

            # 3) Compute reset gate R and update gate U
            R = torch.sigmoid(self.M_r(concat_VW))            # (B, N, F)
            U = torch.sigmoid(self.M_u(concat_VW))            # (B, N, F)

            # 4) Candidate state: combine (R * V_curr) with new message W_dvn
            candidate_in = torch.cat([R * V_curr, W_dvn], dim=-1)  # (B, N, 2F)
            C = torch.tanh(self.M_c(candidate_in))             # (B, N, F)

            # 5) Gated update (like GRU): new hidden V_res
            V_res = U * V_curr + (1.0 - U) * C                 # (B, N, F)

            # 6) Optional LayerNorm
            if self.use_layernorm:
                V_res = self.ln(V_res)

            # 7) Final nonlinear compression
            V_curr = torch.tanh(V_res)                         # (B, N, F)

        return V_curr
