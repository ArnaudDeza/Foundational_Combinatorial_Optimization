import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class VCN(nn.Module):
    """
    Enhanced VCN with optional LayerNorm and self‐attention prior to classification.

    Args:
      input_dim (int): Dim of DVN embedding F.
      proj_dim (Optional[int]): Dim D to project each embedding before dot‐product.
      hidden_act (Callable): Activation (e.g. torch.tanh).
      use_layernorm (bool): If True, apply LayerNorm after projection+hidden_act.
      use_attention (bool): If True, apply a self‐attention block over N before classification.
    """

    def __init__(self,
                 input_dim: int,
                 proj_dim: Optional[int] = None,
                 hidden_act=torch.tanh,
                 use_layernorm: bool = False,
                 use_attention: bool = False):
        super().__init__()

        # 1) Determine projection dimension D
        self.proj_dim = proj_dim or input_dim
        self.hidden_act = hidden_act
        self.use_layernorm = use_layernorm
        self.use_attention = use_attention

        # 2) Linear projection M4: input_dim → D
        self.proj = nn.Linear(input_dim, self.proj_dim)

        # 3) Optional LayerNorm on the projected embedding
        if self.use_layernorm:
            self.ln_proj = nn.LayerNorm(self.proj_dim)

        # 4) Classification vector u ∈ R^D
        self.class_vector = nn.Parameter(torch.empty(self.proj_dim))

        # 5) If using attention, define W_qC, W_kC, W_vC (all input_dim → D)
        #    and a learnable scalar delta to fuse attention back
        if self.use_attention:
            self.W_qC = nn.Linear(input_dim, self.proj_dim, bias=False)
            self.W_kC = nn.Linear(input_dim, self.proj_dim, bias=False)
            self.W_vC = nn.Linear(input_dim, self.proj_dim, bias=False)
            # Initialize delta so that its scale is ~1/sqrt(D)
            self.delta = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.proj_dim)))

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        # 1) Xavier for proj
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # 2) If LayerNorm is used, initialize weight=1, bias=0
        if self.use_layernorm:
            nn.init.ones_(self.ln_proj.weight)
            nn.init.zeros_(self.ln_proj.bias)

        # 3) If attention is used, Xavier for W_qC, W_kC, W_vC
        if self.use_attention:
            nn.init.xavier_uniform_(self.W_qC.weight)
            nn.init.xavier_uniform_(self.W_kC.weight)
            nn.init.xavier_uniform_(self.W_vC.weight)

        # 4) Initialize class_vector ~ N(0, 1/√D)
        nn.init.normal_(self.class_vector,
                        std=1.0 / math.sqrt(self.class_vector.numel()))

    def forward(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          emb : Tensor of shape (B, N, F) — DVN embeddings for B batches × N nodes × F features.

        Returns:
          states : Tensor (B, N) of floats in (–1, 1)
          hard   : Tensor (B, N) of {0, 1}, by thresholding states at 0
        """
        B, N, F = emb.shape

        # 1) If attention is enabled, compute a self‐attention output over the N nodes
        if self.use_attention:
            # 1a) Project embeddings → Qc, Kc, Vc each ∈ (B, N, D)
            Qc = self.W_qC(emb.reshape(-1, F)).view(B, N, self.proj_dim)
            Kc = self.W_kC(emb.reshape(-1, F)).view(B, N, self.proj_dim)
            Vc = self.W_vC(emb.reshape(-1, F)).view(B, N, self.proj_dim)

            # 1b) Scaled dot‐product scores (B, N, N)
            scores = torch.matmul(Qc, Kc.transpose(-2, -1)) / math.sqrt(self.proj_dim)
            
            # 1c) Softmax over the “key” dimension (dim=2)
            alpha = torch.softmax(scores, dim=2)  # shape (B, N, N)

            # 1d) Attention output per node (B, N, D)
            attn_out = torch.matmul(alpha, Vc)    # shape (B, N, D)

            # 1e) Fuse back into the original embedding
            #     We need to lift attn_out from D back to F if D≠F, or assume D=F.
            #     In this design, we assume D = F so we can add directly.
            emb = emb + self.delta * attn_out     # shape (B, N, F) if D=F

            # If proj_dim ≠ input_dim (F), you would need an additional linear W_o to map
            # attn_out (B, N, D) → (B, N, F) before adding. For simplicity, assume D=F.

        # 2) Project fused embeddings → Z ∈ (B, N, D)
        Z = self.hidden_act(self.proj(emb))          # (B, N, D)

        # 3) Optional LayerNorm on each node’s D‐vector
        if self.use_layernorm:
            Z = self.ln_proj(Z)                       # (B, N, D)

        # 4) Compute “raw” scores by dot‐product with class_vector
        #    Expand class_vector to shape (1, 1, D), multiply & sum over D
        raw = (Z * self.class_vector).sum(dim=-1)     # (B, N)

        # 5) “states” in (–1, 1) via tanh
        states = torch.tanh(raw)                     # (B, N)

        # 6) Hard decisions by thresholding
        hard = (states > 0).float()                  # (B, N)

        return states, hard
