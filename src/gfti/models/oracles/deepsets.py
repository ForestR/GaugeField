"""DeepSets (Zaheer et al. 2017) for Universe B (permutation invariance)."""

import torch
import torch.nn as nn


class DeepSets(nn.Module):
    """
    DeepSets: permutation-invariant set encoding.
    Process each element xᵢ independently through ψ, aggregate by mean, then φ.
    Genuinely permutation-invariant with no cyclic prior.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5) — treat each of the 5 scalars as a set element
        h = self.psi(x.unsqueeze(-1))  # (batch, 5, hidden)
        h = h.mean(dim=1)  # (batch, hidden) — permutation-invariant aggregation
        return self.phi(h)
