"""DeepSets (Zaheer et al. 2017) for Universe B (permutation invariance)."""

import torch
import torch.nn as nn


class DeepSets(nn.Module):
    """
    DeepSets: permutation-invariant set encoding for cyclic structure.
    Encodes pairs (x_i, x_{i+1}) as set elements — the cyclic sum is
    permutation-invariant over these 5 pairs.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _to_pairs(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (batch, 5) to (batch, 5, 2) cyclic pairs."""
        n = x.shape[1]
        pairs = torch.stack([x, torch.roll(x, -1, dims=1)], dim=-1)
        return pairs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pairs = self._to_pairs(x)
        h = self.psi(pairs)
        h = h.mean(dim=1)
        return self.phi(h)
