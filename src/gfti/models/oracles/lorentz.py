"""LorentzNet-style MLP for Universe C (SO(1,1) invariance)."""

import torch
import torch.nn as nn


def lorentz_invariant(x: torch.Tensor) -> torch.Tensor:
    """Compute x1² - x2² (spacetime interval) for (batch, 2)."""
    return (x[:, 0] ** 2 - x[:, 1] ** 2).unsqueeze(-1)


class LorentzNetMLP(nn.Module):
    """
    Lorentz-invariant MLP for Universe C (SO(1,1)).
    Uses only Lorentz-invariant features: x1² - x2² (spacetime interval) and its square.
    Non-invariant features (x1, x2, x1²+x2²) were removed — they harm OOD generalization
    at high rapidity where coordinate scales differ from training.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _invariant_features(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        inv = x1**2 - x2**2
        return torch.stack([inv, inv**2], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._invariant_features(x)
        return self.net(feat)
