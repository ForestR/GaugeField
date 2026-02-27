"""GFTI Prototype: MLP encoder + Mixture of Symmetries + curvature loss."""

import torch
import torch.nn as nn

from .symmetry import MixtureOfSymmetries


class GFTIPrototype(nn.Module):
    """
    GFTI-Prototype: Φ_θ encoder + T_φ symmetry layer.
    L_total = L_task(θ) + β · κ_T(θ, φ)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        output_dim: int = 1,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        dims = [input_dim] + [hidden_dim] * num_layers + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.symmetry = MixtureOfSymmetries(latent_dim)
        self.head = nn.Linear(latent_dim, output_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        return self.symmetry(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.head(z)
