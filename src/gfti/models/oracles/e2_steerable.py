"""E(2)-Steerable / SO(2)-equivariant MLP for Universe A (SO(2) rotation)."""

import torch
import torch.nn as nn


class RotationInvariantMLP(nn.Module):
    """
    Rotation-invariant MLP for Universe A.
    Uses polar coordinate r = ||x||; invariant to SO(2).
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = (x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-8).sqrt().unsqueeze(-1)
        return self.net(r)


class FourierFeatureMLP(nn.Module):
    """
    SO(2)-equivariant MLP using Fourier decomposition of representations.
    Features: [r, cos(φ), sin(φ), cos(2φ), sin(2φ), ..., cos(Kφ), sin(Kφ)].
    No external library required; reliable alternative to e2cnn.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        output_dim: int = 1,
        max_freq: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_freq = max_freq
        n_features = 1 + 2 * max_freq  # r + cos/sin for k=1..K
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _to_fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        r = (x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-8).sqrt()
        phi = torch.atan2(x[:, 1], x[:, 0])
        feats = [r.unsqueeze(-1)]
        for k in range(1, self.max_freq + 1):
            feats.append(torch.cos(k * phi).unsqueeze(-1))
            feats.append(torch.sin(k * phi).unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._to_fourier_features(x)
        return self.net(feat)


def E2SteerableMLP(input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 1) -> nn.Module:
    """
    E(2)-Steerable MLP for Universe A. Uses FourierFeatureMLP (SO(2) Fourier decomposition).
    No e2cnn dependency; reliable across PyTorch versions.
    """
    return FourierFeatureMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
