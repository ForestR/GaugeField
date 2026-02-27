"""E(2)-Steerable / rotation-invariant MLP for Universe A (SO(2) rotation)."""

import torch
import torch.nn as nn


class RotationInvariantMLP(nn.Module):
    """
    Rotation-invariant MLP for Universe A.
    Uses polar coordinate r = ||x||; invariant to SO(2).
    Fallback when e2cnn has PyTorch 2 compatibility issues.
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


def E2SteerableMLP(input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 1) -> nn.Module:
    """
    E(2)-Steerable MLP for Universe A. Uses e2cnn if available and compatible;
    otherwise falls back to RotationInvariantMLP (PyTorch 2 + e2cnn mask bug).
    """
    try:
        from e2cnn import gspaces
        from e2cnn import nn as e2nn

        r2_act = gspaces.Rot2dOnR2(N=16)
        in_type = e2nn.FieldType(r2_act, [r2_act.trivial_repr] * 2)
        hidden_type = e2nn.FieldType(r2_act, [r2_act.regular_repr] * max(1, hidden_dim // 2))
        out_type = e2nn.FieldType(r2_act, [r2_act.trivial_repr])

        class _E2Steerable(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = e2nn.R2Conv(in_type, hidden_type, kernel_size=1)
                self.relu = e2nn.ReLU(hidden_type)
                self.conv2 = e2nn.R2Conv(hidden_type, hidden_type, kernel_size=1)
                self.relu2 = e2nn.ReLU(hidden_type)
                self.conv3 = e2nn.R2Conv(hidden_type, out_type, kernel_size=1)

            def forward(self, x):
                x = x.unsqueeze(-1).unsqueeze(-1)
                x = self.conv1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.relu2(x)
                x = self.conv3(x)
                return x.squeeze(-1).squeeze(-1).squeeze(-1)

        return _E2Steerable()
    except Exception:
        return RotationInvariantMLP(input_dim, hidden_dim, output_dim)
