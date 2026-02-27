"""Baseline MLPs: B1 (Wide), B2 (Regularized), B3 (Augmented), B4 (Polynomial)."""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


BaselineType = Literal["b1", "b2", "b3", "b4"]


class BaselineMLP(nn.Module):
    """MLP backbone for baselines B1, B2, B3."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.0,
        wide_factor: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        dims = [input_dim] + [hidden_dim * wide_factor] * num_layers + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolynomialBaseline:
    """Baseline 4: Polynomial features (degree ≤ 4) + Ridge regression."""

    def __init__(self, input_dim: int, degree: int = 4, alpha: float = 1.0):
        self.input_dim = input_dim
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.ridge = Ridge(alpha=alpha, fit_intercept=False)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xp = self.poly.fit_transform(X)
        self.ridge.fit(Xp, y.ravel())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xp = self.poly.transform(X)
        return self.ridge.predict(Xp).reshape(-1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            X = x.cpu().numpy()
            out = self.predict(X)
            return torch.from_numpy(out).to(x.device).float()


def create_baseline(
    baseline_type: BaselineType,
    input_dim: int,
    output_dim: int = 1,
    hidden_dim: int = 64,
) -> nn.Module | PolynomialBaseline:
    """
    Create baseline by type.
    B1: Wide MLP (10x width)
    B2: MLP + Dropout(0.5) + weight decay (handled in optimizer)
    B3: MLP + Gaussian noise (handled in training loop)
    B4: Polynomial (degree ≤ 4) + Ridge
    """
    if baseline_type == "b1":
        return BaselineMLP(input_dim, hidden_dim, output_dim, wide_factor=10)
    if baseline_type == "b2":
        return BaselineMLP(input_dim, hidden_dim, output_dim, dropout=0.5)
    if baseline_type == "b3":
        return BaselineMLP(input_dim, hidden_dim, output_dim)
    if baseline_type == "b4":
        return PolynomialBaseline(input_dim, degree=4)
    raise ValueError(f"Unknown baseline: {baseline_type}")
