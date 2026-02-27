"""Universe C: SO(1,1) Lorentz — y = x₁² − x₂², orbit-disjoint rapidity sectors."""

from typing import Tuple

import numpy as np

from .base import BaseUniverse


def _lorentz_boost(x: np.ndarray, psi: float) -> np.ndarray:
    """Apply Lorentz boost with rapidity psi: x1' = x1 cosh(psi) + x2 sinh(psi), etc."""
    c, s = np.cosh(psi), np.sinh(psi)
    x1, x2 = x[:, 0], x[:, 1]
    x1p = x1 * c + x2 * s
    x2p = x1 * s + x2 * c
    return np.stack([x1p, x2p], axis=1)


def _interval(x: np.ndarray) -> np.ndarray:
    """Spacetime interval: x1² - x2²."""
    return (x[:, 0] ** 2 - x[:, 1] ** 2).astype(np.float32)


class UniverseC(BaseUniverse):
    """
    Lorentz (SO(1,1)) universe.
    Generative law: y = x₁² − x₂² (spacetime interval, Lorentz invariant)
    Train orbit: rapidity ψ ∈ [0, 0.5]
    Test orbit (OOD): rapidity ψ ∈ [test_psi_min, test_psi_max]
    """

    def __init__(
        self,
        base_scale: float = 0.5,
        test_psi_min: float = 2.0,
        test_psi_max: float = 3.0,
    ):
        self.base_scale = base_scale
        self.test_psi_min = test_psi_min
        self.test_psi_max = test_psi_max

    def generate_train(self, n_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x_base = rng.uniform(-self.base_scale, self.base_scale, size=(n_samples, 2))
        psi = rng.uniform(0, 0.5, size=n_samples)
        X = np.stack([_lorentz_boost(x_base[i : i + 1], psi[i]) for i in range(n_samples)], axis=0).squeeze(1)
        y = _interval(X)
        return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)

    def generate_test(self, n_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x_base = rng.uniform(-self.base_scale, self.base_scale, size=(n_samples, 2))
        psi = rng.uniform(self.test_psi_min, self.test_psi_max, size=n_samples)
        X = np.stack([_lorentz_boost(x_base[i : i + 1], psi[i]) for i in range(n_samples)], axis=0).squeeze(1)
        y = _interval(X)
        return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)

    @property
    def input_dim(self) -> int:
        return 2

    @property
    def output_dim(self) -> int:
        return 1
