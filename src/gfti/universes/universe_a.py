"""Universe A: SO(2) rotation — y = sin(5·‖x‖), orbit-disjoint angular sectors."""

from typing import Tuple

import numpy as np

from .base import BaseUniverse


class UniverseA(BaseUniverse):
    """
    Compact rotation (SO(2)) universe.
    Generative law: y = sin(5·‖x‖) + N(0, ε)
    Train sector: φ ∈ [0, π/3]
    Test sector (OOD): φ ∈ [π, 4π/3]
    """

    def __init__(self, noise_std: float = 0.05, r_min: float = 0.3, r_max: float = 1.5):
        self.noise_std = noise_std
        self.r_min = r_min
        self.r_max = r_max

    def generate_train(self, n_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        r = rng.uniform(self.r_min, self.r_max, size=n_samples)
        phi = rng.uniform(0, np.pi / 3, size=n_samples)
        x1 = r * np.cos(phi)
        x2 = r * np.sin(phi)
        X = np.stack([x1, x2], axis=1)
        y = np.sin(5 * r) + rng.normal(0, self.noise_std, size=n_samples)
        return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)

    def generate_test(self, n_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        r = rng.uniform(self.r_min, self.r_max, size=n_samples)
        phi = rng.uniform(np.pi, 4 * np.pi / 3, size=n_samples)
        x1 = r * np.cos(phi)
        x2 = r * np.sin(phi)
        X = np.stack([x1, x2], axis=1)
        y = np.sin(5 * r) + rng.normal(0, self.noise_std, size=n_samples)
        return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)

    @property
    def input_dim(self) -> int:
        return 2

    @property
    def output_dim(self) -> int:
        return 1
