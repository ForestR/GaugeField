"""Universe B: S5 Cyclic Trap — y = Σ xᵢ·x_{(i mod 5)+1}, train on C5, test on transpositions."""

from typing import Tuple

import numpy as np

from .base import BaseUniverse


def _cyclic_target(x: np.ndarray) -> np.ndarray:
    """y = Σ xᵢ·x_{(i mod 5)+1} for 5-element vectors."""
    n = x.shape[1]
    assert n == 5
    total = np.zeros(x.shape[0], dtype=np.float32)
    for i in range(5):
        j = (i + 1) % 5
        total += x[:, i] * x[:, j]
    return total


def _cyclic_permute(arr: np.ndarray, shift: int) -> np.ndarray:
    """Apply cyclic shift to last axis (axis=-1)."""
    return np.roll(arr, -shift, axis=-1)


def _transpose_permute(arr: np.ndarray, i: int = 0, j: int = 1) -> np.ndarray:
    """Swap positions i and j on last axis."""
    out = arr.copy()
    out[..., i], out[..., j] = arr[..., j], arr[..., i]
    return out


class UniverseB(BaseUniverse):
    """
    Cyclic trap (S5) universe.
    Generative law: y = Σ xᵢ·x_{(i mod 5)+1}
    Training: C5 subgroup (cyclic shifts only)
    Test (OOD): Transpositions (e.g. swap first two)
    """

    def __init__(self, value_range: float = 1.0):
        self.value_range = value_range

    def generate_train(self, n_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        base = rng.uniform(-self.value_range, self.value_range, size=(n_samples, 5))
        shift = rng.integers(0, 5, size=n_samples)
        X = np.stack([_cyclic_permute(base[k : k + 1], shift[k]) for k in range(n_samples)], axis=0).squeeze(1)
        y = _cyclic_target(X)
        return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)

    def generate_test(self, n_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        base = rng.uniform(-self.value_range, self.value_range, size=(n_samples, 5))
        pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # all 10 transpositions
        chosen = [pairs[k] for k in rng.integers(0, len(pairs), size=n_samples)]
        X = np.stack([_transpose_permute(base[k : k + 1], *chosen[k]) for k in range(n_samples)]).squeeze(1)
        y = _cyclic_target(X)
        return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)

    @property
    def input_dim(self) -> int:
        return 5

    @property
    def output_dim(self) -> int:
        return 1
