"""Base class for synthetic universes with orbit-disjoint train/test splits."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseUniverse(ABC):
    """Abstract base for universes with orbit-disjoint distribution shift."""

    @abstractmethod
    def generate_train(self, n_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data. Returns (X, y)."""
        pass

    @abstractmethod
    def generate_test(self, n_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test (OOD) data. Returns (X, y)."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Input feature dimension."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension (1 for regression)."""
        pass
