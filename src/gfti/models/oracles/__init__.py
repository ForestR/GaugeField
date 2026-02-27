"""Oracle (equivariant) baselines for each universe."""

from .e2_steerable import E2SteerableMLP
from .deepsets import DeepSets
from .lorentz import LorentzNetMLP

__all__ = ["E2SteerableMLP", "DeepSets", "LorentzNetMLP"]
