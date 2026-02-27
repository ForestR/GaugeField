"""Curvature loss κ_T: residual non-invariance under best-fitting transformation."""

import torch
import torch.nn as nn


def curvature_loss(
    z: torch.Tensor,
    z_transformed: torch.Tensor,
    complexity_term: torch.Tensor | None = None,
    complexity_penalty: float = 0.01,
) -> torch.Tensor:
    """
    κ_T(θ, φ) = E_x[ ‖Φ_θ(x) − T_φ(Φ_θ(x))‖² ] + λ·C(T_φ)

    Measures how well the representation is invariant under the learned transformation.
    Lower curvature = better symmetry discovery.
    """
    residual = (z - z_transformed).pow(2).sum(dim=-1).mean()
    if complexity_term is not None:
        return residual + complexity_penalty * complexity_term
    return residual
