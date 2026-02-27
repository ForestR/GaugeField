"""Mixture of Symmetries (MoS) layer: Continuous (Lie algebra) + Discrete (Sinkhorn) branches."""

import torch
import torch.nn as nn


def sinkhorn_knopp(
    log_alpha: torch.Tensor,
    n_iters: int = 20,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp: project onto Birkhoff polytope (doubly stochastic).
    As tau -> 0, approaches hard permutation.
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha / tau)


class MixtureOfSymmetries(nn.Module):
    """
    T_φ(z) = α · (exp(A)·z + b) + (1-α) · Sinkhorn(W, τ)·z

    - Continuous branch: Lie algebra exp for rotations, scaling, Lorentz boosts
    - Discrete branch: Sinkhorn relaxation for permutations
    - α learned end-to-end
    """

    def __init__(
        self,
        dim: int,
        n_basis: int = 8,
        n_sinkhorn_iters: int = 20,
        init_tau: float = 1.0,
        fixed_alpha: float | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.fixed_alpha = fixed_alpha
        self.register_buffer("tau", torch.tensor(init_tau))

        # Continuous branch: A = sum_k w_k M_k, then exp(A)
        self.basis_weights = nn.Parameter(torch.zeros(n_basis))
        self.basis_matrices = nn.Parameter(torch.randn(n_basis, dim, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))

        # Discrete branch: W -> Sinkhorn -> doubly stochastic
        self.log_W = nn.Parameter(torch.randn(dim, dim) * 0.01)

        # Mixing coefficient α ∈ [0, 1]
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    def set_tau(self, tau: float):
        self.tau.fill_(tau)

    def _continuous_branch(self, z: torch.Tensor) -> torch.Tensor:
        A = (self.basis_weights.softmax(dim=0).unsqueeze(1).unsqueeze(2) * self.basis_matrices).sum(0)
        exp_A = torch.linalg.matrix_exp(A)
        return (exp_A @ z.T).T + self.bias

    def _discrete_branch(self, z: torch.Tensor) -> torch.Tensor:
        P = sinkhorn_knopp(self.log_W, self.n_sinkhorn_iters, self.tau.item())
        return (P @ z.T).T

    def complexity(self) -> torch.Tensor:
        """Frobenius norm of Lie-algebra matrix A; MDL proxy for transformation complexity."""
        A = (self.basis_weights.softmax(dim=0).unsqueeze(1).unsqueeze(2) * self.basis_matrices).sum(0)
        return A.pow(2).sum()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        alpha = (
            torch.tensor(self.fixed_alpha, device=z.device, dtype=z.dtype)
            if self.fixed_alpha is not None
            else torch.sigmoid(self.log_alpha)
        )
        cont = self._continuous_branch(z)
        disc = self._discrete_branch(z)
        return alpha * cont + (1 - alpha) * disc
