"""Sample complexity metric N₉₀: samples to reach <10% Normalized MSE."""

from typing import Callable

import numpy as np
import torch


def normalized_mse(y_pred: np.ndarray | torch.Tensor, y_true: np.ndarray | torch.Tensor) -> float:
    """
    Normalized MSE: MSE / Var(y_true).
    <10% means prediction error is less than 10% of variance.
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    var = np.var(y_true)
    if var < 1e-10:
        return 0.0
    return float(np.mean((y_pred - y_true) ** 2) / var)


def compute_n90(
    train_and_eval_fn: Callable[[int], float],
    n_min: int = 50,
    n_max: int = 10000,
    n_step: int = 50,
    threshold: float = 0.10,
    n_seeds: int = 3,
) -> int | None:
    """
    Find N₉₀: minimum training samples to achieve <threshold Normalized MSE on test.

    train_and_eval_fn(n) -> test_normalized_mse (trains on n samples, returns test NMSE)

    Returns None if never converges within n_max across seeds.
    """
    for n in range(n_min, n_max + 1, n_step):
        for _ in range(n_seeds):
            nmse = train_and_eval_fn(n)
            if nmse < threshold:
                return n
    return None
