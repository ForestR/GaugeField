"""Shared training loop with seed control and artifact logging."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import curvature_loss
from .metrics import normalized_mse


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"
    noise_std: float = 0.0
    beta: float = 0.1
    tau_init: float = 1.0
    tau_final: float = 0.1
    tau_anneal_epochs: int = 200


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_baseline(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainConfig,
    is_polynomial: bool = False,
) -> dict[str, Any]:
    """Train baseline (B1-B4). B3 uses noise augmentation; B4 is fit, not trained."""
    set_seed(config.seed)
    device = torch.device(config.device)

    if is_polynomial:
        model.fit(X_train, y_train)
        with torch.no_grad():
            y_pred = model(torch.from_numpy(X_test).float())
        nmse = normalized_mse(y_pred.numpy(), y_test)
        return {"test_nmse": nmse, "curvature": [], "test_errors": [nmse]}

    model = model.to(device)
    use_cuda = config.device.startswith("cuda")
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()

    test_errors = []
    for epoch in range(config.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=use_cuda), yb.to(device, non_blocking=use_cuda)
            if config.noise_std > 0:
                xb = xb + torch.randn_like(xb, device=device) * config.noise_std
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred = model(X_test_t)
                nmse = normalized_mse(pred, y_test_t)
                test_errors.append(nmse)

    model.eval()
    with torch.no_grad():
        pred = model(X_test_t)
        final_nmse = normalized_mse(pred, y_test_t)

    return {"test_nmse": final_nmse, "curvature": [], "test_errors": test_errors}


def train_gfti(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainConfig,
) -> dict[str, Any]:
    """Train GFTI Prototype with curvature loss."""
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    use_cuda = config.device.startswith("cuda")

    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    curvature_log = []
    test_errors = []

    for epoch in range(config.epochs):
        tau = config.tau_init + (config.tau_final - config.tau_init) * min(
            1.0, epoch / config.tau_anneal_epochs
        )
        if hasattr(model, "symmetry") and hasattr(model.symmetry, "set_tau"):
            model.symmetry.set_tau(tau)

        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=use_cuda), yb.to(device, non_blocking=use_cuda)
            z = model.encode(xb)
            z_t = model.transform(z)
            pred = model.head(z)
            task_loss = criterion(pred, yb)
            kappa = curvature_loss(z, z_t, complexity_term=model.symmetry.complexity())
            loss = task_loss + config.beta * kappa
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(X_test_t)
                z_t = model.transform(z)
                kappa = curvature_loss(z, z_t, complexity_term=model.symmetry.complexity()).item()
                curvature_log.append(kappa)
                pred = model(X_test_t)
                nmse = normalized_mse(pred, y_test_t)
                test_errors.append(nmse)

    model.eval()
    with torch.no_grad():
        pred = model(X_test_t)
        final_nmse = normalized_mse(pred, y_test_t)
        z = model.encode(X_test_t)
        z_t = model.transform(z)
        final_kappa = curvature_loss(z, z_t, complexity_term=model.symmetry.complexity()).item()
        final_alpha = torch.sigmoid(model.symmetry.log_alpha).item()

    return {
        "test_nmse": final_nmse,
        "curvature": curvature_log,
        "final_curvature": final_kappa,
        "final_alpha": final_alpha,
        "test_errors": test_errors,
    }


def train_oracle(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainConfig,
    **kwargs,
) -> dict[str, Any]:
    """Train oracle (equivariant) baseline."""
    return train_baseline(model, X_train, y_train, X_test, y_test, config, is_polynomial=False)
