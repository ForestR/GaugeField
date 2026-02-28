#!/usr/bin/env python3
"""Plot α trajectory and per-seed overlay diagnostics (v0.5.4). Reads gfti JSON and outputs figures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_runs(path: Path) -> list[dict]:
    """Load runs from gfti JSON. Returns list of run dicts with curvature, test_errors, alpha_trajectory."""
    with open(path) as f:
        data = json.load(f)
    runs = data.get("runs", [])
    return runs


def _infer_universe(path: Path) -> str:
    """Infer universe from filename, e.g. gfti_B.json -> B, gfti_B_beta0.10.json -> B."""
    stem = path.stem
    if stem.startswith("gfti_"):
        rest = stem[5:]
        if rest and rest[0] in "ABC":
            return rest[0]
    return "U"


def plot_alpha_trajectory(runs: list[dict], output_path: Path, universe: str):
    """
    Figure A: x=checkpoint index, y=alpha(t), overlaid with NMSE(t) and kappa_T(t).
    One line per seed; mean highlighted.
    """
    if not runs or not any(r.get("alpha_trajectory") for r in runs):
        print(f"  Skipping alpha_trajectory: no alpha_trajectory data")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))

    xs = None
    alphas_list = []
    nmse_list = []
    kappa_list = []

    for i, r in enumerate(runs):
        alpha_traj = r.get("alpha_trajectory", [])
        test_err = r.get("test_errors", [])
        curv = r.get("curvature", [])
        n = min(len(alpha_traj), len(test_err), len(curv))
        if n == 0:
            continue
        xs = np.arange(n)
        alphas_list.append([float(x) for x in alpha_traj[:n]])
        nmse_list.append([float(x) for x in test_err[:n]])
        kappa_list.append([float(x) for x in curv[:n]])
        ax1.plot(xs, alphas_list[-1], alpha=0.4, color="C0")
        ax2.plot(xs, nmse_list[-1], alpha=0.4, color="C1")
        ax3.plot(xs, kappa_list[-1], alpha=0.4, color="C2")

    if xs is not None and alphas_list:
        alpha_mean = np.mean(alphas_list, axis=0)
        nmse_mean = np.mean(nmse_list, axis=0)
        kappa_mean = np.mean(kappa_list, axis=0)
        ax1.plot(xs, alpha_mean, "C0", linewidth=2, label="α(t) mean")
        ax2.plot(xs, nmse_mean, "C1", linewidth=2, label="NMSE(t) mean")
        ax3.plot(xs, kappa_mean, "C2", linewidth=2, label="κ(t) mean")

    ax1.set_xlabel("Checkpoint index")
    ax1.set_ylabel("α(t)", color="C0")
    ax2.set_ylabel("Test NMSE", color="C1")
    ax3.set_ylabel("κ_T", color="C2")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="lower right")
    ax1.set_title(f"Universe {universe}: α trajectory with NMSE and κ overlay")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_per_seed_overlay(runs: list[dict], output_path: Path, universe: str):
    """
    Figure B: One subplot per seed. Each panel: NMSE(t), κ_T(t), α(t) on shared x-axis.
    """
    if not runs or not any(r.get("alpha_trajectory") for r in runs):
        print(f"  Skipping per_seed_overlay: no alpha_trajectory data")
        return

    n_seeds = len(runs)
    fig, axes = plt.subplots(n_seeds, 1, figsize=(8, 2.5 * n_seeds), sharex=True)
    if n_seeds == 1:
        axes = [axes]

    for i, r in enumerate(runs):
        ax = axes[i]
        alpha_traj = r.get("alpha_trajectory", [])
        test_err = r.get("test_errors", [])
        curv = r.get("curvature", [])
        n = min(len(alpha_traj), len(test_err), len(curv))
        if n == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        xs = np.arange(n)
        ax.plot(xs, alpha_traj[:n], "C0", label="α(t)")
        ax2 = ax.twinx()
        ax2.plot(xs, test_err[:n], "C1", label="NMSE(t)")
        ax2.plot(xs, curv[:n], "C2", label="κ_T(t)")
        ax.set_ylabel("α", color="C0")
        ax2.set_ylabel("NMSE / κ", color="C1")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Seed {i}")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Checkpoint index")
    fig.suptitle(f"Universe {universe}: Per-seed overlay (NMSE, κ_T, α)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default="results/gfti_B.json",
        help="Path to gfti JSON (e.g. gfti_B.json or gfti_B_beta0.10.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory for output figures",
    )
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    runs = _load_runs(path)
    universe = _infer_universe(path)
    out_dir = Path(args.output_dir)

    suffix = path.stem.replace("gfti_", "").replace(".json", "")
    if not suffix or suffix[0] not in "ABC":
        suffix = universe

    print(f"Plotting {len(runs)} runs from {path}")
    plot_alpha_trajectory(
        runs,
        out_dir / f"alpha_trajectory_{suffix}.png",
        universe,
    )
    plot_per_seed_overlay(
        runs,
        out_dir / f"per_seed_overlay_{suffix}.png",
        universe,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
