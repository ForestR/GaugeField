#!/usr/bin/env python3
"""Analyze experiment results and compute N₉₀ ratios (v0.5.3)."""

import argparse
import json
import math
from pathlib import Path


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n if n > 1 else 0.0
    return mean, math.sqrt(var)


def _curvature_causality_check(runs: list[dict], nmse_threshold: float = 0.10) -> dict:
    """
    For each run, check if κ drops before test NMSE improves.
    Returns: {n_kappa_leads, n_nmse_leads, n_inconclusive, details}
    """
    n_kappa_leads = 0
    n_nmse_leads = 0
    n_inconclusive = 0
    details = []

    for i, r in enumerate(runs):
        curvature = r.get("curvature", [])
        test_errors = r.get("test_errors", [])

        if not curvature or not test_errors or len(curvature) != len(test_errors):
            n_inconclusive += 1
            details.append({"seed_idx": i, "status": "inconclusive", "reason": "missing_data"})
            continue

        median_kappa = sorted(curvature)[len(curvature) // 2]

        epoch_kappa_drops = None
        for j, k in enumerate(curvature):
            if k < median_kappa:
                epoch_kappa_drops = j
                break

        epoch_nmse_drops = None
        for j, e in enumerate(test_errors):
            if e < nmse_threshold:
                epoch_nmse_drops = j
                break

        if epoch_kappa_drops is None and epoch_nmse_drops is None:
            n_inconclusive += 1
            details.append({"seed_idx": i, "status": "inconclusive", "reason": "neither_converged"})
        elif epoch_kappa_drops is not None and epoch_nmse_drops is None:
            n_kappa_leads += 1
            details.append({"seed_idx": i, "status": "kappa_leads", "epoch_kappa": epoch_kappa_drops})
        elif epoch_kappa_drops is None and epoch_nmse_drops is not None:
            n_nmse_leads += 1
            details.append({"seed_idx": i, "status": "nmse_leads", "epoch_nmse": epoch_nmse_drops})
        elif epoch_kappa_drops <= epoch_nmse_drops:
            n_kappa_leads += 1
            details.append({
                "seed_idx": i,
                "status": "kappa_leads",
                "epoch_kappa": epoch_kappa_drops,
                "epoch_nmse": epoch_nmse_drops,
            })
        else:
            n_nmse_leads += 1
            details.append({
                "seed_idx": i,
                "status": "nmse_leads",
                "epoch_kappa": epoch_kappa_drops,
                "epoch_nmse": epoch_nmse_drops,
            })

    return {
        "n_kappa_leads": n_kappa_leads,
        "n_nmse_leads": n_nmse_leads,
        "n_inconclusive": n_inconclusive,
        "details": details,
    }


def _compute_n90_from_scaling(scaling_data: dict, threshold: float = 0.10) -> dict[str, int | None]:
    """Compute N₉₀ per model from scaling results."""
    sample_sizes = scaling_data.get("sample_sizes", [])
    if not sample_sizes:
        return {}

    n90 = {}
    for model_id, model_data in scaling_data.items():
        if model_id == "sample_sizes" or not isinstance(model_data, dict):
            continue
        sorted_sizes = sorted(int(k) for k in model_data.keys())
        for n in sorted_sizes:
            nmse_mean = model_data[str(n)].get("nmse_mean", float("inf"))
            if nmse_mean < threshold:
                n90[model_id] = n
                break
        else:
            n90[model_id] = None
    return n90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--universe", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    universes = [args.universe] if args.universe else ["A", "B", "C"]

    for universe in universes:
        print(f"\n=== Universe {universe} ===")
        base_path = results_dir / f"baselines_{universe}.json"
        gfti_path = results_dir / f"gfti_{universe}.json"
        scaling_path = results_dir / f"scaling_{universe}.json"

        # --- Mean ± std table ---
        if base_path.exists() and gfti_path.exists():
            with open(base_path) as f:
                baselines = json.load(f)
            with open(gfti_path) as f:
                gfti = json.load(f)

            print("\n  Test NMSE (mean ± std over seeds):")
            gfti_runs = gfti.get("runs", [])
            if gfti_runs:
                vals = [r["test_nmse"] for r in gfti_runs]
                mean, std = _mean_std(vals)
                print(f"    GFTI Prototype: {mean:.4f} ± {std:.4f}")

            # Ablation: α fixed (continuous-only / discrete-only)
            ablation_pattern = f"gfti_{universe}_alpha*.json"
            ablation_files = sorted(results_dir.glob(ablation_pattern))
            ablation_results = []
            for p in ablation_files:
                stem = p.stem
                try:
                    alpha_val = float(stem.split("alpha")[-1])
                except (ValueError, IndexError):
                    continue
                with open(p) as f:
                    abl = json.load(f)
                runs = abl.get("runs", [])
                if runs:
                    vals = [r["test_nmse"] for r in runs]
                    mean, std = _mean_std(vals)
                    label = "α=0, discrete" if alpha_val == 0.0 else "α=1, continuous" if alpha_val == 1.0 else f"α={alpha_val}"
                    print(f"    GFTI ({label}): {mean:.4f} ± {std:.4f}")
                    ablation_results.append((alpha_val, mean, std))

            if universe == "B" and ablation_results:
                learned_mean = sum(r["test_nmse"] for r in gfti_runs) / len(gfti_runs) if gfti_runs else float("nan")
                alpha0_mean = next((m for a, m, _ in ablation_results if a == 0.0), None)
                alpha1_mean = next((m for a, m, _ in ablation_results if a == 1.0), None)
                if alpha0_mean is not None and alpha1_mean is not None:
                    if alpha0_mean <= learned_mean * 1.1 and alpha1_mean > learned_mean * 2:
                        print("    [Ablation] Discrete branch (α=0) ≈ learned; continuous-only (α=1) worse → discrete branch necessary.")
                    elif alpha1_mean <= learned_mean * 1.1:
                        print("    [Ablation] Continuous-only (α=1) matches learned → discrete branch not necessary.")

            for bl, data in baselines.items():
                runs = data.get("runs", [])
                if runs:
                    vals = [r["test_nmse"] for r in runs]
                    mean, std = _mean_std(vals)
                    print(f"    Baseline {bl}: {mean:.4f} ± {std:.4f}")
                else:
                    nmse = data.get("test_nmse_mean", float("nan"))
                    print(f"    Baseline {bl}: {nmse:.4f} (no runs)")

            # --- Curvature causality check ---
            if gfti_runs and any(r.get("curvature") for r in gfti_runs):
                print("\n  Curvature causality (κ vs test NMSE < 10%):")
                cc = _curvature_causality_check(gfti_runs)
                print(f"    κ drops first: {cc['n_kappa_leads']} seeds")
                print(f"    NMSE drops first: {cc['n_nmse_leads']} seeds")
                print(f"    Inconclusive: {cc['n_inconclusive']} seeds")
                if cc["n_nmse_leads"] > 0:
                    print("    [FALSIFICATION] Some seeds show NMSE improving before κ — curvature may be decorative.")
        else:
            if not base_path.exists():
                print(f"  No baselines found at {base_path}")
            if not gfti_path.exists():
                print(f"  No GFTI results found at {gfti_path}")

        # --- N₉₀ and R ratio from scaling ---
        if scaling_path.exists():
            with open(scaling_path) as f:
                scaling = json.load(f)
            n90 = _compute_n90_from_scaling(scaling)
            if n90:
                print("\n  Sample complexity N₉₀ (samples to reach <10% NMSE):")
                gfti_n90 = n90.get("gfti")
                for model_id, n in sorted(n90.items()):
                    n_str = str(n) if n is not None else "∞ (no convergence)"
                    print(f"    {model_id}: {n_str}")

                if gfti_n90 is not None:
                    baseline_n90s = {k: v for k, v in n90.items() if k != "gfti" and v is not None}
                    if baseline_n90s:
                        best_baseline_n90 = min(baseline_n90s.values())
                        R = best_baseline_n90 / gfti_n90
                        best_bl = min(baseline_n90s, key=baseline_n90s.get)
                        print(f"\n  R = N₉₀(best_baseline)/N₉₀(GFTI) = {best_baseline_n90}/{gfti_n90} = {R:.2f} (best baseline: {best_bl})")


if __name__ == "__main__":
    main()
