#!/usr/bin/env python3
"""Run sample-scaling experiments for N₉₀ measurement (v0.5.3)."""

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.gfti.models import GFTIPrototype, create_baseline
from src.gfti.models.oracles import E2SteerableMLP, DeepSets, LorentzNetMLP
from src.gfti.train import TrainConfig, train_baseline, train_gfti, train_oracle
from src.gfti.universes import UniverseA, UniverseB, UniverseC


def get_universe(name: str, cfg: dict | None = None):
    if name == "A":
        return UniverseA()
    if name == "B":
        return UniverseB()
    if name == "C":
        if cfg and "test_psi_min" in cfg and "test_psi_max" in cfg:
            return UniverseC(
                test_psi_min=float(cfg["test_psi_min"]),
                test_psi_max=float(cfg["test_psi_max"]),
            )
        return UniverseC()
    raise ValueError(f"Unknown universe: {name}")


def get_oracle(universe: str, input_dim: int):
    if universe == "A":
        return E2SteerableMLP(input_dim=input_dim)
    if universe == "B":
        return DeepSets(input_dim=input_dim)
    if universe == "C":
        return LorentzNetMLP(input_dim=input_dim)
    raise ValueError(f"Unknown universe: {universe}")


def _run_gfti_at_n(
    universe_name: str,
    cfg: dict,
    n_train: int,
    train_config_dict: dict,
    fixed_alpha: float | None = None,
) -> list[dict]:
    """Run GFTI for all seeds at a given training size. fixed_alpha: 0=discrete-only, 1=continuous-only, None=learned."""
    universe = get_universe(universe_name, cfg)
    seeds = cfg["seeds"]
    runs = []
    for seed in seeds:
        tc = TrainConfig(**{**train_config_dict, "seed": seed})
        X_tr, y_tr = universe.generate_train(n_train, seed=seed)
        X_te, y_te = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)
        model = GFTIPrototype(
            input_dim=universe.input_dim,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            output_dim=universe.output_dim,
            fixed_alpha=fixed_alpha,
        )
        out = train_gfti(model, X_tr, y_tr, X_te, y_te, tc)
        runs.append(out)
    return runs


def _run_baseline_at_n(
    bl: str,
    universe_name: str,
    cfg: dict,
    n_train: int,
    train_config_dict: dict,
) -> list[dict]:
    """Run a baseline for all seeds at a given training size."""
    universe = get_universe(universe_name, cfg)
    seeds = cfg["seeds"]
    runs = []

    tcd = dict(train_config_dict)
    if bl == "b2":
        tcd["weight_decay"] = 0.01
    elif bl == "b3":
        tcd["noise_std"] = 0.05

    if bl == "b4":
        for seed in seeds:
            tc = TrainConfig(**{**tcd, "seed": seed})
            X_tr, y_tr = universe.generate_train(n_train, seed=seed)
            X_te, y_te = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)
            model = create_baseline(bl, universe.input_dim, universe.output_dim, cfg["hidden_dim"])
            out = train_baseline(
                model, X_tr, y_tr, X_te, y_te, tc, is_polynomial=True
            )
            runs.append(out)
        return runs

    train_fn = train_oracle if bl == "b5" else train_baseline
    for seed in seeds:
        tc = TrainConfig(**{**tcd, "seed": seed})
        X_tr, y_tr = universe.generate_train(n_train, seed=seed)
        X_te, y_te = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)
        if bl == "b5":
            model = get_oracle(universe_name, universe.input_dim)
        else:
            model = create_baseline(bl, universe.input_dim, universe.output_dim, cfg["hidden_dim"])
        out = train_fn(model, X_tr, y_tr, X_te, y_te, tc, is_polynomial=False)
        runs.append(out)
    return runs


def _parse_gfti_fixed_alpha(model_id: str) -> float | None:
    """Parse gfti_alpha0 -> 0.0, gfti_alpha1 -> 1.0, gfti -> None."""
    if model_id == "gfti":
        return None
    if model_id == "gfti_alpha0":
        return 0.0
    if model_id == "gfti_alpha1":
        return 1.0
    return None


def _run_single_task(args_tuple: tuple) -> tuple[str, int, list[dict]]:
    """
    Worker for ProcessPoolExecutor. Runs one (model_id, n) combination with all seeds.
    Returns (model_id, n, runs).
    """
    model_id, universe_name, cfg, n, train_config_dict = args_tuple
    fixed_alpha = _parse_gfti_fixed_alpha(model_id)
    if fixed_alpha is not None or model_id == "gfti":
        runs = _run_gfti_at_n(universe_name, cfg, n, train_config_dict, fixed_alpha=fixed_alpha)
    else:
        runs = _run_baseline_at_n(model_id, universe_name, cfg, n, train_config_dict)
    return (model_id, n, runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--sample_sizes",
        type=str,
        default="50,100,200,400,800,1600",
        help="Comma-separated training sample sizes",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gfti,gfti_alpha0,gfti_alpha1,b1,b2,b3,b4,b5",
        help="Comma-separated model IDs to run (gfti_alpha0=discrete-only, gfti_alpha1=continuous-only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run (model, n) tasks in parallel (uses more GPU memory)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=6,
        help="Max parallel jobs when --parallel (default: 6)",
    )
    args = parser.parse_args()

    sample_sizes = [int(x.strip()) for x in args.sample_sizes.split(",")]
    models = [x.strip() for x in args.models.split(",")]

    config_path = args.config or Path(__file__).parent / "configs" / f"universe_{args.universe.lower()}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_config_dict = {
        "epochs": int(cfg["epochs"]),
        "batch_size": int(cfg["batch_size"]),
        "lr": float(cfg["lr"]),
        "weight_decay": float(cfg.get("weight_decay", 0)),
        "device": args.device,
        "noise_std": float(cfg.get("noise_std", 0)),
        "beta": float(cfg.get("beta", 0.1)),
        "tau_init": 1.0,
        "tau_final": 0.1,
        "tau_anneal_epochs": 200,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"sample_sizes": sample_sizes}
    for model_id in models:
        results[model_id] = {}

    def _aggregate_runs(runs: list[dict]) -> dict:
        nmse_vals = [float(r["test_nmse"]) for r in runs]
        mean_nmse = sum(nmse_vals) / len(nmse_vals)
        std_nmse = (
            (sum((x - mean_nmse) ** 2 for x in nmse_vals) / len(nmse_vals)) ** 0.5
            if len(nmse_vals) > 1
            else 0.0
        )
        runs_ser = [
            {
                "test_nmse": float(r["test_nmse"]),
                "curvature": [float(x) for x in r.get("curvature", [])],
                "test_errors": [float(x) for x in r.get("test_errors", [])],
                "final_alpha": float(r["final_alpha"]) if r.get("final_alpha") is not None else None,
            }
            for r in runs
        ]
        return {
            "nmse_mean": mean_nmse,
            "nmse_std": std_nmse,
            "runs": runs_ser,
        }

    tasks = [(model_id, n) for model_id in models for n in sample_sizes]

    if args.parallel:
        n_workers = args.jobs or len(tasks)
        n_workers = min(n_workers, len(tasks))
        print(f"Running {len(tasks)} tasks in parallel (jobs={n_workers})...")
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _run_single_task,
                    (model_id, args.universe, cfg, n, train_config_dict),
                ): (model_id, n)
                for model_id, n in tasks
            }
            for future in as_completed(futures):
                model_id, n = futures[future]
                try:
                    _, _, runs = future.result()
                    results[model_id][str(n)] = _aggregate_runs(runs)
                    agg = results[model_id][str(n)]
                    print(f"  {model_id} @ n={n} NMSE {agg['nmse_mean']:.4f} ± {agg['nmse_std']:.4f}")
                except Exception as e:
                    print(f"  {model_id} @ n={n} FAILED: {e}")
                    results[model_id][str(n)] = {
                        "nmse_mean": float("nan"),
                        "nmse_std": 0.0,
                        "runs": [],
                        "error": str(e),
                    }
    else:
        for model_id in models:
            for n in sample_sizes:
                print(f"  {model_id} @ n={n}...", end=" ", flush=True)
                fixed_alpha = _parse_gfti_fixed_alpha(model_id)
                if fixed_alpha is not None or model_id == "gfti":
                    runs = _run_gfti_at_n(
                        args.universe, cfg, n, train_config_dict,
                        fixed_alpha=fixed_alpha,
                    )
                else:
                    runs = _run_baseline_at_n(
                        model_id, args.universe, cfg, n, train_config_dict
                    )
                results[model_id][str(n)] = _aggregate_runs(runs)
                agg = results[model_id][str(n)]
                print(f"NMSE {agg['nmse_mean']:.4f} ± {agg['nmse_std']:.4f}")

    out_path = output_dir / f"scaling_{args.universe}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
