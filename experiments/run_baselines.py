#!/usr/bin/env python3
"""Run baseline experiments (B1-B5) for a given universe."""

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.gfti.models import create_baseline
from src.gfti.models.oracles import E2SteerableMLP, DeepSets, LorentzNetMLP
from src.gfti.train import TrainConfig, train_baseline, train_oracle
from src.gfti.universes import UniverseA, UniverseB, UniverseC


def _run_single_baseline(args_tuple: tuple) -> tuple[str, dict]:
    """
    Worker for ProcessPoolExecutor. Runs one baseline with all seeds.
    Returns (baseline_id, results_dict).
    """
    bl, universe_name, cfg, train_config_dict = args_tuple
    universe = get_universe(universe_name, cfg)
    train_config = TrainConfig(**train_config_dict)

    if bl == "b2":
        train_config.weight_decay = 0.01
    elif bl == "b3":
        train_config.noise_std = 0.05

    if bl == "b4":
        runs = []
        for seed in cfg["seeds"]:
            X_tr, y_tr = universe.generate_train(cfg["n_train_samples"], seed=seed)
            X_te, y_te = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)
            model = create_baseline(bl, universe.input_dim, universe.output_dim, cfg["hidden_dim"])
            out = train_baseline(
                model, X_tr, y_tr, X_te, y_te, train_config, is_polynomial=True
            )
            runs.append(out)
        results = {"runs": runs, "test_nmse_mean": sum(r["test_nmse"] for r in runs) / len(runs)}
        return (bl, results)

    train_fn = train_oracle if bl == "b5" else train_baseline
    runs = []
    for seed in cfg["seeds"]:
        train_config.seed = seed
        X_tr, y_tr = universe.generate_train(cfg["n_train_samples"], seed=seed)
        X_te, y_te = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)
        if bl == "b5":
            model = get_oracle(universe_name, universe.input_dim)
        else:
            model = create_baseline(bl, universe.input_dim, universe.output_dim, cfg["hidden_dim"])
        out = train_fn(model, X_tr, y_tr, X_te, y_te, train_config, is_polynomial=False)
        runs.append(out)
    results = {"runs": runs, "test_nmse_mean": sum(r["test_nmse"] for r in runs) / len(runs)}
    return (bl, results)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--baseline", type=str, default=None, choices=["b1", "b2", "b3", "b4", "b5"])
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run baselines in parallel (uses more GPU memory)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=5,
        help="Max parallel jobs when --parallel (default: 5)",
    )
    args = parser.parse_args()

    config_path = args.config or Path(__file__).parent / "configs" / f"universe_{args.universe.lower()}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    universe = get_universe(args.universe, cfg)
    X_train, y_train = universe.generate_train(cfg["n_train_samples"], seed=cfg["seeds"][0])
    X_test, y_test = universe.generate_test(cfg["n_test_samples"], seed=cfg["seeds"][0] + 1)

    train_config = TrainConfig(
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0)),
        device=args.device,
        noise_std=float(cfg.get("noise_std", 0)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines_to_run = [args.baseline] if args.baseline else ["b1", "b2", "b3", "b4", "b5"]
    results = {}

    train_config_dict = {
        "epochs": int(cfg["epochs"]),
        "batch_size": int(cfg["batch_size"]),
        "lr": float(cfg["lr"]),
        "weight_decay": float(cfg.get("weight_decay", 0)),
        "seed": cfg["seeds"][0],
        "device": args.device,
        "noise_std": float(cfg.get("noise_std", 0)),
        "beta": float(cfg.get("beta", 0.1)),
        "tau_init": 1.0,
        "tau_final": 0.1,
        "tau_anneal_epochs": 200,
    }

    if args.parallel:
        n_workers = min(args.jobs, len(baselines_to_run))
        print(f"Running {len(baselines_to_run)} baselines in parallel (jobs={n_workers})...")
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _run_single_baseline,
                    (bl, args.universe, cfg, train_config_dict),
                ): bl
                for bl in baselines_to_run
            }
            for future in as_completed(futures):
                bl = futures[future]
                try:
                    _, res = future.result()
                    results[bl] = res
                    print(f"  {bl} test NMSE (mean): {results[bl]['test_nmse_mean']:.4f}")
                except Exception as e:
                    print(f"  {bl} FAILED: {e}")
                    results[bl] = {"runs": [], "test_nmse_mean": float("nan"), "error": str(e)}
    else:
        for bl in baselines_to_run:
            print(f"Running baseline {bl}...")
            if bl == "b4":
                runs = []
                for seed in cfg["seeds"]:
                    X_tr, y_tr = universe.generate_train(cfg["n_train_samples"], seed=seed)
                    X_te, y_te = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)
                    model = create_baseline(bl, universe.input_dim, universe.output_dim, cfg["hidden_dim"])
                    out = train_baseline(
                        model, X_tr, y_tr, X_te, y_te, train_config, is_polynomial=True
                    )
                    runs.append(out)
                results[bl] = {"runs": runs, "test_nmse_mean": sum(r["test_nmse"] for r in runs) / len(runs)}
                print(f"  {bl} test NMSE (mean): {results[bl]['test_nmse_mean']:.4f}")
                continue

            if bl == "b5":
                train_fn = train_oracle
            else:
                train_fn = train_baseline
                if bl == "b2":
                    train_config.weight_decay = 0.01
                elif bl == "b3":
                    train_config.noise_std = 0.05

            runs = []
            for seed in cfg["seeds"]:
                train_config.seed = seed
                X_tr, y_tr = universe.generate_train(cfg["n_train_samples"], seed=seed)
                X_te, y_te = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)
                if bl == "b5":
                    model = get_oracle(args.universe, universe.input_dim)
                else:
                    model = create_baseline(bl, universe.input_dim, universe.output_dim, cfg["hidden_dim"])
                out = train_fn(model, X_tr, y_tr, X_te, y_te, train_config, is_polynomial=False)
                runs.append(out)

            results[bl] = {"runs": runs, "test_nmse_mean": sum(r["test_nmse"] for r in runs) / len(runs)}
            print(f"  {bl} test NMSE (mean): {results[bl]['test_nmse_mean']:.4f}")

    out_path = output_dir / f"baselines_{args.universe}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
