#!/usr/bin/env python3
"""Run GFTI Prototype experiments for a given universe."""

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.gfti.models import GFTIPrototype
from src.gfti.train import TrainConfig, train_gfti
from src.gfti.universes import UniverseA, UniverseB, UniverseC


def _run_single_gfti(args_tuple: tuple) -> dict:
    """
    Worker for ProcessPoolExecutor. Runs one GFTI seed.
    Returns the result dict for that run.
    """
    seed, universe_name, cfg, train_config_dict = args_tuple
    universe = get_universe(universe_name, cfg)
    train_config = TrainConfig(**train_config_dict)
    train_config.seed = seed

    X_train, y_train = universe.generate_train(cfg["n_train_samples"], seed=seed)
    X_test, y_test = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)

    fixed_alpha = train_config_dict.get("fixed_alpha")
    model = GFTIPrototype(
        input_dim=universe.input_dim,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        output_dim=universe.output_dim,
        fixed_alpha=fixed_alpha,
    )
    return train_gfti(model, X_train, y_train, X_test, y_test, train_config)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run seeds in parallel (uses more GPU memory)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=5,
        help="Max parallel jobs when --parallel (default: 5)",
    )
    parser.add_argument(
        "--fixed_alpha",
        type=float,
        default=None,
        help="Fix branch: 1.0=continuous-only, 0.0=discrete-only, omit=learned",
    )
    args = parser.parse_args()

    config_path = args.config or Path(__file__).parent / "configs" / f"universe_{args.universe.lower()}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    universe = get_universe(args.universe, cfg)
    train_config = TrainConfig(
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0)),
        device=args.device,
        beta=float(cfg.get("beta", 0.1)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config_dict = {
        "epochs": int(cfg["epochs"]),
        "batch_size": int(cfg["batch_size"]),
        "lr": float(cfg["lr"]),
        "weight_decay": float(cfg.get("weight_decay", 0)),
        "device": args.device,
        "beta": float(cfg.get("beta", 0.1)),
        "fixed_alpha": args.fixed_alpha,
    }

    results = {"runs": []}
    seeds = cfg["seeds"]

    if args.parallel:
        n_workers = min(args.jobs, len(seeds))
        print(f"Running GFTI {len(seeds)} seeds in parallel (jobs={n_workers})...")
        ctx = multiprocessing.get_context("spawn")
        seed_to_out = {}
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _run_single_gfti,
                    (seed, args.universe, cfg, {**train_config_dict, "seed": seed}),
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    out = future.result()
                    seed_to_out[seed] = out
                    print(f"  seed {seed} test NMSE: {out['test_nmse']:.4f}")
                except Exception as e:
                    print(f"  seed {seed} FAILED: {e}")
                    raise
        results["runs"] = [seed_to_out[s] for s in seeds]
    else:
        for seed in seeds:
            print(f"Running GFTI seed {seed}...")
            train_config.seed = seed
            X_train, y_train = universe.generate_train(cfg["n_train_samples"], seed=seed)
            X_test, y_test = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)

            model = GFTIPrototype(
                input_dim=universe.input_dim,
                hidden_dim=cfg["hidden_dim"],
                latent_dim=cfg["latent_dim"],
                output_dim=universe.output_dim,
                fixed_alpha=args.fixed_alpha,
            )
            out = train_gfti(model, X_train, y_train, X_test, y_test, train_config)
            results["runs"].append(out)

    nmse = [r["test_nmse"] for r in results["runs"]]
    results["test_nmse_mean"] = sum(nmse) / len(nmse)
    results["curvature_trajectories"] = [r["curvature"] for r in results["runs"]]
    results["alpha_trajectories"] = [r.get("final_alpha") for r in results["runs"]]
    print(f"GFTI test NMSE (mean): {results['test_nmse_mean']:.4f}")

    alpha_suffix = f"_alpha{args.fixed_alpha}" if args.fixed_alpha is not None else ""
    out_path = output_dir / f"gfti_{args.universe}{alpha_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
