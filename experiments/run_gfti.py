#!/usr/bin/env python3
"""Run GFTI Prototype experiments for a given universe."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.gfti.models import GFTIPrototype
from src.gfti.train import TrainConfig, train_gfti
from src.gfti.universes import UniverseA, UniverseB, UniverseC


def get_universe(name: str):
    if name == "A":
        return UniverseA()
    if name == "B":
        return UniverseB()
    if name == "C":
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
    args = parser.parse_args()

    config_path = args.config or Path(__file__).parent / "configs" / f"universe_{args.universe.lower()}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    universe = get_universe(args.universe)
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

    results = {"runs": []}
    for seed in cfg["seeds"]:
        print(f"Running GFTI seed {seed}...")
        train_config.seed = seed
        X_train, y_train = universe.generate_train(cfg["n_train_samples"], seed=seed)
        X_test, y_test = universe.generate_test(cfg["n_test_samples"], seed=seed + 1000)

        model = GFTIPrototype(
            input_dim=universe.input_dim,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            output_dim=universe.output_dim,
        )
        out = train_gfti(model, X_train, y_train, X_test, y_test, train_config)
        results["runs"].append(out)

    nmse = [r["test_nmse"] for r in results["runs"]]
    results["test_nmse_mean"] = sum(nmse) / len(nmse)
    results["curvature_trajectories"] = [r["curvature"] for r in results["runs"]]
    print(f"GFTI test NMSE (mean): {results['test_nmse_mean']:.4f}")

    out_path = output_dir / f"gfti_{args.universe}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
