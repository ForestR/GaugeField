#!/usr/bin/env python3
"""Analyze experiment results and compute N₉₀ ratios."""

import argparse
import json
from pathlib import Path


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

        if not base_path.exists():
            print(f"  No baselines found at {base_path}")
            continue
        if not gfti_path.exists():
            print(f"  No GFTI results found at {gfti_path}")
            continue

        with open(base_path) as f:
            baselines = json.load(f)
        with open(gfti_path) as f:
            gfti = json.load(f)

        gfti_nmse = gfti["test_nmse_mean"]
        print(f"  GFTI Prototype: NMSE = {gfti_nmse:.4f}")

        for bl, data in baselines.items():
            if "test_nmse" in data:
                nmse = data["test_nmse"]
            else:
                nmse = data.get("test_nmse_mean", float("nan"))
            print(f"  Baseline {bl}: NMSE = {nmse:.4f}")

        if "curvature_trajectories" in gfti:
            print(f"  GFTI curvature trajectories: {len(gfti['curvature_trajectories'])} runs")


if __name__ == "__main__":
    main()
