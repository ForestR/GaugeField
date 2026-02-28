#!/usr/bin/env python3
"""Run β-strength ablation for GFTI (v0.5.4). Loops β ∈ {0.0, 0.01, 0.1, 1.0, 5.0} and saves gfti_{U}_beta{b}.json per β."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


def _run_beta(args_tuple: tuple) -> float:
    """Worker: run run_gfti.py for one β value. Returns beta."""
    beta, universe, config, output_dir, device = args_tuple
    script_dir = Path(__file__).resolve().parent
    run_gfti = script_dir / "run_gfti.py"
    cmd = [
        sys.executable,
        "-u",  # unbuffered stdout for real-time log visibility
        str(run_gfti),
        "--universe",
        universe,
        "--beta",
        str(beta),
        "--output_dir",
        output_dir,
        "--device",
        device,
    ]
    if config:
        cmd.extend(["--config", config])

    prefix = f"[β={beta}] "
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(f"{prefix}{line}", end="", flush=True)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    return beta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="B", choices=["A", "B", "C"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--betas",
        type=str,
        default="0.0,0.01,0.1,1.0,5.0",
        help="Comma-separated β values for ablation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run β values in parallel (uses more GPU memory)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=5,
        help="Max parallel jobs when --parallel (default: 5)",
    )
    args = parser.parse_args()

    betas = [float(x.strip()) for x in args.betas.split(",")]

    if args.parallel:
        n_workers = min(args.jobs, len(betas))
        print(f"Running {len(betas)} β values in parallel (jobs={n_workers})...")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _run_beta,
                    (beta, args.universe, args.config, args.output_dir, args.device),
                ): beta
                for beta in betas
            }
            for future in as_completed(futures):
                beta = futures[future]
                try:
                    future.result()
                    print(f"  β = {beta} done")
                except subprocess.CalledProcessError as e:
                    print(f"  β = {beta} FAILED: {e}")
                    raise
    else:
        for beta in betas:
            print(f"\n--- β = {beta} ---")
            _run_beta((beta, args.universe, args.config, args.output_dir, args.device))

    print(f"\nβ-ablation complete. Results in {args.output_dir}/gfti_{args.universe}_beta*.json")


if __name__ == "__main__":
    main()
