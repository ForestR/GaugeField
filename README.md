# GaugeField (GFTI)

**Gauge Field Theory of Intelligence** — Experimental validation of symmetry-discovery learners under orbit-disjoint distribution shift.

## Overview

This repository implements **v0.5.2: The Separating Construction Protocol**, a controlled experiment to test whether learners that explicitly discover symmetries (Class 3) outperform strong statistical baselines (Class 2) on out-of-distribution generalization.

## Theory Summary

The GFTI framework proposes a four-class taxonomy of intelligence:

- **Class 1 & 2 (Statistical)**: Efficient compression and routing — well-served by current Transformer architectures.
- **Class 3 & 4 (Structural)**: Symmetry discovery and topological creativity — require inductive bias beyond scaling.

The central claim: **statistical learners cannot emerge structural intelligence through data scaling alone** (topological obstruction). This experiment isolates the **inductive bias of symmetry discovery** in minimal toy universes.

## Experiment Structure

### Three Synthetic Universes

| Universe | Group | Generative Law | Train Orbit | Test Orbit (OOD) |
|----------|-------|----------------|-------------|------------------|
| **A** | SO(2) | y = sin(5·‖x‖) | φ ∈ [0, π/3] | φ ∈ [π, 4π/3] |
| **B** | S₅ (Cyclic Trap) | y = Σ xᵢ·x_{(i mod 5)+1} | C₅ subgroup | Transpositions |
| **C** | SO(1,1) Lorentz | y = x₁² − x₂² | ψ ∈ [0, 0.5] | ψ ∈ [2.0, 3.0] |

### Models

- **Baselines 1–4**: Wide MLP, Regularized MLP, Augmented MLP, Polynomial (degree ≤ 4)
- **Baseline 5 (Oracle)**: Equivariant architectures (E2-Steerable, DeepSets, LorentzNet)
- **GFTI Prototype**: MLP + Mixture of Symmetries (MoS) layer with curvature loss

### Success Criterion

- **N₉₀**: Samples to reach <10% Normalized MSE on test orbit
- **R = N₉₀(Baseline) / N₉₀(GFTI)** — success if R > 10 or baseline never converges

## Installation

```bash
conda create -n gfti python=3.10
conda activate gfti
pip install -r requirements.txt
```

Or use existing `torch_env`:

```bash
conda activate torch_env
pip install -r requirements.txt
```

## Usage

```bash
# Run baselines (5 seeds each) — uses GPU automatically when available
python experiments/run_baselines.py --universe A --output_dir results/

# Run baselines in parallel (faster, uses more GPU memory)
python experiments/run_baselines.py --universe A --parallel --jobs 5

# Run GFTI prototype
python experiments/run_gfti.py --universe A --output_dir results/

# Force CPU if needed
python experiments/run_baselines.py --universe A --device cpu

# Analyze results
python experiments/analyze_results.py --results_dir results/
```

## Project Structure

```
GaugeField/
├── src/gfti/
│   ├── universes/       # Data generators (A, B, C)
│   ├── models/          # MLP, MoS, Prototype, Oracles
│   ├── losses/          # Curvature loss κ_T
│   ├── metrics/         # N₉₀ sample complexity
│   └── train.py         # Shared training loop
├── experiments/
│   ├── configs/         # YAML configs per universe
│   ├── run_baselines.py
│   ├── run_gfti.py
│   └── analyze_results.py
└── notebooks/
    └── 01_universe_visualization.ipynb
```

## Execution Timeline (v0.5.2)

- Universe Generation: Feb 27 – Mar 1
- Baseline Runs (5 seeds): Mar 2 – Mar 10
- GFTI Prototype Runs (5 seeds): Mar 11 – Mar 20
- Report: Mar 25

## License

See [LICENSE](LICENSE).
