#!/usr/bin/env python3
"""Monte‑Carlo study of the hybrid DE optimiser on the noisy laser mock‑up.

For each run we print the usual vectors and *normalised* MSEs:
    nmse(a, b) = mean( ((clip(a,0,1023) – clip(b,0,1023)) / 1023)² )
so a 1‑count error contributes ≈9.56 × 10⁻⁷, and the theoretical maximum is 1.

After *N_RUNS* repetitions we display mean ± σ for
* nmse(init, found)
* nmse(true, found)
* fitness(found)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution
from laser_mockup import MockupACFPulse

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SEED_BASE        = 44        # base seed, run‑index is added for each repetition
N_RUNS           = 10        # number of independent experiments
DE_MAXITER       = 500       # generations → evaluations ≈ MAXITER×popsize
DE_POPSIZE       = 15        # individuals per generation
INITIAL_VECTOR   = np.zeros  # callable: np.zeros or lambda n,dtype: np.full(n,500,dtype)

np.set_printoptions(threshold=1000, linewidth=120, formatter={"int": "{:d}".format})

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def nmse(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised MSE in [0,1] for vectors limited to 0–1023."""
    a_clip = np.clip(a.astype(float), 0, 1023)
    b_clip = np.clip(b.astype(float), 0, 1023)
    diff = (a_clip - b_clip) / 1023.0
    return float(np.mean(diff ** 2))


def device_factory(seed: int) -> MockupACFPulse:
    return MockupACFPulse(seed=seed)


def fitness(vec_int: np.ndarray, device: MockupACFPulse) -> float:
    delay, acf = device.get_pulse(vec_int)
    return device.fitness(np.asarray(delay), np.asarray(acf))


def objective(vec: np.ndarray, device: MockupACFPulse) -> float:
    return fitness(np.rint(vec).astype(int), device)


def vec_to_str(v: np.ndarray) -> str:
    return np.array2string(v, separator=" ", formatter={"int": "{:4d}".format})


def print_summary(title: str, vec: np.ndarray, fit: float) -> None:
    print(f"{title:<9s}: {vec_to_str(vec)}   fitness={fit:8.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────────────

def single_run(run_id: int) -> tuple[float, float, float]:
    dev_seed = SEED_BASE + run_id          # secret optimum varies per run
    de_seed  = SEED_BASE + 1000 + run_id   # DE population randomness

    device = device_factory(dev_seed)

    # 0) initial mask -----------------------------------------------------
    x0 = INITIAL_VECTOR(device.GENES, dtype=float)
    f0 = fitness(x0.astype(int), device)

    # 1) Differential evolution ------------------------------------------
    bounds = [(device.GENE_MIN, device.GENE_MAX)] * device.GENES
    res_de = differential_evolution(
        objective,
        bounds,
        args=(device,),
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        polish=False,
        seed=de_seed,
        updating="deferred",
        disp=False,
    )
    found = np.rint(res_de.x).astype(int)
    f_found = fitness(found, device)

    true_vec = device.true_state

    nmse_init_found  = nmse(x0,   found)
    nmse_true_found  = nmse(true_vec, found)

    # 2) console output ---------------------------------------------------
    print(f"\nRun {run_id+1}/{N_RUNS}")
    print("-------------------------------------------------------------------")
    print_summary("Initial", x0.astype(int), f0)
    print_summary("Found",   found,          f_found)
    print_summary("True",    true_vec,       fitness(true_vec, device))
    print(f"nMSE(init,found)={nmse_init_found:.4f}   nMSE(true,found)={nmse_true_found:.4f}\n")

    return nmse_init_found, nmse_true_found, f_found

# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    nmse_init_all = []
    nmse_true_all = []
    fitness_all   = []

    for run in range(N_RUNS):
        n_init, n_true, f = single_run(run)
        nmse_init_all.append(n_init)
        nmse_true_all.append(n_true)
        fitness_all.append(f)

    nmse_init_all = np.asarray(nmse_init_all)
    nmse_true_all = np.asarray(nmse_true_all)
    fitness_all   = np.asarray(fitness_all)

    print("=== Aggregate results ================================================" )
    print(f"nMSE(init,found):  mean={nmse_init_all.mean():.4f}  std={nmse_init_all.std(ddof=1):.4f}")
    print(f"nMSE(true,found):  mean={nmse_true_all.mean():.4f}  std={nmse_true_all.std(ddof=1):.4f}")
    print(f"Fitness(found):    mean={fitness_all.mean():.3f}   std={fitness_all.std(ddof=1):.3f}")


if __name__ == "__main__":
    print("old fitnesss")
    main()
