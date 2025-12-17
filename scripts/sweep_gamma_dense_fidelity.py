from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy


IDEAL_QPY = Path("outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")


def load_qpy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("rb") as f:
        return qpy.load(f)[0]


def strip_measurements(circ):
    return circ.remove_final_measurements(inplace=False)


def count_injection_sites(circ) -> int:
    # Each "site" = (gate, qubit-in-its-qargs) where we may insert a Pauli jump after the gate.
    n_sites = 0
    for ci in circ.data:
        op = ci.operation
        if op.name in ("barrier", "measure"):
            continue
        n_sites += len(ci.qubits)
    return int(n_sites)


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    w = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, w, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-min", type=float, default=1e-3)
    ap.add_argument("--gamma-max", type=float, default=1e-2)
    ap.add_argument("--num", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=1, help="Monte-Carlo samples per gamma (fast, does NOT build circuits)")
    ap.add_argument("--smooth-window", type=int, default=201)
    args = ap.parse_args()

    ideal_meas = load_qpy(IDEAL_QPY)
    ideal = strip_measurements(ideal_meas)
    n = ideal.num_qubits

    n_sites = count_injection_sites(ideal)

    gammas = np.logspace(np.log10(args.gamma_min), np.log10(args.gamma_max), args.num)

    # Paper-style: for Pauli-jump with independent insertion per site,
    # the "trace-fidelity proxy" (for unitary Pauli errors) is essentially Prob(no jump).
    f_theory = (1.0 - gammas) ** n_sites

    rng = np.random.default_rng(args.seed)

    # Raw single-shot: Bernoulli(p = f_theory)
    u = rng.random(args.num)
    f_raw = (u < f_theory).astype(float)

    # Optional MC mean per gamma (still cheap: only RNG)
    if args.shots > 1:
        # mean of Bernoulli trials per gamma
        uu = rng.random((args.shots, args.num))
        f_mc_mean = (uu < f_theory[None, :]).mean(axis=0)
    else:
        f_mc_mean = None

    f_smooth = rolling_mean(f_raw, args.smooth_window)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("outputs/sweeps") / f"dense_gamma_{args.gamma_min:.0e}_to_{args.gamma_max:.0e}_N{args.num}_shots{args.shots}" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- FIG1: raw points ---
    plt.figure()
    plt.scatter(gammas, f_raw, s=4)
    plt.plot(gammas, f_theory, linestyle="--")
    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Single-shot fidelity vs gamma (raw points), n={n}, sites={n_sites}")
    plt.xlabel("gamma (log scale)")
    plt.ylabel("trace fidelity proxy (single-shot)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    fig1 = out_dir / "fidelity_vs_gamma_points.png"
    plt.savefig(fig1, dpi=200, bbox_inches="tight")
    plt.close()

    # --- FIG2: smooth curve ---
    plt.figure()
    plt.plot(gammas, f_smooth, label=f"rolling mean (window={args.smooth_window})")
    if f_mc_mean is not None:
        plt.plot(gammas, f_mc_mean, label=f"MC mean (shots={args.shots})")
    plt.plot(gammas, f_theory, linestyle="--", label=f"theory: (1-gamma)^N_sites, N_sites={n_sites}")
    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Fidelity vs gamma (smooth curve), n={n}, sites={n_sites}")
    plt.xlabel("gamma (log scale)")
    plt.ylabel("fidelity")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    fig2 = out_dir / "fidelity_vs_gamma_curve.png"
    plt.savefig(fig2, dpi=200, bbox_inches="tight")
    plt.close()

    payload = {
        "ideal_qpy": str(IDEAL_QPY),
        "n_qubits": int(n),
        "n_sites": int(n_sites),
        "gamma_min": float(args.gamma_min),
        "gamma_max": float(args.gamma_max),
        "num": int(args.num),
        "seed": int(args.seed),
        "shots": int(args.shots),
        "smooth_window": int(args.smooth_window),
        "gammas": gammas.tolist(),
        "f_raw": f_raw.tolist(),
        "f_smooth": f_smooth.tolist(),
        "f_theory": f_theory.tolist(),
        "f_mc_mean": None if f_mc_mean is None else f_mc_mean.tolist(),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(">>> sweep_gamma_dense_fidelity.py")
    print("ideal qubits:", n)
    print("injection sites:", n_sites)
    print("WROTE:", out_dir.resolve())
    print("FIG1 :", fig1.resolve())
    print("FIG2 :", fig2.resolve())


if __name__ == "__main__":
    main()
