from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qiskit import qpy
from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


IDEAL_DIR = Path("outputs/ideal/twolocal_n6_seed0")
IDEAL_QPY = IDEAL_DIR / "circuit_ideal.qpy"


def load_qpy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run: python scripts/make_ideal_twolocal_n6.py")
    with path.open("rb") as f:
        return qpy.load(f)[0]


def count_injection_sites(circ) -> int:
    # Count all non-measurement, non-barrier operations as potential injection sites
    n = 0
    for ci in circ.data:
        op = ci.operation
        name = getattr(op, "name", "")
        if name in ("measure", "barrier"):
            continue
        n += 1
    return n


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(ypad, kernel, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-min", type=float, default=1e-3)
    ap.add_argument("--gamma-max", type=float, default=1e-2)
    ap.add_argument("--num", type=int, default=4000)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth-window", type=int, default=401)
    args = ap.parse_args()

    ideal = load_qpy(IDEAL_QPY)
    n_qubits = ideal.num_qubits
    d = 2 ** n_qubits

    n_sites = count_injection_sites(ideal)
    print("ideal qubits:", n_qubits)
    print("injection sites:", n_sites)

    gammas = np.logspace(np.log10(args.gamma_min), np.log10(args.gamma_max), args.num)

    # A) "process-fidelity proxy" here = probability of zero injected jumps (MC estimate)
    p0_mc = np.zeros_like(gammas, dtype=float)

    # B) convert A to average gate fidelity (useful to compare with many papers)
    # F_avg = (d * F_e + 1) / (d + 1) with F_e ~ entanglement/process fidelity
    favg_mc = np.zeros_like(gammas, dtype=float)

    for i, g in enumerate(gammas):
        n0 = 0
        base = args.seed + i * args.shots
        for s in range(args.shots):
            _, stats = apply_pauli_jump_after_each_gate(
                ideal,
                float(g),
                int(base + s),
                include_measurements=False,
            )
            if getattr(stats, "n_noise_ops", 0) == 0:
                n0 += 1

        p0 = n0 / float(args.shots)
        p0_mc[i] = p0
        favg_mc[i] = (d * p0 + 1.0) / (d + 1.0)

    # Theory line used in your current plots: probability of no jump anywhere
    p0_theory = (1.0 - gammas) ** float(n_sites)
    favg_theory = (d * p0_theory + 1.0) / (d + 1.0)

    p0_smooth = rolling_mean(p0_mc, args.smooth_window)
    favg_smooth = rolling_mean(favg_mc, args.smooth_window)

    tag = f"dense_gamma_{args.gamma_min:.0e}_to_{args.gamma_max:.0e}_n{n_qubits}_seed{args.seed}"
    outdir = Path("outputs/sweeps") / tag / datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- FIG A1: A raw points (MC mean per gamma) ----
    fig = plt.figure()
    plt.plot(gammas, p0_mc, marker=".", linestyle="none")
    plt.plot(gammas, p0_theory, linestyle="--")
    plt.xscale("log")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("gamma (log scale)")
    plt.ylabel("P(no jump)  [MC mean per gamma]")
    plt.title(f"P(no jump) vs gamma (points), n={n_qubits}, sites={n_sites}, shots={args.shots}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    pA1 = outdir / "A_p0_vs_gamma_points.png"
    fig.savefig(pA1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- FIG A2: A smooth curve ----
    fig = plt.figure()
    plt.plot(gammas, p0_smooth, linewidth=2)
    plt.plot(gammas, p0_theory, linestyle="--")
    plt.xscale("log")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("gamma (log scale)")
    plt.ylabel("P(no jump)  [smoothed]")
    plt.title(f"P(no jump) vs gamma (smooth), n={n_qubits}, sites={n_sites}, window={args.smooth_window}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    pA2 = outdir / "A_p0_vs_gamma_smooth.png"
    fig.savefig(pA2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- FIG B1: B raw points ----
    fig = plt.figure()
    plt.plot(gammas, favg_mc, marker=".", linestyle="none")
    plt.plot(gammas, favg_theory, linestyle="--")
    plt.xscale("log")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("gamma (log scale)")
    plt.ylabel("F_avg  [MC mean per gamma]")
    plt.title(f"Average gate fidelity vs gamma (points), n={n_qubits}, sites={n_sites}, shots={args.shots}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    pB1 = outdir / "B_favg_vs_gamma_points.png"
    fig.savefig(pB1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- FIG B2: B smooth curve ----
    fig = plt.figure()
    plt.plot(gammas, favg_smooth, linewidth=2)
    plt.plot(gammas, favg_theory, linestyle="--")
    plt.xscale("log")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("gamma (log scale)")
    plt.ylabel("F_avg  [smoothed]")
    plt.title(f"Average gate fidelity vs gamma (smooth), n={n_qubits}, sites={n_sites}, window={args.smooth_window}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    pB2 = outdir / "B_favg_vs_gamma_smooth.png"
    fig.savefig(pB2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_qubits": int(n_qubits),
        "dim": int(d),
        "n_sites": int(n_sites),
        "gamma_min": float(args.gamma_min),
        "gamma_max": float(args.gamma_max),
        "num": int(args.num),
        "shots": int(args.shots),
        "seed": int(args.seed),
        "smooth_window": int(args.smooth_window),
        "gammas": gammas.tolist(),
        "A_p0_mc": p0_mc.tolist(),
        "A_p0_smooth": p0_smooth.tolist(),
        "A_p0_theory": p0_theory.tolist(),
        "B_favg_mc": favg_mc.tolist(),
        "B_favg_smooth": favg_smooth.tolist(),
        "B_favg_theory": favg_theory.tolist(),
        "fig_A_points": str(pA1),
        "fig_A_smooth": str(pA2),
        "fig_B_points": str(pB1),
        "fig_B_smooth": str(pB2),
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("WROTE:", outdir.resolve())
    print("FIG A1:", pA1.resolve())
    print("FIG A2:", pA2.resolve())
    print("FIG B1:", pB1.resolve())
    print("FIG B2:", pB2.resolve())


if __name__ == "__main__":
    main()
