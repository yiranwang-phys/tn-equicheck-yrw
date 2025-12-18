from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from qiskit import qpy
from qiskit.quantum_info import Statevector

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


def load_qpy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("rb") as f:
        return qpy.load(f)[0]


def strip_measurements(circ):
    c = circ.copy()
    try:
        c.remove_final_measurements(inplace=True)
    except Exception:
        pass

    out = c.__class__(c.num_qubits, 0)
    out.global_phase = getattr(c, "global_phase", 0.0)
    for ci in c.data:
        if ci.operation.name == "measure":
            continue
        out.append(ci.operation, ci.qubits, [])
    return out


def count_injection_sites(circ) -> int:
    n_sites = 0
    for ci in circ.data:
        if ci.operation.name in ("measure", "barrier"):
            continue
        n_sites += len(ci.qubits)
    return n_sites


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    k = np.ones(window, dtype=float) / float(window)
    ypad = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(ypad, k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ideal-qpy", type=str, default="outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")
    ap.add_argument("--gamma-min", type=float, default=1e-3)
    ap.add_argument("--gamma-max", type=float, default=1e-2)
    ap.add_argument("--num", type=int, default=4000)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth-window", type=int, default=401)
    ap.add_argument("--out-root", type=str, default="outputs/sweeps")
    args = ap.parse_args()

    ideal = strip_measurements(load_qpy(Path(args.ideal_qpy)))
    n = ideal.num_qubits
    dim = 2**n
    n_sites = count_injection_sites(ideal)

    gammas = np.logspace(np.log10(args.gamma_min), np.log10(args.gamma_max), args.num)

    rng = default_rng(args.seed)
    psi0 = Statevector.from_label("0" * n)
    psi_ideal = psi0.evolve(ideal)

    single = np.zeros(args.num, dtype=float)
    mean = np.zeros(args.num, dtype=float)

    for i, g in enumerate(gammas):
        vals = []
        for _ in range(args.shots):
            noisy, _stats = apply_pauli_jump_after_each_gate(ideal, g, rng=rng)
            psi_noisy = psi0.evolve(noisy)
            f = float(abs(np.vdot(psi_ideal.data, psi_noisy.data)) ** 2)
            vals.append(f)
        single[i] = vals[0]
        mean[i] = float(np.mean(vals))

    smooth = rolling_mean(mean, args.smooth_window)
    p_no_jump = (1.0 - gammas) ** n_sites

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / f"statefid_gamma_{args.gamma_min:.0e}_to_{args.gamma_max:.0e}_N{args.num}_shots{args.shots}_n{n}_seed{args.seed}" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    res = {
        "metric": "output_state_fidelity",
        "n_qubits": n,
        "dim": dim,
        "n_sites": int(n_sites),
        "gamma_min": args.gamma_min,
        "gamma_max": args.gamma_max,
        "num": int(args.num),
        "shots": int(args.shots),
        "seed": int(args.seed),
        "gammas": gammas.tolist(),
        "single_shot": single.tolist(),
        "mean": mean.tolist(),
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    fig1 = out_dir / "fidelity_vs_gamma_raw.png"
    plt.figure()
    plt.scatter(gammas, single, s=6)
    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("gamma (log scale)")
    plt.ylabel("F_state (single-shot)")
    plt.title(f"Output-state fidelity (raw), n={n}, shots={args.shots}")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    fig2 = out_dir / "fidelity_vs_gamma_curve.png"
    plt.figure()
    plt.plot(gammas, mean, linewidth=1.0, label=f"MC mean (shots={args.shots})")
    plt.plot(gammas, smooth, linewidth=2.0, label=f"rolling mean (window={args.smooth_window})")
    plt.plot(gammas, p_no_jump, linestyle="--", linewidth=2.0, label=f"no-jump prob (1-gamma)^N, N={n_sites}")
    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("gamma (log scale)")
    plt.ylabel("F_state")
    plt.title(f"Output-state fidelity (curve), n={n}, sites={n_sites}")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    print(f"OUTPUT DIR = {out_dir.resolve()}")
    print(f"FIG1 = {fig1.resolve()}")
    print(f"FIG2 = {fig2.resolve()}")


if __name__ == "__main__":
    main()
