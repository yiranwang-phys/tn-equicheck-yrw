import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def load_qpy_first(path: Path) -> QuantumCircuit:
    with path.open("rb") as f:
        return qpy.load(f)[0]


def unitary_part(circ: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(circ.num_qubits)
    for ci in circ.data:
        op = ci.operation
        if op.name in ("measure", "reset", "barrier"):
            continue
        out.append(op, ci.qubits, [])
    return out


def count_injection_sites(circ_unitary: QuantumCircuit) -> int:
    n = 0
    for ci in circ_unitary.data:
        op = ci.operation
        if op.name == "barrier":
            continue
        if op.num_qubits > 0:
            n += 1
    return n


def smooth_edge_safe(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    w = int(window)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(ypad, kernel, mode="valid")


def unitary_overlap_metrics(U_ideal: np.ndarray, U_noisy: np.ndarray) -> tuple[float, float]:
    d = U_ideal.shape[0]
    tr = np.trace(U_ideal.conj().T @ U_noisy)
    phi = float(np.abs(tr) / float(d))
    f_avg = float((float(d) * (phi * phi) + 1.0) / (float(d) + 1.0))
    return phi, f_avg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-min", type=float, default=1e-4)
    ap.add_argument("--gamma-max", type=float, default=1.0)
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth-window", type=int, default=31)
    ap.add_argument("--mode", choices=["proxy", "unitary"], default="proxy")
    args = ap.parse_args()

    ideal_qpy = Path("outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")
    if not ideal_qpy.exists():
        raise FileNotFoundError(f"Missing: {ideal_qpy}. Run: python scripts/make_ideal_twolocal_n6.py")

    ideal_meas = load_qpy_first(ideal_qpy)
    ideal = unitary_part(ideal_meas)

    n_qubits = ideal.num_qubits
    d = 2**n_qubits
    n_sites = count_injection_sites(ideal)

    gammas = np.logspace(np.log10(args.gamma_min), np.log10(args.gamma_max), args.num)

    out_dir = Path("outputs/sweeps") / f"gamma_{args.gamma_min:.0e}_to_{args.gamma_max:.0e}_N{args.num}_shots{args.shots}_n{n_qubits}_seed{args.seed}" / now_ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ideal qubits: {n_qubits}")
    print(f"injection sites: {n_sites}")
    print(f"mode: {args.mode}")
    print(f"OUTPUT DIR: {out_dir.resolve()}")

    U_ideal = None
    if args.mode == "unitary":
        U_ideal = Operator(ideal).data

    p0_list = []
    phi_mean_list = []
    favg_mean_list = []

    raw_gamma = []
    raw_favg = []

    for i, g in enumerate(gammas):
        n_nojump = 0
        favg_shots = []
        phi_shots = []

        for s in range(args.shots):
            shot_seed = int(args.seed + i * args.shots + s)
            noisy_meas, stats = apply_pauli_jump_after_each_gate(
                ideal_meas,
                float(g),
                seed=shot_seed,
                include_measurements=False,
            )
            noisy = unitary_part(noisy_meas)

            nojump = int(getattr(stats, "n_noise_ops", 0) == 0)
            n_nojump += nojump

            if args.mode == "proxy":
                phi = 1.0 if nojump else 0.0
                favg = (float(d) * (phi * phi) + 1.0) / (float(d) + 1.0)
            else:
                U_noisy = Operator(noisy).data
                phi, favg = unitary_overlap_metrics(U_ideal, U_noisy)

            favg_shots.append(float(favg))
            phi_shots.append(float(phi))

            raw_gamma.append(float(g))
            raw_favg.append(float(favg))

        p0 = float(n_nojump) / float(args.shots)
        p0_list.append(p0)
        phi_mean_list.append(float(np.mean(phi_shots)))
        favg_mean_list.append(float(np.mean(favg_shots)))

        if (i + 1) % max(1, args.num // 10) == 0:
            print(f"{i+1}/{args.num} gamma={g:.3e}  p0={p0:.4f}  favg_mean={favg_mean_list[-1]:.4f}")

    p0_arr = np.array(p0_list, dtype=float)
    phi_mean = np.array(phi_mean_list, dtype=float)
    favg_mean = np.array(favg_mean_list, dtype=float)

    p0_theory = (1.0 - gammas) ** float(n_sites)
    favg_theory = (1.0 / (float(d) + 1.0)) + (float(d) / (float(d) + 1.0)) * p0_theory

    favg_smooth = smooth_edge_safe(favg_mean, args.smooth_window)

    results = {
        "n_qubits": n_qubits,
        "d": d,
        "n_sites": n_sites,
        "mode": args.mode,
        "gamma_min": args.gamma_min,
        "gamma_max": args.gamma_max,
        "num": args.num,
        "shots": args.shots,
        "seed": args.seed,
        "smooth_window": args.smooth_window,
        "gammas": gammas.tolist(),
        "p0_mc": p0_arr.tolist(),
        "phi_mean_mc": phi_mean.tolist(),
        "favg_mean_mc": favg_mean.tolist(),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig1 = out_dir / "fidelity_vs_gamma_raw.png"
    plt.figure()
    plt.xscale("log")
    plt.scatter(raw_gamma, raw_favg, s=2)
    plt.xlabel("gamma (log scale)")
    plt.ylabel("fidelity (per-shot)")
    plt.title(f"Raw per-shot fidelity vs gamma, n={n_qubits}, sites={n_sites}, mode={args.mode}")
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    fig2 = out_dir / "fidelity_vs_gamma_curve.png"
    plt.figure()
    plt.xscale("log")
    plt.plot(gammas, favg_mean, label=f"MC mean (shots={args.shots})")
    plt.plot(gammas, favg_smooth, label=f"smooth (window={args.smooth_window})")
    plt.plot(gammas, favg_theory, "--", label=f"theory from p0=(1-gamma)^sites, sites={n_sites}")
    plt.xlabel("gamma (log scale)")
    plt.ylabel("fidelity")
    plt.title(f"Fidelity vs gamma, n={n_qubits}, mode={args.mode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    print(f"FIG1: {fig1.resolve()}")
    print(f"FIG2: {fig2.resolve()}")


if __name__ == "__main__":
    main()
