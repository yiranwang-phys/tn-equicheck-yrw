import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.quantum_info import Statevector

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


def _load_ideal_qpy(n: int, seed: int) -> tuple:
    qpy_path = Path("outputs") / "ideal" / f"twolocal_n{n}_seed{seed}" / "circuit_ideal.qpy"
    if not qpy_path.exists():
        raise FileNotFoundError(str(qpy_path))
    with qpy_path.open("rb") as f:
        circ = qpy.load(f)[0]
    return circ, str(qpy_path)


def _pure_state_fidelity(psi: Statevector, phi: Statevector) -> float:
    v = np.vdot(psi.data, phi.data)
    return float(np.abs(v) ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma-min-exp", type=int, default=-3)
    p.add_argument("--gamma-max-exp", type=int, default=-1)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smooth-window", type=int, default=11)
    args = p.parse_args()

    ideal, ideal_qpy = _load_ideal_qpy(args.n, args.seed)
    psi_ideal = Statevector.from_instruction(ideal)

    gammas = np.logspace(args.gamma_min_exp, args.gamma_max_exp, args.num)

    meanF = np.zeros_like(gammas, dtype=float)
    stdF = np.zeros_like(gammas, dtype=float)

    base_seed = int(args.seed)

    for k, g in enumerate(gammas):
        vals = []
        for s in range(args.shots):
            noisy, _stats = apply_pauli_jump_after_each_gate(
                ideal, float(g), seed=base_seed + 100000 * k + s
            )
            psi_noisy = Statevector.from_instruction(noisy)
            vals.append(_pure_state_fidelity(psi_ideal, psi_noisy))
        vals = np.asarray(vals, dtype=float)
        meanF[k] = float(vals.mean())
        stdF[k] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if (k + 1) % max(1, args.num // 10) == 0:
            print(f"{k+1}/{args.num} gamma={g:.3e} meanF={meanF[k]:.6f}")

    err = 1.0 - meanF

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path("outputs") / "experiments" / f"error_vs_gamma_log_n{args.n}_seed{args.seed}" / ts
    outdir.mkdir(parents=True, exist_ok=True)

    data = {
        "ideal_qpy": ideal_qpy,
        "n": int(args.n),
        "seed": int(args.seed),
        "shots": int(args.shots),
        "gamma_min_exp": int(args.gamma_min_exp),
        "gamma_max_exp": int(args.gamma_max_exp),
        "num": int(args.num),
        "gammas": gammas.tolist(),
        "mean_state_fidelity": meanF.tolist(),
        "std_state_fidelity": stdF.tolist(),
        "mean_error": err.tolist(),
    }
    (outdir / "results.json").write_text(json.dumps(data, indent=2))

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(gammas, err + 1e-16)
    plt.xlabel("gamma (log scale)")
    plt.ylabel("error = 1 - state fidelity (log scale)")
    plt.title(f"Error vs gamma (state fidelity), n={args.n}, shots={args.shots}")
    fig1 = outdir / "error_vs_gamma_log.png"
    plt.savefig(fig1, dpi=200, bbox_inches="tight")
    plt.close()

    if args.smooth_window >= 3 and args.smooth_window % 2 == 1 and args.smooth_window < len(err):
        w = args.smooth_window
        kernel = np.ones(w, dtype=float) / float(w)
        err_smooth = np.convolve(err, kernel, mode="same")
        plt.figure()
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(gammas, err_smooth + 1e-16)
        plt.xlabel("gamma (log scale)")
        plt.ylabel("smoothed error (log scale)")
        plt.title(f"Smoothed error vs gamma, n={args.n}, window={w}")
        fig2 = outdir / "error_vs_gamma_log_smooth.png"
        plt.savefig(fig2, dpi=200, bbox_inches="tight")
        plt.close()
        print("FIG2:", str(fig2))

    print("OUTPUT DIR:", str(outdir))
    print("FIG1:", str(fig1))


if __name__ == "__main__":
    main()