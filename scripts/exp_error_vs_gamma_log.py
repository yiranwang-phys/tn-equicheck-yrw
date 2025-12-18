import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate

def _count_sites(circ):
    n = 0
    for inst, qargs, _ in circ.data:
        name = inst.name.lower()
        if name in ("barrier", "measure", "reset"):
            continue
        n += len(qargs)
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-min-exp", type=int, default=-3)
    ap.add_argument("--gamma-max-exp", type=int, default=-1)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    qpy_path = root / "outputs" / "ideal" / "twolocal_n6_seed0" / "circuit_ideal.qpy"
    if not qpy_path.exists():
        raise FileNotFoundError(f"Missing ideal QPY: {qpy_path}. Run make_ideal_twolocal_n6.py first.")

    with open(qpy_path, "rb") as f:
        ideal = qpy.load(f)[0]

    n = ideal.num_qubits
    d = 2 ** n
    n_sites = _count_sites(ideal)

    gammas = np.logspace(args.gamma_min_exp, args.gamma_max_exp, args.num)
    meanF = np.zeros_like(gammas, dtype=float)
    meanErr = np.zeros_like(gammas, dtype=float)

    for i, g in enumerate(gammas):
        vals = []
        for s in range(args.shots):
            shot_seed = args.seed * 1000003 + i * 7919 + s
            _, st = apply_pauli_jump_after_each_gate(ideal, float(g), seed=int(shot_seed), include_measurements=False)
            v = 1.0 if st.n_noise_ops == 0 else (1.0 / (d + 1.0))
            vals.append(v)
        m = float(np.mean(vals))
        meanF[i] = m
        meanErr[i] = 1.0 - m

        if (i + 1) % max(1, args.num // 10) == 0:
            print(f"{i+1}/{args.num} gamma={g:.3e} meanF={m:.6f} err={1.0-m:.6f}", flush=True)

    p0 = (1.0 - gammas) ** n_sites
    theoryF = (1.0 / (d + 1.0)) + (d / (d + 1.0)) * p0
    theoryErr = 1.0 - theoryF

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = root / "outputs" / "experiments" / f"error_vs_gamma_log_n{n}_seed{args.seed}" / ts
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "results.json", "w") as f:
        json.dump(
            {
                "n": n,
                "d": d,
                "n_sites": n_sites,
                "gamma_min_exp": args.gamma_min_exp,
                "gamma_max_exp": args.gamma_max_exp,
                "num": args.num,
                "shots": args.shots,
                "seed": args.seed,
                "gammas": gammas.tolist(),
                "mean_fidelity": meanF.tolist(),
                "mean_error": meanErr.tolist(),
                "theory_error": theoryErr.tolist(),
            },
            f,
            indent=2,
        )

    plt.figure()
    plt.loglog(gammas, meanErr)
    plt.loglog(gammas, theoryErr, linestyle="--")
    plt.xlabel("gamma")
    plt.ylabel("error = 1 - fidelity_proxy")
    plt.title(f"Error vs gamma (log-log), n={n}, sites={n_sites}, shots={args.shots}")
    plt.tight_layout()
    plt.savefig(outdir / "error_vs_gamma_loglog.png", dpi=200)
    plt.close()

    print(f"WROTE: {outdir}", flush=True)
    print(f"FIG : {outdir/'error_vs_gamma_loglog.png'}", flush=True)

if __name__ == "__main__":
    main()
