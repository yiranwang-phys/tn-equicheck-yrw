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

def _parse_gamma_list(s):
    xs = []
    for p in s.split(","):
        p = p.strip()
        if p:
            xs.append(float(p))
    return xs

def _batch_sizes(tmax):
    a = list(range(10, min(110, tmax + 1), 10))
    b = []
    for k in (200, 300, 500, 800, 1000, 2000, 5000, 10000):
        if k <= tmax and k not in a:
            b.append(k)
    c = [tmax] if tmax not in a and tmax not in b else []
    return a + b + c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=int, default=10000)
    ap.add_argument("--gamma-list", type=str, default="1e-3,1e-2,1e-1")
    ap.add_argument("--n-batches", type=int, default=50)
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
    gammas = _parse_gamma_list(args.gamma_list)
    sizes = _batch_sizes(args.tmax)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = root / "outputs" / "experiments" / f"mc_convergence_n{n}_seed{args.seed}" / ts
    outdir.mkdir(parents=True, exist_ok=True)

    results = {"n": n, "d": d, "n_sites": n_sites, "tmax": args.tmax, "n_batches": args.n_batches, "seed": args.seed, "gammas": gammas, "sizes": sizes, "curves": {}}

    plt.figure()

    for gi, g in enumerate(gammas):
        vals = np.zeros(args.tmax, dtype=float)
        for t in range(args.tmax):
            shot_seed = args.seed * 1000003 + gi * 104729 + t
            _, st = apply_pauli_jump_after_each_gate(ideal, float(g), seed=int(shot_seed), include_measurements=False)
            vals[t] = 1.0 if st.n_noise_ops == 0 else (1.0 / (d + 1.0))
        ref = float(np.mean(vals))

        mean_err = []
        std_err = []
        rng = np.random.default_rng(args.seed + gi * 99991)

        for k in sizes:
            errs = []
            for _ in range(args.n_batches):
                idx = rng.choice(args.tmax, size=int(k), replace=False)
                m = float(np.mean(vals[idx]))
                errs.append(abs(m - ref))
            mean_err.append(float(np.mean(errs)))
            std_err.append(float(np.std(errs)))

        results["curves"][str(g)] = {"ref": ref, "mean_abs_err": mean_err, "std_abs_err": std_err}
        plt.loglog(sizes, mean_err)

        print(f"gamma={g:.3e} ref={ref:.6f} done", flush=True)

    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    plt.xlabel("num_trajectories")
    plt.ylabel("mean |batch_mean - ref_mean|")
    plt.title(f"MC convergence (proxy fidelity), n={n}, sites={n_sites}, tmax={args.tmax}")
    plt.tight_layout()
    plt.savefig(outdir / "mc_convergence.png", dpi=200)
    plt.close()

    print(f"WROTE: {outdir}", flush=True)
    print(f"FIG : {outdir/'mc_convergence.png'}", flush=True)

if __name__ == "__main__":
    main()
