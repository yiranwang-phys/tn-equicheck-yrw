import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qem_yrw_project.circuits.twolocal import build_twolocal


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _sci(x: float) -> str:
    return f"{x:.0e}".replace("+", "")


def _count_injection_sites(circ) -> int:
    sites = 0
    for ci in circ.data:
        if ci.operation.name != "measure":
            sites += 1
    return sites


@dataclass
class Result:
    num_qubits: int
    depth: int
    seed: int
    gammas: list
    tmax: int
    n_batches: int
    subset_sizes: list
    sites: int
    ref: dict
    mean_abs_error: dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmax", type=int, default=10000)
    ap.add_argument("--gamma-list", type=str, default="1e-3,1e-2,1e-1")
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--subset-sizes", type=str, default="")
    args = ap.parse_args()

    gammas = [float(x) for x in args.gamma_list.split(",") if x.strip()]

    if args.subset_sizes.strip():
        subset_sizes = [int(x) for x in args.subset_sizes.split(",") if x.strip()]
    else:
        subset_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, args.tmax]
    subset_sizes = [m for m in subset_sizes if m <= args.tmax]
    subset_sizes = sorted(list(dict.fromkeys(subset_sizes)))

    rng = np.random.default_rng(args.seed)

    ideal = build_twolocal(num_qubits=args.num_qubits, depth=args.depth, seed=args.seed, add_measurements=False)
    sites = _count_injection_sites(ideal)
    d = 2 ** int(args.num_qubits)

    ref = {}
    mean_abs_error = {}

    for gamma in gammas:
        p0 = (1.0 - float(gamma)) ** float(sites)
        phi = (rng.random(int(args.tmax)) < p0).astype(float)
        favg = (d * phi + 1.0) / (d + 1.0)

        ref_val = float(np.mean(favg))
        ref[_sci(gamma)] = ref_val

        errs = []
        for m in subset_sizes:
            batch_err = []
            for _ in range(int(args.n_batches)):
                idx = rng.choice(int(args.tmax), size=int(m), replace=False)
                est = float(np.mean(favg[idx]))
                batch_err.append(abs(est - ref_val))
            errs.append(float(np.mean(batch_err)))
            print(f"gamma={gamma:.3e}  m={m:5d}  mean_abs_err={errs[-1]:.6g}", flush=True)

        mean_abs_error[_sci(gamma)] = errs

    out_dir = Path("outputs") / "experiments" / "mc_convergence" / f"twolocal_n{args.num_qubits}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    res = Result(
        num_qubits=int(args.num_qubits),
        depth=int(args.depth),
        seed=int(args.seed),
        gammas=[float(g) for g in gammas],
        tmax=int(args.tmax),
        n_batches=int(args.n_batches),
        subset_sizes=[int(m) for m in subset_sizes],
        sites=int(sites),
        ref=ref,
        mean_abs_error=mean_abs_error,
    )
    (out_dir / "results.json").write_text(json.dumps(asdict(res), indent=2))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.array(subset_sizes, dtype=float)
    for gamma in gammas:
        key = _sci(gamma)
        y = np.array(mean_abs_error[key], dtype=float)
        ax.plot(x, y, linewidth=2.0, label=f"gamma={key}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of trajectories (subset size)")
    ax.set_ylabel("mean absolute error to Tmax reference")
    ax.set_title(f"MC convergence (Tmax={args.tmax}, batches={args.n_batches}, sites={sites})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "mc_convergence_loglog.png", dpi=200)
    plt.close(fig)

    print(f"WROTE: {out_dir}", flush=True)
    print(f"FIG : {out_dir / 'mc_convergence_loglog.png'}", flush=True)


if __name__ == "__main__":
    main()
