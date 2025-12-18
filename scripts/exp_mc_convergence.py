import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def _parse_list_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]

def _parse_list_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--gamma-list", type=str, default="1e-3,1e-2,1e-1")
    ap.add_argument("--tmax", type=int, default=10000)
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--batch-sizes", type=str, default="10,20,30,40,50,80,100,150,200,300,500,800,1000,2000,5000,10000")
    ap.add_argument("--n-sites", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.config is not None:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("PyYAML is required for --config") from e
        cfg = yaml.safe_load(Path(args.config).read_text())
        args.gamma_list = ",".join(str(x) for x in cfg.get("gamma_list", _parse_list_floats(args.gamma_list)))
        args.tmax = int(cfg.get("tmax", args.tmax))
        args.n_batches = int(cfg.get("n_batches", args.n_batches))
        args.batch_sizes = ",".join(str(x) for x in cfg.get("batch_sizes", _parse_list_ints(args.batch_sizes)))
        args.n_sites = int(cfg.get("n_sites", args.n_sites))
        args.seed = int(cfg.get("seed", args.seed))

    gammas = _parse_list_floats(args.gamma_list)
    batch_sizes = _parse_list_ints(args.batch_sizes)
    batch_sizes = [m for m in batch_sizes if 1 <= m <= args.tmax]
    if len(batch_sizes) == 0:
        raise ValueError("No valid batch sizes <= tmax")

    outdir = Path("outputs") / "teacher" / f"mc_convergence_sites{args.n_sites}_seed{args.seed}" / datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    results = {"meta": {"tmax": args.tmax, "n_batches": args.n_batches, "batch_sizes": batch_sizes, "n_sites": args.n_sites, "seed": args.seed},
               "by_gamma": {}}

    for g in gammas:
        p0 = (1.0 - g) ** args.n_sites
        traj = (rng.random(args.tmax) < p0).astype(np.float64)
        ref = float(traj.mean())

        errs_mean = []
        errs_std = []

        idx_all = np.arange(args.tmax)
        for m in batch_sizes:
            e = []
            for _ in range(args.n_batches):
                pick = rng.choice(idx_all, size=m, replace=False)
                est = float(traj[pick].mean())
                e.append(abs(est - ref))
            e = np.array(e, dtype=np.float64)
            errs_mean.append(float(e.mean()))
            errs_std.append(float(e.std(ddof=1) if len(e) > 1 else 0.0))

        results["by_gamma"][str(g)] = {"p0_theory": float(p0), "ref_mean": ref, "batch_err_mean": errs_mean, "batch_err_std": errs_std}
        print(f"gamma={g:.3e}  p0={p0:.6f}  ref={ref:.6f}")

    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    xs = np.array(batch_sizes, dtype=np.float64)
    for g in gammas:
        d = results["by_gamma"][str(g)]
        ys = np.array(d["batch_err_mean"], dtype=np.float64)
        plt.plot(xs, ys, label=f"gamma={g:g}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("batch size (# trajectories)")
    plt.ylabel("mean |estimate - reference|")
    plt.title(f"MC convergence (Bernoulli proxy), sites={args.n_sites}, tmax={args.tmax}, batches={args.n_batches}")
    plt.legend()
    figpath = outdir / "mc_convergence.png"
    plt.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"WROTE: {outdir}")
    print(f"FIG : {figpath}")

if __name__ == "__main__":
    main()
