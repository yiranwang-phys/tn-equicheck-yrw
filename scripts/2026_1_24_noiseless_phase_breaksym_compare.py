# scripts/2026_1_24_noiseless_phase_breaksym_compare.py
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common_shift_tools_2026_1 import (
    build_heisenberg,
    build_ising,
    build_observables,
    cache_paths,
    count_shift_targets,
    extract_ops,
    load_pickle,
    make_symlog_deltas,
    noiseless_fidelity_and_obs,
    plot_fidelity,
    plot_obs_vs_delta,
    save_pickle,
    stable_hash,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--delta_points", type=int, default=401)  # noiseless, cheap
    ap.add_argument("--prep", type=str, default="neel", choices=["zero", "plus", "neel", "random"])
    ap.add_argument("--prep_seed", type=int, default=0)
    ap.add_argument("--outroot", type=str, default=r"outputs\2026-1")
    args = ap.parse_args()

    outroot = Path(args.outroot)
    figdir, pkldir = cache_paths(outroot)
    deltas = make_symlog_deltas(delta_max=2 * math.pi, points=args.delta_points, min_abs=1e-6)

    mode = "global"
    obs_pick = "ZZ_bond_avg"  # 浣犺鈥滀竴涓渶鍚堢悊鐨?observable鈥濓細杩欓噷閫?ZZ 鏈€杩戦偦鐩稿叧

    results = {}

    for model in ("heisenberg", "ising"):
        qc = build_heisenberg(args.n, args.depth) if model == "heisenberg" else build_ising(args.n, args.depth)
        counts = count_shift_targets(qc, mode)
        print(f"[SANITY] shift gate counts ({model}): {counts} total={sum(counts.values())}")

        base_ops = extract_ops(qc)
        obs = build_observables(args.n, model=model)

        meta = dict(
            exp="noiseless_phase_breaksym_compare",
            model=model,
            n=args.n,
            depth=args.depth,
            mode=mode,
            delta_points=int(args.delta_points if args.delta_points % 2 == 1 else args.delta_points + 1),
            prep=args.prep,
            prep_seed=args.prep_seed,
            obs_pick=obs_pick,
        )
        key = stable_hash(meta)
        pkl_path = pkldir / f"noiseless_phase_{model}_{key}.pkl"

        if pkl_path.exists():
            print(f"[CACHE] load {pkl_path.name}")
            pack = load_pickle(pkl_path)
            fid = pack["fidelity"]
            obs_arr = pack["obs_arr"]
        else:
            fid, obs_arr = noiseless_fidelity_and_obs(
                n=args.n,
                base_ops=base_ops,
                deltas=deltas,
                shift_mode=mode,
                prep=args.prep,          # <-- symmetry-breaking here
                prep_seed=args.prep_seed,
                obs=obs,
                obs_pick=obs_pick,
            )
            save_pickle(pkl_path, dict(meta=meta, deltas=deltas, fidelity=fid, obs_arr=obs_arr))
            print(f"[CACHE] saved {pkl_path.name}")

        results[model] = (fid, obs_arr)

        plot_fidelity(
            title=f"{model} | global shift | fidelity (noiseless) | prep={args.prep}",
            deltas=deltas,
            fid=fid,
            figpath=figdir / f"noiseless_fidelity_{model}.png",
        )

        plot_obs_vs_delta(
            title=f"{model} | {obs_pick} vs delta (noiseless) | global shift | prep={args.prep}",
            deltas=deltas,
            obs_name=obs_pick,
            obs_vals=obs_arr,
            ref_val=float(obs_arr[np.where(deltas == 0.0)[0][0]]),
            figpath=figdir / f"noiseless_obs_{obs_pick}_{model}.png",
        )

    # combined compare plot (fidelity)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.gca()
    for model in ("heisenberg", "ising"):
        fid, _ = results[model]
        ax.plot(deltas, fid, label=f"{model} fidelity(未)")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("delta")
    ax.set_ylabel("fidelity (0..1)")
    ax.set_title(f"Noiseless phase response | global shift | prep={args.prep}")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "noiseless_phase_fidelity_compare_models.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
