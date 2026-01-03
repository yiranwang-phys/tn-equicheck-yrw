# scripts/2026_1_22_noisy_ising_global_vs_rzz_YAQS.py
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common_shift_tools_2026_1 import (
    build_ising,
    build_observables,
    cache_paths,
    count_shift_targets,
    extract_ops,
    load_pickle,
    make_symlog_deltas,
    mse_vs_ref,
    plot_mse,
    plot_obs_vs_delta,
    save_pickle,
    stable_hash,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--traj", type=int, default=1000)
    ap.add_argument("--traj_ref", type=int, default=10000)
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--delta_points", type=int, default=161)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outroot", type=str, default=r"outputs\2026-1")
    args = ap.parse_args()

    outroot = Path(args.outroot)
    figdir, pkldir = cache_paths(outroot)
    deltas = make_symlog_deltas(delta_max=2 * math.pi, points=args.delta_points, min_abs=1e-4)

    qc = build_ising(args.n, args.depth)
    obs = build_observables(args.n, model="ising")
    base_ops = extract_ops(qc)

    for mode in ("global", "rzz"):
        counts = count_shift_targets(qc, mode)
        print(f"[SANITY] shift gate counts (ising, mode={mode}): {counts} total={sum(counts.values())}")

        meta = dict(
            exp="noisy_ising_global_vs_rzz",
            model="ising",
            mode=mode,
            n=args.n,
            depth=args.depth,
            gamma=args.gamma,
            traj=args.traj,
            traj_ref=args.traj_ref,
            delta_points=int(args.delta_points if args.delta_points % 2 == 1 else args.delta_points + 1),
            seed=args.seed,
            obs_names=obs.names,
        )
        key = stable_hash(meta)
        pkl_path = pkldir / f"noisy_ising_{mode}_{key}.pkl"

        if pkl_path.exists():
            print(f"[CACHE] load {pkl_path.name}")
            pack = load_pickle(pkl_path)
            ref = pack["ref_obs"]
            means = pack["means"]
        else:
            from common_shift_tools_2026_1 import estimate_noisy_means

            ref = estimate_noisy_means(
                n=args.n,
                base_ops=base_ops,
                deltas=np.array([0.0]),
                shift_mode=mode,
                gamma=args.gamma,
                traj=args.traj_ref,
                workers=args.workers,
                seed0=args.seed + 12345,
                prep="zero",
                prep_seed=args.seed,
                obs=obs,
            )[0]

            means = estimate_noisy_means(
                n=args.n,
                base_ops=base_ops,
                deltas=deltas,
                shift_mode=mode,
                gamma=args.gamma,
                traj=args.traj,
                workers=args.workers,
                seed0=args.seed + 777,
                prep="zero",
                prep_seed=args.seed,
                obs=obs,
            )

            save_pickle(pkl_path, dict(meta=meta, deltas=deltas, ref_obs=ref, means=means))
            print(f"[CACHE] saved {pkl_path.name}")

        from common_shift_tools_2026_1 import estimate_noisy_means

        est0 = estimate_noisy_means(
            n=args.n,
            base_ops=base_ops,
            deltas=np.array([0.0]),
            shift_mode=mode,
            gamma=args.gamma,
            traj=args.traj,
            workers=args.workers,
            seed0=args.seed + 999,
            prep="zero",
            prep_seed=args.seed,
            obs=obs,
        )[0]
        floor = float(np.mean((est0 - ref) ** 2))

        mse = mse_vs_ref(means, ref)
        best_delta = float(deltas[int(np.argmin(mse))])

        plot_mse(
            title=f"ising | shift={mode} | noise gamma={args.gamma} | traj={args.traj} ref={args.traj_ref}",
            deltas=deltas,
            mse=mse,
            figpath=figdir / f"noisy_ising_mse_{mode}.png",
            floor=floor,
            best_delta=best_delta,
        )

        # 浣犺鐨勶細鍐嶉€変竴涓?observable锛堟垜榛樿鐢?ZZ_bond_avg锛夊苟鐢?refline
        pick = "ZZ_bond_avg"
        j = obs.names.index(pick)
        plot_obs_vs_delta(
            title=f"ising | {pick} vs delta | shift={mode} | noise gamma={args.gamma}",
            deltas=deltas,
            obs_name=pick,
            obs_vals=means[:, j],
            ref_val=float(ref[j]),
            figpath=figdir / f"noisy_ising_{pick}_{mode}.png",
        )

        print(f"[DONE] ising mode={mode}: best delta={best_delta:.6f}, floor~{floor:.3e}")


if __name__ == "__main__":
    main()
