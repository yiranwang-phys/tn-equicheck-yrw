# scripts/2026_1_21_noisy_compare_global_YAQS.py
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

    for model in ("heisenberg", "ising"):
        qc = build_heisenberg(args.n, args.depth) if model == "heisenberg" else build_ising(args.n, args.depth)
        mode = "global"

        counts = count_shift_targets(qc, mode)
        print(f"[SANITY] shift gate counts ({model}): {counts} total={sum(counts.values())}")

        base_ops = extract_ops(qc)
        obs = build_observables(args.n, model=model)  # E_bond, Mz, ZZ/XX/YY

        meta = dict(
            exp="noisy_compare_global",
            model=model,
            n=args.n,
            depth=args.depth,
            gamma=args.gamma,
            traj=args.traj,
            traj_ref=args.traj_ref,
            delta_points=int(args.delta_points if args.delta_points % 2 == 1 else args.delta_points + 1),
            mode=mode,
            seed=args.seed,
            obs_names=obs.names,
        )
        key = stable_hash(meta)
        pkl_path = pkldir / f"noisy_global_{model}_{key}.pkl"

        if pkl_path.exists():
            print(f"[CACHE] load {pkl_path.name}")
            pack = load_pickle(pkl_path)
            ref = pack["ref_obs"]
            means = pack["means"]
        else:
            # --- ref: noisy delta=0 with traj_ref ---
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

            # --- sweep means: noisy with traj ---
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

            save_pickle(
                pkl_path,
                dict(meta=meta, deltas=deltas, ref_obs=ref, means=means),
            )
            print(f"[CACHE] saved {pkl_path.name}")

        # --- MC floor: run delta=0 with traj (same estimator) vs ref(traj_ref) ---
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
        best_idx = int(np.argmin(mse))
        best_delta = float(deltas[best_idx])

        plot_mse(
            title=f"{model} | global shift | noise pauli_xyz gamma={args.gamma} | traj={args.traj} ref={args.traj_ref}",
            deltas=deltas,
            mse=mse,
            figpath=figdir / f"noisy_global_mse_{model}.png",
            floor=floor,
            best_delta=best_delta,
        )

        # Per-observable curves with ref line
        for j, name in enumerate(obs.names):
            plot_obs_vs_delta(
                title=f"{model} | {name} vs delta | noise gamma={args.gamma} | mode=global",
                deltas=deltas,
                obs_name=name,
                obs_vals=means[:, j],
                ref_val=float(ref[j]),
                figpath=figdir / f"noisy_global_{model}_{name}.png",
            )

        print(f"[DONE] {model}: best delta={best_delta:.6f}, floor~{floor:.3e}")

    # extra: compare the two models on a single figure (MSE)
    # (only makes sense because both are "global")
    import matplotlib.pyplot as plt
    from common_shift_tools_2026_1 import load_pickle

    # reload from cache
    heis = sorted(pkldir.glob("noisy_global_heisenberg_*.pkl"))[-1]
    ising = sorted(pkldir.glob("noisy_global_ising_*.pkl"))[-1]
    A = load_pickle(heis)
    B = load_pickle(ising)
    deltas = A["deltas"]
    mse_h = mse_vs_ref(A["means"], A["ref_obs"])
    mse_i = mse_vs_ref(B["means"], B["ref_obs"])

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(deltas, mse_h, label="heisenberg MSE vs noisy ref")
    ax.plot(deltas, mse_i, label="ising MSE vs noisy ref")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_yscale("log")
    ax.set_xlabel("delta")
    ax.set_ylabel("MSE loss vs noisy ref (未=0)")
    ax.set_title(f"Global shift sweep | same noise gamma={args.gamma} | traj={args.traj} ref={args.traj_ref}")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "noisy_global_mse_compare_models.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
