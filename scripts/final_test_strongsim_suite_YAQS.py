# scripts/final_test_strongsim_suite_YAQS.py
from __future__ import annotations

# ==========================================================
# Windows spawn safety: set non-GUI matplotlib backend early
# ==========================================================
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile

# -----------------------------
# Repo + src import (src-layout)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from qem_yrw_project.circuits.twolocal import build_twolocal
from qem_yrw_project.noise import depolarizing_xyz_processes, yaqs_example_processes, apply_processes_after_each_gate
from qem_yrw_project.sim import (
    ts,
    strip_measurements,
    qpy_bytes,
    qpy_load_bytes,
    fidelity_pure,
    worker_init,
    shot_seed,
    run_strongsim_statevector,
)


# -----------------------------
# Angle shifting (family-specific)
# -----------------------------
def shifted_circuit_family(circ: QuantumCircuit, delta: float, family: str) -> QuantumCircuit:
    fam = family.strip().lower()
    delta = float(delta)

    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    out.global_phase = getattr(circ, "global_phase", 0.0)

    for inst, qargs, cargs in circ.data:
        name = inst.name.lower()

        # map bits by index (robust)
        nq = [out.qubits[circ.find_bit(q).index] for q in qargs]
        nc = [out.clbits[circ.find_bit(c).index] for c in cargs]

        if name in ("measure", "barrier", "reset"):
            out.append(inst, nq, nc)
            continue

        if fam == "rzz" and name == "rzz" and len(inst.params) == 1:
            theta = float(inst.params[0]) + delta
            out.rzz(theta, nq[0], nq[1])
        elif fam == "rx" and name == "rx" and len(inst.params) == 1:
            theta = float(inst.params[0]) + delta
            out.rx(theta, nq[0])
        else:
            out.append(inst, nq, nc)

    return out


# -----------------------------
# MC worker: chunk fidelities
# -----------------------------
def _chunk_fidelities(
    circ_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    processes: List[dict],
    start: int,
    shots: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
) -> np.ndarray:
    worker_init(int(blas_threads))

    base = strip_measurements(qpy_load_bytes(circ_qpy))
    out = np.empty(int(shots), dtype=np.float64)

    for i in range(int(shots)):
        global_idx = int(start + i)
        s = shot_seed(int(seed_base), global_idx)

        # sample a noisy circuit for this trajectory (explicit Pauli gates after each gate)
        noisy, _stats = apply_processes_after_each_gate(
            base,
            processes,
            seed=int(s),
            restrict_to_touched_sites=True,
        )
        noisy = strip_measurements(noisy)

        vec = run_strongsim_statevector(
            noisy,
            n=int(n),
            max_bond_dim=int(max_bond_dim),
            threshold=float(threshold),
            noise_model=None,
        )
        out[i] = fidelity_pure(psi_ideal, vec)

    return out


def mc_fidelities_parallel(
    ex,
    circ_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    processes: List[dict],
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
) -> np.ndarray:
    traj = int(traj)
    chunk = int(max(1, chunk))
    n_chunks = (traj + chunk - 1) // chunk

    futs = []
    for k in range(n_chunks):
        start = k * chunk
        shots = min(chunk, traj - start)
        futs.append(
            ex.submit(
                _chunk_fidelities,
                circ_qpy, n, psi_ideal, processes,
                int(start), int(shots), int(seed_base),
                int(blas_threads), int(max_bond_dim), float(threshold),
            )
        )

    arrs = [f.result() for f in futs]
    return np.concatenate(arrs, axis=0) if arrs else np.empty((0,), dtype=float)


def mc_mean_parallel(*args, **kwargs) -> float:
    fids = mc_fidelities_parallel(*args, **kwargs)
    return float(np.mean(fids)) if fids.size else float("nan")


# -----------------------------
# Tasks
# -----------------------------
def task_angle_shift(
    ex,
    base: QuantumCircuit,
    ideal_vec: np.ndarray,
    *,
    n: int,
    gamma: float,
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    delta_min: float,
    delta_max: float,
    delta_num: int,
    families: List[str],
    noise_kind: str,
    out_dir: Path,
) -> Dict[str, object]:
    base_qpy = qpy_bytes(base)

    # pick process list (our “probability per gate-step” sampling model)
    if noise_kind == "yaqs_example":
        processes = yaqs_example_processes(n, gamma)
    else:
        processes = depolarizing_xyz_processes(n, gamma)

    deltas = np.logspace(np.log10(float(delta_min)), np.log10(float(delta_max)), int(delta_num))
    results: Dict[str, object] = {"deltas": deltas.tolist(), "families": {}}

    fid0 = mc_mean_parallel(
        ex,
        base_qpy,
        n,
        ideal_vec,
        processes,
        traj=int(traj),
        chunk=int(chunk),
        seed_base=int(seed_base),
        blas_threads=int(blas_threads),
        max_bond_dim=int(max_bond_dim),
        threshold=float(threshold),
    )
    results["baseline"] = {"delta": 0.0, "mean_fidelity": float(fid0)}

    fam_curves: Dict[str, Dict[str, object]] = {}
    for fam in families:
        curve = np.empty_like(deltas, dtype=np.float64)
        for i, dlt in enumerate(deltas):
            shifted = shifted_circuit_family(base, float(dlt), fam)
            shifted_qpy = qpy_bytes(shifted)

            curve[i] = mc_mean_parallel(
                ex,
                shifted_qpy,
                n,
                ideal_vec,
                processes,
                traj=int(traj),
                chunk=int(chunk),
                seed_base=int(seed_base),  # paired across deltas & families
                blas_threads=int(blas_threads),
                max_bond_dim=int(max_bond_dim),
                threshold=float(threshold),
            )

        best_idx = int(np.argmax(curve))
        fam_curves[fam] = {
            "mean_fidelity": curve.tolist(),
            "best_delta": float(deltas[best_idx]),
            "best_fidelity": float(curve[best_idx]),
            "delta0_baseline": float(fid0),
            "deltaF_best_minus_base": float(curve[best_idx] - fid0),
        }

        # individual plot
        plt.figure()
        plt.plot(deltas, curve, marker="o", markersize=3, label=f"shift only {fam.upper()}")
        plt.axhline(float(fid0), linestyle="--", label="delta=0 baseline")
        plt.xscale("log")
        plt.xlabel("angle shift delta (log)")
        plt.ylabel("mean state fidelity")
        txt = (
            f"baseline (δ=0): F={fid0:.6f}\n"
            f"best: δ={deltas[best_idx]:.3e}, F={curve[best_idx]:.6f}\n"
            f"ΔF(best-base)={curve[best_idx]-fid0:+.3e}"
        )
        plt.gca().text(
            0.02, 0.02, txt,
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.25),
            va="bottom",
        )
        plt.title(f"Angle-shift scan: only {fam.upper()} (gamma={gamma:g}, paired, {noise_kind})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"angle_shift_only_{fam}_logx.png", dpi=200, bbox_inches="tight")
        plt.close()

    results["families"] = fam_curves

    # combined plot
    plt.figure()
    for fam in families:
        curve = np.asarray(fam_curves[fam]["mean_fidelity"], dtype=float)
        plt.plot(deltas, curve, marker="o", markersize=3, label=f"shift only {fam.upper()}")
    plt.axhline(float(fid0), linestyle="--", label="delta=0 baseline")
    plt.xscale("log")
    plt.xlabel("angle shift delta (log)")
    plt.ylabel("mean state fidelity")
    plt.title(f"Angle-shift families: {', '.join([f.upper() for f in families])} (gamma={gamma:g}, paired, {noise_kind})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "angle_shift_families_logx.png", dpi=200, bbox_inches="tight")
    plt.close()

    return results


def task_error_vs_gamma(
    ex,
    base: QuantumCircuit,
    ideal_vec: np.ndarray,
    *,
    n: int,
    gammas: np.ndarray,
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    noise_kind: str,
    out_dir: Path,
) -> Dict[str, object]:
    base_qpy = qpy_bytes(base)

    meanF = np.zeros_like(gammas, dtype=float)
    err = np.zeros_like(gammas, dtype=float)

    for i, g in enumerate(gammas):
        if noise_kind == "yaqs_example":
            processes = yaqs_example_processes(n, float(g))
        else:
            processes = depolarizing_xyz_processes(n, float(g))

        meanF[i] = mc_mean_parallel(
            ex,
            base_qpy,
            n,
            ideal_vec,
            processes,
            traj=int(traj),
            chunk=int(chunk),
            seed_base=int(seed_base),  # paired across gamma points
            blas_threads=int(blas_threads),
            max_bond_dim=int(max_bond_dim),
            threshold=float(threshold),
        )
        err[i] = 1.0 - meanF[i]

        if (i + 1) % max(1, len(gammas) // 10) == 0:
            print(f"[gamma] {i+1}/{len(gammas)}  gamma={g:.3e}  meanF={meanF[i]:.6f}  err={err[i]:.6f}", flush=True)

    # plots
    plt.figure()
    plt.plot(gammas, err, marker="o", markersize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("gamma (log)")
    plt.ylabel("error = 1 - mean fidelity (log)")
    plt.title(f"Error vs gamma (log-log), traj={traj}, {noise_kind}")
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_gamma_loglog.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(gammas, meanF, marker="o", markersize=3)
    plt.xscale("log")
    plt.xlabel("gamma (log)")
    plt.ylabel("mean fidelity")
    plt.title(f"Mean fidelity vs gamma, traj={traj}, {noise_kind}")
    plt.tight_layout()
    plt.savefig(out_dir / "meanF_vs_gamma_logx.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "gammas": gammas.tolist(),
        "mean_fidelity": meanF.tolist(),
        "error_1_minus_fidelity": err.tolist(),
    }


def task_mc_convergence(
    ex,
    base: QuantumCircuit,
    ideal_vec: np.ndarray,
    *,
    n: int,
    gamma: float,
    traj_ref: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    batch_sizes: List[int],
    noise_kind: str,
    out_dir: Path,
) -> Dict[str, object]:
    base_qpy = qpy_bytes(base)

    if noise_kind == "yaqs_example":
        processes = yaqs_example_processes(n, gamma)
    else:
        processes = depolarizing_xyz_processes(n, gamma)

    # 1) generate a large reference set of fidelities
    fids = mc_fidelities_parallel(
        ex,
        base_qpy,
        n,
        ideal_vec,
        processes,
        traj=int(traj_ref),
        chunk=int(chunk),
        seed_base=int(seed_base),
        blas_threads=int(blas_threads),
        max_bond_dim=int(max_bond_dim),
        threshold=float(threshold),
    )
    ref_mean = float(np.mean(fids))

    # 2) for each batch size m: split into disjoint batches of size m and average |batch_mean - ref_mean|
    xs = []
    ys = []
    ystd = []
    for m in batch_sizes:
        m = int(m)
        if m <= 0:
            continue
        nb = int(len(fids) // m)
        if nb <= 0:
            continue
        batches = fids[: nb * m].reshape(nb, m)
        means = np.mean(batches, axis=1)
        abs_err = np.abs(means - ref_mean)

        xs.append(m)
        ys.append(float(np.mean(abs_err)))
        ystd.append(float(np.std(means)))

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ystd = np.asarray(ystd, dtype=float)

    # plots
    plt.figure()
    plt.plot(xs, ys, marker="o", markersize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("batch size m (log)")
    plt.ylabel("avg |mean(F)_batch - mean(F)_ref| (log)")
    plt.title(f"MC convergence (ref={traj_ref} traj), gamma={gamma:g}, {noise_kind}")
    plt.tight_layout()
    plt.savefig(out_dir / "mc_convergence_loglog.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(xs, ystd, marker="o", markersize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("batch size m (log)")
    plt.ylabel("std of batch means (log)")
    plt.title(f"Std(batch means) vs m, gamma={gamma:g}, {noise_kind}")
    plt.tight_layout()
    plt.savefig(out_dir / "mc_batchmean_std_loglog.png", dpi=200, bbox_inches="tight")
    plt.close()

    # also save raw fids (so you can reuse without recompute)
    np.save(out_dir / "fidelities_ref.npy", fids)

    return {
        "ref_traj": int(traj_ref),
        "ref_mean_fidelity": ref_mean,
        "batch_sizes": xs.tolist(),
        "avg_abs_error_vs_ref": ys.tolist(),
        "std_batch_means": ystd.tolist(),
        "saved": ["fidelities_ref.npy", "mc_convergence_loglog.png", "mc_batchmean_std_loglog.png"],
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, default="all", choices=["all", "mc", "gamma", "shift"])

    ap.add_argument("--num-qubits", type=int, default=10)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # compilation
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    # YAQS strongsim controls
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)

    # parallel
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--blas-threads", type=int, default=1)

    # noise model choice for sampling
    ap.add_argument("--noise-kind", type=str, default="depolarizing_xyz", choices=["depolarizing_xyz", "yaqs_example"])

    # base gamma + traj for single-point tasks (mc, shift)
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=50)

    # gamma sweep
    ap.add_argument("--gamma-min-exp", type=int, default=-3)
    ap.add_argument("--gamma-max-exp", type=int, default=-1)
    ap.add_argument("--gamma-num", type=int, default=60)

    # angle shift
    ap.add_argument("--delta-min", type=float, default=1e-6)
    ap.add_argument("--delta-max", type=float, default=2 * np.pi)
    ap.add_argument("--delta-num", type=int, default=60)
    ap.add_argument("--families", type=str, default="rzz,rx")

    # MC convergence
    ap.add_argument("--traj-ref", type=int, default=10000)
    ap.add_argument("--batch-sizes", type=str, default="10,20,30,40,50,60,80,100,150,200,300,400,600,800,1000")

    args = ap.parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    # prepare output dir
    n = int(args.num_qubits)
    out_dir = (
        REPO_ROOT
        / "outputs"
        / "experiments"
        / "final_test_strongsim_suite"
        / f"n{n}_d{int(args.depth)}_seed{int(args.seed)}"
        / ts()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # build base circuit
    base = build_twolocal(num_qubits=n, depth=int(args.depth), seed=int(args.seed), add_measurements=False)
    base = strip_measurements(base)

    if int(args.use_transpile) == 1:
        base = transpile(
            base,
            optimization_level=int(args.opt_level),
            seed_transpiler=int(args.seed_transpiler),
        )
        base = strip_measurements(base)

    # compute ideal statevector once
    ideal_vec = run_strongsim_statevector(
        base,
        n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
        noise_model=None,
    )

    # paired seeds across the entire run directory
    seed_base = int(args.seed + 20251220)

    meta = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "workers": int(workers),
        "blas_threads": int(args.blas_threads),
        "n": n,
        "depth": int(args.depth),
        "seed": int(args.seed),
        "use_transpile": int(args.use_transpile),
        "opt_level": int(args.opt_level),
        "seed_transpiler": int(args.seed_transpiler),
        "yaqs": {"max_bond_dim": int(args.max_bond_dim), "threshold": float(args.threshold)},
        "noise_kind": str(args.noise_kind),
        "seed_base_paired": int(seed_base),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    t0 = time.time()
    results: Dict[str, object] = {"meta": meta}

    families = [x.strip().lower() for x in str(args.families).split(",") if x.strip()]
    if not families:
        families = ["rzz", "rx"]

    batch_sizes = [int(x.strip()) for x in str(args.batch_sizes).split(",") if x.strip()]

    with ProcessPoolExecutor(
        max_workers=min(int(workers), 256),
        initializer=worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        if args.task in ("all", "gamma"):
            gammas = np.logspace(int(args.gamma_min_exp), int(args.gamma_max_exp), int(args.gamma_num))
            results["error_vs_gamma"] = task_error_vs_gamma(
                ex,
                base,
                ideal_vec,
                n=n,
                gammas=gammas,
                traj=int(args.traj),
                chunk=int(args.chunk),
                seed_base=seed_base,
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                noise_kind=str(args.noise_kind),
                out_dir=out_dir,
            )

        if args.task in ("all", "shift"):
            results["angle_shift"] = task_angle_shift(
                ex,
                base,
                ideal_vec,
                n=n,
                gamma=float(args.gamma),
                traj=int(args.traj),
                chunk=int(args.chunk),
                seed_base=seed_base,
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                delta_min=float(args.delta_min),
                delta_max=float(args.delta_max),
                delta_num=int(args.delta_num),
                families=families,
                noise_kind=str(args.noise_kind),
                out_dir=out_dir,
            )

        if args.task in ("all", "mc"):
            results["mc_convergence"] = task_mc_convergence(
                ex,
                base,
                ideal_vec,
                n=n,
                gamma=float(args.gamma),
                traj_ref=int(args.traj_ref),
                chunk=int(args.chunk),
                seed_base=seed_base,
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                batch_sizes=batch_sizes,
                noise_kind=str(args.noise_kind),
                out_dir=out_dir,
            )

    elapsed = time.time() - t0
    results["elapsed_sec"] = float(elapsed)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"task={args.task}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        f"noise_kind={args.noise_kind}",
        "",
        "Saved: meta.json, results.json",
        "If task includes gamma:  error_vs_gamma_loglog.png, meanF_vs_gamma_logx.png",
        "If task includes shift:  angle_shift_families_logx.png, angle_shift_only_*.png",
        "If task includes mc:     fidelities_ref.npy, mc_convergence_loglog.png, mc_batchmean_std_loglog.png",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    # macOS: force spawn for safety
    if platform.system() == "Darwin":
        mp.set_start_method("spawn", force=True)

    mp.freeze_support()
    main()
