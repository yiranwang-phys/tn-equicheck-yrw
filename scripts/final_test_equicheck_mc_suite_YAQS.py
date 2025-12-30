# scripts/final_test_equicheck_mc_suite_YAQS.py
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Operator


# -----------------------------
# Repo + src import (src-layout)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Project circuit builder (fallback if missing)
# -----------------------------
try:
    from qem_yrw_project.circuits.twolocal import build_twolocal  # type: ignore
except Exception:
    def build_twolocal(num_qubits: int, depth: int, seed: int, add_measurements: bool = False) -> QuantumCircuit:
        rng = np.random.default_rng(seed)
        qc = QuantumCircuit(num_qubits)
        for _ in range(depth):
            for i in range(num_qubits):
                qc.rx(float(rng.uniform(0, 2 * np.pi)), i)
            for i in range(num_qubits - 1):
                qc.rzz(float(rng.uniform(0, 2 * np.pi)), i, i + 1)
        if add_measurements:
            qc.measure_all()
        return qc


# -----------------------------
# Prefer your src noise implementation (StrongSim-friendly)
# -----------------------------
try:
    from qem_yrw_project.noise import apply_pauli_jump_after_each_gate  # type: ignore
except Exception:
    def apply_pauli_jump_after_each_gate(circ: QuantumCircuit, gamma: float, seed: int):
        rng = np.random.default_rng(int(seed))
        out = QuantumCircuit(circ.num_qubits)
        for ci in circ.data:
            op = ci.operation
            if op.name in ("measure", "barrier", "reset"):
                continue
            out.append(op, list(ci.qubits), list(ci.clbits))
            for q in list(ci.qubits):
                if rng.random() < gamma:
                    r = int(rng.integers(0, 3))
                    if r == 0:
                        out.x(q)
                    elif r == 1:
                        out.y(q)
                    else:
                        out.z(q)
        return out, None


# -----------------------------
# Helpers
# -----------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _set_thread_env(blas_threads: int) -> None:
    n = str(int(max(1, blas_threads)))
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[k] = n

def _try_set_high_priority_windows() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        HIGH_PRIORITY_CLASS = 0x00000080
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, HIGH_PRIORITY_CLASS)
    except Exception:
        pass

def _worker_init(blas_threads: int) -> None:
    _set_thread_env(blas_threads)
    _try_set_high_priority_windows()

def strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    return circ.remove_final_measurements(inplace=False)

def qpy_bytes(circ: QuantumCircuit) -> bytes:
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    return buf.getvalue()

def qpy_load_bytes(b: bytes) -> QuantumCircuit:
    buf = io.BytesIO(b)
    return qpy.load(buf)[0]

def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * shot_index) % (2**31 - 1))


# -----------------------------
# Equivalence metrics
# -----------------------------
def trace_fidelity_unitaries(ideal: QuantumCircuit, noisy: QuantumCircuit) -> float:
    """
    Circuit-level metric (global-phase insensitive):
      F_tr = |Tr(U * U'^†)| / 2^n
    """
    u_ideal = Operator(ideal).data
    u_noisy = Operator(noisy).data
    d = u_ideal.shape[0]
    val = np.trace(u_ideal @ u_noisy.conj().T) / d
    return float(np.abs(val))

def yaqs_equivcheck_bool(ideal: QuantumCircuit, noisy: QuantumCircuit) -> Optional[bool]:
    """
    Try calling mqt.yaqs.digital.equivalence_checker.run.
    We only guarantee a boolean if the API returns one.
    """
    try:
        from mqt.yaqs.digital.equivalence_checker import run as yaqs_run
    except Exception:
        return None

    sig = inspect.signature(yaqs_run)
    kwargs = {}
    # pass only supported kwargs (keep minimal)
    for k in ["parallel", "max_bond_dim", "threshold", "eps", "epsilon", "tol", "tolerance"]:
        if k in sig.parameters:
            if k == "parallel":
                kwargs[k] = False
            elif k == "max_bond_dim":
                kwargs[k] = 64
            elif k == "threshold":
                kwargs[k] = 1e-10
            else:
                kwargs[k] = 1e-6

    try:
        res = yaqs_run(ideal, noisy, **kwargs)
    except TypeError:
        res = yaqs_run(ideal, noisy)

    if isinstance(res, bool):
        return res
    if isinstance(res, dict):
        for key in ["equivalent", "is_equivalent", "result"]:
            if key in res and isinstance(res[key], bool):
                return res[key]
    if isinstance(res, (tuple, list)):
        for x in res:
            if isinstance(x, bool):
                return x
    return None


# -----------------------------
# MC worker (compute trace fidelity + yaqs bool if available)
# -----------------------------
def _chunk_equiv_metrics(
    circ_qpy: bytes,
    n: int,
    gamma: float,
    start: int,
    shots: int,
    seed_base: int,
    blas_threads: int,
    do_yaqs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    _worker_init(int(blas_threads))

    ideal = strip_measurements(qpy_load_bytes(circ_qpy))

    ftr = np.empty(int(shots), dtype=np.float64)
    eqb = np.empty(int(shots), dtype=np.float64)  # 1/0 or nan

    for i in range(int(shots)):
        global_idx = int(start + i)
        s = shot_seed(int(seed_base), global_idx)

        noisy, _ = apply_pauli_jump_after_each_gate(ideal, float(gamma), seed=int(s))
        noisy = strip_measurements(noisy)

        ftr[i] = trace_fidelity_unitaries(ideal, noisy)

        if int(do_yaqs) == 1:
            b = yaqs_equivcheck_bool(ideal, noisy)
            eqb[i] = np.nan if b is None else (1.0 if b else 0.0)
        else:
            eqb[i] = np.nan

    return ftr, eqb


def mc_mean_equiv_parallel(
    ex,
    circ_qpy: bytes,
    n: int,
    gamma: float,
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    do_yaqs: int,
) -> Dict[str, float]:
    traj = int(traj)
    chunk = int(max(1, chunk))
    n_chunks = (traj + chunk - 1) // chunk

    futs = []
    for k in range(n_chunks):
        start = k * chunk
        shots = min(chunk, traj - start)
        futs.append(
            ex.submit(
                _chunk_equiv_metrics,
                circ_qpy, n, float(gamma),
                int(start), int(shots), int(seed_base),
                int(blas_threads), int(do_yaqs),
            )
        )

    all_f = []
    all_b = []
    for f in futs:
        ff, bb = f.result()
        all_f.append(ff)
        all_b.append(bb)

    F = np.concatenate(all_f) if all_f else np.array([], dtype=float)
    B = np.concatenate(all_b) if all_b else np.array([], dtype=float)

    out = {
        "mean_trace_fidelity": float(np.mean(F)) if F.size else float("nan"),
        "std_trace_fidelity": float(np.std(F)) if F.size else float("nan"),
        "traj": float(traj),
    }
    if np.isfinite(B).any():
        good = B[np.isfinite(B)]
        out["yaqs_pass_rate"] = float(np.mean(good)) if good.size else float("nan")
    else:
        out["yaqs_pass_rate"] = float("nan")
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, default="mc", choices=["mc", "gamma", "all"])
    ap.add_argument("--num-qubits", type=int, default=5)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=50)

    # reference for mc convergence
    ap.add_argument("--traj-ref", type=int, default=10000)
    ap.add_argument("--batch-list", type=str, default="10,20,30,40,50,80,100,150,200,300,500,800,1000")

    # gamma sweep
    ap.add_argument("--gamma-min-exp", type=int, default=-3)
    ap.add_argument("--gamma-max-exp", type=int, default=-1)
    ap.add_argument("--gamma-num", type=int, default=40)

    # parallel
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--blas-threads", type=int, default=1)

    # compilation
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    # YAQS equicheck (bool only, if available)
    ap.add_argument("--do-yaqs-equicheck", type=int, default=1)

    args = ap.parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    n = int(args.num_qubits)

    base = build_twolocal(num_qubits=n, depth=int(args.depth), seed=int(args.seed), add_measurements=False)
    base = strip_measurements(base)

    if int(args.use_transpile) == 1:
        base = transpile(
            base,
            optimization_level=int(args.opt_level),
            seed_transpiler=int(args.seed_transpiler),
        )
        base = strip_measurements(base)

    base_qpy = qpy_bytes(base)
    seed_base = int(args.seed + 20251230)

    out_dir = REPO_ROOT / "outputs" / "experiments" / "final_test_equicheck_mc_suite" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "task": str(args.task),
        "n": n,
        "depth": int(args.depth),
        "seed": int(args.seed),
        "traj": int(args.traj),
        "traj_ref": int(args.traj_ref),
        "chunk": int(args.chunk),
        "workers": int(workers),
        "blas_threads": int(args.blas_threads),
        "do_yaqs_equicheck": int(args.do_yaqs_equicheck),
        "gamma": float(args.gamma),
        "gamma_sweep": {
            "min_exp": int(args.gamma_min_exp),
            "max_exp": int(args.gamma_max_exp),
            "num": int(args.gamma_num),
        },
        "batch_list": str(args.batch_list),
        "notes": {
            "metric_main": "trace_fidelity = |Tr(U U'^†)| / 2^n (global phase insensitive)",
            "yaqs_equicheck": "if available, report pass_rate over trajectories (bool output).",
            "mc_goal": "mentor request: average equivalence metric over many noisy circuits (Monte Carlo).",
        }
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    from concurrent.futures import ProcessPoolExecutor

    results: Dict[str, object] = {"meta": meta}

    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=min(int(workers), 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        if args.task in ("mc", "all"):
            # Reference value with traj_ref
            ref = mc_mean_equiv_parallel(
                ex,
                base_qpy, n,
                gamma=float(args.gamma),
                traj=int(args.traj_ref),
                chunk=int(args.chunk),
                seed_base=seed_base,
                blas_threads=int(args.blas_threads),
                do_yaqs=int(args.do_yaqs_equicheck),
            )
            results["ref"] = ref

            # Convergence curve: batch means vs ref
            batch_list = [int(x) for x in str(args.batch_list).split(",") if x.strip()]
            errs = []
            stds = []
            means = []

            # We compute for each m: average over disjoint batches from traj_ref samples:
            # Here we approximate by re-running MC with traj=m (paired seed_base),
            # then compare to ref mean (works well as a clean pipeline).
            ref_mean = float(ref["mean_trace_fidelity"])

            for m in batch_list:
                cur = mc_mean_equiv_parallel(
                    ex,
                    base_qpy, n,
                    gamma=float(args.gamma),
                    traj=int(m),
                    chunk=int(args.chunk),
                    seed_base=seed_base,  # paired across m
                    blas_threads=int(args.blas_threads),
                    do_yaqs=int(args.do_yaqs_equicheck),
                )
                mu = float(cur["mean_trace_fidelity"])
                sd = float(cur["std_trace_fidelity"])
                err = abs(mu - ref_mean)

                means.append(mu)
                stds.append(sd)
                errs.append(err)

                print(f"[mc] m={m:5d}  mean={mu:.6f}  std={sd:.6f}  |mean-ref|={err:.3e}", flush=True)

            results["mc_convergence"] = {
                "batch_list": batch_list,
                "mean_trace_fidelity": means,
                "std_trace_fidelity": stds,
                "abs_mean_minus_ref": errs,
            }

            # Plot convergence
            plt.figure()
            plt.plot(batch_list, errs, marker="o")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("batch size m (log)")
            plt.ylabel("|mean(F_tr) - ref| (log)")
            plt.title(f"Equiv MC convergence (trace fidelity), gamma={float(args.gamma):g}")
            plt.tight_layout()
            plt.savefig(out_dir / "equiv_mc_convergence_loglog.png", dpi=200, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(batch_list, stds, marker="o")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("batch size m (log)")
            plt.ylabel("std(F_tr) within m (log)")
            plt.title(f"Std vs m (trace fidelity), gamma={float(args.gamma):g}")
            plt.tight_layout()
            plt.savefig(out_dir / "equiv_std_vs_m_loglog.png", dpi=200, bbox_inches="tight")
            plt.close()

        if args.task in ("gamma", "all"):
            gammas = np.logspace(int(args.gamma_min_exp), int(args.gamma_max_exp), int(args.gamma_num))
            meanF = np.empty_like(gammas, dtype=float)
            stdF = np.empty_like(gammas, dtype=float)
            passrate = np.empty_like(gammas, dtype=float)

            for i, g in enumerate(gammas):
                cur = mc_mean_equiv_parallel(
                    ex,
                    base_qpy, n,
                    gamma=float(g),
                    traj=int(args.traj),
                    chunk=int(args.chunk),
                    seed_base=seed_base,  # paired across gammas
                    blas_threads=int(args.blas_threads),
                    do_yaqs=int(args.do_yaqs_equicheck),
                )
                meanF[i] = float(cur["mean_trace_fidelity"])
                stdF[i] = float(cur["std_trace_fidelity"])
                passrate[i] = float(cur["yaqs_pass_rate"])

                print(f"[gamma] gamma={g:.3e}  meanFtr={meanF[i]:.6f}  std={stdF[i]:.6f}  yaqs_pass={passrate[i]}", flush=True)

            results["gamma_sweep"] = {
                "gammas": gammas.tolist(),
                "mean_trace_fidelity": meanF.tolist(),
                "std_trace_fidelity": stdF.tolist(),
                "yaqs_pass_rate": passrate.tolist(),
            }

            plt.figure()
            plt.plot(gammas, meanF, marker="o")
            plt.xscale("log")
            plt.xlabel("gamma (log)")
            plt.ylabel("mean trace fidelity")
            plt.title(f"Mean trace fidelity vs gamma (traj={int(args.traj)})")
            plt.tight_layout()
            plt.savefig(out_dir / "equiv_mean_traceF_vs_gamma_logx.png", dpi=200, bbox_inches="tight")
            plt.close()

            if np.isfinite(passrate).any():
                plt.figure()
                plt.plot(gammas, passrate, marker="o")
                plt.xscale("log")
                plt.xlabel("gamma (log)")
                plt.ylabel("YAQS equicheck pass rate")
                plt.title(f"YAQS equicheck pass rate vs gamma (traj={int(args.traj)})")
                plt.tight_layout()
                plt.savefig(out_dir / "yaqs_equicheck_passrate_vs_gamma_logx.png", dpi=200, bbox_inches="tight")
                plt.close()

    elapsed = time.time() - t0
    results["elapsed_sec"] = float(elapsed)
    results["workers_used"] = int(workers)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"task={args.task}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        "",
        "Saved: meta.json, results.json",
        "If task includes mc:     equiv_mc_convergence_loglog.png, equiv_std_vs_m_loglog.png",
        "If task includes gamma:  equiv_mean_traceF_vs_gamma_logx.png (+ yaqs passrate plot if available)",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
