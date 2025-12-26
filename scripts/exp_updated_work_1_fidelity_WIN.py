# scripts/exp_updated_work_1_fidelity_WIN.py
from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Statevector

# ---- gate classes for safe param shifting ----
from qiskit.circuit.library.standard_gates import (
    RXGate, RYGate, RZGate, RZZGate, RXXGate, RZXGate
)

# -----------------------------
# Repo layout
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Import your project functions (fallback if missing)
# -----------------------------
try:
    from qem_yrw_project.circuits.twolocal import build_twolocal  # type: ignore
except Exception:
    def build_twolocal(num_qubits: int, depth: int, seed: int, add_measurements: bool = False) -> QuantumCircuit:
        rng = np.random.default_rng(seed)
        qc = QuantumCircuit(num_qubits)
        for _ in range(depth):
            for i in range(num_qubits):
                qc.rx(float(rng.uniform(0, 2*np.pi)), i)
            for i in range(num_qubits - 1):
                qc.rzz(float(rng.uniform(0, 2*np.pi)), i, i + 1)
        if add_measurements:
            qc.measure_all()
        return qc

try:
    from qem_yrw_project.pauli_jump import apply_pauli_jump_after_each_gate  # type: ignore
except Exception:
    # minimal fallback: insert random Pauli after each gate site with prob gamma
    def apply_pauli_jump_after_each_gate(circuit: QuantumCircuit, gamma: float, seed: int):
        rng = np.random.default_rng(int(seed))
        noisy = QuantumCircuit(circuit.num_qubits)
        for inst, qargs, cargs in circuit.data:
            if inst.name in ("measure", "barrier", "reset"):
                noisy.append(inst, qargs, cargs)
                continue
            noisy.append(inst, qargs, cargs)
            for q in qargs:
                if rng.random() < gamma:
                    pa = int(rng.integers(0, 3))
                    if pa == 0:
                        noisy.x(q)
                    elif pa == 1:
                        noisy.y(q)
                    else:
                        noisy.z(q)
        return noisy, None

# -----------------------------
# CPU helpers
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

# -----------------------------
# Utils
# -----------------------------
def strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    return circ.remove_final_measurements(inplace=False)

def qpy_bytes(circ: QuantumCircuit) -> bytes:
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    return buf.getvalue()

def qpy_load_bytes(b: bytes) -> QuantumCircuit:
    buf = io.BytesIO(b)
    return qpy.load(buf)[0]

def prep_basis_state(n: int, bitstring: str) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i, b in enumerate(bitstring):
        if b == "1":
            qc.x(i)
    return qc

def pure_state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    amp = np.vdot(psi, phi)
    return float(np.real(amp * np.conjugate(amp)))

# -----------------------------
# Angle shift (global)
# -----------------------------
_SHIFTABLE = {
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "rzz": RZZGate,
    "rxx": RXXGate,
    "rzx": RZXGate,
}

def shifted_circuit(circ: QuantumCircuit, delta: float) -> QuantumCircuit:
    """Shift parameters of common rotation gates by +delta. Others unchanged."""
    delta = float(delta)
    out = QuantumCircuit(circ.num_qubits)
    for inst, qargs, cargs in circ.data:
        name = inst.name.lower()
        if name in ("measure", "barrier", "reset"):
            out.append(inst, qargs, cargs)
            continue
        if name in _SHIFTABLE and len(inst.params) == 1:
            theta = float(inst.params[0]) + delta
            gate_cls = _SHIFTABLE[name]
            out.append(gate_cls(theta), qargs, cargs)
        else:
            out.append(inst, qargs, cargs)
    return out

# -----------------------------
# Chunk scheduling to "fill CPU"
# -----------------------------
def choose_chunk(total_shots: int, workers: int, tasks_per_worker: int, min_chunk: int = 1) -> int:
    total_shots = int(total_shots)
    workers = int(max(1, workers))
    tasks_per_worker = int(max(1, tasks_per_worker))
    target_tasks = workers * tasks_per_worker
    chunk = int(math.ceil(total_shots / max(1, target_tasks)))
    return int(max(min_chunk, chunk))

# -----------------------------
# Monte Carlo worker (returns array of fidelities)
# -----------------------------
def _mc_chunk_fidelities(
    circ_qpy: bytes,
    n: int,
    bitstring: str,
    psi_ideal: np.ndarray,
    gamma: float,
    shots: int,
    seed0: int,
    blas_threads: int,
) -> np.ndarray:
    _worker_init(blas_threads)

    circ = strip_measurements(qpy_load_bytes(circ_qpy))
    prep = prep_basis_state(n, bitstring)
    rng = np.random.default_rng(int(seed0))

    fids = np.empty(int(shots), dtype=np.float64)
    for i in range(int(shots)):
        s = int(rng.integers(0, 2**31 - 1))
        noisy, _ = apply_pauli_jump_after_each_gate(circ, float(gamma), s)
        phi = Statevector.from_instruction(prep.compose(noisy, inplace=False)).data
        fids[i] = pure_state_fidelity(psi_ideal, phi)
    return fids

def mc_mean_fidelity_parallel(
    ex,
    circ_qpy: bytes,
    n: int,
    bitstring: str,
    psi_ideal: np.ndarray,
    gamma: float,
    total_shots: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
) -> float:
    total_shots = int(total_shots)
    chunk = int(max(1, chunk))
    n_chunks = (total_shots + chunk - 1) // chunk

    futs = []
    for k in range(n_chunks):
        s = min(chunk, total_shots - k * chunk)
        seed0 = int(seed_base + 97 * k)
        futs.append(ex.submit(
            _mc_chunk_fidelities, circ_qpy, n, bitstring, psi_ideal,
            float(gamma), int(s), int(seed0), int(blas_threads)
        ))

    sum_f = 0.0
    cnt = 0
    for fut in futs:
        arr = fut.result()
        sum_f += float(np.sum(arr))
        cnt += int(arr.size)
    return sum_f / max(1, cnt)

def mc_pool_fidelities_parallel(
    ex,
    circ_qpy: bytes,
    n: int,
    bitstring: str,
    psi_ideal: np.ndarray,
    gamma: float,
    total_shots: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
) -> np.ndarray:
    total_shots = int(total_shots)
    chunk = int(max(1, chunk))
    n_chunks = (total_shots + chunk - 1) // chunk

    futs = []
    for k in range(n_chunks):
        s = min(chunk, total_shots - k * chunk)
        seed0 = int(seed_base + 97 * k)
        futs.append(ex.submit(
            _mc_chunk_fidelities, circ_qpy, n, bitstring, psi_ideal,
            float(gamma), int(s), int(seed0), int(blas_threads)
        ))
    parts = [f.result() for f in futs]
    return np.concatenate(parts, axis=0)

# -----------------------------
# Resampling convergence curve
# -----------------------------
def convergence_curve_from_pool(
    pool: np.ndarray,
    reference: float,
    Ns: List[int],
    resamples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each N, estimate E[ |mean(sample_N) - reference| ] using bootstrap resampling.
    Returns (mean_error[N], std_error[N]) over resamples.
    """
    rng = np.random.default_rng(int(seed))
    pool = np.asarray(pool, dtype=np.float64)
    M = pool.size

    mean_err = []
    std_err = []
    for N in Ns:
        N = int(N)
        errs = np.empty(int(resamples), dtype=np.float64)
        for r in range(int(resamples)):
            idx = rng.integers(0, M, size=N)  # with replacement
            m = float(np.mean(pool[idx]))
            errs[r] = abs(m - reference)
        mean_err.append(float(np.mean(errs)))
        std_err.append(float(np.std(errs)))
    return np.asarray(mean_err), np.asarray(std_err)

def add_info_box(ax, text: str) -> None:
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # model
    ap.add_argument("--num-qubits", type=int, default=10)     # supervisor: 10 sites
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # noise gamma sets
    ap.add_argument("--gamma-conv", type=str, default="1e-3,1e-2,1e-1")
    ap.add_argument("--gamma-sweep-min", type=float, default=1e-3)
    ap.add_argument("--gamma-sweep-max", type=float, default=1e-1)
    ap.add_argument("--gamma-sweep-num", type=int, default=100)

    # Monte Carlo budgets (supervisor-style)
    ap.add_argument("--conv-max-traj", type=int, default=10000)    # reference pool size
    ap.add_argument("--conv-resamples", type=int, default=200)     # bootstrap repeats
    ap.add_argument("--conv-Ns", type=str, default="10,20,30,40,50,60,70,80,90,100,200,300,500,800,1000,2000,5000")

    # sweep/angle trajectories (you can crank these up)
    ap.add_argument("--sweep-traj", type=int, default=2000)        # per gamma
    ap.add_argument("--angle-traj", type=int, default=2000)        # per delta
    ap.add_argument("--angle-gamma", type=float, default=1e-2)

    # angle scan
    ap.add_argument("--angle-min", type=float, default=1e-6)
    ap.add_argument("--angle-max", type=float, default=2*np.pi)
    ap.add_argument("--angle-num", type=int, default=40)

    # parallel
    ap.add_argument("--workers", type=int, default=0)              # 0 => all logical cores
    ap.add_argument("--blas-threads", type=int, default=1)         # avoid oversubscription
    ap.add_argument("--fill-cpu", type=int, default=1)             # force many small tasks to occupy all cores
    ap.add_argument("--tasks-per-worker", type=int, default=8)     # more => smaller chunks

    # transpile
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    # input state (single reference input)
    ap.add_argument("--input", type=str, default="")  # default = all zeros

    args = ap.parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    n = int(args.num_qubits)
    bitstring = args.input.strip() if args.input.strip() else ("0" * n)
    if len(bitstring) != n or any(c not in "01" for c in bitstring):
        raise ValueError(f"--input must be a bitstring of length {n} (only 0/1). Got: {bitstring}")

    def chunk_for(total_shots: int) -> int:
        if int(args.fill_cpu) == 1:
            return choose_chunk(total_shots, workers, int(args.tasks_per_worker), min_chunk=1)
        return max(1, int(math.ceil(total_shots / workers)))

    # Build base circuit (Qiskit)
    circ = build_twolocal(num_qubits=n, depth=int(args.depth), seed=int(args.seed), add_measurements=False)
    circ = strip_measurements(circ)
    if int(args.use_transpile) == 1:
        circ = transpile(
            circ,
            optimization_level=int(args.opt_level),
            seed_transpiler=int(args.seed_transpiler),
        )
        circ = strip_measurements(circ)

    # Ideal state for this input
    prep = prep_basis_state(n, bitstring)
    psi_ideal = Statevector.from_instruction(prep.compose(circ, inplace=False)).data

    out_dir = REPO_ROOT / "outputs" / "experiments" / "updated_work_1_fidelity" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "workers": workers,
        "blas_threads": int(args.blas_threads),
        "fill_cpu": int(args.fill_cpu),
        "tasks_per_worker": int(args.tasks_per_worker),
        "input": bitstring,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    circ_qpy = qpy_bytes(circ)

    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    results: Dict[str, object] = {"args": vars(args), "meta": meta}

    # We will compute angle-scan first to know "best shift" info box, then reuse that box on all plots.
    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        # ==========================================================
        # 3) Angle shifts (logspace) -> baseline + best delta
        # ==========================================================
        deltas = np.logspace(np.log10(float(args.angle_min)), np.log10(float(args.angle_max)), int(args.angle_num))

        # baseline (no shift)
        fid_base = mc_mean_fidelity_parallel(
            ex, circ_qpy, n, bitstring, psi_ideal,
            gamma=float(args.angle_gamma),
            total_shots=int(args.angle_traj),
            chunk=chunk_for(int(args.angle_traj)),
            seed_base=int(args.seed + 333000),
            blas_threads=int(args.blas_threads),
        )

        angle_mean_fid = np.empty_like(deltas, dtype=np.float64)
        for i, dlt in enumerate(deltas):
            shifted = shifted_circuit(circ, float(dlt))
            shifted_qpy = qpy_bytes(shifted)
            mu = mc_mean_fidelity_parallel(
                ex, shifted_qpy, n, bitstring, psi_ideal,
                gamma=float(args.angle_gamma),
                total_shots=int(args.angle_traj),
                chunk=chunk_for(int(args.angle_traj)),
                seed_base=int(args.seed + 334000 + i * 1000),
                blas_threads=int(args.blas_threads),
            )
            angle_mean_fid[i] = mu

        best_idx = int(np.argmax(angle_mean_fid))
        best_delta = float(deltas[best_idx])
        best_fid = float(angle_mean_fid[best_idx])
        improve = best_fid - float(fid_base)

        info_box = (
            f"Fidelity metric (pure-state overlap)\n"
            f"angle_gamma = {float(args.angle_gamma):.3g}\n"
            f"baseline (δ=0): {fid_base:.6f}\n"
            f"best δ*: {best_delta:.3g}\n"
            f"best F(δ*): {best_fid:.6f}\n"
            f"Δ(best-base): {improve:+.6f}"
        )

        results["angle_shift_fidelity"] = {
            "gamma": float(args.angle_gamma),
            "baseline_fid_delta0": float(fid_base),
            "deltas": deltas.tolist(),
            "mean_fidelity": angle_mean_fid.tolist(),
            "best_delta": best_delta,
            "best_fidelity": best_fid,
            "improvement": improve,
        }

        # plot angle scan
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(deltas, angle_mean_fid, marker="o", markersize=3, label="shifted")
        ax.axhline(float(fid_base), linestyle="--", label="baseline (delta=0)")
        ax.axvline(best_delta, linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("angle shift delta (log)")
        ax.set_ylabel("mean state fidelity")
        ax.set_title(f"Angle-shift scan (Fidelity), gamma={float(args.angle_gamma):g}")
        ax.legend()
        add_info_box(ax, info_box)
        fig.tight_layout()
        fig.savefig(out_dir / "angle_shift_logx_fidelity.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ==========================================================
        # 1) MC convergence (smooth curve via resampling from pool)
        # ==========================================================
        gammas_conv = [float(x) for x in args.gamma_conv.split(",") if x.strip()]
        Ns = [int(x) for x in args.conv_Ns.split(",") if x.strip()]
        Ns = sorted(set([N for N in Ns if N > 0]))

        conv_data = {}
        for g in gammas_conv:
            pool = mc_pool_fidelities_parallel(
                ex, circ_qpy, n, bitstring, psi_ideal,
                gamma=g,
                total_shots=int(args.conv_max_traj),
                chunk=chunk_for(int(args.conv_max_traj)),
                seed_base=int(args.seed + 123456 + int(1e6*g)),
                blas_threads=int(args.blas_threads),
            )
            ref = float(np.mean(pool))
            mean_err, std_err = convergence_curve_from_pool(
                pool=pool,
                reference=ref,
                Ns=Ns,
                resamples=int(args.conv_resamples),
                seed=int(args.seed + 999 + int(1e6*g)),
            )
            conv_data[str(g)] = {
                "reference_mean_fid": ref,
                "Ns": Ns,
                "mean_abs_error": mean_err.tolist(),
                "std_abs_error": std_err.tolist(),
            }

        results["mc_convergence_fidelity"] = conv_data

        fig = plt.figure()
        ax = fig.gca()
        for g in gammas_conv:
            d = conv_data[str(g)]
            x = np.asarray(d["Ns"], dtype=float)
            y = np.asarray(d["mean_abs_error"], dtype=float)
            ax.plot(x, y, marker="o", label=f"gamma={g:g} (refF={float(d['reference_mean_fid']):.4f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N trajectories (log)")
        ax.set_ylabel("E[ |mean_N - reference| ] (log)")
        ax.set_title("Monte Carlo convergence (Fidelity) — resampling from N_max pool")
        ax.legend()
        add_info_box(ax, info_box)
        fig.tight_layout()
        fig.savefig(out_dir / "mc_convergence_loglog_fidelity.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ==========================================================
        # 2) Error vs noise strength (logspace gamma)
        # ==========================================================
        gmin = float(args.gamma_sweep_min)
        gmax = float(args.gamma_sweep_max)
        num = int(args.gamma_sweep_num)
        gammas = np.logspace(np.log10(gmin), np.log10(gmax), num=num)

        sweep_mean_fid = np.empty(num, dtype=np.float64)
        for i, g in enumerate(gammas):
            mu = mc_mean_fidelity_parallel(
                ex, circ_qpy, n, bitstring, psi_ideal,
                gamma=float(g),
                total_shots=int(args.sweep_traj),
                chunk=chunk_for(int(args.sweep_traj)),
                seed_base=int(args.seed + 222000 + i * 1000),
                blas_threads=int(args.blas_threads),
            )
            sweep_mean_fid[i] = mu

        sweep_error = 1.0 - sweep_mean_fid
        best_gamma_idx = int(np.argmax(sweep_mean_fid))
        best_gamma = float(gammas[best_gamma_idx])
        best_gamma_fid = float(sweep_mean_fid[best_gamma_idx])

        results["error_vs_gamma_fidelity"] = {
            "gamma": gammas.tolist(),
            "mean_fidelity": sweep_mean_fid.tolist(),
            "error_1_minus_fid": sweep_error.tolist(),
            "best_gamma": best_gamma,
            "best_gamma_fidelity": best_gamma_fid,
        }

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(gammas, sweep_error, marker="o", markersize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("gamma (log)")
        ax.set_ylabel("error = 1 - mean fidelity (log)")
        ax.set_title("Error vs noise strength (Fidelity, log-log)")
        # annotate best gamma
        ax.axvline(best_gamma, linestyle=":", linewidth=1)
        add_info_box(ax, info_box + f"\n(best gamma in sweep: {best_gamma:.3g}, F={best_gamma_fid:.6f})")
        fig.tight_layout()
        fig.savefig(out_dir / "error_vs_gamma_loglog_fidelity.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    elapsed = time.time() - t0
    results["elapsed_sec"] = float(elapsed)
    results["workers_used"] = int(workers)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}",
        f"fill_cpu={int(args.fill_cpu)}, tasks_per_worker={int(args.tasks_per_worker)}",
        f"input={bitstring}",
        "",
        f"[Angle scan] baseline_fid={fid_base:.8f}",
        f"[Angle scan] best_delta={best_delta:.8g}, best_fid={best_fid:.8f}, improvement={improve:+.8f}",
        "",
        "Saved figures:",
        "  mc_convergence_loglog_fidelity.png",
        "  error_vs_gamma_loglog_fidelity.png",
        "  angle_shift_logx_fidelity.png",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
