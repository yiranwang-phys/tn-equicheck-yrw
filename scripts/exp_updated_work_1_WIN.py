# scripts/exp_updated_work_1_WIN.py
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy, transpile

# ---- gate classes for safe param shifting ----
from qiskit.circuit.library.standard_gates import (
    RXGate, RYGate, RZGate, RZZGate, RXXGate, RZXGate
)

# -----------------------------
# Make src/ importable (src-layout)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Import your project functions
# -----------------------------
try:
    from qem_yrw_project.circuits.twolocal import build_twolocal  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Cannot import build_twolocal. Make sure your repo uses src-layout and you run from repo root."
    ) from e

try:
    # Prefer the richer one with include_measurements flag
    from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Cannot import apply_pauli_jump_after_each_gate from qem_yrw_project.noise.pauli_jump"
    ) from e

# -----------------------------
# YAQS imports (StrongSim)
# -----------------------------
try:
    from mqt.yaqs import simulator
    from mqt.yaqs.core.data_structures.networks import MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
    from mqt.yaqs.core.libraries.gate_library import Z
except Exception as e:
    raise RuntimeError(
        "Cannot import mqt.yaqs. Install YAQS (mqt-yaqs) in this venv first."
    ) from e

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

def prepend_prep_x(circ: QuantumCircuit, bitstring: str) -> QuantumCircuit:
    """Return (prep(bitstring) ∘ circ). bitstring indexed qubit-0..n-1."""
    n = circ.num_qubits
    if len(bitstring) != n or any(c not in "01" for c in bitstring):
        raise ValueError(f"input bitstring must be length {n} of 0/1. Got {bitstring}")
    if set(bitstring) == {"0"} or bitstring == "0" * n:
        return circ
    prep = QuantumCircuit(n)
    for i, b in enumerate(bitstring):
        if b == "1":
            prep.x(i)
    return prep.compose(circ, inplace=False)

def strongsim_all_Z(circ: QuantumCircuit, *, n: int, max_bond_dim: int, threshold: float) -> np.ndarray:
    """
    YAQS StrongSim: compute <Z_i> for all sites on |0...0>.
    """
    state = MPS(n, state="zeros")
    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(n)],
        num_traj=1,
        max_bond_dim=int(max_bond_dim),
        threshold=float(threshold),
    )
    simulator.run(state, circ, sim_params, noise_model=None, parallel=False)
    return np.array([obs.results[0] for obs in sim_params.observables], dtype=float)

def z_match_score(z_noisy: np.ndarray, z_ideal: np.ndarray) -> float:
    """
    Score in [0,1]. 1 means identical Z expectation profile.
    """
    z_noisy = np.asarray(z_noisy, dtype=float)
    z_ideal = np.asarray(z_ideal, dtype=float)
    return float(1.0 - np.mean(np.abs(z_noisy - z_ideal) / 2.0))

# -----------------------------
# Angle shift
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
# Monte Carlo worker (returns array of scores)
# -----------------------------
def _apply_noise(circ: QuantumCircuit, gamma: float, seed: int) -> QuantumCircuit:
    """
    Signature-tolerant wrapper around your apply_pauli_jump_after_each_gate.
    """
    try:
        noisy, _ = apply_pauli_jump_after_each_gate(circ, float(gamma), int(seed), include_measurements=False)
    except TypeError:
        noisy, _ = apply_pauli_jump_after_each_gate(circ, float(gamma), int(seed))
        noisy = strip_measurements(noisy)
    return noisy

def _mc_chunk_scores(
    circ_qpy: bytes,
    z_ideal: np.ndarray,
    gamma: float,
    shots: int,
    seed0: int,
    max_bond_dim: int,
    threshold: float,
    blas_threads: int,
) -> np.ndarray:
    _worker_init(blas_threads)

    base = strip_measurements(qpy_load_bytes(circ_qpy))
    n = base.num_qubits

    rng = np.random.default_rng(int(seed0))
    out = np.empty(int(shots), dtype=np.float64)

    for i in range(int(shots)):
        s = int(rng.integers(0, 2**31 - 1))
        noisy = _apply_noise(base, float(gamma), s)
        z_noisy = strongsim_all_Z(noisy, n=n, max_bond_dim=max_bond_dim, threshold=threshold)
        out[i] = z_match_score(z_noisy, z_ideal)
    return out

def mc_pool_scores_parallel(
    ex,
    circ_qpy: bytes,
    z_ideal: np.ndarray,
    gamma: float,
    total_shots: int,
    chunk: int,
    seed_base: int,
    max_bond_dim: int,
    threshold: float,
    blas_threads: int,
) -> np.ndarray:
    total_shots = int(total_shots)
    chunk = int(max(1, chunk))
    n_chunks = (total_shots + chunk - 1) // chunk

    futs = []
    for k in range(n_chunks):
        s = min(chunk, total_shots - k * chunk)
        seed0 = int(seed_base + 97 * k)
        futs.append(
            ex.submit(
                _mc_chunk_scores,
                circ_qpy,
                z_ideal,
                float(gamma),
                int(s),
                int(seed0),
                int(max_bond_dim),
                float(threshold),
                int(blas_threads),
            )
        )
    parts = [f.result() for f in futs]
    return np.concatenate(parts, axis=0)

def mc_mean_score_parallel(
    ex,
    circ_qpy: bytes,
    z_ideal: np.ndarray,
    gamma: float,
    total_shots: int,
    chunk: int,
    seed_base: int,
    max_bond_dim: int,
    threshold: float,
    blas_threads: int,
) -> float:
    pool = mc_pool_scores_parallel(
        ex, circ_qpy, z_ideal, gamma, total_shots, chunk, seed_base, max_bond_dim, threshold, blas_threads
    )
    return float(np.mean(pool))

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

# -----------------------------
# Main
# -----------------------------
@dataclass
class RunSummary:
    args: Dict
    workers_used: int
    elapsed_sec: float
    z_ideal: List[float]
    mc_convergence: Dict
    error_vs_gamma: Dict
    angle_shift: Dict

def main() -> None:
    ap = argparse.ArgumentParser()

    # model
    ap.add_argument("--num-qubits", type=int, default=10)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", type=str, default="")  # bitstring, default all-zeros

    # YAQS strongsim params
    ap.add_argument("--max-bond-dim", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=1e-10)

    # compilation
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    # 1) MC convergence
    ap.add_argument("--gamma-list", type=str, default="1e-3,1e-2,1e-1")
    ap.add_argument("--conv-max-traj", type=int, default=10000)
    ap.add_argument("--conv-resamples", type=int, default=200)
    ap.add_argument("--conv-Ns", type=str, default="10,20,30,40,50,60,70,80,90,100,200,300,500,800,1000,2000,5000,10000")

    # 2) error vs gamma sweep (logspace)
    ap.add_argument("--gamma-sweep-min", type=float, default=1e-3)
    ap.add_argument("--gamma-sweep-max", type=float, default=1e-1)
    ap.add_argument("--gamma-sweep-num", type=int, default=100)
    ap.add_argument("--sweep-traj", type=int, default=200)  # per gamma

    # 3) angle shift scan (logspace delta)
    ap.add_argument("--angle-gamma", type=float, default=1e-2)
    ap.add_argument("--angle-traj", type=int, default=200)  # per delta
    ap.add_argument("--angle-min", type=float, default=1e-6)
    ap.add_argument("--angle-max", type=float, default=2 * np.pi)
    ap.add_argument("--angle-num", type=int, default=40)

    # parallel
    ap.add_argument("--workers", type=int, default=0)        # 0 => all logical cores
    ap.add_argument("--blas-threads", type=int, default=1)   # avoid oversubscription
    ap.add_argument("--chunk", type=int, default=10)         # trajectories per task

    args = ap.parse_args()

    # workers
    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    n = int(args.num_qubits)
    bitstring = args.input.strip() if args.input.strip() else ("0" * n)
    if len(bitstring) != n or any(c not in "01" for c in bitstring):
        raise ValueError(f"--input must be a bitstring of length {n} (only 0/1). Got: {bitstring}")

    # Build circuit (Qiskit)
    circ = build_twolocal(num_qubits=n, depth=int(args.depth), seed=int(args.seed), add_measurements=False)
    circ = strip_measurements(circ)
    if int(args.use_transpile) == 1:
        # IMPORTANT: do NOT set routing_method="stochastic" (your earlier error).
        circ = transpile(
            circ,
            optimization_level=int(args.opt_level),
            seed_transpiler=int(args.seed_transpiler),
        )
        circ = strip_measurements(circ)

    # prepend input preparation if needed
    circ = prepend_prep_x(circ, bitstring)

    # output directory
    out_dir = REPO_ROOT / "outputs" / "experiments" / "updated_work_1" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "meta.txt").write_text(
        f"time={datetime.now().isoformat()}\n"
        f"platform={platform.platform()}\n"
        f"python={platform.python_version()}\n"
        f"workers={workers}\n"
        f"blas_threads={int(args.blas_threads)}\n"
        f"chunk={int(args.chunk)}\n"
        f"input={bitstring}\n"
        f"max_bond_dim={int(args.max_bond_dim)}\n"
        f"threshold={float(args.threshold)}\n",
        encoding="utf-8"
    )

    # Ideal Z profile (YAQS StrongSim, noiseless)
    z_ideal = strongsim_all_Z(circ, n=n, max_bond_dim=int(args.max_bond_dim), threshold=float(args.threshold))
    np.savetxt(out_dir / "z_ideal.txt", z_ideal)

    circ_qpy = qpy_bytes(circ)

    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()

    # =========================
    # Run all tasks
    # =========================
    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        # ----------------------------------------------------------
        # 1) MC convergence
        # ----------------------------------------------------------
        gammas_conv = [float(x) for x in args.gamma_list.split(",") if x.strip()]
        Ns = [int(x) for x in args.conv_Ns.split(",") if x.strip()]
        Ns = sorted({N for N in Ns if N > 0 and N <= int(args.conv_max_traj)})

        conv_data: Dict[str, Dict] = {}
        for g in gammas_conv:
            pool = mc_pool_scores_parallel(
                ex,
                circ_qpy,
                z_ideal,
                gamma=g,
                total_shots=int(args.conv_max_traj),
                chunk=int(args.chunk),
                seed_base=int(args.seed + 123456 + int(1e6 * g)),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                blas_threads=int(args.blas_threads),
            )
            ref = float(np.mean(pool))
            mean_err, std_err = convergence_curve_from_pool(
                pool=pool,
                reference=ref,
                Ns=Ns,
                resamples=int(args.conv_resamples),
                seed=int(args.seed + 999 + int(1e6 * g)),
            )
            conv_data[str(g)] = {
                "reference_mean_score": ref,
                "Ns": Ns,
                "mean_abs_error": mean_err.tolist(),
                "std_abs_error": std_err.tolist(),
            }

        plt.figure()
        for g in gammas_conv:
            d = conv_data[str(g)]
            x = np.asarray(d["Ns"], dtype=float)
            y = np.asarray(d["mean_abs_error"], dtype=float)
            plt.plot(x, y, marker="o", label=f"gamma={g:g}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("N trajectories (log)")
        plt.ylabel("E[ |mean_N - reference| ] (log)")
        plt.title("MC convergence (YAQS StrongSim Z-match score)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "mc_convergence_loglog.png", dpi=200, bbox_inches="tight")
        plt.close()

        # ----------------------------------------------------------
        # 2) error vs gamma (logspace sweep)
        # ----------------------------------------------------------
        gmin = float(args.gamma_sweep_min)
        gmax = float(args.gamma_sweep_max)
        num = int(args.gamma_sweep_num)
        gammas = np.logspace(np.log10(gmin), np.log10(gmax), num=num)

        sweep_mean_score = np.empty(num, dtype=np.float64)
        for i, g in enumerate(gammas):
            mu = mc_mean_score_parallel(
                ex,
                circ_qpy,
                z_ideal,
                gamma=float(g),
                total_shots=int(args.sweep_traj),
                chunk=int(args.chunk),
                seed_base=int(args.seed + 222000 + i * 1000),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                blas_threads=int(args.blas_threads),
            )
            sweep_mean_score[i] = mu

        sweep_error = 1.0 - sweep_mean_score

        plt.figure()
        plt.plot(gammas, sweep_error, marker="o", markersize=3)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("gamma (log)")
        plt.ylabel("error = 1 - mean(Z-match score) (log)")
        plt.title("Error vs noise strength (YAQS StrongSim)")
        plt.tight_layout()
        plt.savefig(out_dir / "error_vs_gamma_loglog.png", dpi=200, bbox_inches="tight")
        plt.close()

        # ----------------------------------------------------------
        # 3) angle shift scan (logspace delta)
        # ----------------------------------------------------------
        deltas = np.logspace(np.log10(float(args.angle_min)), np.log10(float(args.angle_max)), int(args.angle_num))

        # baseline at delta=0
        base_score = mc_mean_score_parallel(
            ex,
            circ_qpy,
            z_ideal,
            gamma=float(args.angle_gamma),
            total_shots=int(args.angle_traj),
            chunk=int(args.chunk),
            seed_base=int(args.seed + 333000),
            max_bond_dim=int(args.max_bond_dim),
            threshold=float(args.threshold),
            blas_threads=int(args.blas_threads),
        )

        angle_mean_score = np.empty_like(deltas, dtype=np.float64)
        for i, dlt in enumerate(deltas):
            shifted = shifted_circuit(circ, float(dlt))
            shifted_qpy = qpy_bytes(shifted)
            mu = mc_mean_score_parallel(
                ex,
                shifted_qpy,
                z_ideal,  # target stays ORIGINAL ideal Z-profile
                gamma=float(args.angle_gamma),
                total_shots=int(args.angle_traj),
                chunk=int(args.chunk),
                seed_base=int(args.seed + 334000 + i * 1000),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                blas_threads=int(args.blas_threads),
            )
            angle_mean_score[i] = mu

        plt.figure()
        plt.plot(deltas, angle_mean_score, marker="o", markersize=3, label="shifted")
        plt.axhline(float(base_score), linestyle="--", label="delta=0 baseline")
        plt.xscale("log")
        plt.xlabel("angle shift delta (log)")
        plt.ylabel("mean Z-match score")
        plt.title(f"Angle-shift scan (gamma={float(args.angle_gamma):g})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "angle_shift_logx.png", dpi=200, bbox_inches="tight")
        plt.close()

    elapsed = time.time() - t0

    summary = RunSummary(
        args=vars(args),
        workers_used=int(workers),
        elapsed_sec=float(elapsed),
        z_ideal=z_ideal.tolist(),
        mc_convergence=conv_data,
        error_vs_gamma={
            "gamma": gammas.tolist(),
            "mean_score": sweep_mean_score.tolist(),
            "error_1_minus_score": sweep_error.tolist(),
        },
        angle_shift={
            "gamma": float(args.angle_gamma),
            "delta0": 0.0,
            "base_score": float(base_score),
            "deltas": deltas.tolist(),
            "mean_score": angle_mean_score.tolist(),
        },
    )

    (out_dir / "results.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        f"input={bitstring}",
        "",
        "Saved figures:",
        "  mc_convergence_loglog.png",
        "  error_vs_gamma_loglog.png",
        "  angle_shift_logx.png",
        "",
        "Notes:",
        "- This uses YAQS StrongSim to compute <Z_i> and defines a Z-match score in [0,1].",
        "- For Pauli-jump (stochastic) noise, global parameter shifts typically do NOT improve this score.",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)

if __name__ == "__main__":
    main()
