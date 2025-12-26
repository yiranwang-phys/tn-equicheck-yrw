# scripts/exp_angle_shift_families_strongsim_YAQS_WIN.py
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy, transpile

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
# Pauli-jump after each gate (fallback if missing)
# -----------------------------
try:
    from qem_yrw_project.pauli_jump import apply_pauli_jump_after_each_gate  # type: ignore
except Exception:
    def apply_pauli_jump_after_each_gate(circuit: QuantumCircuit, gamma: float, seed: int):
        rng = np.random.default_rng(int(seed))
        noisy = QuantumCircuit(circuit.num_qubits)
        for ci in circuit.data:
            op = ci.operation
            if op.name in ("measure", "barrier", "reset"):
                continue
            qargs = list(ci.qubits)
            cargs = list(ci.clbits)
            noisy.append(op, qargs, cargs)
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

def fidelity_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    amp = np.vdot(psi, phi)
    f = float(np.real(amp * np.conjugate(amp)))
    if f < 0.0:
        f = 0.0
    if f > 1.0 + 1e-9:
        f = 1.0
    return f

# -----------------------------
# YAQS StrongSim: MPS -> dense statevector
# (match the working approach in your updated_work_1 script)
# -----------------------------
def mps_to_statevector(mps) -> np.ndarray:
    tensors = mps.tensors
    psi = np.asarray(tensors[0])
    psi = psi[:, 0, :]  # (2, chi)
    for t in tensors[1:]:
        t = np.asarray(t)  # (2, chi_prev, chi_next)
        psi = np.tensordot(psi, t, axes=([1], [1]))  # (2^k, 2, chi_next)
        psi = psi.reshape(psi.shape[0] * psi.shape[1], psi.shape[2])  # (2^(k+1), chi_next)
    psi = psi[:, 0]
    return psi.astype(np.complex128, copy=False)

def _make_mps(n: int):
    from mqt.yaqs.core.data_structures.networks import MPS
    # try the init style you used successfully before
    try:
        return MPS(length=n, state="zeros")
    except TypeError:
        return MPS(n)

def _make_strongsim_params(max_bond_dim: int, threshold: float):
    from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
    sig = inspect.signature(StrongSimParams)
    params = sig.parameters

    kwargs = {}
    # max bond dim name
    if "max_bond_dim" in params:
        kwargs["max_bond_dim"] = int(max_bond_dim)
    elif "maxBondDim" in params:
        kwargs["maxBondDim"] = int(max_bond_dim)

    # truncation/threshold name (your YAQS seems to use 'threshold')
    if "threshold" in params:
        kwargs["threshold"] = float(threshold)
    elif "svd_cut" in params:
        kwargs["svd_cut"] = float(threshold)
    elif "svdCut" in params:
        kwargs["svdCut"] = float(threshold)

    # get_state flag
    if "get_state" in params:
        kwargs["get_state"] = True

    # num_traj if exists
    if "num_traj" in params:
        kwargs["num_traj"] = 1

    # show_progress if exists
    if "show_progress" in params:
        kwargs["show_progress"] = False

    return StrongSimParams(**kwargs)

def run_strongsim_statevector(
    circ: QuantumCircuit,
    n: int,
    max_bond_dim: int,
    threshold: float,
) -> np.ndarray:
    """
    Run YAQS strong circuit simulation (no noise_model!) and return final statevector.
    Noise must be baked into the circuit as explicit Pauli gates per trajectory.
    """
    from mqt.yaqs import simulator

    state = _make_mps(int(n))
    sim_params = _make_strongsim_params(int(max_bond_dim), float(threshold))

    # run signature differs by version; try both
    try:
        simulator.run(state, circ, sim_params, noise_model=None, parallel=False)
    except TypeError:
        simulator.run(state, circ, sim_params, noise_model=None)

    out_state = getattr(sim_params, "output_state", None)
    if out_state is None:
        out_state = state
    return mps_to_statevector(out_state)

# -----------------------------
# Angle shifting (family-specific): shift only RZZ or only RX
# -----------------------------
def shifted_circuit_family(circ: QuantumCircuit, delta: float, family: str) -> QuantumCircuit:
    fam = family.strip().lower()
    delta = float(delta)

    out = QuantumCircuit(circ.num_qubits)
    for ci in circ.data:
        op = ci.operation
        if op.name in ("measure", "barrier", "reset"):
            continue

        name = op.name.lower()
        if fam == "rzz" and name == "rzz" and len(op.params) == 1:
            theta = float(op.params[0]) + delta
            q0, q1 = ci.qubits[0], ci.qubits[1]
            out.rzz(theta, q0, q1)
        elif fam == "rx" and name == "rx" and len(op.params) == 1:
            theta = float(op.params[0]) + delta
            q0 = ci.qubits[0]
            out.rx(theta, q0)
        else:
            out.append(op, list(ci.qubits), list(ci.clbits))

    return out

# -----------------------------
# Paired seeds across variants
# -----------------------------
def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * shot_index) % (2**31 - 1))

# -----------------------------
# MC chunk worker
# -----------------------------
def _chunk_fidelities(
    circ_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    start: int,
    shots: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
) -> np.ndarray:
    _worker_init(int(blas_threads))

    base = strip_measurements(qpy_load_bytes(circ_qpy))
    out = np.empty(int(shots), dtype=np.float64)

    for i in range(int(shots)):
        global_idx = int(start + i)
        s = shot_seed(int(seed_base), global_idx)

        noisy, _ = apply_pauli_jump_after_each_gate(base, float(gamma), int(s))
        noisy = strip_measurements(noisy)

        vec = run_strongsim_statevector(
            noisy,
            n=int(n),
            max_bond_dim=int(max_bond_dim),
            threshold=float(threshold),
        )
        out[i] = fidelity_pure(psi_ideal, vec)

    return out

def mc_mean_parallel(
    ex,
    circ_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
) -> float:
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
                circ_qpy, n, psi_ideal, float(gamma),
                int(start), int(shots), int(seed_base),
                int(blas_threads), int(max_bond_dim), float(threshold),
            )
        )

    s = 0.0
    cnt = 0
    for f in futs:
        arr = f.result()
        s += float(np.sum(arr))
        cnt += int(arr.size)
    return s / max(1, cnt)

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--num-qubits", type=int, default=10)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=50)

    ap.add_argument("--delta-min", type=float, default=1e-6)
    ap.add_argument("--delta-max", type=float, default=2 * np.pi)
    ap.add_argument("--delta-num", type=int, default=60)

    # order matters: rzz first then rx
    ap.add_argument("--families", type=str, default="rzz,rx")

    # YAQS strongsim controls (match your working style: threshold)
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)

    # parallel
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--blas-threads", type=int, default=1)

    # compilation
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

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

    # Ideal vector by YAQS StrongSim (no noise, no shift)
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
    )

    deltas = np.logspace(np.log10(float(args.delta_min)), np.log10(float(args.delta_max)), int(args.delta_num))
    families = [x.strip().lower() for x in args.families.split(",") if x.strip()]
    if not families:
        families = ["rzz", "rx"]

    out_dir = REPO_ROOT / "outputs" / "experiments" / "angle_shift_families_strongsim" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "workers": workers,
        "blas_threads": int(args.blas_threads),
        "chunk": int(args.chunk),
        "n": n,
        "depth": int(args.depth),
        "seed": int(args.seed),
        "gamma": float(args.gamma),
        "traj": int(args.traj),
        "delta_min": float(args.delta_min),
        "delta_max": float(args.delta_max),
        "delta_num": int(args.delta_num),
        "families": families,
        "yaqs": {"max_bond_dim": int(args.max_bond_dim), "threshold": float(args.threshold)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_qpy = qpy_bytes(base)

    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    results: Dict[str, object] = {"meta": meta, "deltas": deltas.tolist()}

    seed_base = int(args.seed + 20251220)  # paired across all deltas/families

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        fid0 = mc_mean_parallel(
            ex, base_qpy, n, ideal_vec,
            gamma=float(args.gamma),
            traj=int(args.traj),
            chunk=int(args.chunk),
            seed_base=seed_base,
            blas_threads=int(args.blas_threads),
            max_bond_dim=int(args.max_bond_dim),
            threshold=float(args.threshold),
        )
        results["baseline"] = {"delta": 0.0, "mean_fidelity": float(fid0)}

        fam_curves: Dict[str, Dict[str, object]] = {}
        for fam in families:
            curve = np.empty_like(deltas, dtype=np.float64)
            for i, dlt in enumerate(deltas):
                shifted = shifted_circuit_family(base, float(dlt), fam)
                shifted_qpy = qpy_bytes(shifted)
                curve[i] = mc_mean_parallel(
                    ex, shifted_qpy, n, ideal_vec,
                    gamma=float(args.gamma),
                    traj=int(args.traj),
                    chunk=int(args.chunk),
                    seed_base=seed_base,  # paired
                    blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim),
                    threshold=float(args.threshold),
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
            plt.title(f"Angle-shift scan: only {fam.upper()} (gamma={float(args.gamma):g}, paired)")
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
        plt.title(f"Angle-shift families: {', '.join([f.upper() for f in families])} (gamma={float(args.gamma):g}, paired)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "angle_shift_families_logx.png", dpi=200, bbox_inches="tight")
        plt.close()

    elapsed = time.time() - t0
    results["elapsed_sec"] = float(elapsed)
    results["workers_used"] = int(workers)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        f"gamma={float(args.gamma):g}, traj={int(args.traj)}",
        "",
        "Saved: angle_shift_families_logx.png",
        "Saved: angle_shift_only_rzz_logx.png (if rzz in families)",
        "Saved: angle_shift_only_rx_logx.png  (if rx  in families)",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
