# scripts/exp_compensation_mixed_strongsim_YAQS_WIN.py
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
from typing import Dict, List, Tuple

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
        """
        After each gate application site, independently insert X/Y/Z with prob gamma on each touched qubit.
        This "bakes" stochastic noise into the circuit (so YAQS sees a unitary circuit).
        """
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
    try:
        return MPS(length=n, state="zeros")
    except TypeError:
        return MPS(n)

def _make_strongsim_params(max_bond_dim: int, threshold: float):
    """
    Robust across YAQS versions:
    - Some versions use StrongSimParams(max_bond_dim=..., threshold=..., get_state=True)
    """
    from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
    sig = inspect.signature(StrongSimParams)
    params = sig.parameters

    kwargs = {}
    if "max_bond_dim" in params:
        kwargs["max_bond_dim"] = int(max_bond_dim)
    if "threshold" in params:
        kwargs["threshold"] = float(threshold)
    if "get_state" in params:
        kwargs["get_state"] = True
    if "show_progress" in params:
        kwargs["show_progress"] = False
    if "num_traj" in params:
        kwargs["num_traj"] = 1

    return StrongSimParams(**kwargs)

def run_strongsim_statevector(
    circ: QuantumCircuit,
    n: int,
    max_bond_dim: int,
    threshold: float,
) -> np.ndarray:
    """
    YAQS strong circuit simulation, NO noise_model (must bake noise into circuit).
    Returns final statevector (dense) converted from MPS.
    """
    from mqt.yaqs import simulator

    state = _make_mps(int(n))
    sim_params = _make_strongsim_params(int(max_bond_dim), float(threshold))

    # YAQS examples: simulator.run(state, circ, sim_params, noise_model=None)
    try:
        simulator.run(state, circ, sim_params, noise_model=None, parallel=False)
    except TypeError:
        simulator.run(state, circ, sim_params, noise_model=None)

    out_state = getattr(sim_params, "output_state", None)
    if out_state is None:
        out_state = state
    return mps_to_statevector(out_state)

# -----------------------------
# Coherent drift + compilation compensation
# -----------------------------
def apply_param_offsets(
    circ: QuantumCircuit,
    offset_rx: float,
    offset_rzz: float,
) -> QuantumCircuit:
    """
    Return a new circuit where:
      RX(theta)  -> RX(theta + offset_rx)
      RZZ(theta) -> RZZ(theta + offset_rzz)
    Other gates unchanged.
    """
    out = QuantumCircuit(circ.num_qubits)
    for ci in circ.data:
        op = ci.operation
        if op.name in ("measure", "barrier", "reset"):
            continue

        name = op.name.lower()
        if name == "rx" and len(op.params) == 1:
            theta = float(op.params[0]) + float(offset_rx)
            out.rx(theta, ci.qubits[0])
        elif name == "rzz" and len(op.params) == 1:
            theta = float(op.params[0]) + float(offset_rzz)
            out.rzz(theta, ci.qubits[0], ci.qubits[1])
        else:
            out.append(op, list(ci.qubits), list(ci.clbits))
    return out

# -----------------------------
# Paired seeds across all candidates
# -----------------------------
def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * shot_index) % (2**31 - 1))

# -----------------------------
# MC worker: for a given (delta_rx, delta_rzz), run trajectories and return fidelities
# -----------------------------
def _chunk_fidelities(
    base_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    start: int,
    shots: int,
    seed_base: int,
    blas_threads: int,
    # YAQS params
    max_bond_dim: int,
    threshold: float,
    # coherent error + compensation
    eps_rx: float,
    eps_rzz: float,
    delta_rx: float,
    delta_rzz: float,
) -> np.ndarray:
    _worker_init(int(blas_threads))

    base = strip_measurements(qpy_load_bytes(base_qpy))

    # "compiled" compensation (delta) then "hardware drift" (eps)
    # net effect = theta + delta + eps
    net_rx = float(delta_rx) + float(eps_rx)
    net_rzz = float(delta_rzz) + float(eps_rzz)
    drifted = apply_param_offsets(base, offset_rx=net_rx, offset_rzz=net_rzz)

    out = np.empty(int(shots), dtype=np.float64)
    for i in range(int(shots)):
        global_idx = int(start + i)
        s = shot_seed(int(seed_base), global_idx)

        noisy, _ = apply_pauli_jump_after_each_gate(drifted, float(gamma), int(s))
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
    base_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    eps_rx: float,
    eps_rzz: float,
    delta_rx: float,
    delta_rzz: float,
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
                base_qpy, n, psi_ideal, float(gamma),
                int(start), int(shots), int(seed_base),
                int(blas_threads), int(max_bond_dim), float(threshold),
                float(eps_rx), float(eps_rzz),
                float(delta_rx), float(delta_rzz),
            )
        )

    s = 0.0
    cnt = 0
    for f in futs:
        arr = f.result()
        s += float(np.sum(arr))
        cnt += int(arr.size)
    return s / max(1, cnt)

def mc_pool_parallel(
    ex,
    base_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    eps_rx: float,
    eps_rzz: float,
    delta_rx: float,
    delta_rzz: float,
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
                base_qpy, n, psi_ideal, float(gamma),
                int(start), int(shots), int(seed_base),
                int(blas_threads), int(max_bond_dim), float(threshold),
                float(eps_rx), float(eps_rzz),
                float(delta_rx), float(delta_rzz),
            )
        )

    parts = [f.result() for f in futs]
    return np.concatenate(parts, axis=0)

# -----------------------------
# Convergence curve from pool (smooth)
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
# Plot helpers
# -----------------------------
def plot_heatmap(
    Z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    out_path: Path,
    baseline_f: float,
    best_xyf: Tuple[float, float, float],
) -> None:
    # Z shape = (len(ys), len(xs)) where rows correspond to ys
    plt.figure()
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
    )
    plt.colorbar(label="mean fidelity")
    bx, by, bf = best_xyf
    plt.scatter([bx], [by], marker="x", s=120, linewidths=2)
    txt = (
        f"baseline (δrx=0, δrzz=0): F={baseline_f:.6f}\n"
        f"best: δrx={bx:+.4f}, δrzz={by:+.4f}\n"
        f"F_best={bf:.6f}, ΔF={bf-baseline_f:+.3e}"
    )
    plt.gca().text(
        0.02, 0.02, txt,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.25),
        va="bottom",
    )
    plt.xlabel("delta_rx (rad)")
    plt.ylabel("delta_rzz (rad)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # circuit
    ap.add_argument("--num-qubits", type=int, default=10)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # YAQS StrongSim
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)

    # stochastic Pauli noise strength sweep
    ap.add_argument("--gamma-min", type=float, default=1e-3)
    ap.add_argument("--gamma-max", type=float, default=1e-1)
    ap.add_argument("--gamma-num", type=int, default=60)

    # drift (coherent, systematic) — THIS is what makes angle-shift compensation meaningful
    ap.add_argument("--eps-rx", type=float, default=0.02, help="systematic RX over-rotation (rad)")
    ap.add_argument("--eps-rzz", type=float, default=0.02, help="systematic RZZ over-rotation (rad)")

    # grid search gamma (pick one gamma to search (δrx, δrzz))
    ap.add_argument("--grid-gamma", type=float, default=1e-2)

    # MC budgets
    ap.add_argument("--grid-traj", type=int, default=600)
    ap.add_argument("--baseline-traj", type=int, default=2000)
    ap.add_argument("--sweep-traj", type=int, default=400)

    # convergence
    ap.add_argument("--conv-gammas", type=str, default="1e-3,1e-2,1e-1")
    ap.add_argument("--conv-max-traj", type=int, default=10000)
    ap.add_argument("--conv-resamples", type=int, default=200)
    ap.add_argument("--conv-Ns", type=str, default="10,20,30,40,50,60,70,80,90,100,200,300,500,800,1000,2000,5000")

    # grid (coarse)
    ap.add_argument("--grid-points", type=int, default=9)
    ap.add_argument("--grid-span", type=float, default=0.06,
                    help="grid span around center (rad), range=[center-span, center+span]")

    # refine around best
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--refine-points", type=int, default=11)
    ap.add_argument("--refine-span", type=float, default=0.02)

    # parallel
    ap.add_argument("--workers", type=int, default=0)     # 0 => all logical cores
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=50)

    # transpile
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

    # Build base circuit
    base = build_twolocal(num_qubits=n, depth=int(args.depth), seed=int(args.seed), add_measurements=False)
    base = strip_measurements(base)

    if int(args.use_transpile) == 1:
        base = transpile(
            base,
            optimization_level=int(args.opt_level),
            seed_transpiler=int(args.seed_transpiler),
        )
        base = strip_measurements(base)

    # Print YAQS signature (so terminal proves it's used)
    try:
        import mqt.yaqs
        from mqt.yaqs import simulator
        from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
        print("[YAQS]", getattr(mqt.yaqs, "__version__", "unknown"))
        print("[YAQS] simulator =", getattr(simulator, "__file__", "unknown"))
        print("[YAQS] StrongSimParams =", inspect.signature(StrongSimParams), flush=True)
    except Exception as e:
        raise RuntimeError(f"YAQS import failed in this venv: {e}")

    # Ideal target: intended circuit, no drift, no stochastic noise
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
    )

    out_dir = REPO_ROOT / "outputs" / "experiments" / "compensation_mixed_strongsim" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
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
        "yaqs": {"max_bond_dim": int(args.max_bond_dim), "threshold": float(args.threshold)},
        "noise": {
            "pauli_jump": True,
            "eps_rx": float(args.eps_rx),
            "eps_rzz": float(args.eps_rzz),
        },
        "grid": {
            "gamma": float(args.grid_gamma),
            "grid_traj": int(args.grid_traj),
            "baseline_traj": int(args.baseline_traj),
            "points": int(args.grid_points),
            "span": float(args.grid_span),
            "refine": int(args.refine),
            "refine_points": int(args.refine_points),
            "refine_span": float(args.refine_span),
        }
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_qpy = qpy_bytes(base)

    from concurrent.futures import ProcessPoolExecutor

    seed_base = int(args.seed + 20251220)  # paired across all candidates

    t0 = time.time()
    results: Dict[str, object] = {"meta": meta}

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        # ==========================================================
        # A) Baseline at grid gamma (delta=0,0), with drift + stochastic noise
        # ==========================================================
        baseline_f = mc_mean_parallel(
            ex,
            base_qpy, n, ideal_vec,
            gamma=float(args.grid_gamma),
            traj=int(args.baseline_traj),
            chunk=int(args.chunk),
            seed_base=seed_base,
            blas_threads=int(args.blas_threads),
            max_bond_dim=int(args.max_bond_dim),
            threshold=float(args.threshold),
            eps_rx=float(args.eps_rx), eps_rzz=float(args.eps_rzz),
            delta_rx=0.0, delta_rzz=0.0,
        )

        # ==========================================================
        # B) Coarse grid around the expected optimum: delta ~ -eps
        # ==========================================================
        center_rx = -float(args.eps_rx)
        center_rzz = -float(args.eps_rzz)
        span = float(args.grid_span)
        P = int(args.grid_points)

        xs = np.linspace(center_rx - span, center_rx + span, P)  # delta_rx
        ys = np.linspace(center_rzz - span, center_rzz + span, P)  # delta_rzz

        Z = np.empty((len(ys), len(xs)), dtype=np.float64)

        best = (0.0, 0.0, baseline_f)  # (delta_rx, delta_rzz, fidelity)
        for iy, drzz in enumerate(ys):
            for ix, drx in enumerate(xs):
                f = mc_mean_parallel(
                    ex,
                    base_qpy, n, ideal_vec,
                    gamma=float(args.grid_gamma),
                    traj=int(args.grid_traj),
                    chunk=int(args.chunk),
                    seed_base=seed_base,  # paired seeds across grid
                    blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim),
                    threshold=float(args.threshold),
                    eps_rx=float(args.eps_rx), eps_rzz=float(args.eps_rzz),
                    delta_rx=float(drx), delta_rzz=float(drzz),
                )
                Z[iy, ix] = f
                if f > best[2]:
                    best = (float(drx), float(drzz), float(f))

        results["grid_coarse"] = {
            "xs_delta_rx": xs.tolist(),
            "ys_delta_rzz": ys.tolist(),
            "Z_fidelity": Z.tolist(),
            "baseline_fidelity": float(baseline_f),
            "best_delta_rx": best[0],
            "best_delta_rzz": best[1],
            "best_fidelity": best[2],
            "deltaF_best_minus_base": float(best[2] - baseline_f),
        }

        plot_heatmap(
            Z=Z, xs=xs, ys=ys,
            title=f"Coarse compensation grid (gamma={float(args.grid_gamma):g})",
            out_path=out_dir / "compensation_heatmap_coarse.png",
            baseline_f=float(baseline_f),
            best_xyf=best,
        )

        # ==========================================================
        # C) Refine grid around coarse best
        # ==========================================================
        best2 = best
        if int(args.refine) == 1:
            span2 = float(args.refine_span)
            P2 = int(args.refine_points)
            xs2 = np.linspace(best[0] - span2, best[0] + span2, P2)
            ys2 = np.linspace(best[1] - span2, best[1] + span2, P2)
            Z2 = np.empty((len(ys2), len(xs2)), dtype=np.float64)

            best2 = best
            for iy, drzz in enumerate(ys2):
                for ix, drx in enumerate(xs2):
                    f = mc_mean_parallel(
                        ex,
                        base_qpy, n, ideal_vec,
                        gamma=float(args.grid_gamma),
                        traj=int(args.grid_traj),
                        chunk=int(args.chunk),
                        seed_base=seed_base,  # paired
                        blas_threads=int(args.blas_threads),
                        max_bond_dim=int(args.max_bond_dim),
                        threshold=float(args.threshold),
                        eps_rx=float(args.eps_rx), eps_rzz=float(args.eps_rzz),
                        delta_rx=float(drx), delta_rzz=float(drzz),
                    )
                    Z2[iy, ix] = f
                    if f > best2[2]:
                        best2 = (float(drx), float(drzz), float(f))

            results["grid_refine"] = {
                "xs_delta_rx": xs2.tolist(),
                "ys_delta_rzz": ys2.tolist(),
                "Z_fidelity": Z2.tolist(),
                "best_delta_rx": best2[0],
                "best_delta_rzz": best2[1],
                "best_fidelity": best2[2],
                "deltaF_best_minus_base": float(best2[2] - baseline_f),
            }

            plot_heatmap(
                Z=Z2, xs=xs2, ys=ys2,
                title=f"Refined compensation grid (gamma={float(args.grid_gamma):g})",
                out_path=out_dir / "compensation_heatmap_refine.png",
                baseline_f=float(baseline_f),
                best_xyf=best2,
            )

        # ==========================================================
        # D) Error vs gamma: baseline vs best (fixed best deltas)
        # ==========================================================
        gammas = np.logspace(np.log10(float(args.gamma_min)), np.log10(float(args.gamma_max)), int(args.gamma_num))

        base_curve = np.empty_like(gammas, dtype=np.float64)
        best_curve = np.empty_like(gammas, dtype=np.float64)

        for i, g in enumerate(gammas):
            base_curve[i] = mc_mean_parallel(
                ex, base_qpy, n, ideal_vec,
                gamma=float(g),
                traj=int(args.sweep_traj),
                chunk=int(args.chunk),
                seed_base=seed_base,
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                eps_rx=float(args.eps_rx), eps_rzz=float(args.eps_rzz),
                delta_rx=0.0, delta_rzz=0.0,
            )
            best_curve[i] = mc_mean_parallel(
                ex, base_qpy, n, ideal_vec,
                gamma=float(g),
                traj=int(args.sweep_traj),
                chunk=int(args.chunk),
                seed_base=seed_base,  # paired
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                eps_rx=float(args.eps_rx), eps_rzz=float(args.eps_rzz),
                delta_rx=float(best2[0]), delta_rzz=float(best2[1]),
            )

        results["error_vs_gamma"] = {
            "gamma": gammas.tolist(),
            "baseline_mean_fidelity": base_curve.tolist(),
            "best_mean_fidelity": best_curve.tolist(),
            "best_delta_rx": float(best2[0]),
            "best_delta_rzz": float(best2[1]),
        }

        plt.figure()
        err_base = 1.0 - base_curve
        err_best = 1.0 - best_curve
        plt.plot(gammas, err_base, marker="o", markersize=3, label="baseline (δ=0,0)")
        plt.plot(gammas, err_best, marker="o", markersize=3, label=f"best fixed (δrx={best2[0]:+.4f}, δrzz={best2[1]:+.4f})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("gamma (log)")
        plt.ylabel("error = 1 - mean fidelity (log)")
        txt = (
            f"drift: eps_rx={float(args.eps_rx):+.4f}, eps_rzz={float(args.eps_rzz):+.4f}\n"
            f"grid gamma={float(args.grid_gamma):g}\n"
            f"best @ grid: δrx={best2[0]:+.4f}, δrzz={best2[1]:+.4f}"
        )
        plt.gca().text(
            0.02, 0.02, txt,
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.25),
            va="bottom",
        )
        plt.title("Error vs noise strength: baseline vs best-compensated (YAQS StrongSim)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "error_vs_gamma_baseline_vs_best_loglog.png", dpi=200, bbox_inches="tight")
        plt.close()

        # ==========================================================
        # E) MC convergence (smooth curve via resampling pool)
        # ==========================================================
        conv_gammas = [float(x) for x in args.conv_gammas.split(",") if x.strip()]
        Ns = sorted(set(int(x) for x in args.conv_Ns.split(",") if x.strip() and int(x) > 0))

        conv_data = {}
        for g in conv_gammas:
            pool = mc_pool_parallel(
                ex, base_qpy, n, ideal_vec,
                gamma=float(g),
                traj=int(args.conv_max_traj),
                chunk=int(args.chunk),
                seed_base=seed_base,
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                eps_rx=float(args.eps_rx), eps_rzz=float(args.eps_rzz),
                delta_rx=0.0, delta_rzz=0.0,
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
                "reference_mean_fidelity": ref,
                "Ns": Ns,
                "mean_abs_error": mean_err.tolist(),
                "std_abs_error": std_err.tolist(),
            }

        results["mc_convergence"] = conv_data

        plt.figure()
        for g in conv_gammas:
            d = conv_data[str(g)]
            x = np.asarray(d["Ns"], dtype=float)
            y = np.asarray(d["mean_abs_error"], dtype=float)
            plt.plot(x, y, marker="o", label=f"gamma={g:g}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("N trajectories (log)")
        plt.ylabel("E[ |mean_N - reference| ] (log)")
        plt.title("MC convergence (baseline, drift+Pauli baked, YAQS StrongSim)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "mc_convergence_loglog.png", dpi=200, bbox_inches="tight")
        plt.close()

    elapsed = time.time() - t0
    results["elapsed_sec"] = float(elapsed)
    results["workers_used"] = int(workers)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        "",
        f"drift eps_rx={float(args.eps_rx):+.6f}, eps_rzz={float(args.eps_rzz):+.6f}",
        f"grid gamma={float(args.grid_gamma):g}, baseline_traj={int(args.baseline_traj)}, grid_traj={int(args.grid_traj)}",
        "",
        "Saved: compensation_heatmap_coarse.png",
        "Saved: compensation_heatmap_refine.png (if --refine 1)",
        "Saved: error_vs_gamma_baseline_vs_best_loglog.png",
        "Saved: mc_convergence_loglog.png",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
