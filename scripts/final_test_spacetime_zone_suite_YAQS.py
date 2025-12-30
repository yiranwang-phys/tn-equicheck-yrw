# scripts/final_test_spacetime_zone_suite_YAQS.py
from __future__ import annotations

# ==========================================================
# CRITICAL: Force non-GUI Matplotlib backend (Windows spawn safe)
# Must be set BEFORE importing matplotlib.pyplot
# ==========================================================
import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import io
import json
import platform
import sys
import time
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
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
        rng = np.random.default_rng(int(seed))
        qc = QuantumCircuit(num_qubits)
        for _ in range(int(depth)):
            # 1Q RX layer
            for i in range(num_qubits):
                qc.rx(float(rng.uniform(0, 2 * np.pi)), i)
            # 2Q RZZ ring
            for i in range(num_qubits - 1):
                qc.rzz(float(rng.uniform(0, 2 * np.pi)), i, i + 1)
            if num_qubits > 2:
                qc.rzz(float(rng.uniform(0, 2 * np.pi)), num_qubits - 1, 0)
        if add_measurements:
            qc.measure_all()
        return qc


# -----------------------------
# Small utilities
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
    _set_thread_env(int(blas_threads))
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


def _qb_index(qb) -> int:
    # Qiskit versions differ
    if hasattr(qb, "index"):
        return int(qb.index)
    return int(getattr(qb, "_index"))


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
    from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
    sig = inspect.signature(StrongSimParams)
    params = sig.parameters

    kwargs = {}
    if "max_bond_dim" in params:
        kwargs["max_bond_dim"] = int(max_bond_dim)
    elif "maxBondDim" in params:
        kwargs["maxBondDim"] = int(max_bond_dim)

    if "threshold" in params:
        kwargs["threshold"] = float(threshold)
    elif "svd_cut" in params:
        kwargs["svd_cut"] = float(threshold)
    elif "svdCut" in params:
        kwargs["svdCut"] = float(threshold)

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
    YAQS strong circuit simulation (no noise_model).
    Noise must be baked into the circuit as explicit Pauli gates per trajectory.
    """
    from mqt.yaqs import simulator
    state = _make_mps(int(n))
    sim_params = _make_strongsim_params(int(max_bond_dim), float(threshold))

    try:
        simulator.run(state, circ, sim_params, noise_model=None, parallel=False)
    except TypeError:
        simulator.run(state, circ, sim_params, noise_model=None)

    out_state = getattr(sim_params, "output_state", None)
    if out_state is None:
        out_state = state
    return mps_to_statevector(out_state)


# -----------------------------
# Masks helpers (pack bits for multiprocessing)
# -----------------------------
def pack_mask(mask_bool: np.ndarray) -> bytes:
    mask_bool = np.asarray(mask_bool, dtype=np.uint8)
    return np.packbits(mask_bool).tobytes()


def unpack_mask(mask_bytes: bytes, length: int) -> np.ndarray:
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    bits = np.unpackbits(arr)[:int(length)]
    return bits.astype(bool, copy=False)


def count_steps(circ: QuantumCircuit) -> int:
    m = 0
    for ci in circ.data:
        name = ci.operation.name.lower()
        if name in ("measure", "barrier", "reset"):
            continue
        m += 1
    return m


def make_time_mask_prefix(m_steps: int, frac: float) -> np.ndarray:
    frac = float(frac)
    frac = min(max(frac, 0.0), 1.0)
    k = int(round(frac * int(m_steps)))
    mask = np.zeros(int(m_steps), dtype=bool)
    mask[:k] = True
    return mask


def make_time_mask_window(m_steps: int, start_frac: float, end_frac: float) -> np.ndarray:
    a = min(max(float(start_frac), 0.0), 1.0)
    b = min(max(float(end_frac), 0.0), 1.0)
    if b < a:
        a, b = b, a
    i0 = int(round(a * int(m_steps)))
    i1 = int(round(b * int(m_steps)))
    mask = np.zeros(int(m_steps), dtype=bool)
    mask[i0:i1] = True
    return mask


def make_qubit_mask_prefix(n: int, k: int) -> np.ndarray:
    k = int(k)
    mask = np.zeros(int(n), dtype=bool)
    mask[: max(0, min(int(n), k))] = True
    return mask


def make_qubit_mask_random(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    k = int(k)
    mask = np.zeros(int(n), dtype=bool)
    k = max(0, min(int(n), k))
    if k > 0:
        idx = rng.choice(int(n), size=k, replace=False)
        mask[idx] = True
    return mask


def count_active_sites(circ: QuantumCircuit, time_mask: np.ndarray, qubit_mask: np.ndarray) -> int:
    m = 0
    S = 0
    for ci in circ.data:
        op = ci.operation
        name = op.name.lower()
        if name in ("measure", "barrier", "reset"):
            continue
        if time_mask[m]:
            for qb in list(ci.qubits):
                if qubit_mask[_qb_index(qb)]:
                    S += 1
        m += 1
    return int(S)


# -----------------------------
# Noise injection: Pauli jump only AFTER gate, only inside spacetime zone
# gamma is jump probability per active (gate, touched qubit)
# -----------------------------
def bake_pauli_jump_spacetime_zone(
    circuit: QuantumCircuit,
    gamma: float,
    seed: int,
    time_mask: np.ndarray,
    qubit_mask: np.ndarray,
) -> QuantumCircuit:
    rng = np.random.default_rng(int(seed))
    noisy = QuantumCircuit(circuit.num_qubits)

    m = 0  # step index over non-(measure/barrier/reset) ops
    for ci in circuit.data:
        op = ci.operation
        name = op.name.lower()
        if name in ("measure", "barrier", "reset"):
            continue

        qargs = list(ci.qubits)
        cargs = list(ci.clbits)
        noisy.append(op, qargs, cargs)

        if time_mask[m]:
            for qb in qargs:
                q = _qb_index(qb)
                if qubit_mask[q]:
                    if rng.random() < float(gamma):
                        pa = int(rng.integers(0, 3))
                        if pa == 0:
                            noisy.x(qb)
                        elif pa == 1:
                            noisy.y(qb)
                        else:
                            noisy.z(qb)

        m += 1

    return noisy


# -----------------------------
# Paired seeds across variants
# -----------------------------
def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * shot_index) % (2**31 - 1))


# -----------------------------
# MC chunk worker
# -----------------------------
def _chunk_fidelities_zone(
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
    time_mask_bytes: bytes,
    time_len: int,
    qubit_mask_bytes: bytes,
    qubit_len: int,
) -> np.ndarray:
    _worker_init(int(blas_threads))

    base = strip_measurements(qpy_load_bytes(circ_qpy))
    time_mask = unpack_mask(time_mask_bytes, int(time_len))
    qubit_mask = unpack_mask(qubit_mask_bytes, int(qubit_len))

    out = np.empty(int(shots), dtype=np.float64)
    for i in range(int(shots)):
        global_idx = int(start + i)
        s = shot_seed(int(seed_base), global_idx)

        noisy = bake_pauli_jump_spacetime_zone(
            base, float(gamma), int(s),
            time_mask=time_mask,
            qubit_mask=qubit_mask,
        )

        vec = run_strongsim_statevector(
            noisy,
            n=int(n),
            max_bond_dim=int(max_bond_dim),
            threshold=float(threshold),
        )
        out[i] = fidelity_pure(psi_ideal, vec)

    return out


def mc_mean_std_parallel(
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
    time_mask: np.ndarray,
    qubit_mask: np.ndarray,
) -> Tuple[float, float]:
    traj = int(traj)
    chunk = int(max(1, chunk))
    n_chunks = (traj + chunk - 1) // chunk

    time_mask_bytes = pack_mask(time_mask.astype(bool))
    qubit_mask_bytes = pack_mask(qubit_mask.astype(bool))

    futs = []
    for k in range(n_chunks):
        start = k * chunk
        shots = min(chunk, traj - start)
        futs.append(
            ex.submit(
                _chunk_fidelities_zone,
                circ_qpy, n, psi_ideal, float(gamma),
                int(start), int(shots), int(seed_base),
                int(blas_threads), int(max_bond_dim), float(threshold),
                time_mask_bytes, int(time_mask.size),
                qubit_mask_bytes, int(qubit_mask.size),
            )
        )

    all_vals = []
    for f in futs:
        arr = f.result()
        all_vals.append(arr)

    vals = np.concatenate(all_vals) if all_vals else np.array([], dtype=float)
    if vals.size == 0:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if vals.size >= 2 else 0.0)


# -----------------------------
# Plot helpers
# -----------------------------
def plot_line_with_band(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    ystd: Optional[np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    labels: Optional[List[str]] = None,
) -> None:
    plt.figure()
    if y.ndim == 1:
        plt.plot(x, y, marker="o", markersize=3)
        if ystd is not None:
            plt.fill_between(x, y - ystd, y + ystd, alpha=0.2)
    else:
        # multiple curves in y: shape (C, len(x))
        C = y.shape[0]
        for c in range(C):
            lab = labels[c] if (labels is not None and c < len(labels)) else f"curve{c}"
            plt.plot(x, y[c], marker="o", markersize=3, label=lab)
            if ystd is not None:
                plt.fill_between(x, y[c] - ystd[c], y[c] + ystd[c], alpha=0.15)
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_heatmap(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure()
    # Z shape: (len(y), len(x))
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
    )
    plt.colorbar(label="error = 1 - mean fidelity")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # tasks
    ap.add_argument("--task", type=str, default="all",
                    help="time | space | grid | all")

    # circuit
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # YAQS strongsim
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)

    # MC
    ap.add_argument("--traj", type=int, default=2000)   # DON'T reduce (per your requirement)
    ap.add_argument("--chunk", type=int, default=50)

    # gamma
    ap.add_argument("--gamma", type=float, default=1e-2)

    # time zone
    ap.add_argument("--time-mode", type=str, default="prefix", help="prefix | window")
    ap.add_argument("--time-fracs", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--time-window", type=str, default="0.25,0.75", help="start_frac,end_frac (used if time-mode=window)")

    # space zone
    ap.add_argument("--space-mode", type=str, default="both", help="prefix | random | both")
    ap.add_argument("--space-K-list", type=str, default="", help="e.g. 0,1,2,3,4,5,6  (empty => 0..n)")
    ap.add_argument("--space-random-reps", type=int, default=5)
    ap.add_argument("--space-random-seed-base", type=int, default=20251230)

    # parallel
    ap.add_argument("--workers", type=int, default=0)      # 0 => all logical cores
    ap.add_argument("--blas-threads", type=int, default=1)

    # transpile
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    args = ap.parse_args()

    task = args.task.strip().lower()
    do_time = task in ("time", "all")
    do_space = task in ("space", "all")
    do_grid = task in ("grid", "all")

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

    # Ideal
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
    )

    m_steps = count_steps(base)
    gamma = float(args.gamma)

    # Parse lists
    time_fracs = [float(x) for x in args.time_fracs.split(",") if x.strip()]
    if not time_fracs:
        time_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]

    if args.space_K_list.strip():
        space_K_list = [int(x) for x in args.space_K_list.split(",") if x.strip()]
    else:
        space_K_list = list(range(0, n + 1))

    space_K_list = sorted(list(set(max(0, min(n, int(k))) for k in space_K_list)))

    # Output dir
    out_dir = REPO_ROOT / "outputs" / "experiments" / "final_test_spacetime_zone_suite" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_base = int(args.seed + 20251220)  # paired across all zone settings

    meta: Dict[str, object] = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "task": task,
        "workers": workers,
        "blas_threads": int(args.blas_threads),
        "chunk": int(args.chunk),
        "n": n,
        "depth": int(args.depth),
        "seed": int(args.seed),
        "yaqs": {"max_bond_dim": int(args.max_bond_dim), "threshold": float(args.threshold)},
        "noise": {
            "kind": "pauli_jump_xyz_after_gate_only",
            "gamma": gamma,
            "site_definition": "active (step, touched_qubit) only",
        },
        "traj": int(args.traj),
        "time_mode": args.time_mode,
        "time_fracs": time_fracs,
        "time_window": args.time_window,
        "space_mode": args.space_mode,
        "space_K_list": space_K_list,
        "space_random_reps": int(args.space_random_reps),
        "space_random_seed_base": int(args.space_random_seed_base),
        "paired_seed_base": seed_base,
        "m_steps": int(m_steps),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_qpy = qpy_bytes(base)

    from concurrent.futures import ProcessPoolExecutor

    results: Dict[str, object] = {"meta": meta}

    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        # -------------------------
        # TIME sweep (full space)
        # -------------------------
        if do_time:
            time_mode = args.time_mode.strip().lower()
            tw = [float(x) for x in args.time_window.split(",") if x.strip()]
            if len(tw) != 2:
                tw = [0.25, 0.75]

            qubit_mask_all = np.ones(n, dtype=bool)

            meanF = []
            stdF = []
            activeS = []

            for f in time_fracs:
                if time_mode == "window":
                    time_mask = make_time_mask_window(m_steps, tw[0], tw[1])
                else:
                    time_mask = make_time_mask_prefix(m_steps, f)

                S = count_active_sites(base, time_mask=time_mask, qubit_mask=qubit_mask_all)
                mu, sd = mc_mean_std_parallel(
                    ex, base_qpy, n, ideal_vec,
                    gamma=gamma,
                    traj=int(args.traj),
                    chunk=int(args.chunk),
                    seed_base=seed_base,
                    blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim),
                    threshold=float(args.threshold),
                    time_mask=time_mask,
                    qubit_mask=qubit_mask_all,
                )
                meanF.append(mu)
                stdF.append(sd)
                activeS.append(S)
                print(f"[TIME] frac={f:.3f}  active_sites={S}  meanF={mu:.6f}  std={sd:.6f}", flush=True)

            meanF = np.array(meanF, dtype=float)
            stdF = np.array(stdF, dtype=float)
            activeS = np.array(activeS, dtype=int)

            results["time_sweep"] = {
                "time_fracs": time_fracs,
                "active_sites": activeS.tolist(),
                "mean_fidelity": meanF.tolist(),
                "std_fidelity": stdF.tolist(),
                "error": (1.0 - meanF).tolist(),
            }

            plot_line_with_band(
                out_dir / "time_sweep_error_vs_timefrac.png",
                x=np.array(time_fracs, dtype=float),
                y=(1.0 - meanF),
                ystd=stdF,
                xlabel="time fraction (prefix steps)",
                ylabel="error = 1 - mean fidelity",
                title=f"Time-zone sweep (gamma={gamma:g}, traj={int(args.traj)})",
            )

            plot_line_with_band(
                out_dir / "time_sweep_error_vs_active_sites.png",
                x=activeS.astype(float),
                y=(1.0 - meanF),
                ystd=stdF,
                xlabel="active sites S (count of active (step,qubit))",
                ylabel="error = 1 - mean fidelity",
                title=f"Time-zone sweep by active sites (gamma={gamma:g}, traj={int(args.traj)})",
            )

            # collapse-like x = gamma*S
            plot_line_with_band(
                out_dir / "time_sweep_collapse_error_vs_gammaS.png",
                x=(gamma * activeS.astype(float)),
                y=(1.0 - meanF),
                ystd=stdF,
                xlabel="gamma * S",
                ylabel="error = 1 - mean fidelity",
                title=f"Collapse test (time-zone): error vs gamma*S",
            )

        # -------------------------
        # SPACE sweep (full time)
        # -------------------------
        if do_space:
            time_mask_all = np.ones(m_steps, dtype=bool)

            space_mode = args.space_mode.strip().lower()
            want_prefix = space_mode in ("prefix", "both")
            want_random = space_mode in ("random", "both")

            # prefix curve
            if want_prefix:
                meanF_p = []
                stdF_p = []
                activeS_p = []
                for K in space_K_list:
                    qubit_mask = make_qubit_mask_prefix(n, K)
                    S = count_active_sites(base, time_mask=time_mask_all, qubit_mask=qubit_mask)

                    mu, sd = mc_mean_std_parallel(
                        ex, base_qpy, n, ideal_vec,
                        gamma=gamma,
                        traj=int(args.traj),
                        chunk=int(args.chunk),
                        seed_base=seed_base,
                        blas_threads=int(args.blas_threads),
                        max_bond_dim=int(args.max_bond_dim),
                        threshold=float(args.threshold),
                        time_mask=time_mask_all,
                        qubit_mask=qubit_mask,
                    )
                    meanF_p.append(mu)
                    stdF_p.append(sd)
                    activeS_p.append(S)
                    print(f"[SPACE:prefix] K={K}  active_sites={S}  meanF={mu:.6f}  std={sd:.6f}", flush=True)

                meanF_p = np.array(meanF_p, dtype=float)
                stdF_p = np.array(stdF_p, dtype=float)
                activeS_p = np.array(activeS_p, dtype=int)

                results["space_sweep_prefix"] = {
                    "K_list": space_K_list,
                    "active_sites": activeS_p.tolist(),
                    "mean_fidelity": meanF_p.tolist(),
                    "std_fidelity": stdF_p.tolist(),
                    "error": (1.0 - meanF_p).tolist(),
                }

                plot_line_with_band(
                    out_dir / "space_sweep_prefix_error_vs_K.png",
                    x=np.array(space_K_list, dtype=float),
                    y=(1.0 - meanF_p),
                    ystd=stdF_p,
                    xlabel="active qubits K (prefix {0..K-1})",
                    ylabel="error = 1 - mean fidelity",
                    title=f"Space-zone sweep (prefix), gamma={gamma:g}, traj={int(args.traj)}",
                )

                plot_line_with_band(
                    out_dir / "space_sweep_prefix_collapse_error_vs_gammaS.png",
                    x=(gamma * activeS_p.astype(float)),
                    y=(1.0 - meanF_p),
                    ystd=stdF_p,
                    xlabel="gamma * S",
                    ylabel="error = 1 - mean fidelity",
                    title=f"Collapse test (space-zone prefix): error vs gamma*S",
                )

            # random curve (mean±std over reps)
            if want_random:
                reps = int(args.space_random_reps)
                base_seed = int(args.space_random_seed_base)

                meanF_r = []
                stdF_r = []
                activeS_r = []

                for K in space_K_list:
                    mus = []
                    sds = []
                    Ss = []
                    for r in range(reps):
                        rng = np.random.default_rng(base_seed + 1009 * r + 17 * K)
                        qubit_mask = make_qubit_mask_random(n, K, rng=rng)
                        S = count_active_sites(base, time_mask=time_mask_all, qubit_mask=qubit_mask)

                        mu, sd = mc_mean_std_parallel(
                            ex, base_qpy, n, ideal_vec,
                            gamma=gamma,
                            traj=int(args.traj),
                            chunk=int(args.chunk),
                            seed_base=seed_base,  # keep paired seeds
                            blas_threads=int(args.blas_threads),
                            max_bond_dim=int(args.max_bond_dim),
                            threshold=float(args.threshold),
                            time_mask=time_mask_all,
                            qubit_mask=qubit_mask,
                        )
                        mus.append(mu)
                        sds.append(sd)
                        Ss.append(S)

                    # average across masks (replicates)
                    meanF_r.append(float(np.mean(mus)))
                    stdF_r.append(float(np.std(mus, ddof=1) if len(mus) >= 2 else 0.0))
                    activeS_r.append(int(round(np.mean(Ss))))

                    print(f"[SPACE:random] K={K}  reps={reps}  "
                          f"meanF={meanF_r[-1]:.6f}  std_over_masks={stdF_r[-1]:.6f}", flush=True)

                meanF_r = np.array(meanF_r, dtype=float)
                stdF_r = np.array(stdF_r, dtype=float)
                activeS_r = np.array(activeS_r, dtype=int)

                results["space_sweep_random"] = {
                    "K_list": space_K_list,
                    "active_sites_mean": activeS_r.tolist(),
                    "mean_fidelity_mean_over_masks": meanF_r.tolist(),
                    "std_over_masks": stdF_r.tolist(),
                    "error": (1.0 - meanF_r).tolist(),
                }

                plot_line_with_band(
                    out_dir / "space_sweep_random_error_vs_K.png",
                    x=np.array(space_K_list, dtype=float),
                    y=(1.0 - meanF_r),
                    ystd=stdF_r,
                    xlabel="active qubits K (random subset)",
                    ylabel="error = 1 - mean fidelity",
                    title=f"Space-zone sweep (random masks), gamma={gamma:g}, traj={int(args.traj)}",
                )

                plot_line_with_band(
                    out_dir / "space_sweep_random_collapse_error_vs_gammaS.png",
                    x=(gamma * activeS_r.astype(float)),
                    y=(1.0 - meanF_r),
                    ystd=stdF_r,
                    xlabel="gamma * S (S averaged over random masks)",
                    ylabel="error = 1 - mean fidelity",
                    title=f"Collapse test (space-zone random): error vs gamma*S",
                )

        # -------------------------
        # SPACETIME grid (prefix time + prefix space)
        # -------------------------
        if do_grid:
            # Use prefix time fractions and prefix space (fast + deterministic)
            # Z[y_i, x_j] where y_i = K, x_j = time_frac
            Ks = np.array(space_K_list, dtype=int)
            Ts = np.array(time_fracs, dtype=float)

            Zerr = np.zeros((Ks.size, Ts.size), dtype=float)

            for iy, K in enumerate(Ks):
                qubit_mask = make_qubit_mask_prefix(n, int(K))
                for ix, f in enumerate(Ts):
                    time_mask = make_time_mask_prefix(m_steps, float(f))

                    mu, sd = mc_mean_std_parallel(
                        ex, base_qpy, n, ideal_vec,
                        gamma=gamma,
                        traj=int(args.traj),
                        chunk=int(args.chunk),
                        seed_base=seed_base,
                        blas_threads=int(args.blas_threads),
                        max_bond_dim=int(args.max_bond_dim),
                        threshold=float(args.threshold),
                        time_mask=time_mask,
                        qubit_mask=qubit_mask,
                    )
                    Zerr[iy, ix] = 1.0 - mu
                    print(f"[GRID] K={K}  time_frac={f:.3f}  error={Zerr[iy, ix]:.6f}  (std={sd:.6f})", flush=True)

            results["spacetime_grid_prefix_prefix"] = {
                "K_list": Ks.tolist(),
                "time_fracs": Ts.tolist(),
                "error_grid": Zerr.tolist(),
            }

            plot_heatmap(
                out_dir / "grid_error_heatmap_prefix_prefix.png",
                x=Ts,
                y=Ks.astype(float),
                Z=Zerr,
                xlabel="time fraction (prefix steps)",
                ylabel="active qubits K (prefix)",
                title=f"Spacetime-zone heatmap (gamma={gamma:g}, traj={int(args.traj)})",
            )

    elapsed = time.time() - t0
    results["elapsed_sec"] = float(elapsed)
    results["workers_used"] = int(workers)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"task={task}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        f"n={n}, depth={int(args.depth)}, seed={int(args.seed)}",
        f"gamma={gamma:g}, traj={int(args.traj)}",
        f"m_steps={m_steps}",
        "",
        "Saved: meta.json, results.json",
        "If task includes time: time_sweep_*.png",
        "If task includes space: space_sweep_*.png",
        "If task includes grid:  grid_error_heatmap_prefix_prefix.png",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
