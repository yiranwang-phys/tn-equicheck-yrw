# scripts/exp_updated_work_1_fidelity_strongsim_YAQS_WIN.py
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

# -----------------------------
# YAQS imports (correct style: import simulator module)
# -----------------------------
try:
    from mqt.yaqs import simulator  # official examples import this way
    from mqt.yaqs.core.data_structures.networks import MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
except Exception as e:
    raise RuntimeError(
        "Failed to import mqt.yaqs. Make sure YAQS is installed in this venv.\n"
        "Try: python -m pip install mqt.yaqs\n"
        f"Original import error: {e}"
    )

# ---- gate classes for safe param shifting ----
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate, RZZGate, RXXGate, RZXGate

# -----------------------------
# Make src/ importable (src-layout)
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

# -----------------------------
# QPY helpers
# -----------------------------
def qpy_bytes(circ: QuantumCircuit) -> bytes:
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    return buf.getvalue()

def qpy_load_bytes(b: bytes) -> QuantumCircuit:
    buf = io.BytesIO(b)
    return qpy.load(buf)[0]

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
# Circuit utils
# -----------------------------
def strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    return circ.remove_final_measurements(inplace=False)

def prep_basis_state_circuit(n: int, bitstring: str) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i, b in enumerate(bitstring):
        if b == "1":
            qc.x(i)
    return qc

# -----------------------------
# Noise model (explicit Pauli insertion) => still a deterministic unitary circuit per trajectory
# -----------------------------
def apply_pauli_jump_after_each_gate_unitary(circuit: QuantumCircuit, gamma: float, seed: int) -> QuantumCircuit:
    """
    After each non-measure gate, apply X/Y/Z on each touched qubit with probability gamma.
    This produces a *unitary* noisy circuit (one trajectory).
    """
    rng = np.random.default_rng(int(seed))
    noisy = QuantumCircuit(circuit.num_qubits)
    for inst, qargs, cargs in circuit.data:
        name = inst.name.lower()
        if name in ("measure", "barrier", "reset"):
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
    return noisy

# -----------------------------
# Angle shift (global +delta)
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
    delta = float(delta)
    out = QuantumCircuit(circ.num_qubits)
    for inst, qargs, cargs in circ.data:
        name = inst.name.lower()
        if name in ("measure", "barrier", "reset"):
            continue
        if name in _SHIFTABLE and len(inst.params) == 1:
            theta = float(inst.params[0]) + delta
            out.append(_SHIFTABLE[name](theta), qargs, cargs)
        else:
            out.append(inst, qargs, cargs)
    return out

# -----------------------------
# YAQS: MPS -> dense statevector (n<=~12 is fine)
# -----------------------------
def mps_to_statevector(mps: MPS) -> np.ndarray:
    """
    Contract MPS tensors into a dense vector.
    MPS tensors have index order (sigma, chi_left, chi_right). (YAQS docs) 
    """
    tensors = mps.tensors  # list of rank-3 arrays
    # first: (2,1,chi1) -> (2,chi1)
    psi = np.asarray(tensors[0])
    psi = psi[:, 0, :]  # (2, chi)
    for t in tensors[1:]:
        t = np.asarray(t)  # (2, chi_prev, chi_next)
        psi = np.tensordot(psi, t, axes=([1], [1]))  # (2^k, 2, chi_next)
        psi = psi.reshape(psi.shape[0] * psi.shape[1], psi.shape[2])  # (2^(k+1), chi_next)
    psi = psi[:, 0]  # last bond dim should be 1
    return psi.astype(np.complex128, copy=False)

def run_strongsim_statevector(
    circ: QuantumCircuit,
    n: int,
    max_bond_dim: int,
    svd_cut: float,
) -> np.ndarray:
    """
    Run YAQS strong circuit simulation (no noise_model) and return final statevector.
    """
    state = MPS(length=n, state="zeros")
    sim_params = StrongSimParams(
        num_traj=1,
        max_bond_dim=int(max_bond_dim),
        threshold=float(svd_cut),
        get_state=True,
        show_progress=False,
    )
    # YAQS examples use simulator.run(...) in this style. 
    # Keep parallel=False to avoid nested parallelism.
    try:
        simulator.run(state, circ, sim_params, noise_model=None, parallel=False)
    except TypeError:
        # older signature without 'parallel'
        simulator.run(state, circ, sim_params, noise_model=None)

    out_state = sim_params.output_state if sim_params.output_state is not None else state
    return mps_to_statevector(out_state)

def fidelity(ideal_vec: np.ndarray, vec: np.ndarray) -> float:
    amp = np.vdot(ideal_vec, vec)
    f = float(np.real(amp * np.conjugate(amp)))
    # numerical guard
    if f < 0.0:
        f = 0.0
    if f > 1.0 + 1e-9:
        f = 1.0
    return f

# -----------------------------
# Monte Carlo chunks (paired seeds via global index)
# -----------------------------
def _chunk_fidelities(
    circ_qpy: bytes,
    n: int,
    ideal_vec: np.ndarray,
    gamma: float,
    shots: int,
    start_index: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    svd_cut: float,
) -> np.ndarray:
    _worker_init(int(blas_threads))

    base = qpy_load_bytes(circ_qpy)
    out = np.empty(int(shots), dtype=np.float64)

    for i in range(int(shots)):
        seed = int(seed_base + start_index + i)  # paired across variants if same seed_base
        noisy = apply_pauli_jump_after_each_gate_unitary(base, float(gamma), seed)
        vec = run_strongsim_statevector(noisy, n=n, max_bond_dim=max_bond_dim, svd_cut=svd_cut)
        out[i] = fidelity(ideal_vec, vec)

    return out

def mc_pool_parallel(
    ex,
    circ_qpy: bytes,
    n: int,
    ideal_vec: np.ndarray,
    gamma: float,
    total_shots: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    svd_cut: float,
) -> np.ndarray:
    total_shots = int(total_shots)
    chunk = int(max(1, chunk))
    n_chunks = (total_shots + chunk - 1) // chunk

    futs = []
    for k in range(n_chunks):
        start = k * chunk
        s = min(chunk, total_shots - start)
        futs.append(
            ex.submit(
                _chunk_fidelities,
                circ_qpy, n, ideal_vec,
                float(gamma), int(s), int(start), int(seed_base),
                int(blas_threads), int(max_bond_dim), float(svd_cut),
            )
        )
    parts = [f.result() for f in futs]
    return np.concatenate(parts, axis=0)

def mc_mean_parallel(*args, **kwargs) -> float:
    pool = mc_pool_parallel(*args, **kwargs)
    return float(np.mean(pool))

# -----------------------------
# Convergence curve (random subsets from pool)
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
            if N <= M:
                idx = rng.choice(M, size=N, replace=False)
            else:
                idx = rng.integers(0, M, size=N)
            m = float(np.mean(pool[idx]))
            errs[r] = abs(m - reference)
        mean_err.append(float(np.mean(errs)))
        std_err.append(float(np.std(errs)))
    return np.asarray(mean_err), np.asarray(std_err)

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # model (per supervisor)
    ap.add_argument("--num-qubits", type=int, default=10)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", type=str, default="")  # basis bitstring; default all-zeros

    # YAQS strongsim controls
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--svd-cut", type=float, default=1e-12)

    # noise strengths (per supervisor)
    ap.add_argument("--gamma-conv", type=str, default="1e-3,1e-2,1e-1")
    ap.add_argument("--gamma-sweep-min", type=float, default=1e-3)
    ap.add_argument("--gamma-sweep-max", type=float, default=1e-1)
    ap.add_argument("--gamma-sweep-num", type=int, default=100)

    # Monte Carlo budgets (per supervisor)
    ap.add_argument("--conv-max-traj", type=int, default=10000)
    ap.add_argument("--conv-resamples", type=int, default=200)
    ap.add_argument("--conv-Ns", type=str, default="10,20,30,40,50,60,70,80,90,100,200,300,500,800,1000,2000,5000")

    ap.add_argument("--sweep-traj", type=int, default=2000)
    ap.add_argument("--angle-traj", type=int, default=2000)
    ap.add_argument("--angle-gamma", type=float, default=1e-2)

    # angle scan (logspace up to 2pi)
    ap.add_argument("--angle-min", type=float, default=1e-6)
    ap.add_argument("--angle-max", type=float, default=2*np.pi)
    ap.add_argument("--angle-num", type=int, default=60)

    # parallel
    ap.add_argument("--workers", type=int, default=0)        # 0 => all logical cores
    ap.add_argument("--blas-threads", type=int, default=1)   # avoid oversubscription
    ap.add_argument("--chunk", type=int, default=50)         # shots per task

    # optional transpile (keep OFF by default to avoid unsupported gates)
    ap.add_argument("--use-transpile", type=int, default=0)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    args = ap.parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    n = int(args.num_qubits)
    bitstring = args.input.strip() if args.input.strip() else ("0" * n)
    if len(bitstring) != n or any(c not in "01" for c in bitstring):
        raise ValueError(f"--input must be a bitstring of length {n}. Got: {bitstring}")

    # Build circuit
    base = build_twolocal(num_qubits=n, depth=int(args.depth), seed=int(args.seed), add_measurements=False)
    base = strip_measurements(base)

    # prepare input via circuit prefix (keeps MPS init simple)
    prep = prep_basis_state_circuit(n, bitstring)
    base = prep.compose(base, inplace=False)

    if int(args.use_transpile) == 1:
        # safer basis (still: if YAQS complains, turn this OFF)
        base = transpile(
            base,
            optimization_level=int(args.opt_level),
            seed_transpiler=int(args.seed_transpiler),
            basis_gates=["rx", "ry", "rz", "rxx", "rzz", "rzx", "cx", "id"],
        )
        base = strip_measurements(base)

    # Output dir
    out_dir = REPO_ROOT / "outputs" / "experiments" / "updated_work_1_fidelity_strongsim" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ideal vector (YAQS StrongSim, no noise)
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        svd_cut=float(args.svd_cut),
    )

    base_qpy = qpy_bytes(base)

    results: Dict[str, object] = {"args": vars(args)}
    t0 = time.time()

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        # ==========================================================
        # 1) MC convergence (pool -> resampled subset error)
        # ==========================================================
        gammas_conv = [float(x) for x in args.gamma_conv.split(",") if x.strip()]
        Ns = sorted({int(x) for x in args.conv_Ns.split(",") if x.strip() and int(x) > 0})

        conv_data = {}
        for g in gammas_conv:
            pool = mc_pool_parallel(
                ex,
                base_qpy, n, ideal_vec,
                gamma=float(g),
                total_shots=int(args.conv_max_traj),
                chunk=int(args.chunk),
                seed_base=int(args.seed + 1000000 + int(1e6*g)),
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                svd_cut=float(args.svd_cut),
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
        for g in gammas_conv:
            d = conv_data[str(g)]
            x = np.asarray(d["Ns"], dtype=float)
            y = np.asarray(d["mean_abs_error"], dtype=float)
            plt.plot(x, y, marker="o", label=f"gamma={g:g}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("N trajectories (log)")
        plt.ylabel("E[ |mean_N - reference| ] (log)")
        plt.title("MC convergence (YAQS StrongSim, Fidelity)")
        # annotate reference for one gamma (middle one if exists)
        g_anno = gammas_conv[min(1, len(gammas_conv)-1)]
        ref_anno = conv_data[str(g_anno)]["reference_mean_fidelity"]
        plt.gca().text(
            0.02, 0.02,
            f"reference (gamma={g_anno:g}) = {ref_anno:.6f}\nN_max={int(args.conv_max_traj)}",
            transform=plt.gca().transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.2),
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "mc_convergence_loglog.png", dpi=200, bbox_inches="tight")
        plt.close()

        # ==========================================================
        # 2) Error vs noise strength (logspace gamma)
        # ==========================================================
        gammas = np.logspace(np.log10(float(args.gamma_sweep_min)), np.log10(float(args.gamma_sweep_max)), int(args.gamma_sweep_num))
        mean_fids = np.empty_like(gammas, dtype=np.float64)

        for i, g in enumerate(gammas):
            mean_fids[i] = mc_mean_parallel(
                ex,
                base_qpy, n, ideal_vec,
                gamma=float(g),
                total_shots=int(args.sweep_traj),
                chunk=int(args.chunk),
                seed_base=int(args.seed + 2000000 + i * 100000),
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                svd_cut=float(args.svd_cut),
            )

        errs = 1.0 - mean_fids
        results["error_vs_gamma"] = {
            "gamma": gammas.tolist(),
            "mean_fidelity": mean_fids.tolist(),
            "error_1_minus_fid": errs.tolist(),
        }

        plt.figure()
        plt.plot(gammas, errs, marker="o", markersize=3)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("gamma (log)")
        plt.ylabel("error = 1 - mean fidelity (log)")
        plt.title("Error vs noise strength (YAQS StrongSim, Fidelity)")
        # annotate best fidelity in sweep (typically smallest gamma)
        idx_best = int(np.argmax(mean_fids))
        plt.gca().text(
            0.02, 0.02,
            f"best in sweep: gamma={gammas[idx_best]:.3e}\nF={mean_fids[idx_best]:.6f}\nideal baseline F=1",
            transform=plt.gca().transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.2),
        )
        plt.tight_layout()
        plt.savefig(out_dir / "error_vs_gamma_loglog.png", dpi=200, bbox_inches="tight")
        plt.close()

        # ==========================================================
        # 3) Angle-shift scan (paired seeds) at fixed gamma
        # ==========================================================
        deltas = np.logspace(np.log10(float(args.angle_min)), np.log10(float(args.angle_max)), int(args.angle_num))

        # baseline (delta=0) with paired seeds
        seed_base_angle = int(args.seed + 3000000)  # same for all deltas => paired estimator

        fid0 = mc_mean_parallel(
            ex,
            base_qpy, n, ideal_vec,
            gamma=float(args.angle_gamma),
            total_shots=int(args.angle_traj),
            chunk=int(args.chunk),
            seed_base=seed_base_angle,
            blas_threads=int(args.blas_threads),
            max_bond_dim=int(args.max_bond_dim),
            svd_cut=float(args.svd_cut),
        )

        angle_mean = np.empty_like(deltas, dtype=np.float64)

        for i, dlt in enumerate(deltas):
            sh = shifted_circuit(base, float(dlt))
            sh_qpy = qpy_bytes(sh)
            angle_mean[i] = mc_mean_parallel(
                ex,
                sh_qpy, n, ideal_vec,
                gamma=float(args.angle_gamma),
                total_shots=int(args.angle_traj),
                chunk=int(args.chunk),
                seed_base=seed_base_angle,  # paired!
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                svd_cut=float(args.svd_cut),
            )

        results["angle_shift"] = {
            "gamma": float(args.angle_gamma),
            "baseline_delta0": 0.0,
            "baseline_fidelity": float(fid0),
            "deltas": deltas.tolist(),
            "mean_fidelity": angle_mean.tolist(),
        }

        idx = int(np.argmax(angle_mean))
        best_delta = float(deltas[idx])
        best_fid = float(angle_mean[idx])

        plt.figure()
        plt.plot(deltas, angle_mean, marker="o", markersize=3, label="shifted")
        plt.axhline(float(fid0), linestyle="--", label="delta=0 baseline")
        plt.xscale("log")
        plt.xlabel("angle shift delta (log)")
        plt.ylabel("mean state fidelity")
        plt.title(f"Angle-shift scan (gamma={float(args.angle_gamma):g}, paired seeds)")
        plt.gca().text(
            0.02, 0.02,
            f"baseline (δ=0): F={fid0:.6f}\n"
            f"best: δ={best_delta:.3e}, F={best_fid:.6f}\n"
            f"ΔF(best-base)={best_fid - fid0:+.6e}",
            transform=plt.gca().transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.2),
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "angle_shift_logx.png", dpi=200, bbox_inches="tight")
        plt.close()

    elapsed = time.time() - t0
    results["elapsed_sec"] = float(elapsed)
    results["workers_used"] = int(workers)
    results["platform"] = platform.platform()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        f"input(bitstring)={bitstring}",
        "",
        "Saved: mc_convergence_loglog.png",
        "Saved: error_vs_gamma_loglog.png",
        "Saved: angle_shift_logx.png",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
