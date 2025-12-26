# scripts/updated_work_1_WIN.py
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import XGate


# -----------------------------
# repo paths + src-layout (fix ModuleNotFoundError on Windows)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


# -----------------------------
# your project imports (must exist in repo)
# -----------------------------
from qem_yrw_project.circuits.twolocal import build_twolocal
from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate, PauliJumpStats


# -----------------------------
# YAQS imports (must be installed in your venv)
# -----------------------------
def _import_yaqs():
    from mqt.yaqs import simulator
    from mqt.yaqs.core.data_structures.networks import MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
    from mqt.yaqs.core.libraries.gate_library import Z
    return simulator, MPS, Observable, StrongSimParams, Z


# -----------------------------
# helpers
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


def save_circuit_png_or_txt(circ: QuantumCircuit, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig = circ.draw(output="mpl", fold=120)
        fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (out_dir / f"{stem}.txt").write_text(str(circ.draw(output="text")), encoding="utf-8")
        (out_dir / f"{stem}.draw_error.txt").write_text(str(e), encoding="utf-8")


def qpy_bytes(circ: QuantumCircuit) -> bytes:
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    return buf.getvalue()


def qpy_load_bytes(b: bytes) -> QuantumCircuit:
    buf = io.BytesIO(b)
    return qpy.load(buf)[0]


def strip_measurements(c: QuantumCircuit) -> QuantumCircuit:
    return c.remove_final_measurements(inplace=False)


def prep_circuit_from_bitstring(n: int, bitstring: str) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i, b in enumerate(bitstring):
        if b == "1":
            qc.append(XGate(), [i])
    return qc


def fidelity_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    amp = np.vdot(psi, phi)
    return float(np.real(amp * np.conjugate(amp)))


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y.copy()
    k = np.ones(window, dtype=float) / float(window)
    ypad = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(ypad, k, mode="valid")


def _chunk_list(xs: List[int], chunk: int) -> List[List[int]]:
    return [xs[i:i + chunk] for i in range(0, len(xs), chunk)]


def _mps_to_statevector_try(mps) -> Optional[np.ndarray]:
    """
    Try common YAQS MPS export methods.
    Returns ndarray (complex) if possible, else None.
    """
    cand = [
        "to_statevector",
        "to_state_vector",
        "statevector",
        "to_vector",
        "vector",
        "to_dense",
        "to_numpy",
        "as_vector",
        "as_statevector",
    ]
    for name in cand:
        if hasattr(mps, name):
            obj = getattr(mps, name)
            try:
                v = obj() if callable(obj) else obj
                v = np.asarray(v, dtype=np.complex128).reshape(-1)
                return v
            except Exception:
                continue
    return None


def strongsim_final_statevector(
    circ: QuantumCircuit,
    n: int,
    bitstring: str,
    max_bond_dim: int,
    threshold: float,
    allow_qiskit_fallback: bool,
    _warn_once: Dict[str, bool],
) -> np.ndarray:
    """
    Run YAQS StrongSim and return final statevector if YAQS can export it.
    If export is not available and allow_qiskit_fallback is True, fall back to Qiskit statevector.
    """
    simulator, MPS, Observable, StrongSimParams, Z = _import_yaqs()

    prep = prep_circuit_from_bitstring(n, bitstring)
    full = prep.compose(circ, inplace=False)

    state = MPS(n, state="zeros")

    # Some YAQS versions may require >=1 observable; we add Z on qubit 0 and ignore it.
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        num_traj=1,
        max_bond_dim=int(max_bond_dim),
        threshold=float(threshold),
    )
    simulator.run(state, full, sim_params, noise_model=None, parallel=False)

    v = _mps_to_statevector_try(state)
    if v is not None:
        return v

    if not allow_qiskit_fallback:
        raise RuntimeError(
            "YAQS StrongSim ran, but MPS export to statevector is not available in your YAQS version.\n"
            "Please either:\n"
            "  (1) upgrade YAQS to a version where MPS can export the statevector, or\n"
            "  (2) rerun with --allow-qiskit-fallback 1 (will use Qiskit Statevector only for overlap).\n"
        )

    if not _warn_once.get("fallback", False):
        print(
            "[WARN] YAQS MPS cannot export statevector in this environment. "
            "Falling back to Qiskit Statevector for fidelity overlap ONLY.",
            flush=True,
        )
        _warn_once["fallback"] = True

    # fallback
    from qiskit.quantum_info import Statevector
    return Statevector.from_instruction(full).data


# -----------------------------
# Worker: compute fidelities for a chunk of seeds (YAQS StrongSim per trajectory)
# -----------------------------
def _fids_for_seed_chunk_yaqs(
    circ_qpy: bytes,
    psi_ideal_ref: np.ndarray,
    gamma: float,
    seeds: List[int],
    n: int,
    bitstring: str,
    max_bond_dim: int,
    threshold: float,
    blas_threads: int,
    allow_qiskit_fallback: bool,
) -> np.ndarray:
    _worker_init(blas_threads)

    circ = strip_measurements(qpy_load_bytes(circ_qpy))

    # local warn flag in each process
    warn_once: Dict[str, bool] = {}

    out = np.empty(len(seeds), dtype=np.float64)
    for i, s in enumerate(seeds):
        noisy, _st = apply_pauli_jump_after_each_gate(
            circ, float(gamma), int(s), include_measurements=False
        )
        phi = strongsim_final_statevector(
            noisy, n, bitstring,
            max_bond_dim=max_bond_dim,
            threshold=threshold,
            allow_qiskit_fallback=allow_qiskit_fallback,
            _warn_once=warn_once,
        )
        out[i] = fidelity_pure(psi_ideal_ref, phi)
    return out


# -----------------------------
# Angle shift (NOTE: not ideal-equivalent; for comparison only)
# -----------------------------
def apply_angle_shift_all_params(circ: QuantumCircuit, delta: float) -> QuantumCircuit:
    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    for ci in circ.data:
        op = ci.operation
        q_idx = [circ.find_bit(q).index for q in ci.qubits]
        c_idx = [circ.find_bit(c).index for c in ci.clbits]
        if getattr(op, "params", None) and len(op.params) > 0:
            new_op = op.copy()
            new_op.params = [float(p) + float(delta) for p in op.params]
        else:
            new_op = op
        out.append(new_op, [out.qubits[i] for i in q_idx], [out.clbits[i] for i in c_idx])
    return out


# -----------------------------
# Variant compilation candidates (robust across Qiskit versions)
# -----------------------------
@dataclass
class Candidate:
    name: str
    circ: QuantumCircuit
    meta: Dict[str, object]


def circuit_stats(c: QuantumCircuit) -> Dict[str, int]:
    ops = c.count_ops()
    twoq = 0
    for inst, qargs, _ in c.data:
        if len(qargs) == 2 and inst.name not in ("barrier", "measure"):
            twoq += 1
    return {
        "depth": c.depth(),
        "size": c.size(),
        "num_twoq": twoq,
        "num_rzz": int(ops.get("rzz", 0)),
        "num_cx": int(ops.get("cx", 0)),
        "num_rz": int(ops.get("rz", 0)),
        "num_rx": int(ops.get("rx", 0)),
        "num_sx": int(ops.get("sx", 0)),
        "num_x": int(ops.get("x", 0)),
    }


def build_candidates(base: QuantumCircuit, seed_transpiler: int) -> List[Candidate]:
    base = strip_measurements(base)
    cands: List[Candidate] = []
    cands.append(Candidate("raw", base, {"how": "no transpile"}))

    for opt in [0, 1, 2, 3]:
        c_rzz = transpile(
            base,
            optimization_level=opt,
            seed_transpiler=seed_transpiler,
            basis_gates=["rz", "rx", "x", "rzz"],
        )
        cands.append(Candidate(f"basis_rzz_opt{opt}", strip_measurements(c_rzz), {"basis": "rzz", "opt": opt}))

    for opt in [0, 1, 2, 3]:
        c_cx = transpile(
            base,
            optimization_level=opt,
            seed_transpiler=seed_transpiler,
            basis_gates=["rz", "sx", "x", "cx"],
        )
        cands.append(Candidate(f"basis_cx_opt{opt}", strip_measurements(c_cx), {"basis": "cx", "opt": opt}))

    # de-duplicate
    seen: Dict[bytes, str] = {}
    uniq: List[Candidate] = []
    for cand in cands:
        b = qpy_bytes(cand.circ)
        if b not in seen:
            seen[b] = cand.name
            uniq.append(cand)
    return uniq


# -----------------------------
# Main
# -----------------------------
from concurrent.futures import ProcessPoolExecutor


def main() -> None:
    ap = argparse.ArgumentParser()

    # model
    ap.add_argument("--num-qubits", type=int, default=10)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", type=str, default="")  # bitstring, default all-zeros

    # noise
    ap.add_argument("--gamma-single", type=float, default=1e-2)

    ap.add_argument("--gamma-min", type=float, default=1e-3)
    ap.add_argument("--gamma-max", type=float, default=1e-1)
    ap.add_argument("--gamma-num", type=int, default=60)

    # MC budgets
    ap.add_argument("--traj-gamma", type=int, default=300)      # per gamma
    ap.add_argument("--traj-angle", type=int, default=300)      # per delta
    ap.add_argument("--traj-mc", type=int, default=10000)       # for convergence curve

    ap.add_argument("--n-batches", type=int, default=50)

    # angle shift sweep
    ap.add_argument("--delta-min", type=float, default=1e-6)
    ap.add_argument("--delta-max", type=float, default=2 * math.pi)
    ap.add_argument("--delta-num", type=int, default=30)

    # compilation search
    ap.add_argument("--compilation-traj", type=int, default=600)  # per candidate
    ap.add_argument("--seed-transpiler", type=int, default=123)

    # YAQS StrongSim params
    ap.add_argument("--max-bond-dim", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=1e-10)

    # parallel
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=25)

    # keep CPU busy if traj too small
    ap.add_argument("--fill-cpu", type=int, default=1)
    ap.add_argument("--fill-factor", type=int, default=8)

    # fallback behavior
    ap.add_argument("--allow-qiskit-fallback", type=int, default=1)

    args = ap.parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    allow_qiskit_fallback = bool(int(args.allow_qiskit_fallback))

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    n = int(args.num_qubits)
    bitstring = args.input.strip() if args.input.strip() else ("0" * n)
    if len(bitstring) != n or any(c not in "01" for c in bitstring):
        raise ValueError(f"--input must be a bitstring length {n}. Got: {bitstring}")

    out_dir = REPO_ROOT / "outputs" / "updated_work_1" / f"n{n}_d{int(args.depth)}_seed{int(args.seed)}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "repo_root": str(REPO_ROOT.resolve()),
        "num_qubits": n,
        "depth": int(args.depth),
        "seed": int(args.seed),
        "input": bitstring,
        "workers": workers,
        "blas_threads": int(args.blas_threads),
        "chunk": int(args.chunk),
        "fill_cpu": int(args.fill_cpu),
        "fill_factor": int(args.fill_factor),
        "gamma_single": float(args.gamma_single),
        "gamma_min": float(args.gamma_min),
        "gamma_max": float(args.gamma_max),
        "gamma_num": int(args.gamma_num),
        "traj_gamma": int(args.traj_gamma),
        "traj_angle": int(args.traj_angle),
        "traj_mc": int(args.traj_mc),
        "n_batches": int(args.n_batches),
        "delta_min": float(args.delta_min),
        "delta_max": float(args.delta_max),
        "delta_num": int(args.delta_num),
        "compilation_traj": int(args.compilation_traj),
        "seed_transpiler": int(args.seed_transpiler),
        "max_bond_dim": int(args.max_bond_dim),
        "threshold": float(args.threshold),
        "allow_qiskit_fallback": allow_qiskit_fallback,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"OUTPUT DIR: {out_dir.resolve()}", flush=True)
    print(f"workers={workers}, blas_threads={int(args.blas_threads)} chunk={int(args.chunk)}", flush=True)

    # ------------------------------------------------------------
    # 0) Build ideal circuit + save
    # ------------------------------------------------------------
    ideal = build_twolocal(num_qubits=n, depth=int(args.depth), seed=int(args.seed), add_measurements=False)
    ideal = strip_measurements(ideal)

    with (out_dir / "circuit_ideal.qpy").open("wb") as f:
        qpy.dump(ideal, f)
    save_circuit_png_or_txt(ideal, out_dir, "circuit_ideal")

    # ------------------------------------------------------------
    # 1) Compute ideal reference state (YAQS StrongSim)
    # ------------------------------------------------------------
    warn_once_main: Dict[str, bool] = {}
    psi_ideal = strongsim_final_statevector(
        ideal, n, bitstring,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
        allow_qiskit_fallback=allow_qiskit_fallback,
        _warn_once=warn_once_main,
    )

    # ------------------------------------------------------------
    # Shared executor for all MC-heavy parts
    # ------------------------------------------------------------
    ex = ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    )

    try:
        # ------------------------------------------------------------
        # 2) Single-trajectory demo + YAQS EquiCheck (if available)
        # ------------------------------------------------------------
        gamma0 = float(args.gamma_single)
        noisy = None
        st_best: Optional[PauliJumpStats] = None
        seed_used = None
        for s in range(0, 5000):
            cand, st = apply_pauli_jump_after_each_gate(ideal, gamma0, s, include_measurements=False)
            if st.n_noise_ops > 0:
                noisy, st_best, seed_used = cand, st, s
                break

        if noisy is None:
            raise RuntimeError("Could not find a seed that inserts noise ops (unexpected).")

        with (out_dir / "circuit_noisy_singletraj.qpy").open("wb") as f:
            qpy.dump(noisy, f)
        save_circuit_png_or_txt(noisy, out_dir, "circuit_noisy_singletraj")

        phi = strongsim_final_statevector(
            noisy, n, bitstring,
            max_bond_dim=int(args.max_bond_dim),
            threshold=float(args.threshold),
            allow_qiskit_fallback=allow_qiskit_fallback,
            _warn_once=warn_once_main,
        )
        fid_single = fidelity_pure(psi_ideal, phi)

        # also compute all-Z expectations (as in your old YAQS demo)
        try:
            simulator, MPS, Observable, StrongSimParams, Z = _import_yaqs()
            prep = prep_circuit_from_bitstring(n, bitstring)
            state = MPS(n, state="zeros")
            sim_params = StrongSimParams(
                observables=[Observable(Z(), site) for site in range(n)],
                num_traj=1,
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
            )
            simulator.run(state, prep.compose(noisy, inplace=False), sim_params, noise_model=None, parallel=False)
            z_vals = [float(obs.results[0]) for obs in sim_params.observables]
        except Exception as e:
            z_vals = []
            (out_dir / "singletraj_Z_error.txt").write_text(str(e), encoding="utf-8")

        (out_dir / "singletraj_report.txt").write_text(
            "tag=UPDATED_WORK_1_SINGLETRAJ\n"
            f"gamma={gamma0}\n"
            f"seed_used={seed_used}\n"
            f"n_gates_seen={st_best.n_gates_seen if st_best else None}\n"
            f"n_noise_ops={st_best.n_noise_ops if st_best else None}\n"
            f"state_fidelity={fid_single}\n"
            f"Z_expectations={z_vals}\n",
            encoding="utf-8",
        )
        print(f"[singletraj] seed={seed_used} n_noise_ops={st_best.n_noise_ops if st_best else None} fid={fid_single:.6f}", flush=True)

        # YAQS EquiCheck (best effort)
        try:
            from mqt.yaqs.digital.equivalence_checker import run as equiv_run
            sig = str(equiv_run)
            result = None
            err = None
            try:
                result = equiv_run(ideal, noisy)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            rep = []
            rep.append("tag=UPDATED_WORK_1_YAQS_EQUIVCHECK\n")
            rep.append(f"gamma={gamma0}\nseed_used={seed_used}\n")
            rep.append(f"n_noise_ops={st_best.n_noise_ops if st_best else None}\n")
            rep.append(f"equiv_run={sig}\n")
            rep.append(f"result={result}\n")
            if err:
                rep.append("\nERROR:\n" + err + "\n")
            (out_dir / "yaqs_equivcheck_report.txt").write_text("".join(rep), encoding="utf-8")
        except Exception as e:
            (out_dir / "yaqs_equivcheck_report.txt").write_text(
                "tag=UPDATED_WORK_1_YAQS_EQUIVCHECK\n"
                "status=SKIPPED\n"
                f"reason={type(e).__name__}: {e}\n",
                encoding="utf-8",
            )

        # ------------------------------------------------------------
        # 3) Gamma sweep: state fidelity vs gamma (YAQS StrongSim + Pauli-jump)
        # ------------------------------------------------------------
        gammas = np.logspace(math.log10(float(args.gamma_min)), math.log10(float(args.gamma_max)), int(args.gamma_num))
        meanF = np.zeros_like(gammas, dtype=float)
        seF = np.zeros_like(gammas, dtype=float)

        traj_gamma = int(args.traj_gamma)
        if int(args.fill_cpu) == 1:
            traj_gamma = max(traj_gamma, workers * int(args.fill_factor))

        rng = np.random.default_rng(int(args.seed))
        base_seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=traj_gamma, dtype=np.int64).tolist()]
        seed_chunks = _chunk_list(base_seeds, int(args.chunk))

        ideal_qpy = qpy_bytes(ideal)

        for i, g in enumerate(gammas):
            futs = []
            for ch in seed_chunks:
                futs.append(
                    ex.submit(
                        _fids_for_seed_chunk_yaqs,
                        ideal_qpy, psi_ideal, float(g), ch, n, bitstring,
                        int(args.max_bond_dim), float(args.threshold),
                        int(args.blas_threads),
                        allow_qiskit_fallback,
                    )
                )
            parts = [f.result() for f in futs]
            fids = np.concatenate(parts, axis=0)

            meanF[i] = float(np.mean(fids))
            if fids.size > 1:
                seF[i] = float(np.std(fids, ddof=1) / math.sqrt(fids.size))
            else:
                seF[i] = float("nan")

            if (i + 1) % max(1, len(gammas) // 10) == 0:
                print(f"[gamma] {i+1}/{len(gammas)} gamma={g:.3e} F={meanF[i]:.6f} ± {seF[i]:.6f}", flush=True)

        (out_dir / "gamma_sweep_state_fidelity.json").write_text(
            json.dumps(
                {
                    "tag": "UPDATED_WORK_1_GAMMA_SWEEP",
                    "gammas": gammas.tolist(),
                    "mean_fidelity": meanF.tolist(),
                    "se": seF.tolist(),
                    "traj": traj_gamma,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # plots (xlog and loglog for error)
        err = 1.0 - meanF

        plt.figure()
        plt.semilogx(gammas, err)
        plt.xlabel("gamma")
        plt.ylabel("error = 1 - fidelity")
        plt.title("Error vs gamma (xlog)")
        plt.tight_layout()
        plt.savefig(out_dir / "error_vs_gamma_xlog.png", dpi=200)
        plt.close()

        plt.figure()
        plt.loglog(gammas, err)
        plt.xlabel("gamma")
        plt.ylabel("error = 1 - fidelity")
        plt.title("Error vs gamma (loglog)")
        plt.tight_layout()
        plt.savefig(out_dir / "error_vs_gamma_loglog.png", dpi=200)
        plt.close()

        # also save mean fidelity curve
        plt.figure()
        plt.semilogx(gammas, meanF)
        plt.xlabel("gamma")
        plt.ylabel("mean state fidelity")
        plt.title("Mean state fidelity vs gamma (xlog)")
        plt.tight_layout()
        plt.savefig(out_dir / "state_fidelity_vs_gamma_xlog.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------
        # 4) MC convergence (fix gamma=gamma_single): error to Tmax reference
        # ------------------------------------------------------------
        traj_mc = int(args.traj_mc)
        if int(args.fill_cpu) == 1:
            traj_mc = max(traj_mc, workers * int(args.fill_factor))

        rng2 = np.random.default_rng(int(args.seed) + 999)
        seeds_mc = [int(x) for x in rng2.integers(0, 2**31 - 1, size=traj_mc, dtype=np.int64).tolist()]
        chunks_mc = _chunk_list(seeds_mc, int(args.chunk))

        # compute reference using all traj_mc
        futs = [
            ex.submit(
                _fids_for_seed_chunk_yaqs,
                ideal_qpy, psi_ideal, gamma0, ch, n, bitstring,
                int(args.max_bond_dim), float(args.threshold),
                int(args.blas_threads),
                allow_qiskit_fallback,
            )
            for ch in chunks_mc
        ]
        fids_all = np.concatenate([f.result() for f in futs], axis=0)
        ref_val = float(np.mean(fids_all))

        # batch sizes
        n_batches = int(args.n_batches)
        batch_sizes = np.unique(np.round(np.logspace(1, math.log10(traj_mc), n_batches)).astype(int))
        mean_abs_err = []
        for bs in batch_sizes:
            # take first bs samples
            est = float(np.mean(fids_all[:bs]))
            mean_abs_err.append(abs(est - ref_val))

        (out_dir / "mc_convergence.json").write_text(
            json.dumps(
                {
                    "tag": "UPDATED_WORK_1_MC_CONVERGENCE",
                    "gamma": gamma0,
                    "traj_mc": traj_mc,
                    "ref_val": ref_val,
                    "batch_sizes": batch_sizes.tolist(),
                    "abs_error": mean_abs_err,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        plt.figure()
        plt.loglog(batch_sizes, mean_abs_err, marker="o")
        plt.xlabel("sample size")
        plt.ylabel("mean absolute error to Tmax reference")
        plt.title(f"MC convergence (gamma={gamma0}, Tmax={traj_mc})")
        plt.tight_layout()
        plt.savefig(out_dir / "mc_convergence_loglog.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------
        # 5) Angle shift sweep (NOT ideal-equivalent; for contrast)
        # ------------------------------------------------------------
        deltas = np.logspace(math.log10(float(args.delta_min)), math.log10(float(args.delta_max)), int(args.delta_num))

        traj_angle = int(args.traj_angle)
        if int(args.fill_cpu) == 1:
            traj_angle = max(traj_angle, workers * int(args.fill_factor))

        rng3 = np.random.default_rng(int(args.seed) + 2025)
        angle_seeds = [int(x) for x in rng3.integers(0, 2**31 - 1, size=traj_angle, dtype=np.int64).tolist()]
        angle_chunks = _chunk_list(angle_seeds, int(args.chunk))

        # baseline (delta=0)
        futs0 = [
            ex.submit(
                _fids_for_seed_chunk_yaqs,
                ideal_qpy, psi_ideal, gamma0, ch, n, bitstring,
                int(args.max_bond_dim), float(args.threshold),
                int(args.blas_threads),
                allow_qiskit_fallback,
            )
            for ch in angle_chunks
        ]
        f0 = np.concatenate([f.result() for f in futs0], axis=0)
        base_mean = float(np.mean(f0))

        means = np.zeros_like(deltas, dtype=float)
        ses = np.zeros_like(deltas, dtype=float)

        for i, dlt in enumerate(deltas):
            shifted = apply_angle_shift_all_params(ideal, float(dlt))
            shifted_qpy = qpy_bytes(shifted)

            futs = [
                ex.submit(
                    _fids_for_seed_chunk_yaqs,
                    shifted_qpy, psi_ideal, gamma0, ch, n, bitstring,
                    int(args.max_bond_dim), float(args.threshold),
                    int(args.blas_threads),
                    allow_qiskit_fallback,
                )
                for ch in angle_chunks
            ]
            fids = np.concatenate([f.result() for f in futs], axis=0)
            means[i] = float(np.mean(fids))
            ses[i] = float(np.std(fids, ddof=1) / math.sqrt(fids.size)) if fids.size > 1 else float("nan")

        best_i = int(np.argmax(means))
        (out_dir / "angle_shift.json").write_text(
            json.dumps(
                {
                    "tag": "UPDATED_WORK_1_ANGLE_SHIFT",
                    "gamma": gamma0,
                    "traj": traj_angle,
                    "baseline_mean": base_mean,
                    "deltas": deltas.tolist(),
                    "mean_fidelity": means.tolist(),
                    "se": ses.tolist(),
                    "best_delta": float(deltas[best_i]),
                    "best_mean": float(means[best_i]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        plt.figure()
        plt.semilogx(deltas, means, marker="o", markersize=3)
        plt.axhline(base_mean, linestyle="--")
        plt.xlabel("delta (log scale)")
        plt.ylabel("mean state fidelity")
        plt.title("Angle shift sweep (NOT ideal-equivalent) — for contrast")
        plt.tight_layout()
        plt.savefig(out_dir / "angle_shift_logx.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------
        # 6) Variant compilation search (ideal-equivalent; changes gate counts)
        # ------------------------------------------------------------
        cands = build_candidates(ideal, seed_transpiler=int(args.seed_transpiler))

        traj_comp = int(args.compilation_traj)
        if int(args.fill_cpu) == 1:
            traj_comp = max(traj_comp, workers * int(args.fill_factor))

        rng4 = np.random.default_rng(int(args.seed) + 424242)
        comp_seeds = [int(x) for x in rng4.integers(0, 2**31 - 1, size=traj_comp, dtype=np.int64).tolist()]
        comp_chunks = _chunk_list(comp_seeds, int(args.chunk))

        rows = []
        for cand in cands:
            cq = qpy_bytes(cand.circ)

            futs = [
                ex.submit(
                    _fids_for_seed_chunk_yaqs,
                    cq, psi_ideal, gamma0, ch, n, bitstring,
                    int(args.max_bond_dim), float(args.threshold),
                    int(args.blas_threads),
                    allow_qiskit_fallback,
                )
                for ch in comp_chunks
            ]
            fids = np.concatenate([f.result() for f in futs], axis=0)
            mean = float(np.mean(fids))
            se = float(np.std(fids, ddof=1) / math.sqrt(fids.size)) if fids.size > 1 else float("nan")

            # sanity: candidate ideal state should match reference (global phase cancels)
            psi_cand = strongsim_final_statevector(
                cand.circ, n, bitstring,
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                allow_qiskit_fallback=allow_qiskit_fallback,
                _warn_once=warn_once_main,
            )
            ideal_eq = fidelity_pure(psi_ideal, psi_cand)

            st = circuit_stats(cand.circ)
            rows.append(
                {
                    "name": cand.name,
                    "mean_fidelity": mean,
                    "se": se,
                    "ideal_equiv_fidelity": ideal_eq,
                    **st,
                    **{f"meta_{k}": v for k, v in cand.meta.items()},
                }
            )
            print(f"[compile] {cand.name:16s} F={mean:.6f} ± {se:.6f} size={st['size']} twoq={st['num_twoq']}", flush=True)

        rows.sort(key=lambda r: r["mean_fidelity"], reverse=True)

        (out_dir / "compilation_search.json").write_text(
            json.dumps({"tag": "UPDATED_WORK_1_COMPILATION_SEARCH", "rows": rows}, indent=2),
            encoding="utf-8",
        )
        with (out_dir / "compilation_search.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        labels = [r["name"] for r in rows]
        means2 = [r["mean_fidelity"] for r in rows]
        ses2 = [r["se"] for r in rows]

        plt.figure(figsize=(12, 5))
        x = np.arange(len(labels))
        plt.bar(x, means2, yerr=ses2, capsize=3)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("mean state fidelity")
        plt.title(f"Variant compilation search (gamma={gamma0:g})")
        plt.tight_layout()
        plt.savefig(out_dir / "compilation_fidelity_by_candidate.png", dpi=200, bbox_inches="tight")
        plt.close()

        sizes = [r["size"] for r in rows]
        twoq = [r["num_twoq"] for r in rows]

        plt.figure(figsize=(6, 5))
        plt.scatter(sizes, means2)
        plt.xlabel("total gates (size)")
        plt.ylabel("mean state fidelity")
        plt.title("Fidelity vs total gates (variant compilation)")
        plt.tight_layout()
        plt.savefig(out_dir / "compilation_fidelity_vs_size.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.scatter(twoq, means2)
        plt.xlabel("two-qubit gates")
        plt.ylabel("mean state fidelity")
        plt.title("Fidelity vs 2Q gates (variant compilation)")
        plt.tight_layout()
        plt.savefig(out_dir / "compilation_fidelity_vs_twoq.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------
        # DONE
        # ------------------------------------------------------------
        print("DONE. WROTE:", out_dir.resolve(), flush=True)
        print("FIG :", out_dir / "error_vs_gamma_xlog.png", flush=True)
        print("FIG :", out_dir / "error_vs_gamma_loglog.png", flush=True)
        print("FIG :", out_dir / "mc_convergence_loglog.png", flush=True)
        print("FIG :", out_dir / "angle_shift_logx.png", flush=True)
        print("FIG :", out_dir / "compilation_fidelity_by_candidate.png", flush=True)

    finally:
        ex.shutdown(wait=True)


if __name__ == "__main__":
    main()
