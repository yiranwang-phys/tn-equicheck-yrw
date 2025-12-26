# scripts/exp_sites_mask_strongsim_YAQS_WIN.py
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
# Sites definition + masked Pauli-jump (baked) noise
# -----------------------------
def count_sites(circ: QuantumCircuit) -> int:
    """site = (a gate application) x (each touched qubit)."""
    s = 0
    for ci in circ.data:
        op = ci.operation
        if op.name in ("measure", "barrier", "reset"):
            continue
        s += len(ci.qubits)
    return int(s)

def apply_pauli_jump_with_site_mask(
    circuit: QuantumCircuit,
    gamma: float,
    seed: int,
    active_mask: np.ndarray,
) -> QuantumCircuit:
    """
    After each gate, for each touched qubit (site) with active_mask[site_id]=True:
      with prob gamma, insert X/Y/Z uniformly.
    This bakes stochastic noise into the circuit => YAQS sees a unitary circuit per trajectory.
    """
    rng = np.random.default_rng(int(seed))
    noisy = QuantumCircuit(circuit.num_qubits)
    site_id = 0

    for ci in circuit.data:
        op = ci.operation
        if op.name in ("measure", "barrier", "reset"):
            continue

        qargs = list(ci.qubits)
        cargs = list(ci.clbits)
        noisy.append(op, qargs, cargs)

        for q in qargs:
            if site_id >= active_mask.size:
                raise RuntimeError(f"site_id overflow: {site_id} >= mask size {active_mask.size}")
            if bool(active_mask[site_id]):
                if rng.random() < gamma:
                    pa = int(rng.integers(0, 3))
                    if pa == 0:
                        noisy.x(q)
                    elif pa == 1:
                        noisy.y(q)
                    else:
                        noisy.z(q)
            site_id += 1

    return noisy

# -----------------------------
# Paired seeds across all candidates
# -----------------------------
def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * int(shot_index)) % (2**31 - 1))

def split_traj(traj_total: int, reps: int) -> List[Tuple[int, int]]:
    """
    Split [0, traj_total) into reps contiguous segments:
      returns list of (offset, shots).
    This keeps *exactly the same* set of shot seeds across schemes.
    """
    traj_total = int(traj_total)
    reps = int(max(1, reps))
    base = traj_total // reps
    rem = traj_total % reps
    out = []
    off = 0
    for r in range(reps):
        shots = base + (1 if r < rem else 0)
        out.append((off, shots))
        off += shots
    return out

# -----------------------------
# MC worker
# -----------------------------
def _chunk_fidelities(
    base_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    start: int,
    shots: int,
    seed_base: int,
    seed_offset: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    mask_bytes: bytes,
) -> np.ndarray:
    _worker_init(int(blas_threads))

    base = strip_measurements(qpy_load_bytes(base_qpy))
    active_mask = np.frombuffer(mask_bytes, dtype=np.uint8).astype(bool, copy=False)

    out = np.empty(int(shots), dtype=np.float64)
    for i in range(int(shots)):
        global_idx = int(seed_offset + start + i)
        s = shot_seed(int(seed_base), global_idx)

        noisy = apply_pauli_jump_with_site_mask(base, float(gamma), int(s), active_mask)
        noisy = strip_measurements(noisy)

        vec = run_strongsim_statevector(
            noisy,
            n=int(n),
            max_bond_dim=int(max_bond_dim),
            threshold=float(threshold),
        )
        out[i] = fidelity_pure(psi_ideal, vec)

    return out

def mc_mean_parallel_masked(
    ex,
    base_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    traj: int,
    chunk: int,
    seed_base: int,
    seed_offset: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    active_mask: np.ndarray,
) -> float:
    traj = int(traj)
    chunk = int(max(1, chunk))
    n_chunks = (traj + chunk - 1) // chunk

    mask_bytes = np.asarray(active_mask, dtype=np.uint8).tobytes()

    futs = []
    for k in range(n_chunks):
        start = k * chunk
        shots = min(chunk, traj - start)
        futs.append(
            ex.submit(
                _chunk_fidelities,
                base_qpy, int(n), psi_ideal, float(gamma),
                int(start), int(shots),
                int(seed_base), int(seed_offset),
                int(blas_threads),
                int(max_bond_dim), float(threshold),
                mask_bytes,
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
# Mask builders
# -----------------------------
def mask_prefix(S_total: int, S_active: int) -> np.ndarray:
    m = np.zeros(int(S_total), dtype=np.uint8)
    S_active = int(max(0, min(S_total, S_active)))
    if S_active > 0:
        m[:S_active] = 1
    return m

def mask_random(S_total: int, S_active: int, rng: np.random.Generator) -> np.ndarray:
    m = np.zeros(int(S_total), dtype=np.uint8)
    S_active = int(max(0, min(S_total, S_active)))
    if S_active > 0:
        idx = rng.choice(int(S_total), size=int(S_active), replace=False)
        m[idx] = 1
    return m

# -----------------------------
# Plotting
# -----------------------------
def plot_error_vs_sites(
    out_path: Path,
    S_list: np.ndarray,
    err_prefix: np.ndarray,
    err_random_mean: np.ndarray,
    err_random_std: np.ndarray,
    gamma: float,
    title: str,
) -> None:
    plt.figure()
    plt.plot(S_list, err_prefix, marker="o", markersize=3, label="Prefix sites")
    plt.errorbar(S_list, err_random_mean, yerr=err_random_std, marker="o", markersize=3,
                 linestyle="-", capsize=3, label="Random sites (mean ± std over masks)")
    plt.xlabel("Active sites S")
    plt.ylabel("Error = 1 - mean fidelity")
    plt.title(title + f"  (gamma={gamma:g})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_collapse_gammaS(
    out_path: Path,
    xs_prefix: np.ndarray,
    ys_prefix: np.ndarray,
    xs_random: np.ndarray,
    ys_random: np.ndarray,
    title: str,
) -> None:
    plt.figure()
    plt.scatter(xs_prefix, ys_prefix, s=18, label="Prefix")
    plt.scatter(xs_random, ys_random, s=18, label="Random")
    plt.xlabel("gamma * S (expected jump count scale)")
    plt.ylabel("Error = 1 - mean fidelity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # circuit
    ap.add_argument("--num-qubits", type=int, default=5)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    # YAQS StrongSim
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)

    # gammas (comma separated)
    ap.add_argument("--gammas", type=str, default="1e-3,3e-3,1e-2")

    # S grid
    ap.add_argument("--S-fracs", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="fractions of total sites, comma separated")
    ap.add_argument("--S-list", type=str, default="",
                    help="explicit S list (overrides --S-fracs), comma separated integers")

    # MC budget per (gamma, S, scheme)
    ap.add_argument("--traj", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=50)

    # random mask reps (random scheme only)
    ap.add_argument("--mask-reps", type=int, default=5)
    ap.add_argument("--mask-seed-base", type=int, default=20251221)

    # parallel
    ap.add_argument("--workers", type=int, default=0)     # 0 => all logical cores
    ap.add_argument("--blas-threads", type=int, default=1)

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

    # Print YAQS signature
    try:
        import mqt.yaqs
        from mqt.yaqs import simulator
        from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
        print("[YAQS]", getattr(mqt.yaqs, "__version__", "unknown"))
        print("[YAQS] simulator =", getattr(simulator, "__file__", "unknown"))
        print("[YAQS] StrongSimParams =", inspect.signature(StrongSimParams), flush=True)
    except Exception as e:
        raise RuntimeError(f"YAQS import failed in this venv: {e}")

    # Ideal target
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
    )

    S_total = count_sites(base)
    print(f"[SITES] total sites S_total = {S_total} for n={n}, depth={int(args.depth)}")

    # Parse gamma list
    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    if len(gammas) == 0:
        raise ValueError("Empty --gammas")

    # Build S list
    if args.S_list.strip():
        S_list = np.array([int(x) for x in args.S_list.split(",") if x.strip()], dtype=int)
    else:
        fracs = [float(x) for x in args.S_fracs.split(",") if x.strip()]
        S_list = np.array([int(round(f * S_total)) for f in fracs], dtype=int)

    S_list = np.unique(np.clip(S_list, 0, S_total))
    S_list.sort()

    out_dir = REPO_ROOT / "outputs" / "experiments" / "sites_mask_strongsim" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_base = int(args.seed + 20251220)  # paired seeds across all candidates

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
        "noise": {"type": "pauli_jump_baked_masked_sites", "site_definition": "each (gate,touched_qubit)"},
        "S_total": int(S_total),
        "S_list": S_list.tolist(),
        "gammas": gammas,
        "traj_per_point": int(args.traj),
        "random_mask_reps": int(args.mask_reps),
        "random_mask_seed_base": int(args.mask_seed_base),
        "paired_seed_base": seed_base,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_qpy = qpy_bytes(base)

    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    results: Dict[str, object] = {"meta": meta, "by_gamma": {}}

    xs_collapse_prefix = []
    ys_collapse_prefix = []
    xs_collapse_random = []
    ys_collapse_random = []

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        for g in gammas:
            err_prefix = np.empty_like(S_list, dtype=float)
            err_random_mean = np.empty_like(S_list, dtype=float)
            err_random_std = np.empty_like(S_list, dtype=float)

            for i, S_active in enumerate(S_list):
                # -------------------------
                # Prefix scheme (single mask)
                # -------------------------
                m_pref = mask_prefix(S_total, int(S_active))
                f_pref = mc_mean_parallel_masked(
                    ex,
                    base_qpy, n, ideal_vec,
                    gamma=float(g),
                    traj=int(args.traj),
                    chunk=int(args.chunk),
                    seed_base=seed_base,
                    seed_offset=0,
                    blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim),
                    threshold=float(args.threshold),
                    active_mask=m_pref,
                )
                err_prefix[i] = 1.0 - float(f_pref)

                xs_collapse_prefix.append(float(g) * float(S_active))
                ys_collapse_prefix.append(err_prefix[i])

                # -------------------------
                # Random scheme (mask reps, but total shots = args.traj)
                # Keep EXACT same set of shot seeds by splitting [0,traj) contiguously.
                # -------------------------
                reps = int(max(1, args.mask_reps))
                segments = split_traj(int(args.traj), reps)

                rep_means = []
                weighted_sum = 0.0
                weighted_cnt = 0

                for r, (offset, shots) in enumerate(segments):
                    rng = np.random.default_rng(int(args.mask_seed_base + 10007 * r + 1000003 * int(S_active)))
                    m_rand = mask_random(S_total, int(S_active), rng)

                    if shots <= 0:
                        continue

                    f_r = mc_mean_parallel_masked(
                        ex,
                        base_qpy, n, ideal_vec,
                        gamma=float(g),
                        traj=int(shots),
                        chunk=int(args.chunk),
                        seed_base=seed_base,
                        seed_offset=int(offset),
                        blas_threads=int(args.blas_threads),
                        max_bond_dim=int(args.max_bond_dim),
                        threshold=float(args.threshold),
                        active_mask=m_rand,
                    )

                    rep_means.append(float(f_r))
                    weighted_sum += float(f_r) * int(shots)
                    weighted_cnt += int(shots)

                f_rand = weighted_sum / max(1, weighted_cnt)
                err_random_mean[i] = 1.0 - float(f_rand)

                # std over mask replicates (means), purely “mask position” variability indicator
                rep_means = np.asarray(rep_means, dtype=float)
                err_random_std[i] = float(np.std(1.0 - rep_means)) if rep_means.size > 1 else 0.0

                xs_collapse_random.append(float(g) * float(S_active))
                ys_collapse_random.append(err_random_mean[i])

                print(f"[gamma={g:g}] S={int(S_active):4d}/{S_total}  "
                      f"err_prefix={err_prefix[i]:.4e}  err_random={err_random_mean[i]:.4e} ± {err_random_std[i]:.2e}",
                      flush=True)

            # Save per-gamma figure
            fig_path = out_dir / f"error_vs_sites_gamma_{g:g}.png"
            plot_error_vs_sites(
                out_path=fig_path,
                S_list=S_list.astype(float),
                err_prefix=err_prefix,
                err_random_mean=err_random_mean,
                err_random_std=err_random_std,
                gamma=float(g),
                title="Error vs active sites (Prefix vs Random)",
            )

            results["by_gamma"][str(g)] = {
                "S_list": S_list.tolist(),
                "error_prefix": err_prefix.tolist(),
                "error_random_mean": err_random_mean.tolist(),
                "error_random_std_over_masks": err_random_std.tolist(),
            }

    # Collapse plot across all gammas
    xs_collapse_prefix = np.asarray(xs_collapse_prefix, dtype=float)
    ys_collapse_prefix = np.asarray(ys_collapse_prefix, dtype=float)
    xs_collapse_random = np.asarray(xs_collapse_random, dtype=float)
    ys_collapse_random = np.asarray(ys_collapse_random, dtype=float)

    plot_collapse_gammaS(
        out_path=out_dir / "collapse_error_vs_gammaS.png",
        xs_prefix=xs_collapse_prefix,
        ys_prefix=ys_collapse_prefix,
        xs_random=xs_collapse_random,
        ys_random=ys_collapse_random,
        title="Collapse test: error vs gamma*S (dose scaling)",
    )

    elapsed = time.time() - t0
    results["elapsed_sec"] = float(elapsed)
    results["workers_used"] = int(workers)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"workers={workers}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        "",
        f"n={n}, depth={int(args.depth)}, seed={int(args.seed)}",
        f"S_total={S_total}, S_list={S_list.tolist()}",
        f"gammas={gammas}, traj_per_point={int(args.traj)}, random_mask_reps={int(args.mask_reps)}",
        "",
        "Saved per-gamma figures: error_vs_sites_gamma_*.png",
        "Saved collapse figure: collapse_error_vs_gammaS.png",
        "Saved meta.json, results.json",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
