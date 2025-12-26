# scripts/exp_mc_convergence_by_sites_strongsim_YAQS_WIN.py
from __future__ import annotations

import argparse
import csv
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
from typing import List, Tuple, Dict

import numpy as np

# --- IMPORTANT (Windows + multiprocessing): avoid Tk backend issues ---
import matplotlib
matplotlib.use("Agg")
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
# Sites definition
# -----------------------------
@dataclass(frozen=True)
class Site:
    site_id: int
    inst_index: int
    op_name: str
    qubit: int

def enumerate_sites(circ: QuantumCircuit) -> List[Site]:
    """
    A site = (instruction index, one touched qubit).
    Deterministic order:
      for inst_idx, instruction in circuit.data:
         for q in instruction.qubits (in order):
            yield one site
    """
    sites: List[Site] = []
    sid = 0
    for inst_idx, ci in enumerate(circ.data):
        op = ci.operation
        name = op.name.lower()
        if name in ("measure", "barrier", "reset"):
            continue
        for qb in list(ci.qubits):
            sites.append(Site(site_id=sid, inst_index=inst_idx, op_name=op.name, qubit=int(qb._index)))
            sid += 1
    return sites

def pack_mask(mask_bool: np.ndarray) -> bytes:
    mask_bool = np.asarray(mask_bool, dtype=np.uint8)
    return np.packbits(mask_bool).tobytes()

def unpack_mask(mask_bytes: bytes, length: int) -> np.ndarray:
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    bits = np.unpackbits(arr)[:int(length)]
    return bits.astype(bool, copy=False)

# -----------------------------
# Pauli-jump baked, but only on ACTIVE sites
# -----------------------------
def apply_pauli_jump_active_sites(
    circuit: QuantumCircuit,
    gamma: float,
    seed: int,
    active_mask_bytes: bytes,
    active_len: int,
) -> QuantumCircuit:
    """
    After each gate site (gate, touched qubit), if that site is active:
      with prob gamma apply X/Y/Z uniformly
    """
    rng = np.random.default_rng(int(seed))
    active = unpack_mask(active_mask_bytes, int(active_len))

    noisy = QuantumCircuit(circuit.num_qubits)
    sid = 0
    for ci in circuit.data:
        op = ci.operation
        name = op.name.lower()
        if name in ("measure", "barrier", "reset"):
            continue

        qargs = list(ci.qubits)
        cargs = list(ci.clbits)
        noisy.append(op, qargs, cargs)

        for qb in qargs:
            if sid < active.size and active[sid]:
                if rng.random() < float(gamma):
                    pa = int(rng.integers(0, 3))
                    if pa == 0:
                        noisy.x(qb)
                    elif pa == 1:
                        noisy.y(qb)
                    else:
                        noisy.z(qb)
            sid += 1

    return noisy

# -----------------------------
# Paired seeds across shots
# -----------------------------
def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * shot_index) % (2**31 - 1))

# -----------------------------
# Worker: produce fidelity samples for a chunk
# -----------------------------
def _chunk_fidelities_active_sites(
    base_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    start: int,
    shots: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    active_mask_bytes: bytes,
    active_len: int,
) -> np.ndarray:
    _worker_init(int(blas_threads))
    base = strip_measurements(qpy_load_bytes(base_qpy))

    out = np.empty(int(shots), dtype=np.float64)
    for i in range(int(shots)):
        global_idx = int(start + i)
        s = shot_seed(int(seed_base), global_idx)

        noisy = apply_pauli_jump_active_sites(
            base, float(gamma), int(s),
            active_mask_bytes=active_mask_bytes,
            active_len=int(active_len),
        )
        noisy = strip_measurements(noisy)

        vec = run_strongsim_statevector(
            noisy, n=int(n),
            max_bond_dim=int(max_bond_dim),
            threshold=float(threshold),
        )
        out[i] = fidelity_pure(psi_ideal, vec)

    return out

def auto_chunk_size(traj: int, workers: int, tasks_per_worker: int) -> int:
    """
    Make enough tasks so cores stay busy:
      n_tasks ~ workers * tasks_per_worker
      chunk ~ traj / n_tasks
    """
    workers = max(1, int(workers))
    traj = max(1, int(traj))
    tasks_per_worker = max(2, int(tasks_per_worker))
    n_tasks = workers * tasks_per_worker
    ch = max(1, traj // n_tasks)
    return int(ch)

def mc_samples_parallel_active_sites(
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
    active_mask_bytes: bytes,
    active_len: int,
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
                _chunk_fidelities_active_sites,
                base_qpy, n, psi_ideal,
                float(gamma),
                int(start), int(shots), int(seed_base),
                int(blas_threads),
                int(max_bond_dim), float(threshold),
                active_mask_bytes, int(active_len),
            )
        )

    # collect in order
    arrs = [f.result() for f in futs]
    return np.concatenate(arrs, axis=0)

# -----------------------------
# Batch-resampling convergence curve
# -----------------------------
def convergence_curve_from_samples(
    fids: np.ndarray,
    N_list: List[int],
    num_batches: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Given fidelity samples of length Nmax:
      F_ref = mean(fids)
      For each N in N_list:
        repeat num_batches times:
           choose N indices uniformly (without replacement) from [0..Nmax-1]
           compute mean, take abs diff to F_ref
        y = mean(abs diff), yerr = std(abs diff)
    """
    rng = np.random.default_rng(int(seed))
    Nmax = int(fids.size)
    F_ref = float(np.mean(fids))

    xs = []
    ys = []
    ystd = []

    for N in N_list:
        N = int(N)
        if N <= 0:
            continue
        if N > Nmax:
            N = Nmax

        diffs = np.empty(int(num_batches), dtype=np.float64)
        for b in range(int(num_batches)):
            idx = rng.choice(Nmax, size=N, replace=False)
            m = float(np.mean(fids[idx]))
            diffs[b] = abs(m - F_ref)

        xs.append(N)
        ys.append(float(np.mean(diffs)))
        ystd.append(float(np.std(diffs)))

    return np.array(xs), np.array(ys), np.array(ystd), F_ref

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--num-qubits", type=int, default=5)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # YAQS
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)

    # fixed gamma
    ap.add_argument("--gamma", type=float, default=0.01)

    # MC max trajectories (reference)
    ap.add_argument("--traj-max", type=int, default=10000)

    # S list (active sites) will be generated from these
    ap.add_argument("--S-min", type=int, default=10)
    ap.add_argument("--S-step", type=int, default=10)
    ap.add_argument("--num-S-points", type=int, default=10)

    # convergence N list
    ap.add_argument("--N-list", type=str,
                    default="10,20,30,40,50,60,70,80,90,100,200,300,500,800,1000,2000,5000,10000")
    ap.add_argument("--num-batches", type=int, default=60)
    ap.add_argument("--resample-seed", type=int, default=777)

    # parallel
    ap.add_argument("--workers", type=int, default=0)  # 0 => all logical cores
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--tasks-per-worker", type=int, default=10)
    ap.add_argument("--auto-chunk", type=int, default=1)

    # transpile
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    # save samples
    ap.add_argument("--save-samples", type=int, default=1)

    # seed base offset
    ap.add_argument("--seed-base-offset", type=int, default=20251221)

    args = ap.parse_args()

    n = int(args.num_qubits)
    depth = int(args.depth)

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    # Build base circuit
    base = build_twolocal(num_qubits=n, depth=depth, seed=int(args.seed), add_measurements=False)
    base = strip_measurements(base)

    if int(args.use_transpile) == 1:
        base = transpile(
            base,
            optimization_level=int(args.opt_level),
            seed_transpiler=int(args.seed_transpiler),
        )
        base = strip_measurements(base)

    # Ideal target: no noise
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
    )

    # Enumerate sites and define S_list
    sites = enumerate_sites(base)
    S_total = int(len(sites))

    S_list = []
    cur = int(args.S_min)
    for _ in range(int(args.num_S_points)):
        if cur > S_total:
            break
        S_list.append(cur)
        cur += int(args.S_step)
    # ensure last point not too small: add S_total if not present
    if S_total not in S_list:
        S_list.append(S_total)
    # de-dup + sort
    S_list = sorted(set(int(x) for x in S_list if int(x) >= 0))

    # Parse N_list
    N_list = [int(x) for x in args.N_list.split(",") if x.strip()]
    N_list = sorted(set([x for x in N_list if x > 0]))
    # always cap to traj-max later
    traj_max = int(args.traj_max)

    gamma = float(args.gamma)

    # Report settings
    print("\n===== SETTINGS (MC convergence by sites) =====")
    print(f"n={n}, depth={depth}")
    print(f"gamma={gamma:g}")
    print(f"S_total={S_total}")
    print(f"S_list({len(S_list)} pts)={S_list}")
    print(f"traj_max(reference)={traj_max}")
    print(f"N_list={N_list}")
    print(f"num_batches={int(args.num_batches)}")
    print(f"workers={workers}, blas_threads={int(args.blas_threads)}, tasks_per_worker={int(args.tasks_per_worker)}")
    print("=============================================\n", flush=True)

    # Output dir
    out_dir = REPO_ROOT / "outputs" / "experiments" / "mc_convergence_by_sites" / f"n{n}_d{depth}_seed{args.seed}_gamma{gamma:g}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save site map
    with (out_dir / "sites_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "inst_index", "op_name", "qubit"])
        for s in sites:
            w.writerow([s.site_id, s.inst_index, s.op_name, s.qubit])

    meta = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "n": n,
        "depth": depth,
        "seed": int(args.seed),
        "gamma": gamma,
        "use_transpile": int(args.use_transpile),
        "opt_level": int(args.opt_level),
        "seed_transpiler": int(args.seed_transpiler),
        "yaqs": {"max_bond_dim": int(args.max_bond_dim), "threshold": float(args.threshold)},
        "S_total": S_total,
        "S_list": S_list,
        "traj_max": traj_max,
        "N_list": N_list,
        "num_batches": int(args.num_batches),
        "workers": workers,
        "blas_threads": int(args.blas_threads),
        "chunk": int(args.chunk),
        "auto_chunk": int(args.auto_chunk),
        "tasks_per_worker": int(args.tasks_per_worker),
        "save_samples": int(args.save_samples),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_qpy = qpy_bytes(base)
    seed_base = int(args.seed + int(args.seed_base_offset))

    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()

    # overlay plot data
    overlay = []

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        for S in S_list:
            S = int(S)
            # Prefix mask: first S sites active
            mask_prefix = np.zeros(S_total, dtype=bool)
            if S > 0:
                mask_prefix[:S] = True
            mask_bytes = pack_mask(mask_prefix)

            # chunk sizing for full-core utilization
            chunk = int(args.chunk)
            if int(args.auto_chunk) == 1 or chunk <= 0:
                chunk = auto_chunk_size(traj_max, workers, int(args.tasks_per_worker))

            print(f"--- Running samples: S={S} / {S_total}, traj_max={traj_max}, chunk={chunk} ---", flush=True)

            # 1) generate fidelity samples (length traj_max)
            fids = mc_samples_parallel_active_sites(
                ex,
                base_qpy, n, ideal_vec,
                gamma=gamma,
                traj=traj_max,
                chunk=chunk,
                seed_base=seed_base,   # reproducible across S
                blas_threads=int(args.blas_threads),
                max_bond_dim=int(args.max_bond_dim),
                threshold=float(args.threshold),
                active_mask_bytes=mask_bytes,
                active_len=S_total,
            )

            if fids.size != traj_max:
                raise RuntimeError(f"Internal error: got fids.size={fids.size}, expected {traj_max}")

            if int(args.save_samples) == 1:
                np.save(out_dir / f"fids_S{S}.npy", fids)

            # 2) resampling convergence curve
            xs, ys, ystd, F_ref = convergence_curve_from_samples(
                fids=fids,
                N_list=[min(x, traj_max) for x in N_list],
                num_batches=int(args.num_batches),
                seed=int(args.resample_seed) + 1000 * S,
            )

            # save csv
            csv_path = out_dir / f"mc_convergence_S{S}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["N", "mean_abs_dev", "std_abs_dev", "F_ref(mean over traj_max)"])
                for i in range(xs.size):
                    w.writerow([int(xs[i]), float(ys[i]), float(ystd[i]), float(F_ref)])

            # plot per S
            plt.figure()
            plt.errorbar(xs, ys, yerr=ystd, marker="o", capsize=3)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("N trajectories (subset size)")
            plt.ylabel("E[ | mean(F)_N - mean(F)_ref | ]  (batch-averaged)")
            plt.title(f"MC convergence (gamma={gamma:g}) with prefix active sites S={S} (ref N={traj_max})")
            plt.tight_layout()
            plt.savefig(out_dir / f"mc_convergence_S{S}.png", dpi=220, bbox_inches="tight")
            plt.close()

            overlay.append((S, xs.copy(), ys.copy(), ystd.copy()))

            print(f"[DONE] S={S:4d}  F_ref={F_ref:.6f}  saved {csv_path.name} and mc_convergence_S{S}.png", flush=True)

    # overlay plot (multiple S curves)
    plt.figure()
    for (S, xs, ys, ystd) in overlay:
        plt.plot(xs, ys, marker="o", label=f"S={S}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N trajectories (subset size)")
    plt.ylabel("E[ | mean(F)_N - mean(F)_ref | ]")
    plt.title(f"MC convergence overlay (gamma={gamma:g}, ref N={traj_max})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mc_convergence_overlay.png", dpi=220, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - t0
    (out_dir / "report.txt").write_text(
        "\n".join([
            f"OUTPUT DIR: {out_dir.resolve()}",
            f"elapsed_sec={elapsed:.2f}",
            f"n={n}, depth={depth}, gamma={gamma:g}",
            f"S_total={S_total}",
            f"S_list={S_list}",
            f"traj_max(ref)={traj_max}",
            f"N_list={N_list}",
            f"num_batches={int(args.num_batches)}",
            f"workers={workers}, blas_threads={int(args.blas_threads)}, tasks_per_worker={int(args.tasks_per_worker)}",
            "",
            "Saved per-S plots: mc_convergence_S*.png",
            "Saved overlay plot: mc_convergence_overlay.png",
            "Saved per-S tables: mc_convergence_S*.csv",
            "Saved site map: sites_map.csv",
            "Optionally saved samples: fids_S*.npy",
        ]) + "\n",
        encoding="utf-8",
    )

    print("\nDONE.")
    print(f"OUTPUT DIR: {out_dir.resolve()}")
    print(f"elapsed_sec={elapsed:.2f}", flush=True)

if __name__ == "__main__":
    main()
