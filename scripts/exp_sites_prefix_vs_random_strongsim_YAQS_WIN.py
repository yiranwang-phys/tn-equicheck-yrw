# scripts/exp_sites_prefix_vs_random_strongsim_YAQS_WIN.py
from __future__ import annotations

# ==========================================================
# CRITICAL: Force non-GUI Matplotlib backend on Windows spawn
# Must be set BEFORE importing matplotlib.pyplot
# ==========================================================
import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import csv
import io
import json
import platform
import sys
import time
import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

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
# Paired seeds across all candidates
# -----------------------------
def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * shot_index) % (2**31 - 1))

# -----------------------------
# MC worker
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

def auto_chunk_size(traj: int, workers: int, tasks_per_worker: int = 10) -> int:
    """
    Make enough tasks so cores stay busy:
      n_tasks ~ workers * tasks_per_worker
      chunk ~ traj / n_tasks  (>=1)
    """
    workers = max(1, int(workers))
    traj = max(1, int(traj))
    n_tasks = workers * max(2, int(tasks_per_worker))
    ch = max(1, traj // n_tasks)
    return int(ch)

def mc_mean_parallel_active_sites(
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
                _chunk_fidelities_active_sites,
                base_qpy, n, psi_ideal,
                float(gamma),
                int(start), int(shots), int(seed_base),
                int(blas_threads),
                int(max_bond_dim), float(threshold),
                active_mask_bytes, int(active_len),
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
# S_list generator (exactly N points, smooth; fallback fill random)
# -----------------------------
def make_S_list_randomish_smooth(
    S_total: int,
    num_points: int,
    seed: int,
    include_zero: bool = True,
    include_total: bool = True,
) -> List[int]:
    """
    Build ~smooth S grid with exactly num_points points (as much as possible),
    primarily by linspace + rounding (smooth), then fill with random integers
    if rounding collapses duplicates.

    Returns sorted unique list; if uniqueness reduces count, we add random values.
    """
    S_total = int(S_total)
    num_points = int(max(2, num_points))
    rng = np.random.default_rng(int(seed))

    xs = np.linspace(0, S_total, num_points)
    S_list = [int(round(x)) for x in xs]

    if not include_zero:
        S_list = [x for x in S_list if x != 0]
    if not include_total:
        S_list = [x for x in S_list if x != S_total]

    # ensure boundaries if requested
    if include_zero:
        S_list.append(0)
    if include_total:
        S_list.append(S_total)

    # unique and clamp
    S_list = [int(min(S_total, max(0, x))) for x in S_list]
    S_list = sorted(set(S_list))

    # fill if too few unique points
    target = num_points
    if include_zero and include_total:
        target = max(target, 2)  # still target num_points overall
    while len(S_list) < target:
        S_list.append(int(rng.integers(0, S_total + 1)))
        S_list = sorted(set(S_list))

    # if too many because we forced boundaries, trim but keep 0 and S_total
    if len(S_list) > target:
        keep = set()
        if include_zero:
            keep.add(0)
        if include_total:
            keep.add(S_total)

        middle = [x for x in S_list if x not in keep]
        # pick evenly from middle
        if len(keep) >= target:
            S_list = sorted(list(keep))[:target]
        else:
            need = target - len(keep)
            if need <= 0:
                S_list = sorted(list(keep))
            else:
                pick_idx = np.linspace(0, max(0, len(middle) - 1), need)
                picked = [middle[int(round(i))] for i in pick_idx] if middle else []
                S_list = sorted(set(list(keep) + picked))

    # final safety
    S_list = sorted(set(int(min(S_total, max(0, x))) for x in S_list))
    return S_list

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # fixed goal: n=5, depth default (you can change)
    ap.add_argument("--num-qubits", type=int, default=5)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # YAQS
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)

    # gammas required
    ap.add_argument("--gammas", type=str, default="0.1,0.01,0.001")

    # MC budget
    ap.add_argument("--traj", type=int, default=800, help="MC trajectories per (gamma, S, mask). Set 2000 for final.")
    ap.add_argument("--num-masks", type=int, default=12, help="random masks per S (mean±std over masks)")
    ap.add_argument("--seed-base-offset", type=int, default=20251221)

    # S_list control: "try 10 sites"
    ap.add_argument("--num-S-points", type=int, default=10, help="number of S points to test (try 10 sites)")

    # parallel
    ap.add_argument("--workers", type=int, default=0)  # 0 => all logical cores
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=0, help="0 => auto chunk to keep all cores busy")
    ap.add_argument("--auto-chunk", type=int, default=1)
    ap.add_argument("--tasks-per-worker", type=int, default=10, help="higher => smaller chunks => more parallel tasks")

    # transpile
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

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

    # Ideal target: no noise
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
    )

    # Enumerate sites
    sites = enumerate_sites(base)
    S_total = len(sites)

    # Choose S_list (try 10 points, smooth-ish; fill random if duplicates)
    S_list = make_S_list_randomish_smooth(
        S_total=S_total,
        num_points=int(args.num_S_points),
        seed=int(args.seed) + 777,
        include_zero=True,
        include_total=True,
    )

    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    gammas = list(gammas)

    # Report settings BEFORE running (as you requested)
    print("\n===== SETTINGS =====")
    print(f"n={n}, depth={depth}")
    print(f"S_total={S_total}")
    print(f"S_list({len(S_list)} pts)={S_list}")
    print(f"gammas={gammas}")
    print(f"traj per (gamma,S,mask)={int(args.traj)}, num_masks={int(args.num_masks)}")
    print(f"workers={workers}, blas_threads={int(args.blas_threads)}")
    print(f"auto_chunk={int(args.auto_chunk)}, tasks_per_worker={int(args.tasks_per_worker)}, manual_chunk={int(args.chunk)}")
    print("====================\n", flush=True)

    # Output dir
    out_dir = REPO_ROOT / "outputs" / "experiments" / "sites_prefix_vs_random" / f"n{n}_d{depth}_seed{args.seed}" / _ts()
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
        "use_transpile": int(args.use_transpile),
        "opt_level": int(args.opt_level),
        "seed_transpiler": int(args.seed_transpiler),
        "yaqs": {"max_bond_dim": int(args.max_bond_dim), "threshold": float(args.threshold)},
        "S_total": S_total,
        "S_list": S_list,
        "gammas": gammas,
        "traj": int(args.traj),
        "num_masks": int(args.num_masks),
        "workers": workers,
        "blas_threads": int(args.blas_threads),
        "chunk": int(args.chunk),
        "auto_chunk": int(args.auto_chunk),
        "tasks_per_worker": int(args.tasks_per_worker),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_qpy = qpy_bytes(base)
    seed_base = int(args.seed + int(args.seed_base_offset))

    from concurrent.futures import ProcessPoolExecutor

    # collect collapse points
    collapse_prefix_x = []
    collapse_prefix_y = []
    collapse_random_x = []
    collapse_random_y = []

    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        for gamma in gammas:
            # Table rows: S, prefix_err, random_mean_err, random_std_err
            rows = []

            prefix_errs = []
            random_mean_errs = []
            random_std_errs = []

            # RNG for random masks (mask choice), independent of shot seeds
            # make it depend on gamma + seed so each gamma has stable but different masks
            rng_masks = np.random.default_rng(12345 + int(1e6 * gamma) + int(args.seed))

            for S in S_list:
                S = int(S)

                # chunk auto-tune to keep cores busy
                chunk = int(args.chunk)
                if int(args.auto_chunk) == 1 or chunk <= 0:
                    chunk = auto_chunk_size(
                        traj=int(args.traj),
                        workers=workers,
                        tasks_per_worker=int(args.tasks_per_worker),
                    )

                # -----------------------------
                # Prefix mask: first S sites
                # -----------------------------
                mask_prefix = np.zeros(S_total, dtype=bool)
                if S > 0:
                    mask_prefix[:S] = True
                mask_prefix_bytes = pack_mask(mask_prefix)

                meanF_prefix = mc_mean_parallel_active_sites(
                    ex,
                    base_qpy, n, ideal_vec,
                    gamma=float(gamma),
                    traj=int(args.traj),
                    chunk=int(chunk),
                    seed_base=int(seed_base),  # paired seeds across all candidates
                    blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim),
                    threshold=float(args.threshold),
                    active_mask_bytes=mask_prefix_bytes,
                    active_len=int(S_total),
                )
                err_prefix = 1.0 - float(meanF_prefix)

                # -----------------------------
                # Random masks: mean ± std over masks
                # Note: if S==0 or S==S_total, std may be 0 by construction.
                # -----------------------------
                errs_rand = []
                for m in range(int(args.num_masks)):
                    mask_rand = np.zeros(S_total, dtype=bool)
                    if S > 0:
                        idx = rng_masks.choice(S_total, size=S, replace=False)
                        mask_rand[idx] = True
                    mask_rand_bytes = pack_mask(mask_rand)

                    meanF_rand = mc_mean_parallel_active_sites(
                        ex,
                        base_qpy, n, ideal_vec,
                        gamma=float(gamma),
                        traj=int(args.traj),
                        chunk=int(chunk),
                        seed_base=int(seed_base),  # paired seeds (same shot seeds)
                        blas_threads=int(args.blas_threads),
                        max_bond_dim=int(args.max_bond_dim),
                        threshold=float(args.threshold),
                        active_mask_bytes=mask_rand_bytes,
                        active_len=int(S_total),
                    )
                    errs_rand.append(1.0 - float(meanF_rand))

                err_rand_mean = float(np.mean(errs_rand))
                err_rand_std = float(np.std(errs_rand))

                rows.append([S, err_prefix, err_rand_mean, err_rand_std])

                prefix_errs.append(err_prefix)
                random_mean_errs.append(err_rand_mean)
                random_std_errs.append(err_rand_std)

                # collapse points: x = gamma * S
                collapse_prefix_x.append(float(gamma) * float(S))
                collapse_prefix_y.append(err_prefix)
                collapse_random_x.append(float(gamma) * float(S))
                collapse_random_y.append(err_rand_mean)

                print(
                    f"[gamma={gamma:g}] S={S:4d}  prefix_err={err_prefix:.6f}  "
                    f"random_err={err_rand_mean:.6f} ± {err_rand_std:.6f}  (chunk={chunk})",
                    flush=True
                )

            # save CSV table for this gamma
            csv_path = out_dir / f"table_error_vs_sites_gamma_{gamma:g}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["S_active", "prefix_error", "random_error_mean", "random_error_std"])
                for r in rows:
                    w.writerow(r)

            # plot error vs S for this gamma
            plt.figure()
            plt.plot(S_list, prefix_errs, marker="o", label="Prefix sites")
            plt.errorbar(
                S_list, random_mean_errs, yerr=random_std_errs,
                marker="o", capsize=3, label="Random sites (mean ± std over masks)"
            )
            plt.xlabel("Active sites S")
            plt.ylabel("Error = 1 - mean fidelity")
            plt.title(f"Error vs active sites (Prefix vs Random)  (gamma={gamma:g})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"error_vs_sites_gamma_{gamma:g}.png", dpi=200, bbox_inches="tight")
            plt.close()

    # collapse plot: error vs gamma*S
    plt.figure()
    plt.scatter(collapse_prefix_x, collapse_prefix_y, label="Prefix")
    plt.scatter(collapse_random_x, collapse_random_y, label="Random")
    plt.xlabel("gamma * S (expected jump-count scale)")
    plt.ylabel("Error = 1 - mean fidelity")
    plt.title("Collapse test: error vs gamma*S (dose scaling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "collapse_error_vs_gammaS.png", dpi=200, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - t0
    (out_dir / "report.txt").write_text(
        "\n".join([
            f"OUTPUT DIR: {out_dir.resolve()}",
            f"elapsed_sec={elapsed:.2f}",
            f"n={n}, depth={depth}, S_total={S_total}",
            f"S_list={S_list}",
            f"gammas={gammas}",
            f"traj={int(args.traj)}, num_masks={int(args.num_masks)}",
            f"workers={workers}, blas_threads={int(args.blas_threads)}",
            f"auto_chunk={int(args.auto_chunk)}, tasks_per_worker={int(args.tasks_per_worker)}, chunk={int(args.chunk)}",
            "",
            "Saved per-gamma plots: error_vs_sites_gamma_*.png",
            "Saved collapse plot: collapse_error_vs_gammaS.png",
            "Saved tables: table_error_vs_sites_gamma_*.csv",
            "Saved site map: sites_map.csv",
        ]) + "\n",
        encoding="utf-8"
    )

    print("\nDONE.")
    print(f"OUTPUT DIR: {out_dir.resolve()}")
    print(f"elapsed_sec={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
