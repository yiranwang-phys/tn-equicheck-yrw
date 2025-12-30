# scripts/final_test_spacetime_zone_heatmap_dose_YAQS.py
from __future__ import annotations

import argparse
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator
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

def _make_mps(n: int, pad: int = 0):
    from mqt.yaqs.core.data_structures.networks import MPS
    # Try signature variants (YAQS versions differ)
    try:
        if pad > 0:
            return MPS(n, state="zeros", pad=int(pad))
        return MPS(n, state="zeros")
    except TypeError:
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
    if "num_traj" in params:
        kwargs["num_traj"] = 1
    if "show_progress" in params:
        kwargs["show_progress"] = False

    return StrongSimParams(**kwargs)

def run_strongsim_statevector(
    circ: QuantumCircuit,
    n: int,
    max_bond_dim: int,
    threshold: float,
    pad: int = 0,
) -> np.ndarray:
    """
    YAQS strong circuit simulation (no noise_model) and return final statevector.
    Noise is baked into the circuit as explicit Pauli gates.
    """
    from mqt.yaqs import simulator

    state = _make_mps(int(n), pad=int(pad))
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
# Space-time "active sites"
# site := (instruction index, touched qubit index)
# -----------------------------
@dataclass(frozen=True)
class Site:
    inst_i: int
    q: int

def enumerate_sites(circ: QuantumCircuit, skip_ops: Tuple[str, ...] = ("measure", "barrier", "reset")) -> List[Site]:
    sites: List[Site] = []
    for inst_i, ci in enumerate(circ.data):
        op = ci.operation
        if op.name in skip_ops:
            continue
        for qb in ci.qubits:
            qidx = circ.find_bit(qb).index
            sites.append(Site(inst_i=inst_i, q=int(qidx)))
    return sites

def build_spacetime_mask(
    sites: List[Site],
    *,
    inst_cut: int,
    k_active: int,
) -> np.ndarray:
    """
    Active if:
      - instruction index < inst_cut  (temporal prefix)
      - qubit index < k_active        (spatial prefix)
    """
    mask = np.zeros(len(sites), dtype=np.uint8)
    for i, s in enumerate(sites):
        if s.inst_i < int(inst_cut) and s.q < int(k_active):
            mask[i] = 1
    return mask

def _clone_circuit_structure(circ: QuantumCircuit) -> Tuple[QuantumCircuit, Dict, Dict]:
    out = QuantumCircuit(*circ.qregs, *circ.cregs, name=getattr(circ, "name", None))
    out.global_phase = getattr(circ, "global_phase", 0.0)
    qmap = {circ.qubits[i]: out.qubits[i] for i in range(len(circ.qubits))}
    cmap = {circ.clbits[i]: out.clbits[i] for i in range(len(circ.clbits))}
    return out, qmap, cmap

def apply_pauli_jump_active_sites(
    circ: QuantumCircuit,
    gamma: float,
    rng: Generator,
    active_mask: np.ndarray,
    *,
    skip_ops: Tuple[str, ...] = ("measure", "barrier", "reset"),
) -> Tuple[QuantumCircuit, int]:
    """
    After each gate (except skip), for each touched qubit:
      if active_mask[site_id]==1 and rand<gamma: insert X/Y/Z uniformly.

    Returns:
      (noisy_circuit, n_noise_ops_inserted)
    """
    out, qmap, cmap = _clone_circuit_structure(circ)

    sid = 0
    n_noise = 0

    for inst_i, ci in enumerate(circ.data):
        op = ci.operation
        qargs = [qmap[q] for q in ci.qubits]
        cargs = [cmap[c] for c in ci.clbits]
        out.append(op, qargs, cargs)

        if op.name in skip_ops:
            continue

        for q in qargs:
            if sid >= len(active_mask):
                # should not happen if mask built from enumerate_sites on same circ
                break
            if active_mask[sid] == 1 and rng.random() < float(gamma):
                r = int(rng.integers(0, 3))
                if r == 0:
                    out.x(q)
                elif r == 1:
                    out.y(q)
                else:
                    out.z(q)
                n_noise += 1
            sid += 1

    return out, n_noise


# -----------------------------
# Seeds (paired)
# -----------------------------
def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * shot_index) % (2**31 - 1))


# -----------------------------
# MC worker for one zone cell
# -----------------------------
def _chunk_zone_fidelities(
    circ_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    mask_bytes: bytes,
    start: int,
    shots: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    pad: int,
) -> Tuple[np.ndarray, np.ndarray]:
    _worker_init(int(blas_threads))

    base = strip_measurements(qpy_load_bytes(circ_qpy))
    mask = np.frombuffer(mask_bytes, dtype=np.uint8)

    fids = np.empty(int(shots), dtype=np.float64)
    nnoise = np.empty(int(shots), dtype=np.float64)

    for i in range(int(shots)):
        global_idx = int(start + i)
        s = shot_seed(int(seed_base), global_idx)
        rng = np.random.default_rng(int(s))

        noisy, n_ops = apply_pauli_jump_active_sites(
            base, float(gamma), rng, mask
        )
        noisy = strip_measurements(noisy)

        vec = run_strongsim_statevector(
            noisy,
            n=int(n),
            max_bond_dim=int(max_bond_dim),
            threshold=float(threshold),
            pad=int(pad),
        )
        fids[i] = fidelity_pure(psi_ideal, vec)
        nnoise[i] = float(n_ops)

    return fids, nnoise


def mc_mean_zone_parallel(
    ex,
    circ_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    mask: np.ndarray,
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    pad: int,
) -> Tuple[float, float]:
    """
    Returns (mean_fidelity, mean_noise_ops) for one zone.
    """
    traj = int(traj)
    chunk = int(max(1, chunk))
    n_chunks = (traj + chunk - 1) // chunk

    mask_bytes = mask.astype(np.uint8, copy=False).tobytes()

    futs = []
    for k in range(n_chunks):
        start = k * chunk
        shots = min(chunk, traj - start)
        futs.append(
            ex.submit(
                _chunk_zone_fidelities,
                circ_qpy, n, psi_ideal, float(gamma),
                mask_bytes,
                int(start), int(shots), int(seed_base),
                int(blas_threads), int(max_bond_dim), float(threshold), int(pad),
            )
        )

    s_f = 0.0
    s_n = 0.0
    cnt = 0
    for f in futs:
        fids, nnoise = f.result()
        s_f += float(np.sum(fids))
        s_n += float(np.sum(nnoise))
        cnt += int(fids.size)
    if cnt <= 0:
        return 0.0, 0.0
    return s_f / cnt, s_n / cnt


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--num-qubits", type=int, default=5)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=50)

    # zone grid
    ap.add_argument("--tfrac-num", type=int, default=4, help="number of time fractions in [0,1], mapped to inst_cut via floor")
    ap.add_argument("--k-num", type=int, default=-1, help="if -1 use 0..n; else use 0..k_num")
    ap.add_argument("--use-inst-axis", type=int, default=1, help="also save a heatmap indexed by inst_cut instead of tfrac")

    # YAQS strongsim controls
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)
    ap.add_argument("--pad", type=int, default=0)

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

    # Ideal vector
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
        pad=int(args.pad),
    )

    sites = enumerate_sites(base)
    S_total = len(sites)
    inst_max = max([s.inst_i for s in sites], default=0) + 1

    # Grid axes
    tfracs = np.linspace(0.0, 1.0, int(args.tfrac_num))
    inst_cuts = np.array([int(np.floor(tf * inst_max)) for tf in tfracs], dtype=int)

    if int(args.k_num) < 0:
        ks = np.arange(0, n + 1, dtype=int)
    else:
        ks = np.arange(0, int(args.k_num) + 1, dtype=int)

    # outputs
    out_dir = REPO_ROOT / "outputs" / "experiments" / "final_test_spacetime_zone_heatmap_dose" / f"n{n}_d{args.depth}_seed{args.seed}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "n": n,
        "depth": int(args.depth),
        "seed": int(args.seed),
        "gamma": float(args.gamma),
        "traj": int(args.traj),
        "chunk": int(args.chunk),
        "workers": int(workers),
        "blas_threads": int(args.blas_threads),
        "yaqs": {"max_bond_dim": int(args.max_bond_dim), "threshold": float(args.threshold), "pad": int(args.pad)},
        "grid": {"tfracs": tfracs.tolist(), "inst_cuts": inst_cuts.tolist(), "ks": ks.tolist()},
        "S_total": int(S_total),
        "inst_max": int(inst_max),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_qpy = qpy_bytes(base)
    seed_base = int(args.seed + 20251230)

    from concurrent.futures import ProcessPoolExecutor

    # allocate results
    meanF = np.zeros((len(ks), len(tfracs)), dtype=float)
    meanNoiseOps = np.zeros_like(meanF)
    S_grid = np.zeros_like(meanF)
    dose_grid = np.zeros_like(meanF)

    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=min(int(workers), 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:

        for ix_k, k_active in enumerate(ks):
            for ix_t, inst_cut in enumerate(inst_cuts):
                mask = build_spacetime_mask(sites, inst_cut=int(inst_cut), k_active=int(k_active))
                S = int(np.sum(mask))
                S_grid[ix_k, ix_t] = float(S)
                dose_grid[ix_k, ix_t] = float(args.gamma) * float(S)

                mf, mn = mc_mean_zone_parallel(
                    ex,
                    base_qpy, n, ideal_vec,
                    gamma=float(args.gamma),
                    mask=mask,
                    traj=int(args.traj),
                    chunk=int(args.chunk),
                    seed_base=seed_base,   # paired
                    blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim),
                    threshold=float(args.threshold),
                    pad=int(args.pad),
                )
                meanF[ix_k, ix_t] = float(mf)
                meanNoiseOps[ix_k, ix_t] = float(mn)

                print(
                    f"[k={k_active:2d}, inst_cut={inst_cut:3d}, tfrac={tfracs[ix_t]:.2f}] "
                    f"S={S:5d}, dose=gamma*S={dose_grid[ix_k, ix_t]:.3f}, "
                    f"mean_noise_ops={mn:.3f}, meanF={mf:.6f}",
                    flush=True,
                )

    elapsed = time.time() - t0

    # Save arrays
    np.save(out_dir / "meanF_grid.npy", meanF)
    np.save(out_dir / "meanNoiseOps_grid.npy", meanNoiseOps)
    np.save(out_dir / "S_grid.npy", S_grid)
    np.save(out_dir / "dose_grid.npy", dose_grid)

    # 1) Heatmap: error = 1-meanF
    err = 1.0 - meanF

    plt.figure()
    plt.imshow(
        err,
        origin="lower",
        aspect="auto",
        extent=[tfracs[0], tfracs[-1], ks[0], ks[-1]],
    )
    plt.colorbar(label="error = 1 - mean fidelity")
    plt.xlabel("time fraction (mapped to inst_cut via floor)")
    plt.ylabel("active qubits K (prefix: 0..K-1)")
    plt.title(f"Spacetime-zone heatmap (gamma={float(args.gamma):g}, traj={int(args.traj)})")
    plt.tight_layout()
    plt.savefig(out_dir / "spacetime_zone_error_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Heatmap: dose = gamma*S
    plt.figure()
    plt.imshow(
        dose_grid,
        origin="lower",
        aspect="auto",
        extent=[tfracs[0], tfracs[-1], ks[0], ks[-1]],
    )
    plt.colorbar(label="dose = gamma * (#active sites)")
    plt.xlabel("time fraction (mapped to inst_cut via floor)")
    plt.ylabel("active qubits K (prefix: 0..K-1)")
    plt.title("Dose heatmap (why error is blocky)")
    plt.tight_layout()
    plt.savefig(out_dir / "spacetime_zone_dose_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Sanity heatmap: mean inserted noise ops (should correlate with dose)
    plt.figure()
    plt.imshow(
        meanNoiseOps,
        origin="lower",
        aspect="auto",
        extent=[tfracs[0], tfracs[-1], ks[0], ks[-1]],
    )
    plt.colorbar(label="mean inserted noise ops per trajectory")
    plt.xlabel("time fraction")
    plt.ylabel("active qubits K")
    plt.title("Observed mean noise ops (MC sanity check)")
    plt.tight_layout()
    plt.savefig(out_dir / "spacetime_zone_mean_noise_ops_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Collapse: error vs dose
    x = dose_grid.reshape(-1)
    y = err.reshape(-1)
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("dose = gamma * (#active sites)")
    plt.ylabel("error = 1 - meanF")
    plt.title("Dose collapse (should explain block structure)")
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_dose_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    results = {
        "elapsed_sec": float(elapsed),
        "S_total": int(S_total),
        "inst_max": int(inst_max),
        "meanF_grid_shape": list(meanF.shape),
        "notes": {
            "why_blocks": "Different cells have different #active sites S, so expected noise events ~ gamma*S differ.",
            "active_qubits": "K means allow noise only on qubits 0..K-1 (spatial prefix).",
            "time_fraction": "mapped to inst_cut=floor(tfrac*inst_max), so many small tfrac may map to inst_cut=0 -> no noise.",
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"n={n}, depth={int(args.depth)}, gamma={float(args.gamma):g}, traj={int(args.traj)}",
        f"workers={int(workers)}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        "",
        "Saved:",
        "  spacetime_zone_error_heatmap.png",
        "  spacetime_zone_dose_heatmap.png",
        "  spacetime_zone_mean_noise_ops_heatmap.png",
        "  error_vs_dose_scatter.png",
        "  meanF_grid.npy, meanNoiseOps_grid.npy, S_grid.npy, dose_grid.npy",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
