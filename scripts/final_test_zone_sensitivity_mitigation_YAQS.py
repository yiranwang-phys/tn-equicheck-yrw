# scripts/final_test_zone_sensitivity_mitigation_YAQS.py
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
# Space-time "sites"
# site := (instruction-layer index, touched qubit index)
#
# IMPORTANT FIX:
# We compress "instruction index" to consecutive layers for non-skip ops,
# so inst axis is aligned (no gaps from skipped ops or original circ.data indexing).
# -----------------------------
@dataclass(frozen=True)
class Site:
    layer: int
    q: int

def enumerate_sites(
    circ: QuantumCircuit,
    skip_ops: Tuple[str, ...] = ("measure", "barrier", "reset"),
) -> Tuple[List[Site], int]:
    sites: List[Site] = []
    layer_map: Dict[int, int] = {}
    next_layer = 0

    for inst_i, ci in enumerate(circ.data):
        op = ci.operation
        if op.name in skip_ops:
            continue
        if inst_i not in layer_map:
            layer_map[inst_i] = next_layer
            next_layer += 1
        layer = layer_map[inst_i]
        for qb in ci.qubits:
            qidx = circ.find_bit(qb).index
            sites.append(Site(layer=layer, q=int(qidx)))

    return sites, next_layer  # inst_max = number of layers

def qubit_subset_prefix(n: int, k: int) -> np.ndarray:
    k = int(max(0, min(n, k)))
    mask = np.zeros(n, dtype=np.uint8)
    mask[:k] = 1
    return mask

def qubit_subset_random(n: int, k: int, rng: Generator) -> np.ndarray:
    k = int(max(0, min(n, k)))
    mask = np.zeros(n, dtype=np.uint8)
    if k <= 0:
        return mask
    idx = rng.choice(np.arange(n), size=k, replace=False)
    mask[idx] = 1
    return mask

def build_zone_mask(
    sites: List[Site],
    *,
    layer_lo: int,
    layer_hi: int,
    qmask: np.ndarray,
) -> np.ndarray:
    """zone mask over sites: active if layer_lo <= layer < layer_hi and qmask[q]==1"""
    layer_lo = int(layer_lo)
    layer_hi = int(layer_hi)
    out = np.zeros(len(sites), dtype=np.uint8)
    for i, s in enumerate(sites):
        if layer_lo <= s.layer < layer_hi and int(qmask[s.q]) == 1:
            out[i] = 1
    return out

def layer_from_tfrac(inst_max: int, t: float) -> int:
    t = float(min(1.0, max(0.0, t)))
    return int(np.floor(t * inst_max))

# -----------------------------
# Noise insertion
# -----------------------------
def _clone_circuit_structure(circ: QuantumCircuit) -> Tuple[QuantumCircuit, Dict, Dict]:
    out = QuantumCircuit(*circ.qregs, *circ.cregs, name=getattr(circ, "name", None))
    out.global_phase = getattr(circ, "global_phase", 0.0)
    qmap = {circ.qubits[i]: out.qubits[i] for i in range(len(circ.qubits))}
    cmap = {circ.clbits[i]: out.clbits[i] for i in range(len(circ.clbits))}
    return out, qmap, cmap

def apply_pauli_jump_weighted(
    circ: QuantumCircuit,
    gamma: float,
    rng: Generator,
    *,
    mode: str,
    zone_mask: np.ndarray,
    protect_mask: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    skip_ops: Tuple[str, ...] = ("measure", "barrier", "reset"),
) -> Tuple[QuantumCircuit, int]:
    """
    mode:
      - "zone_only": only sites with zone_mask==1 can have noise with prob=gamma
      - "full_protect": all sites can have noise; if protect_mask==1 use prob=alpha*gamma else gamma
    """
    out, qmap, cmap = _clone_circuit_structure(circ)
    sid = 0
    n_noise = 0

    if protect_mask is None:
        protect_mask = np.zeros_like(zone_mask, dtype=np.uint8)

    gamma = float(gamma)
    alpha = float(alpha)

    for ci in circ.data:
        op = ci.operation
        qargs = [qmap[q] for q in ci.qubits]
        cargs = [cmap[c] for c in ci.clbits]
        out.append(op, qargs, cargs)

        if op.name in skip_ops:
            continue

        for _q in qargs:
            if sid >= len(zone_mask):
                break

            if mode == "zone_only":
                p = gamma if int(zone_mask[sid]) == 1 else 0.0
            elif mode == "full_protect":
                p = gamma * (alpha if int(protect_mask[sid]) == 1 else 1.0)
            else:
                raise ValueError(f"unknown mode={mode}")

            if p > 0.0 and rng.random() < p:
                r = int(rng.integers(0, 3))
                if r == 0:
                    out.x(_q)
                elif r == 1:
                    out.y(_q)
                else:
                    out.z(_q)
                n_noise += 1

            sid += 1

    return out, n_noise

# -----------------------------
# Seeds (paired)
# -----------------------------
def shot_seed(seed_base: int, shot_index: int) -> int:
    return int((seed_base + 1000003 * shot_index) % (2**31 - 1))

# -----------------------------
# Worker
# -----------------------------
def _chunk_mc(
    circ_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    gamma: float,
    mode: str,
    zone_mask_bytes: bytes,
    protect_mask_bytes: bytes,
    alpha: float,
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
    zone_mask = np.frombuffer(zone_mask_bytes, dtype=np.uint8)
    protect_mask = np.frombuffer(protect_mask_bytes, dtype=np.uint8)

    fids = np.empty(int(shots), dtype=np.float64)
    nnoise = np.empty(int(shots), dtype=np.float64)

    for i in range(int(shots)):
        global_idx = int(start + i)
        s = shot_seed(int(seed_base), global_idx)
        rng = np.random.default_rng(int(s))

        noisy, n_ops = apply_pauli_jump_weighted(
            base,
            gamma=float(gamma),
            rng=rng,
            mode=str(mode),
            zone_mask=zone_mask,
            protect_mask=protect_mask,
            alpha=float(alpha),
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

def mc_mean_parallel(
    ex,
    circ_qpy: bytes,
    n: int,
    psi_ideal: np.ndarray,
    *,
    gamma: float,
    mode: str,
    zone_mask: np.ndarray,
    protect_mask: np.ndarray,
    alpha: float,
    traj: int,
    chunk: int,
    seed_base: int,
    blas_threads: int,
    max_bond_dim: int,
    threshold: float,
    pad: int,
) -> Tuple[float, float]:
    traj = int(traj)
    chunk = int(max(1, chunk))
    n_chunks = (traj + chunk - 1) // chunk

    zone_mask_bytes = zone_mask.astype(np.uint8, copy=False).tobytes()
    protect_mask_bytes = protect_mask.astype(np.uint8, copy=False).tobytes()

    futs = []
    for k in range(n_chunks):
        start = k * chunk
        shots = min(chunk, traj - start)
        futs.append(
            ex.submit(
                _chunk_mc,
                circ_qpy, n, psi_ideal,
                float(gamma), str(mode),
                zone_mask_bytes, protect_mask_bytes, float(alpha),
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
# Plot helpers (clean + aligned heatmaps)
# -----------------------------
def centers_to_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges

def plot_heatmap(
    out_path: Path,
    data: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    cbar_label: str,
) -> None:
    x_edges = centers_to_edges(x_centers)
    y_edges = centers_to_edges(y_centers)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(x_edges, y_edges, data, shading="auto")
    fig.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # ticks aligned to centers
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)

    # keep time fraction readable
    if np.min(x_centers) >= 0.0 and np.max(x_centers) <= 1.0:
        ax.set_xlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_protected_map(
    out_path: Path,
    prot_map: np.ndarray,
    *,
    title: str,
    n_qubits: int,
) -> None:
    # prot_map shape: (inst_layers, n_qubits)
    fig, ax = plt.subplots()
    ax.imshow(prot_map, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xlabel("qubit")
    ax.set_ylabel("instruction layer index (compressed)")
    ax.set_title(title)

    ax.set_xticks(np.arange(n_qubits))
    # y ticks: too many layers -> show sparse ticks
    L = prot_map.shape[0]
    if L <= 20:
        ax.set_yticks(np.arange(L))
    else:
        step = max(1, L // 10)
        ax.set_yticks(np.arange(0, L, step))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def annotate_zone_info(
    ax,
    *,
    A_desc: str,
    B_desc: str,
    SA: int,
    SB: int,
    SU: int,
    temporal_rel: str,
    spatial_rel: str,
) -> None:
    txt = (
        "Counts are SPACE–TIME sites (layer × qubit touches)\n"
        f"SA={SA}, SB={SB}, SU={SU}\n"
        f"A: {A_desc}\n"
        f"B: {B_desc}\n"
        f"Separation: temporal={temporal_rel}, spatial={spatial_rel}"
    )
    ax.text(
        0.02, 0.02, txt,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="left",
        alpha=0.9,
    )

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, default="all",
                    choices=["all", "sensitivity", "interaction", "mitigate"])

    # model
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # noise + MC
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=50)

    # YAQS
    ap.add_argument("--max-bond-dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-12)
    ap.add_argument("--pad", type=int, default=0)

    # parallel
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--blas-threads", type=int, default=1)

    # transpile
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    # sensitivity grid
    ap.add_argument("--tfrac-num", type=int, default=7)
    ap.add_argument("--k-num", type=int, default=-1, help="if -1 use 0..n; else use 0..k_num")
    ap.add_argument("--zone-kind", type=str, default="prefix", choices=["prefix", "random"],
                    help="spatial qubit subset: prefix or random (random uses fixed subset per K)")

    # interaction (two zones)
    ap.add_argument("--A-t0", type=float, default=0.0)
    ap.add_argument("--A-t1", type=float, default=0.5)
    ap.add_argument("--A-k", type=int, default=-1)
    ap.add_argument("--B-t0", type=float, default=0.5)
    ap.add_argument("--B-t1", type=float, default=1.0)
    ap.add_argument("--B-k", type=int, default=-1)
    ap.add_argument("--gamma-min-exp", type=float, default=-3)
    ap.add_argument("--gamma-max-exp", type=float, default=-1)
    ap.add_argument("--gamma-num", type=int, default=25)

    # mitigation
    ap.add_argument("--alpha-list", type=str, default="1.0,0.7,0.5,0.3,0.1,0.0")
    ap.add_argument("--protect-mode", type=str, default="topc", choices=["topc", "manual"])
    ap.add_argument("--top-cells", type=int, default=5, help="how many most-sensitive grid cells to protect (topc)")
    ap.add_argument("--protect-t0", type=float, default=0.5, help="manual protect: time window start")
    ap.add_argument("--protect-t1", type=float, default=1.0, help="manual protect: time window end")
    ap.add_argument("--protect-k", type=int, default=-1, help="manual protect: K (prefix or random)")
    ap.add_argument("--protect-zone-kind", type=str, default="prefix", choices=["prefix", "random"])

    args = ap.parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    n = int(args.num_qubits)
    depth = int(args.depth)

    # build base circuit
    base = build_twolocal(num_qubits=n, depth=depth, seed=int(args.seed), add_measurements=False)
    base = strip_measurements(base)

    if int(args.use_transpile) == 1:
        base = transpile(
            base,
            optimization_level=int(args.opt_level),
            seed_transpiler=int(args.seed_transpiler),
        )
        base = strip_measurements(base)

    # ideal
    ideal_vec = run_strongsim_statevector(
        base, n=n,
        max_bond_dim=int(args.max_bond_dim),
        threshold=float(args.threshold),
        pad=int(args.pad),
    )

    sites, inst_max = enumerate_sites(base)  # inst_max = number of layers (compressed)
    S_total = len(sites)

    # grid axes
    tfracs = np.linspace(0.0, 1.0, int(args.tfrac_num))
    layer_cuts = np.array([layer_from_tfrac(inst_max, tf) for tf in tfracs], dtype=int)

    if int(args.k_num) < 0:
        ks = np.arange(0, n + 1, dtype=int)
    else:
        ks = np.arange(0, int(args.k_num) + 1, dtype=int)

    out_dir = (REPO_ROOT / "outputs" / "experiments" / "final_test_zone_sensitivity_mitigation"
               / f"n{n}_d{depth}_seed{int(args.seed)}" / _ts())
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "task": str(args.task),
        "n": n,
        "depth": depth,
        "seed": int(args.seed),
        "gamma": float(args.gamma),
        "traj": int(args.traj),
        "chunk": int(args.chunk),
        "workers": int(workers),
        "blas_threads": int(args.blas_threads),
        "yaqs": {"max_bond_dim": int(args.max_bond_dim), "threshold": float(args.threshold), "pad": int(args.pad)},
        "S_total": int(S_total),
        "inst_layers": int(inst_max),
        "grid": {"tfracs": tfracs.tolist(), "layer_cuts": layer_cuts.tolist(), "ks": ks.tolist(), "zone_kind": str(args.zone_kind)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_qpy = qpy_bytes(base)
    seed_base = int(args.seed + 20251230)

    from concurrent.futures import ProcessPoolExecutor

    # ---------- qmask cache (important for random kind consistency) ----------
    rng_cells = np.random.default_rng(999)
    qmask_cache: Dict[int, np.ndarray] = {}

    def get_qmask(k: int, kind: str) -> np.ndarray:
        k = int(k)
        key = (k, 0) if kind == "prefix" else (k, 1)
        kk = int(key[0] * 10 + key[1])  # small unique key
        if kk in qmask_cache:
            return qmask_cache[kk]
        if kind == "prefix":
            m = qubit_subset_prefix(n, k)
        else:
            m = qubit_subset_random(n, k, rng_cells)
        qmask_cache[kk] = m
        return m

    # ---------- sensitivity cache ----------
    sens_cache: Dict[str, np.ndarray] = {}

    def run_sensitivity(traj_use: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        meanF = np.zeros((len(ks), len(tfracs)), dtype=float)
        meanNoise = np.zeros_like(meanF)
        S_grid = np.zeros_like(meanF)
        dose_grid = np.zeros_like(meanF)

        t0 = time.time()
        with ProcessPoolExecutor(
            max_workers=min(int(workers), 256),
            initializer=_worker_init,
            initargs=(int(args.blas_threads),),
        ) as ex:
            for ix_k, k in enumerate(ks):
                qmask = get_qmask(int(k), str(args.zone_kind))
                for ix_t, layer_cut in enumerate(layer_cuts):
                    zone_mask = build_zone_mask(sites, layer_lo=0, layer_hi=int(layer_cut), qmask=qmask)
                    S = int(np.sum(zone_mask))
                    S_grid[ix_k, ix_t] = float(S)
                    dose_grid[ix_k, ix_t] = float(args.gamma) * float(S)

                    protect = np.zeros_like(zone_mask, dtype=np.uint8)

                    mf, mn = mc_mean_parallel(
                        ex, base_qpy, n, ideal_vec,
                        gamma=float(args.gamma),
                        mode="zone_only",
                        zone_mask=zone_mask,
                        protect_mask=protect,
                        alpha=1.0,
                        traj=int(traj_use),
                        chunk=int(args.chunk),
                        seed_base=seed_base,
                        blas_threads=int(args.blas_threads),
                        max_bond_dim=int(args.max_bond_dim),
                        threshold=float(args.threshold),
                        pad=int(args.pad),
                    )
                    meanF[ix_k, ix_t] = float(mf)
                    meanNoise[ix_k, ix_t] = float(mn)

                    print(f"[sens] K={k:2d}, layer_cut={layer_cut:3d}, S={S:5d}, dose={dose_grid[ix_k,ix_t]:.3f}, meanF={mf:.6f}", flush=True)

        print(f"[sens] elapsed={time.time()-t0:.2f}s", flush=True)
        return meanF, meanNoise, S_grid, dose_grid

    def get_sensitivity_fulltraj() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # returns meanF, meanNoise, S, dose, c_hat  (ALL at traj=args.traj)
        if "meanF" in sens_cache:
            meanF = sens_cache["meanF"]
            meanNoise = sens_cache["meanNoise"]
            S_grid = sens_cache["S_grid"]
            dose_grid = sens_cache["dose_grid"]
        else:
            meanF, meanNoise, S_grid, dose_grid = run_sensitivity(traj_use=int(args.traj))
            sens_cache["meanF"] = meanF
            sens_cache["meanNoise"] = meanNoise
            sens_cache["S_grid"] = S_grid
            sens_cache["dose_grid"] = dose_grid

        err = 1.0 - meanF
        c_hat = np.zeros_like(err)
        for i in range(c_hat.shape[0]):
            for j in range(c_hat.shape[1]):
                d = float(dose_grid[i, j])
                c_hat[i, j] = float(err[i, j]) / d if d > 0 else 0.0
        return meanF, meanNoise, S_grid, dose_grid, c_hat

    # -----------------------------
    # Task B: two-zone interaction
    # -----------------------------
    def run_interaction() -> None:
        A_k = n if int(args.A_k) < 0 else int(args.A_k)
        B_k = n if int(args.B_k) < 0 else int(args.B_k)

        # For interaction, keep subsets consistent with selected kind
        rng_sub = np.random.default_rng(2025)
        if str(args.zone_kind) == "prefix":
            A_qmask = qubit_subset_prefix(n, A_k)
            B_qmask = qubit_subset_prefix(n, B_k)
        else:
            A_qmask = qubit_subset_random(n, A_k, rng_sub)
            B_qmask = qubit_subset_random(n, B_k, rng_sub)

        A_lo = layer_from_tfrac(inst_max, float(args.A_t0))
        A_hi = layer_from_tfrac(inst_max, float(args.A_t1))
        B_lo = layer_from_tfrac(inst_max, float(args.B_t0))
        B_hi = layer_from_tfrac(inst_max, float(args.B_t1))

        zoneA = build_zone_mask(sites, layer_lo=A_lo, layer_hi=A_hi, qmask=A_qmask)
        zoneB = build_zone_mask(sites, layer_lo=B_lo, layer_hi=B_hi, qmask=B_qmask)
        zoneU = np.maximum(zoneA, zoneB).astype(np.uint8)

        SA = int(np.sum(zoneA))
        SB = int(np.sum(zoneB))
        SU = int(np.sum(zoneU))

        # classify separation
        temporal_overlap = max(0, min(A_hi, B_hi) - max(A_lo, B_lo))
        temporal_rel = "disjoint" if temporal_overlap == 0 else "overlap"

        Aq = set(np.where(A_qmask == 1)[0].tolist())
        Bq = set(np.where(B_qmask == 1)[0].tolist())
        spatial_rel = "disjoint" if len(Aq.intersection(Bq)) == 0 else "overlap"

        def qdesc(mask: np.ndarray) -> str:
            idx = np.where(mask == 1)[0].tolist()
            if len(idx) == 0:
                return "K=0 (empty)"
            if len(idx) <= 8:
                return f"qubits={idx}"
            return f"K={len(idx)}"

        A_desc = f"time t∈[{args.A_t0:.2f},{args.A_t1:.2f}] -> layers[{A_lo},{A_hi}), {qdesc(A_qmask)}"
        B_desc = f"time t∈[{args.B_t0:.2f},{args.B_t1:.2f}] -> layers[{B_lo},{B_hi}), {qdesc(B_qmask)}"

        gammas = np.logspace(float(args.gamma_min_exp), float(args.gamma_max_exp), int(args.gamma_num))

        errA = np.zeros_like(gammas, dtype=float)
        errB = np.zeros_like(gammas, dtype=float)
        errU = np.zeros_like(gammas, dtype=float)
        synergy = np.zeros_like(gammas, dtype=float)

        with ProcessPoolExecutor(
            max_workers=min(int(workers), 256),
            initializer=_worker_init,
            initargs=(int(args.blas_threads),),
        ) as ex:
            for i, g in enumerate(gammas):
                protect0 = np.zeros_like(zoneA, dtype=np.uint8)

                mA, _ = mc_mean_parallel(
                    ex, base_qpy, n, ideal_vec,
                    gamma=float(g), mode="zone_only",
                    zone_mask=zoneA, protect_mask=protect0, alpha=1.0,
                    traj=int(args.traj), chunk=int(args.chunk),
                    seed_base=seed_base, blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim), threshold=float(args.threshold), pad=int(args.pad),
                )
                mB, _ = mc_mean_parallel(
                    ex, base_qpy, n, ideal_vec,
                    gamma=float(g), mode="zone_only",
                    zone_mask=zoneB, protect_mask=protect0, alpha=1.0,
                    traj=int(args.traj), chunk=int(args.chunk),
                    seed_base=seed_base, blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim), threshold=float(args.threshold), pad=int(args.pad),
                )
                mU, _ = mc_mean_parallel(
                    ex, base_qpy, n, ideal_vec,
                    gamma=float(g), mode="zone_only",
                    zone_mask=zoneU, protect_mask=protect0, alpha=1.0,
                    traj=int(args.traj), chunk=int(args.chunk),
                    seed_base=seed_base, blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim), threshold=float(args.threshold), pad=int(args.pad),
                )

                errA[i] = 1.0 - float(mA)
                errB[i] = 1.0 - float(mB)
                errU[i] = 1.0 - float(mU)
                synergy[i] = errU[i] - (errA[i] + errB[i])

                print(f"[inter] gamma={g:.3e}  errA={errA[i]:.4f} errB={errB[i]:.4f} errU={errU[i]:.4f}  synergy={synergy[i]:+.4e}", flush=True)

        # Plot synergy vs gamma
        fig, ax = plt.subplots()
        ax.plot(gammas, synergy, marker="o")
        ax.set_xscale("log")
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("gamma (log)")
        ax.set_ylabel("synergy = err(A∪B) - [err(A)+err(B)]")
        ax.set_title("Two-zone interaction (additivity test)")
        annotate_zone_info(ax, A_desc=A_desc, B_desc=B_desc, SA=SA, SB=SB, SU=SU, temporal_rel=temporal_rel, spatial_rel=spatial_rel)
        fig.tight_layout()
        fig.savefig(out_dir / "interaction_synergy_vs_gamma.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Scatter: errU vs errA+errB
        fig, ax = plt.subplots()
        ax.scatter(errA + errB, errU)
        mx = max(1e-9, float(np.max(errA + errB)))
        ax.plot([0, mx], [0, mx], linewidth=1)
        ax.set_xlabel("err(A) + err(B)")
        ax.set_ylabel("err(A∪B)")
        ax.set_title("Additivity scatter (ideal: y=x)")
        annotate_zone_info(ax, A_desc=A_desc, B_desc=B_desc, SA=SA, SB=SB, SU=SU, temporal_rel=temporal_rel, spatial_rel=spatial_rel)
        fig.tight_layout()
        fig.savefig(out_dir / "interaction_additivity_scatter.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        res = {
            "A": {"t0": float(args.A_t0), "t1": float(args.A_t1), "k": int(A_k), "S": SA},
            "B": {"t0": float(args.B_t0), "t1": float(args.B_t1), "k": int(B_k), "S": SB},
            "U": {"S": SU},
            "temporal_rel": temporal_rel,
            "spatial_rel": spatial_rel,
            "gammas": gammas.tolist(),
            "errA": errA.tolist(),
            "errB": errB.tolist(),
            "errU": errU.tolist(),
            "synergy": synergy.tolist(),
        }
        (out_dir / "interaction_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # -----------------------------
    # Task C: mitigation by protecting high-c sites (full_protect)
    #   IMPORTANT: no probe. Use traj=2000 everywhere.
    # -----------------------------
    def run_mitigation() -> None:
        alphas = [float(x.strip()) for x in str(args.alpha_list).split(",") if x.strip() != ""]

        if str(args.protect_mode) == "manual":
            pk = n if int(args.protect_k) < 0 else int(args.protect_k)
            kind = str(args.protect_zone_kind)
            qmask = qubit_subset_prefix(n, pk) if kind == "prefix" else qubit_subset_random(n, pk, rng_cells)
            lo = layer_from_tfrac(inst_max, float(args.protect_t0))
            hi = layer_from_tfrac(inst_max, float(args.protect_t1))
            protect = build_zone_mask(sites, layer_lo=lo, layer_hi=hi, qmask=qmask)
            protect_source = {"mode": "manual", "t0": float(args.protect_t0), "t1": float(args.protect_t1), "k": int(pk), "kind": kind}
        else:
            # topc: use FULL traj sensitivity (no probe)
            meanF, _, S_grid, dose_grid, c_grid = get_sensitivity_fulltraj()

            flat = []
            for ix_k, k in enumerate(ks):
                for ix_t, layer_cut in enumerate(layer_cuts):
                    S = int(S_grid[ix_k, ix_t])
                    if S <= 0:
                        continue
                    flat.append((float(c_grid[ix_k, ix_t]), ix_k, ix_t, int(k), int(layer_cut), int(S)))
            flat.sort(key=lambda x: x[0], reverse=True)
            chosen = flat[: max(1, int(args.top_cells))]

            protect = np.zeros(len(sites), dtype=np.uint8)
            for (_c, _ix_k, _ix_t, k, layer_cut, _S) in chosen:
                qmask = get_qmask(int(k), str(args.zone_kind))
                zone = build_zone_mask(sites, layer_lo=0, layer_hi=int(layer_cut), qmask=qmask)
                protect = np.maximum(protect, zone).astype(np.uint8)

            protect_source = {"mode": "topc", "top_cells": int(args.top_cells), "chosen": chosen}

        S_protect = int(np.sum(protect))
        frac_protect = float(S_protect) / float(max(1, S_total))
        (out_dir / "protect_source.json").write_text(json.dumps(protect_source, indent=2), encoding="utf-8")

        # visualize protected (layer, qubit)
        prot_map = np.zeros((inst_max, n), dtype=np.uint8)
        for sid, s in enumerate(sites):
            if int(protect[sid]) == 1:
                prot_map[s.layer, s.q] = 1

        plot_protected_map(
            out_dir / "protected_sites_inst_qubit.png",
            prot_map,
            title=f"Protected sites (layer × qubit), S_protect={S_protect}/{S_total} ({frac_protect:.2%})",
            n_qubits=n,
        )

        # baseline full noise alpha=1
        zone_all = np.ones(len(sites), dtype=np.uint8)  # unused in full_protect, but keep shape
        meanF_alpha = []
        meanNoise_alpha = []

        with ProcessPoolExecutor(
            max_workers=min(int(workers), 256),
            initializer=_worker_init,
            initargs=(int(args.blas_threads),),
        ) as ex:
            for a in alphas:
                mf, mn = mc_mean_parallel(
                    ex, base_qpy, n, ideal_vec,
                    gamma=float(args.gamma),
                    mode="full_protect",
                    zone_mask=zone_all,
                    protect_mask=protect,
                    alpha=float(a),
                    traj=int(args.traj),
                    chunk=int(args.chunk),
                    seed_base=seed_base,
                    blas_threads=int(args.blas_threads),
                    max_bond_dim=int(args.max_bond_dim),
                    threshold=float(args.threshold),
                    pad=int(args.pad),
                )
                meanF_alpha.append(float(mf))
                meanNoise_alpha.append(float(mn))
                print(f"[mit] alpha={a:.3f}  meanF={mf:.6f}  meanNoiseOps={mn:.3f}", flush=True)

        meanF_alpha = np.array(meanF_alpha, dtype=float)
        err_alpha = 1.0 - meanF_alpha

        # plot meanF vs alpha
        fig, ax = plt.subplots()
        ax.plot(alphas, meanF_alpha, marker="o")
        ax.set_xlabel("alpha (protected sites use gamma' = alpha*gamma)")
        ax.set_ylabel("mean fidelity")
        ax.set_title(f"Mitigation by protecting high-c_hat sites (gamma={float(args.gamma):g})")
        ax.text(
            0.02, 0.02,
            f"Protected fraction: {S_protect}/{S_total} ({frac_protect:.2%})",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="left",
            alpha=0.9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "mitigation_meanF_vs_alpha.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # plot error reduction relative to alpha=1
        base_err = float(err_alpha[0]) if len(err_alpha) > 0 else 1.0
        rel = err_alpha / max(1e-12, base_err)

        fig, ax = plt.subplots()
        ax.plot(alphas, rel, marker="o")
        ax.set_xlabel("alpha")
        ax.set_ylabel("relative error = (1-F)/(1-F at alpha=1)")
        ax.set_title("Error reduction curve (relative)")
        fig.tight_layout()
        fig.savefig(out_dir / "mitigation_relative_error_vs_alpha.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        res = {
            "gamma": float(args.gamma),
            "alphas": alphas,
            "meanF": meanF_alpha.tolist(),
            "meanNoiseOps": np.array(meanNoise_alpha, dtype=float).tolist(),
            "S_protect": int(S_protect),
            "S_total": int(S_total),
            "frac_protect": float(frac_protect),
            "protect_source": protect_source,
        }
        (out_dir / "mitigation_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # -----------------------------
    # Execute
    # -----------------------------
    t0_all = time.time()

    if args.task in ("all", "sensitivity"):
        meanF, meanNoise, S_grid, dose_grid, c_hat = get_sensitivity_fulltraj()
        err = 1.0 - meanF

        x = tfracs
        y = ks.astype(float)

        plot_heatmap(
            out_dir / "sensitivity_error_heatmap.png",
            err,
            x_centers=x,
            y_centers=y,
            xlabel="time fraction (0..1, prefix of instruction layers)",
            ylabel="active qubits K",
            title=f"Sensitivity (zone-only): error heatmap (gamma={float(args.gamma):g}, traj={int(args.traj)})",
            cbar_label="error = 1 - meanF",
        )

        plot_heatmap(
            out_dir / "sensitivity_dose_heatmap.png",
            dose_grid,
            x_centers=x,
            y_centers=y,
            xlabel="time fraction (0..1, prefix of instruction layers)",
            ylabel="active qubits K",
            title="Dose heatmap",
            cbar_label="dose = gamma * S (S = #space-time sites in zone)",
        )

        plot_heatmap(
            out_dir / "sensitivity_c_hat_heatmap.png",
            c_hat,
            x_centers=x,
            y_centers=y,
            xlabel="time fraction (0..1, prefix of instruction layers)",
            ylabel="active qubits K",
            title="Sensitivity map: c_hat (error per unit dose)",
            cbar_label="c_hat = (1-meanF) / (gamma*S)",
        )

        np.save(out_dir / "meanF_grid.npy", meanF)
        np.save(out_dir / "meanNoise_grid.npy", meanNoise)
        np.save(out_dir / "S_grid.npy", S_grid)
        np.save(out_dir / "dose_grid.npy", dose_grid)
        np.save(out_dir / "c_hat_grid.npy", c_hat)

    if args.task in ("all", "interaction"):
        run_interaction()

    if args.task in ("all", "mitigate"):
        run_mitigation()

    elapsed = time.time() - t0_all

    report = [
        f"OUTPUT DIR: {out_dir.resolve()}",
        f"elapsed_sec={elapsed:.2f}",
        f"task={args.task}",
        f"n={n}, depth={depth}, gamma={float(args.gamma):g}, traj={int(args.traj)}",
        f"workers={int(workers)}, blas_threads={int(args.blas_threads)}, chunk={int(args.chunk)}",
        "",
        "Saved (depending on task):",
        "  sensitivity_error_heatmap.png",
        "  sensitivity_dose_heatmap.png",
        "  sensitivity_c_hat_heatmap.png",
        "  interaction_synergy_vs_gamma.png",
        "  interaction_additivity_scatter.png",
        "  protected_sites_inst_qubit.png",
        "  mitigation_meanF_vs_alpha.png",
        "  mitigation_relative_error_vs_alpha.png",
        "  meta.json (+ *_results.json, *.npy)",
    ]
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)

if __name__ == "__main__":
    main()
