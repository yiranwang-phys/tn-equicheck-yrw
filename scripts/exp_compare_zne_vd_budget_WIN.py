# scripts/exp_compare_zne_vd_budget_WIN.py
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Statevector


# -----------------------------
# Make src/ importable (src-layout)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


# -----------------------------
# Try importing your project utils; fallback if not installed
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

try:
    from qem_yrw_project.pauli_jump import apply_pauli_jump_after_each_gate  # type: ignore
except Exception:
    # minimal fallback: per-gate per-qubit Pauli jump
    def apply_pauli_jump_after_each_gate(circuit: QuantumCircuit, gamma: float, seed: int):
        rng = np.random.default_rng(int(seed))
        noisy = QuantumCircuit(circuit.num_qubits)
        for inst, qargs, cargs in circuit.data:
            if inst.name in ("measure", "barrier", "reset"):
                noisy.append(inst, qargs, cargs)
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
        return noisy, None


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
# Circuit helpers
# -----------------------------
def strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    return circ.remove_final_measurements(inplace=False)


def qpy_bytes(circ: QuantumCircuit) -> bytes:
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    return buf.getvalue()


def qpy_load_bytes(b: bytes) -> QuantumCircuit:
    buf = io.BytesIO(b)
    return qpy.load(buf)[0]


def prep_basis_state(n: int, bitstring: str) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i, b in enumerate(bitstring):
        if b == "1":
            qc.x(i)
    return qc


def pure_state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    amp = np.vdot(psi, phi)
    return float(np.real(amp * np.conjugate(amp)))


# -----------------------------
# ZNE folding
# -----------------------------
def global_fold(circ: QuantumCircuit, scale: int) -> QuantumCircuit:
    scale = int(scale)
    if scale < 1 or scale % 2 == 0:
        raise ValueError("folding scale must be odd integer >=1, e.g. 1,3,5,7,...")
    m = (scale - 1) // 2
    out = circ.copy()
    inv = circ.inverse()
    for _ in range(m):
        out = out.compose(inv, inplace=False)
        out = out.compose(circ, inplace=False)
    return out


def poly_extrapolate(scales: np.ndarray, y: np.ndarray, degree: int) -> float:
    degree = int(degree)
    deg = min(degree, len(scales) - 1)
    coeff = np.polyfit(scales.astype(float), y.astype(float), deg=deg)
    p = np.poly1d(coeff)
    return float(p(0.0))


# -----------------------------
# Batch Monte Carlo worker
# -----------------------------
def _batch_mc(
    circ_qpy: bytes,
    n: int,
    bitstring: str,
    psi_ideal: np.ndarray,
    gamma: float,
    shots: int,
    seed0: int,
    collect_rho: bool,
    blas_threads: int,
) -> Tuple[float, int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns: (sum_fidelity, nshots, rho_sum_real, rho_sum_imag)
    If collect_rho=False, rho sums are None.
    """
    _worker_init(blas_threads)

    circ = strip_measurements(qpy_load_bytes(circ_qpy))
    prep = prep_basis_state(n, bitstring)

    rng = np.random.default_rng(int(seed0))
    sum_f = 0.0

    rho_r = None
    rho_i = None
    if collect_rho:
        d = 2**n
        rho_r = np.zeros((d, d), dtype=np.float64)
        rho_i = np.zeros((d, d), dtype=np.float64)

    for _ in range(int(shots)):
        s = int(rng.integers(0, 2**31 - 1))
        noisy, _ = apply_pauli_jump_after_each_gate(circ, float(gamma), s)
        phi = Statevector.from_instruction(prep.compose(noisy, inplace=False)).data
        sum_f += pure_state_fidelity(psi_ideal, phi)

        if collect_rho:
            outer = np.outer(phi, np.conjugate(phi))
            rho_r += np.real(outer)
            rho_i += np.imag(outer)

    return float(sum_f), int(shots), rho_r, rho_i


# -----------------------------
# Main experiment
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=1e-2)

    ap.add_argument("--num-inputs", type=int, default=8)

    # Fair budget: number of trajectories per input for EACH method family
    ap.add_argument("--budget", type=int, default=3000)       # per input
    ap.add_argument("--batch-shots", type=int, default=200)   # chunk size per task

    ap.add_argument("--workers", type=int, default=0)         # 0 => all cores
    ap.add_argument("--blas-threads", type=int, default=1)    # avoid oversubscription

    # ZNE comparison settings
    ap.add_argument("--scales-sets", type=str, default="1,3,5;1,3,5,7")
    ap.add_argument("--degrees", type=str, default="1,2")

    # VD settings
    ap.add_argument("--vd-powers", type=str, default="1,2,3")
    ap.add_argument("--vd-max-n", type=int, default=10)       # VD requires storing rho of size 2^n

    # Circuit selection options (keep it simple + deterministic)
    ap.add_argument("--use-transpile", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--seed-transpiler", type=int, default=123)

    args = ap.parse_args()

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    # Build base circuit (same family as your current experiments)
    base = build_twolocal(
        num_qubits=int(args.num_qubits),
        depth=int(args.depth),
        seed=int(args.seed),
        add_measurements=False,
    )
    base = strip_measurements(base)

    if int(args.use_transpile) == 1:
        base = transpile(base, optimization_level=int(args.opt_level), seed_transpiler=int(args.seed_transpiler))
        base = strip_measurements(base)

    n = int(base.num_qubits)
    if n > int(args.vd_max_n):
        # VD will be disabled beyond this
        pass

    out_dir = REPO_ROOT / "outputs" / "experiments" / "compare_zne_vd_budget" / f"n{n}_d{args.depth}_gamma{args.gamma:.0e}_budget{args.budget}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "meta.txt").write_text(
        f"time={datetime.now().isoformat()}\n"
        f"platform={platform.platform()}\n"
        f"python={platform.python_version()}\n"
        f"workers={workers}\n"
        f"blas_threads={int(args.blas_threads)}\n"
        f"budget_per_input={int(args.budget)}\n",
        encoding="utf-8"
    )

    # Random inputs (computational basis)
    rng = np.random.default_rng(int(args.seed) + 999)
    inputs: List[str] = []
    for _ in range(int(args.num_inputs)):
        bits = rng.integers(0, 2, size=n)
        inputs.append("".join(str(int(b)) for b in bits))

    # Precompute ideal output states for each input
    psi_ideals: Dict[str, np.ndarray] = {}
    for b in inputs:
        prep = prep_basis_state(n, b)
        psi_ideals[b] = Statevector.from_instruction(prep.compose(base, inplace=False)).data

    # Parse ZNE settings
    scales_sets: List[List[int]] = []
    for part in args.scales_sets.split(";"):
        ss = [int(x) for x in part.split(",") if x.strip()]
        ss = sorted(ss)
        if any(s < 1 or s % 2 == 0 for s in ss):
            raise ValueError(f"Invalid scales set: {ss}. Must be odd integers like 1,3,5,...")
        scales_sets.append(ss)

    degrees = [int(x) for x in args.degrees.split(",") if x.strip()]
    degrees = sorted(set(degrees))

    vd_powers = [int(x) for x in args.vd_powers.split(",") if x.strip()]
    vd_powers = sorted(set([p for p in vd_powers if p >= 1]))

    # Serialize base circuit
    base_qpy = qpy_bytes(base)

    # Prepare ZNE folded circuits for each scales set
    zne_circs_qpy: Dict[str, List[bytes]] = {}
    for ss in scales_sets:
        key = ",".join(str(x) for x in ss)
        folded = [global_fold(base, s) for s in ss]
        zne_circs_qpy[key] = [qpy_bytes(c) for c in folded]

    # Parallel execution
    from concurrent.futures import ProcessPoolExecutor, as_completed

    budget = int(args.budget)
    batch = int(max(1, args.batch_shots))
    n_batches = int((budget + batch - 1) // batch)

    t0 = time.time()

    # -------- Baseline (also collect rho for VD) --------
    collect_rho = (n <= int(args.vd_max_n)) and (max(vd_powers) >= 2)
    d = 2**n

    baseline_mean_by_input: Dict[str, float] = {}
    rho_by_input: Dict[str, Optional[np.ndarray]] = {b: None for b in inputs}

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:
        jobs = []
        for bi, b in enumerate(inputs):
            for k in range(n_batches):
                bs = min(batch, budget - k * batch)
                seed0 = int(args.seed + 1000003 * bi + 97 * k)
                jobs.append((b, ex.submit(
                    _batch_mc, base_qpy, n, b, psi_ideals[b],
                    float(args.gamma), int(bs), int(seed0),
                    bool(collect_rho), int(args.blas_threads)
                )))

        sum_f = {b: 0.0 for b in inputs}
        sum_n = {b: 0 for b in inputs}
        rho_r = {b: (np.zeros((d, d), dtype=np.float64) if collect_rho else None) for b in inputs}
        rho_i = {b: (np.zeros((d, d), dtype=np.float64) if collect_rho else None) for b in inputs}

        for b, fut in jobs:
            sf, nn, rr, ii = fut.result()
            sum_f[b] += sf
            sum_n[b] += nn
            if collect_rho and rr is not None and ii is not None:
                rho_r[b] += rr
                rho_i[b] += ii

        for b in inputs:
            baseline_mean_by_input[b] = sum_f[b] / max(1, sum_n[b])
            if collect_rho:
                rho_by_input[b] = (rho_r[b] + 1j * rho_i[b]) / max(1, sum_n[b])

    baseline_vals = np.array([baseline_mean_by_input[b] for b in inputs], dtype=float)
    baseline_mean = float(np.mean(baseline_vals))
    baseline_std = float(np.std(baseline_vals))

    # -------- VD from baseline rho --------
    vd_mean: Dict[int, float] = {}
    vd_std: Dict[int, float] = {}
    if not collect_rho:
        # report p=1 as baseline only
        vd_mean[1] = baseline_mean
        vd_std[1] = baseline_std
        vd_powers = [1]
    else:
        vd_vals_by_p = {p: [] for p in vd_powers}
        for b in inputs:
            rho = rho_by_input[b]
            psi = psi_ideals[b]
            assert rho is not None
            for p in vd_powers:
                # rho^p / Tr(rho^p)
                rp = rho.copy()
                for _ in range(p - 1):
                    rp = rp @ rho
                tr = np.trace(rp)
                if abs(tr) < 1e-18:
                    vd_vals_by_p[p].append(float("nan"))
                    continue
                rp = rp / tr
                vd_vals_by_p[p].append(float(np.real(np.vdot(psi, rp @ psi))))

        for p in vd_powers:
            arr = np.asarray(vd_vals_by_p[p], dtype=float)
            vd_mean[p] = float(np.nanmean(arr))
            vd_std[p] = float(np.nanstd(arr))

    # -------- ZNE (fair budget split across scales) --------
    zne_points: Dict[str, Dict[int, Tuple[float, float]]] = {}  # key=scaleset, scale->(mean,std over inputs)
    zne_extrap: Dict[str, Dict[int, float]] = {}                # key=scaleset, degree->extrapolated

    with ProcessPoolExecutor(
        max_workers=min(workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:
        for ss in scales_sets:
            key = ",".join(str(x) for x in ss)
            qpys = zne_circs_qpy[key]

            # split budget equally over scales (fair total cost)
            per_scale = [budget // len(ss)] * len(ss)
            for i in range(budget - sum(per_scale)):
                per_scale[i % len(ss)] += 1

            # run each scale
            scale_means = []
            scale_stds = []
            zne_points[key] = {}
            zne_extrap[key] = {}

            for si, scale in enumerate(ss):
                bgt = int(per_scale[si])
                nb = int((bgt + batch - 1) // batch)

                sum_f = {b: 0.0 for b in inputs}
                sum_n = {b: 0 for b in inputs}

                jobs = []
                for bi, b in enumerate(inputs):
                    for k in range(nb):
                        bs = min(batch, bgt - k * batch)
                        seed0 = int(args.seed + 3000003 * (si + 1) + 1000003 * bi + 97 * k)
                        jobs.append((b, ex.submit(
                            _batch_mc, qpys[si], n, b, psi_ideals[b],
                            float(args.gamma), int(bs), int(seed0),
                            False, int(args.blas_threads)
                        )))

                for b, fut in jobs:
                    sf, nn, _, _ = fut.result()
                    sum_f[b] += sf
                    sum_n[b] += nn

                vals = np.array([sum_f[b] / max(1, sum_n[b]) for b in inputs], dtype=float)
                mu = float(np.mean(vals))
                sd = float(np.std(vals))
                zne_points[key][scale] = (mu, sd)
                scale_means.append(mu)
                scale_stds.append(sd)

            # extrapolate for each degree using MEAN over inputs per scale
            x = np.asarray(ss, dtype=float)
            y = np.asarray(scale_means, dtype=float)
            for deg in degrees:
                zne_extrap[key][deg] = float(poly_extrapolate(x, y, deg))

    elapsed = time.time() - t0

    # -------- Save results --------
    payload = {
        "tag": "COMPARE_ZNE_VD_BUDGET",
        "args": vars(args),
        "platform": platform.platform(),
        "workers_used": workers,
        "elapsed_sec": float(elapsed),
        "inputs": inputs,
        "baseline": {"mean": baseline_mean, "std_over_inputs": baseline_std},
        "vd": {"powers": vd_powers, "mean": vd_mean, "std_over_inputs": vd_std, "rho_collected": bool(collect_rho)},
        "zne_points": {
            key: {str(scale): {"mean": mu, "std_over_inputs": sd} for scale, (mu, sd) in zne_points[key].items()}
            for key in zne_points
        },
        "zne_extrap": zne_extrap,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -------- Plots --------
    # 1) ZNE points for each scales set
    plt.figure()
    for key in zne_points:
        ss = [int(x) for x in key.split(",")]
        mus = [zne_points[key][s][0] for s in ss]
        plt.plot(np.asarray(ss, float), np.asarray(mus, float), marker="o", label=f"scales={key} (fair budget)")
    plt.title("ZNE points (mean over inputs), fair total budget per scaleset")
    plt.xlabel("folding scale (odd)")
    plt.ylabel("mean state fidelity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "zne_points_compare.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Summary bar: baseline, ZNE extrap (each key/deg), VD p=1/2/3
    labels = ["baseline"]
    values = [baseline_mean]

    for key in zne_extrap:
        for deg in degrees:
            labels.append(f"ZNE scales={key} deg={deg}")
            values.append(zne_extrap[key][deg])

    for p in vd_powers:
        labels.append(f"VD p={p}")
        values.append(float(vd_mean[p]))

    plt.figure(figsize=(max(8, 0.8 * len(values)), 4.8))
    plt.title("Summary: baseline vs ZNE (scales/degree) vs VD (p)")
    plt.bar(np.arange(len(values)), np.asarray(values, float))
    plt.xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    plt.ylim(0.0, 1.02)
    plt.ylabel("fidelity")
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Save a small text report
    rep = []
    rep.append(f"OUTPUT DIR: {out_dir.resolve()}")
    rep.append(f"elapsed_sec={elapsed:.2f}, workers={workers}, blas_threads={int(args.blas_threads)}")
    rep.append(f"baseline mean={baseline_mean:.6f}, std_over_inputs={baseline_std:.6f}")
    for key in zne_extrap:
        for deg in degrees:
            rep.append(f"ZNE extrap scales={key} deg={deg}: {zne_extrap[key][deg]:.6f}")
    for p in vd_powers:
        rep.append(f"VD p={p}: mean={vd_mean[p]:.6f}, std_over_inputs={vd_std[p]:.6f}")
    (out_dir / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print("\n".join(rep), flush=True)


if __name__ == "__main__":
    # Required on Windows for multiprocessing
    main()
