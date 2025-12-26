# scripts/exp_general_mitigation_AB_VD_win.py
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Statevector


# -----------------------------
# Make src/ importable (src-layout project)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


# -----------------------------
# Try importing your project utilities; fallback to local implementations
# -----------------------------
try:
    from qem_yrw_project.circuits.twolocal import build_twolocal  # type: ignore
except Exception:
    def build_twolocal(num_qubits: int, depth: int, seed: int, add_measurements: bool = False) -> QuantumCircuit:
        # fallback: plain TwoLocal-like circuit (rx + rzz chain), deterministic angles
        rng = np.random.default_rng(seed)
        qc = QuantumCircuit(num_qubits)
        for layer in range(depth):
            for i in range(num_qubits):
                qc.rx(float(rng.uniform(0, 2*np.pi)), i)
            for i in range(num_qubits - 1):
                qc.rzz(float(rng.uniform(0, 2*np.pi)), i, i+1)
        if add_measurements:
            qc.measure_all()
        return qc

try:
    from qem_yrw_project.pauli_jump import apply_pauli_jump_after_each_gate  # type: ignore
except Exception:
    @dataclass
    class _PJStats:
        gamma: float
        n_sites: int
        n_noise: int

    def apply_pauli_jump_after_each_gate(circuit: QuantumCircuit, gamma: float, seed: int):
        rng = np.random.default_rng(int(seed))
        noisy = QuantumCircuit(circuit.num_qubits)
        n_sites = 0
        n_noise = 0

        for inst, qargs, cargs in circuit.data:
            if inst.name in ("measure", "barrier", "reset"):
                noisy.append(inst, qargs, cargs)
                continue

            noisy.append(inst, qargs, cargs)

            for q in qargs:
                n_sites += 1
                if rng.random() < gamma:
                    pa = int(rng.integers(0, 3))
                    if pa == 0:
                        noisy.x(q)
                    elif pa == 1:
                        noisy.y(q)
                    else:
                        noisy.z(q)
                    n_noise += 1

        return noisy, _PJStats(gamma=float(gamma), n_sites=int(n_sites), n_noise=int(n_noise))


# -----------------------------
# CPU helpers
# -----------------------------
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
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
    except Exception:
        pass


def _worker_init(blas_threads: int) -> None:
    _set_thread_env(blas_threads)
    _try_set_high_priority_windows()


# -----------------------------
# Utils
# -----------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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


def count_2q_gates(circ: QuantumCircuit) -> int:
    n = 0
    for inst, qargs, _ in circ.data:
        if inst.name in ("measure", "barrier", "reset"):
            continue
        if len(qargs) == 2:
            n += 1
    return n


def count_total_gates(circ: QuantumCircuit) -> int:
    n = 0
    for inst, _, _ in circ.data:
        if inst.name in ("measure", "barrier", "reset"):
            continue
        n += 1
    return n


def save_circuit_draw(circ: QuantumCircuit, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig = circ.draw(output="mpl", fold=140)
        fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (out_dir / f"{stem}.txt").write_text(str(circ.draw(output="text")), encoding="utf-8")
        (out_dir / f"{stem}.draw_err.txt").write_text(str(e), encoding="utf-8")


# -----------------------------
# YAQS EquiCheck wrapper (optional)
# -----------------------------
def yaqs_equivcheck_bool(ideal: QuantumCircuit, cand: QuantumCircuit) -> Optional[bool]:
    try:
        import inspect
        from mqt.yaqs.digital.equivalence_checker import run as yaqs_run
    except Exception:
        return None

    sig = inspect.signature(yaqs_run)
    kwargs = {}
    # be signature-tolerant across yaqs versions
    if "parallel" in sig.parameters:
        kwargs["parallel"] = False
    if "max_bond_dim" in sig.parameters:
        kwargs["max_bond_dim"] = 64
    if "threshold" in sig.parameters:
        kwargs["threshold"] = 1e-10
    if "tol" in sig.parameters:
        kwargs["tol"] = 1e-6
    if "epsilon" in sig.parameters:
        kwargs["epsilon"] = 1e-6

    try:
        res = yaqs_run(ideal, cand, **kwargs)
    except TypeError:
        res = yaqs_run(ideal, cand)

    if isinstance(res, bool):
        return res
    if isinstance(res, (tuple, list)) and res and isinstance(res[0], bool):
        return bool(res[0])
    return None


# -----------------------------
# ZNE folding
# -----------------------------
def global_fold(circ: QuantumCircuit, scale: int) -> QuantumCircuit:
    scale = int(scale)
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be odd integer >= 1, e.g. 1,3,5,...")
    m = (scale - 1) // 2
    out = circ.copy()
    inv = circ.inverse()
    for _ in range(m):
        out = out.compose(inv, inplace=False)
        out = out.compose(circ, inplace=False)
    return out


def poly_extrapolate(scales: np.ndarray, y: np.ndarray, degree: int) -> float:
    deg = min(int(degree), len(scales) - 1)
    coeff = np.polyfit(scales.astype(float), y.astype(float), deg=deg)
    p = np.poly1d(coeff)
    return float(p(0.0))


# -----------------------------
# Candidate compilation + (new) adaptive pilot selection
# -----------------------------
@dataclass
class Candidate:
    idx: int
    opt_level: int
    seed_transpiler: int
    depth: int
    n_total_gates: int
    n_2q_gates: int
    equiv: Optional[bool]
    pilot_mean_fid: Optional[float]


def generate_candidates(
    ideal: QuantumCircuit,
    k: int,
    base_seed: int,
    check_equivalence: bool,
) -> list[Candidate]:
    out: list[Candidate] = []
    ideal0 = strip_measurements(ideal)

    for i in range(int(k)):
        opt_level = int(i % 4)
        st = int(base_seed + 1000 + i)

        cand_circ = transpile(ideal0, optimization_level=opt_level, seed_transpiler=st)
        cand_circ = strip_measurements(cand_circ)

        equiv = None
        if check_equivalence:
            equiv = yaqs_equivcheck_bool(ideal0, cand_circ)

        out.append(
            Candidate(
                idx=i,
                opt_level=opt_level,
                seed_transpiler=st,
                depth=int(cand_circ.depth()),
                n_total_gates=int(count_total_gates(cand_circ)),
                n_2q_gates=int(count_2q_gates(cand_circ)),
                equiv=equiv,
                pilot_mean_fid=None,
            )
        )
    return out


def pick_top_by_structure(cands: list[Candidate], m: int) -> list[int]:
    # prefer fewer 2q, then fewer total, then lower depth
    items = [(c.n_2q_gates, c.n_total_gates, c.depth, c.idx) for c in cands]
    items.sort()
    return [idx for *_ , idx in items[: int(max(1, m))]]


# -----------------------------
# Parallel Monte Carlo batches
# -----------------------------
@dataclass
class BatchOut:
    sum_fid: float
    n: int
    rho_sum_real: Optional[np.ndarray]  # store real+imag separately to be pickle-safe
    rho_sum_imag: Optional[np.ndarray]


def _batch_mc(
    circ_qpy: bytes,
    n_qubits: int,
    bitstring: str,
    psi_ideal: np.ndarray,
    gamma: float,
    shots: int,
    seed0: int,
    collect_rho: bool,
    blas_threads: int,
) -> BatchOut:
    _worker_init(blas_threads)

    circ = strip_measurements(qpy_load_bytes(circ_qpy))
    prep = prep_basis_state(n_qubits, bitstring)

    rng = np.random.default_rng(int(seed0))
    sum_f = 0.0

    rho_r = None
    rho_i = None
    if collect_rho:
        d = 2 ** n_qubits
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

    return BatchOut(sum_fid=float(sum_f), n=int(shots), rho_sum_real=rho_r, rho_sum_imag=rho_i)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ideal-qpy", type=str, default="")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--gamma", type=float, default=1e-2)

    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--batch-shots", type=int, default=200)  # split tasks so CPU can saturate
    ap.add_argument("--num-inputs", type=int, default=8)

    ap.add_argument("--workers", type=int, default=0)       # 0 => all cores
    ap.add_argument("--blas-threads", type=int, default=1)  # set 1 to avoid oversubscription

    # A: equivalence-class search
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--check-equivalence", type=int, default=1)

    # NEW: adaptive pilot selection (breakthrough-ish)
    ap.add_argument("--pilot-shots", type=int, default=300)
    ap.add_argument("--select-m", type=int, default=12)
    ap.add_argument("--select-mode", type=str, default="pilot", choices=["pilot", "structure"])

    # B: ZNE
    ap.add_argument("--zne-scales", type=str, default="1,3,5")
    ap.add_argument("--zne-degree", type=int, default=2)

    # VD
    ap.add_argument("--vd-powers", type=str, default="1,2,3")
    ap.add_argument("--vd-max-dim", type=int, default=256)  # skip VD>1 if 2^n > this

    args = ap.parse_args()

    # workers
    n_workers = int(args.workers)
    if n_workers <= 0:
        n_workers = int(os.cpu_count() or 1)

    _set_thread_env(int(args.blas_threads))
    _try_set_high_priority_windows()

    # build/load ideal
    if args.ideal_qpy.strip():
        p = Path(args.ideal_qpy.strip())
        with p.open("rb") as f:
            ideal = qpy.load(f)[0]
        ideal = strip_measurements(ideal)
    else:
        ideal = build_twolocal(
            num_qubits=int(args.num_qubits),
            depth=int(args.depth),
            seed=int(args.seed),
            add_measurements=False,
        )
        ideal = strip_measurements(ideal)

    n = int(ideal.num_qubits)
    out_dir = Path("outputs") / "experiments" / "general_mitigation_AB_VD" / f"n{n}_seed{args.seed}_gamma{args.gamma:.0e}_shots{args.shots}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "meta.txt").write_text(
        f"time={datetime.now().isoformat()}\n"
        f"platform={platform.platform()}\n"
        f"python={platform.python_version()}\n"
        f"workers={n_workers}\n"
        f"blas_threads={int(args.blas_threads)}\n"
        f"cwd={os.getcwd()}\n"
        f"src_dir_added={SRC_DIR.exists()}\n",
        encoding="utf-8",
    )

    save_circuit_draw(ideal, out_dir, "ideal_circuit")

    # --- generate candidates
    check_eq = bool(int(args.check_equivalence))
    cands = generate_candidates(ideal, int(args.k), int(args.seed), check_eq)

    # filter by equivalence if available; if equiv is None (YAQS missing), keep
    valid = [c for c in cands if (c.equiv is True) or (c.equiv is None)]
    if not valid:
        raise RuntimeError("No valid candidates: EquiCheck said all are non-equivalent. Set --check-equivalence 0 to debug.")

    # choose subset indices (structure prefilter)
    top_idx = pick_top_by_structure(valid, int(args.select_m))
    subset = [valid[i] for i in range(len(valid)) if valid[i].idx in top_idx]

    # prepare pilot inputs
    rng = np.random.default_rng(int(args.seed))
    pilot_inputs = []
    for _ in range(min(4, int(args.num_inputs))):
        bits = rng.integers(0, 2, size=n)
        pilot_inputs.append("".join(str(int(b)) for b in bits))

    # precompute ideal states for pilot inputs
    pilot_psi = {}
    for b in pilot_inputs:
        prep = prep_basis_state(n, b)
        pilot_psi[b] = Statevector.from_instruction(prep.compose(ideal, inplace=False)).data

    # --- pilot evaluation (NEW): black-box selection inside equivalence class
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def pilot_score(circ: QuantumCircuit, seed_base: int) -> float:
        circ_q = qpy_bytes(circ)
        batch = max(50, int(args.pilot_shots) // max(1, len(pilot_inputs)))
        jobs = []
        with ProcessPoolExecutor(
            max_workers=min(n_workers, 32),
            initializer=_worker_init,
            initargs=(int(args.blas_threads),),
        ) as ex:
            j = 0
            for b in pilot_inputs:
                s0 = seed_base + 10007 * j
                jobs.append(
                    ex.submit(
                        _batch_mc,
                        circ_q, n, b, pilot_psi[b],
                        float(args.gamma), int(batch), int(s0),
                        False, int(args.blas_threads),
                    )
                )
                j += 1

            tot = 0.0
            cnt = 0
            for fut in as_completed(jobs):
                r = fut.result()
                tot += r.sum_fid
                cnt += r.n
        return float(tot / max(1, cnt))

    # select best circuit
    if args.select_mode == "pilot":
        # evaluate subset only
        best = None
        best_score = -1.0
        for c in subset:
            # regenerate the actual transpiled circuit deterministically
            cand_circ = transpile(strip_measurements(ideal), optimization_level=c.opt_level, seed_transpiler=c.seed_transpiler)
            cand_circ = strip_measurements(cand_circ)
            sc = pilot_score(cand_circ, seed_base=int(args.seed + 7777 + c.idx))
            c.pilot_mean_fid = sc
            if sc > best_score:
                best_score = sc
                best = cand_circ
        if best is None:
            raise RuntimeError("Pilot selection failed unexpectedly.")
        best_circ = best
    else:
        # purely structure-based (original Best-of-K idea)
        best_idx = pick_top_by_structure(valid, 1)[0]
        chosen = [c for c in valid if c.idx == best_idx][0]
        best_circ = transpile(strip_measurements(ideal), optimization_level=chosen.opt_level, seed_transpiler=chosen.seed_transpiler)
        best_circ = strip_measurements(best_circ)

    save_circuit_draw(best_circ, out_dir, "selected_circuit")

    # dump candidates
    with (out_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx","opt_level","seed_transpiler","depth","n_total_gates","n_2q_gates","equiv","pilot_mean_fid"])
        for c in valid:
            w.writerow([c.idx,c.opt_level,c.seed_transpiler,c.depth,c.n_total_gates,c.n_2q_gates,c.equiv,c.pilot_mean_fid])

    # --- B: ZNE circuits
    scales = [int(x) for x in args.zne_scales.split(",") if x.strip()]
    scales = sorted(scales)
    if any(s < 1 or s % 2 == 0 for s in scales):
        raise ValueError("--zne-scales must be odd integers like 1,3,5")
    zne_circs = [global_fold(best_circ, s) for s in scales]

    # --- VD powers
    vd_powers = [int(x) for x in args.vd_powers.split(",") if x.strip()]
    vd_powers = sorted(set([p for p in vd_powers if p >= 1]))

    # VD feasibility guard
    d = 2 ** n
    collect_rho = (len(vd_powers) > 1) and (d <= int(args.vd_max_dim))
    if len(vd_powers) > 1 and not collect_rho:
        # keep power=1 only
        vd_powers = [1]

    # --- inputs
    rng = np.random.default_rng(int(args.seed) + 12345)
    inputs = []
    for _ in range(int(args.num_inputs)):
        bits = rng.integers(0, 2, size=n)
        inputs.append("".join(str(int(b)) for b in bits))

    psi_ideals = {}
    for b in inputs:
        prep = prep_basis_state(n, b)
        psi_ideals[b] = Statevector.from_instruction(prep.compose(ideal, inplace=False)).data

    # serialize circuits
    main_qpy = qpy_bytes(best_circ)
    zne_qpys = [qpy_bytes(c) for c in zne_circs]

    # --- parallel batches for baseline + ZNE
    shots = int(args.shots)
    batch = int(max(1, args.batch_shots))
    n_batches = int((shots + batch - 1) // batch)

    t0 = time.time()
    baseline_sum = {b: 0.0 for b in inputs}
    baseline_n = {b: 0 for b in inputs}

    # rho accumulator for VD
    rho_r = np.zeros((d, d), dtype=np.float64) if collect_rho else None
    rho_i = np.zeros((d, d), dtype=np.float64) if collect_rho else None
    rho_n_total = 0

    zne_sum = {s: {b: 0.0 for b in inputs} for s in scales}
    zne_n = {s: {b: 0 for b in inputs} for s in scales}

    def submit_all_jobs(ex: ProcessPoolExecutor):
        jobs = []

        # baseline
        for bi, bitstr in enumerate(inputs):
            for k in range(n_batches):
                bs = min(batch, shots - k * batch)
                seed0 = int(args.seed + 1000003 * bi + 97 * k)
                jobs.append(
                    ("baseline", None, bitstr,
                     ex.submit(_batch_mc, main_qpy, n, bitstr, psi_ideals[bitstr],
                               float(args.gamma), int(bs), int(seed0),
                               bool(collect_rho), int(args.blas_threads)))
                )

        # ZNE
        for si, sc in enumerate(scales):
            cq = zne_qpys[si]
            for bi, bitstr in enumerate(inputs):
                for k in range(n_batches):
                    bs = min(batch, shots - k * batch)
                    seed0 = int(args.seed + 2000003 * (si+1) + 1000003 * bi + 97 * k)
                    jobs.append(
                        ("zne", sc, bitstr,
                         ex.submit(_batch_mc, cq, n, bitstr, psi_ideals[bitstr],
                                   float(args.gamma), int(bs), int(seed0),
                                   False, int(args.blas_threads)))
                    )
        return jobs

    with ProcessPoolExecutor(
        max_workers=min(n_workers, 256),
        initializer=_worker_init,
        initargs=(int(args.blas_threads),),
    ) as ex:
        jobs = submit_all_jobs(ex)
        for kind, sc, bitstr, fut in jobs:
            r: BatchOut = fut.result()
            if kind == "baseline":
                baseline_sum[bitstr] += r.sum_fid
                baseline_n[bitstr] += r.n
                if collect_rho and r.rho_sum_real is not None and r.rho_sum_imag is not None:
                    rho_r += r.rho_sum_real
                    rho_i += r.rho_sum_imag
                    rho_n_total += r.n
            else:
                zne_sum[sc][bitstr] += r.sum_fid
                zne_n[sc][bitstr] += r.n

    elapsed = time.time() - t0

    # aggregate baseline
    baseline_fids = []
    for b in inputs:
        baseline_fids.append(baseline_sum[b] / max(1, baseline_n[b]))
    baseline_mean = float(np.mean(baseline_fids))
    baseline_std = float(np.std(baseline_fids))

    # aggregate ZNE
    zne_means_per_scale = []
    zne_stds_per_scale = []
    for sc in scales:
        vals = []
        for b in inputs:
            vals.append(zne_sum[sc][b] / max(1, zne_n[sc][b]))
        zne_means_per_scale.append(float(np.mean(vals)))
        zne_stds_per_scale.append(float(np.std(vals)))

    zne_extrap = poly_extrapolate(np.asarray(scales, float), np.asarray(zne_means_per_scale, float), int(args.zne_degree))

    # VD
    vd_mean = {}
    vd_std = {}
    if vd_powers == [1]:
        # mixed-state fidelity at power=1 is still baseline mean if we are using pure-state shot fidelity;
        # we report baseline under VD=1 for convenience.
        vd_mean[1] = baseline_mean
        vd_std[1] = baseline_std
    else:
        # build rho_hat
        rho = (rho_r + 1j * rho_i) / max(1, rho_n_total)

        # compute per-input VD fidelity
        vd_vals_by_p = {p: [] for p in vd_powers}
        for b in inputs:
            psi = psi_ideals[b]
            for p in vd_powers:
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
            arr = np.asarray(vd_vals_by_p[p], float)
            vd_mean[p] = float(np.nanmean(arr))
            vd_std[p] = float(np.nanstd(arr))

    # save results
    payload = {
        "tag": "GENERAL_MITIGATION_AB_VD",
        "args": vars(args),
        "platform": platform.platform(),
        "workers_used": n_workers,
        "elapsed_sec": float(elapsed),
        "selected_circuit": {
            "depth": int(best_circ.depth()),
            "n_total_gates": int(count_total_gates(best_circ)),
            "n_2q_gates": int(count_2q_gates(best_circ)),
        },
        "inputs": inputs,
        "baseline": {"mean": baseline_mean, "std_over_inputs": baseline_std},
        "zne": {
            "scales": scales,
            "mean_per_scale": zne_means_per_scale,
            "std_over_inputs_per_scale": zne_stds_per_scale,
            "degree": int(args.zne_degree),
            "extrapolated_scale0": float(zne_extrap),
        },
        "vd": {
            "powers": vd_powers,
            "mean": vd_mean,
            "std_over_inputs": vd_std,
            "rho_collected": bool(collect_rho),
            "dim": int(d),
        },
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # plots
    plt.figure()
    plt.title("ZNE (global folding) - mean over inputs")
    plt.plot(np.asarray(scales, float), np.asarray(zne_means_per_scale, float), marker="o")
    plt.xlabel("folding scale (odd)")
    plt.ylabel("mean state fidelity")
    plt.tight_layout()
    plt.savefig(out_dir / "zne_points.png", dpi=200, bbox_inches="tight")
    plt.close()

    labels = ["Selected (baseline)", f"ZNE extrap (deg={int(args.zne_degree)})"]
    values = [baseline_mean, zne_extrap]
    for p in vd_powers:
        labels.append(f"VD p={p}")
        values.append(float(vd_mean[p]))

    plt.figure()
    plt.title("Summary: selection + ZNE + VD")
    plt.bar(np.arange(len(values)), np.asarray(values, float))
    plt.xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    plt.ylim(0.0, 1.02)
    plt.ylabel("fidelity")
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("OUTPUT DIR:", out_dir.resolve(), flush=True)
    print(f"workers={n_workers} blas_threads={int(args.blas_threads)} elapsed={elapsed:.2f}s", flush=True)
    print(f"baseline mean fidelity = {baseline_mean:.6f} (std {baseline_std:.6f})", flush=True)
    print(f"ZNE extrapolated(scale->0) = {zne_extrap:.6f}", flush=True)
    for p in vd_powers:
        print(f"VD p={p}: mean={vd_mean[p]:.6f} std={vd_std[p]:.6f}", flush=True)


if __name__ == "__main__":
    # required on Windows for multiprocessing
    main()
