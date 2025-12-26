# scripts/updated_work_1_suite_YAQS_WIN.py
# Updated Work 1 (no statevector): Qiskit circuit + Qiskit-defined Pauli-jump trajectories
# + YAQS StrongSim observables + YAQS EquiCheck
#
# Main novelty: Equivalence-Guided Compilation Search (EGCS) + Paired comparison (base vs best)

from __future__ import annotations

# ---- early thread env (must happen BEFORE numpy / yaqs heavy imports) ----
import os
import sys
from pathlib import Path

def _early_get_arg(flag: str, default: str) -> str:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return str(sys.argv[i + 1])
    return default

def _set_thread_env(n: int) -> None:
    # Cross-platform variable names are the same.
    # Only the shell syntax differs (PowerShell vs bash). Inside Python it's identical.
    n = int(n)
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[k] = str(n)

_set_thread_env(int(_early_get_arg("--blas-threads", "1")))

# ---- now safe to import heavy deps ----
import io
import json
import math
import time
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Any

import numpy as np
import matplotlib.pyplot as plt

from qiskit import qpy, QuantumCircuit
from qiskit.compiler import transpile

# YAQS
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.digital.equivalence_checker import run as yaqs_equiv_run

# ---- robust import for src-layout package ----
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qem_yrw_project.circuits.twolocal import build_twolocal
from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


# ---------------- utils ----------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def sci(x: float) -> str:
    return f"{x:.0e}".replace("+", "")

def save_circuit_png_or_txt(circ: QuantumCircuit, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig = circ.draw(output="mpl", fold=120)
        fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (out_dir / f"{stem}.txt").write_text(str(circ.draw(output="text")), encoding="utf-8")
        (out_dir / f"{stem}.draw_error.txt").write_text(str(e), encoding="utf-8")

def dump_qpy_bytes(circ: QuantumCircuit) -> bytes:
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    return buf.getvalue()

def load_qpy_bytes(blob: bytes) -> QuantumCircuit:
    buf = io.BytesIO(blob)
    return qpy.load(buf)[0]

def unitary_part(circ: QuantumCircuit) -> QuantumCircuit:
    # remove_final_measurements keeps registers; good for your usage
    try:
        return circ.remove_final_measurements(inplace=False)
    except Exception:
        # fallback: manual strip
        out = QuantumCircuit(circ.num_qubits, 0)
        out.global_phase = getattr(circ, "global_phase", 0.0)
        for ci in circ.data:
            if ci.operation.name == "measure":
                continue
            out.append(ci.operation, ci.qubits, [])
        return out

def count_injection_sites(circ: QuantumCircuit) -> int:
    # Same spirit as your existing scripts: count per-qubit “after gate” noise opportunities.
    sites = 0
    for ci in circ.data:
        name = ci.operation.name
        if name in ("measure", "barrier"):
            continue
        sites += len(ci.qubits)
    return int(sites)

def strongsim_all_Z(circ: QuantumCircuit, n: int, *, max_bond_dim: int, threshold: float) -> np.ndarray:
    state = MPS(n, state="zeros")
    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(n)],
        num_traj=1,
        max_bond_dim=int(max_bond_dim),
        threshold=float(threshold),
    )
    # IMPORTANT: noise_model=None because noise is already compiled into the circuit (trajectory).
    simulator.run(state, circ, sim_params, noise_model=None, parallel=False)
    return np.array([obs.results[0] for obs in sim_params.observables], dtype=float)

def z_match_score(z_ref: np.ndarray, z_test: np.ndarray) -> float:
    # z in [-1,1] => |diff| in [0,2]
    d = float(np.mean(np.abs(z_test - z_ref)))
    return float(np.clip(1.0 - 0.5 * d, 0.0, 1.0))

def chunked(xs: list[int], chunk: int) -> list[list[int]]:
    if chunk <= 0:
        chunk = 1
    return [xs[i : i + chunk] for i in range(0, len(xs), chunk)]


# ---------------- multiprocessing worker ----------------
# On Windows spawn, module is re-imported; keep worker logic top-level and lightweight.
def _worker_eval_chunk(payload: dict) -> dict:
    """
    Evaluate a chunk of trajectory seeds for one candidate circuit under one gamma.
    Returns partial sums for aggregation.
    """
    blob = payload["qpy_blob"]
    gamma = float(payload["gamma"])
    seeds = payload["seeds"]
    z_ref = np.array(payload["z_ref"], dtype=float)
    max_bond_dim = int(payload["max_bond_dim"])
    threshold = float(payload["threshold"])

    circ = load_qpy_bytes(blob)
    n = circ.num_qubits

    sum_score = 0.0
    sum_score2 = 0.0
    sum_nojump = 0.0
    cnt = 0

    for s in seeds:
        noisy, st = apply_pauli_jump_after_each_gate(circ, gamma=gamma, seed=int(s), include_measurements=False)
        z_noisy = strongsim_all_Z(noisy, n, max_bond_dim=max_bond_dim, threshold=threshold)
        sc = z_match_score(z_ref, z_noisy)
        nj = 1.0 if int(getattr(st, "n_noise_ops", 0)) == 0 else 0.0

        sum_score += sc
        sum_score2 += sc * sc
        sum_nojump += nj
        cnt += 1

    return {"sum": sum_score, "sum2": sum_score2, "p0": sum_nojump, "cnt": cnt}


def estimate_candidate(
    circ: QuantumCircuit,
    z_ref: np.ndarray,
    *,
    gamma: float,
    traj: int,
    seed0: int,
    workers: int,
    chunk: int,
    max_bond_dim: int,
    threshold: float,
) -> dict:
    """
    Trajectory Monte Carlo (TJM-style): sample Pauli-jump trajectories, StrongSim each, score by Z-match.
    Parallelized across CPU cores using ProcessPoolExecutor.
    """
    from concurrent.futures import ProcessPoolExecutor

    traj = int(traj)
    if traj <= 0:
        raise ValueError("traj must be positive")

    # workers=0 => all cores
    if workers == 0:
        workers = os.cpu_count() or 1
    workers = max(1, int(workers))

    # Seeds (fixed list so paired experiments can be fair)
    rng = np.random.default_rng(int(seed0))
    seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=traj)]
    seed_chunks = chunked(seeds, int(chunk))

    blob = dump_qpy_bytes(circ)
    payloads = [
        {
            "qpy_blob": blob,
            "gamma": float(gamma),
            "seeds": ch,
            "z_ref": z_ref.tolist(),
            "max_bond_dim": int(max_bond_dim),
            "threshold": float(threshold),
        }
        for ch in seed_chunks
    ]

    t0 = time.time()
    s = 0.0
    s2 = 0.0
    p0 = 0.0
    cnt = 0

    # NOTE: BLAS threads already forced via env; each process runs StrongSim single-threaded.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_worker_eval_chunk, payloads):
            s += float(out["sum"])
            s2 += float(out["sum2"])
            p0 += float(out["p0"])
            cnt += int(out["cnt"])

    mean = s / max(1, cnt)
    var = max(0.0, s2 / max(1, cnt) - mean * mean)
    std = math.sqrt(var)
    p0_mean = p0 / max(1, cnt)

    return {
        "gamma": float(gamma),
        "traj": int(traj),
        "workers": int(workers),
        "chunk": int(chunk),
        "mean_score": float(mean),
        "std_score": float(std),
        "p0": float(p0_mean),
        "wall_s": float(time.time() - t0),
    }


# ---------------- experiment pipeline ----------------
def build_ideal(num_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    circ_meas = build_twolocal(num_qubits=num_qubits, depth=depth, seed=seed, add_measurements=True)
    return circ_meas

def make_candidates(base_unitary: QuantumCircuit, *, n_cands: int, seed: int) -> list[QuantumCircuit]:
    """
    Generate compilation variants (intended to be ideally equivalent).
    Avoid unsupported routing plugins (your previous 'stochastic' crash).
    """
    rng = np.random.default_rng(int(seed))
    cands: list[QuantumCircuit] = []

    # Always include base
    cands.append(base_unitary)

    # Transpile variants
    # Mix optimization levels + random seeds. Keep routing_method default (no 'stochastic').
    opt_levels = [0, 1, 2, 3]
    while len(cands) < n_cands:
        lvl = int(rng.choice(opt_levels))
        s = int(rng.integers(0, 2**31 - 1))
        try:
            v = transpile(
                base_unitary,
                optimization_level=lvl,
                seed_transpiler=s,
                # IMPORTANT: do NOT set routing_method="stochastic"
            )
            cands.append(v)
        except Exception:
            # If transpile fails, skip
            continue

    # Deduplicate by qasm text (cheap heuristic)
    uniq: list[QuantumCircuit] = []
    seen = set()
    for c in cands:
        try:
            key = c.qasm()
        except Exception:
            key = str(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--clean", type=int, default=0)

    ap.add_argument("--num-qubits", type=int, default=10)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--gamma-sweep", type=str, default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")

    ap.add_argument("--traj-rank", type=int, default=400)     # cheap ranking budget per candidate
    ap.add_argument("--traj-paired", type=int, default=2000)  # final paired experiment budget

    ap.add_argument("--workers", type=int, default=0)         # 0 => all cores
    ap.add_argument("--chunk", type=int, default=25)          # seeds per task

    ap.add_argument("--blas-threads", type=int, default=1)    # already applied early, keep for logging

    ap.add_argument("--max-bond-dim", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=1e-10)

    ap.add_argument("--n-candidates", type=int, default=40)
    ap.add_argument("--topk-verify", type=int, default=5)     # run YAQS equivcheck on top-k ranked
    args = ap.parse_args()

    # Ensure thread env reflects args (mostly for logging; early env already set)
    _set_thread_env(int(args.blas_threads))

    out_root = ROOT / "outputs" / "updated_work_1" / f"n{args.num_qubits}_d{args.depth}_seed{args.seed}" / ts()
    if int(args.clean) == 1 and out_root.parent.exists():
        # clean only the parent experiment family, not whole outputs
        import shutil
        shutil.rmtree(out_root.parent, ignore_errors=True)

    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 1) Build ideal reference (measured) + unitary part ----
    ideal_meas = build_ideal(args.num_qubits, args.depth, args.seed)
    ideal_unit = unitary_part(ideal_meas)

    (out_root / "meta.txt").write_text(
        "tag=UPDATED_WORK_1_SUITE\n"
        f"num_qubits={args.num_qubits}\n"
        f"depth={args.depth}\n"
        f"seed={args.seed}\n"
        f"gamma={args.gamma}\n"
        f"workers={args.workers}\n"
        f"chunk={args.chunk}\n"
        f"blas_threads={args.blas_threads}\n"
        f"max_bond_dim={args.max_bond_dim}\n"
        f"threshold={args.threshold}\n",
        encoding="utf-8",
    )

    # Save ideal
    with (out_root / "ideal_measured.qpy").open("wb") as f:
        qpy.dump(ideal_meas, f)
    with (out_root / "ideal_unitary.qpy").open("wb") as f:
        qpy.dump(ideal_unit, f)
    save_circuit_png_or_txt(ideal_unit, out_root, "ideal_unitary")

    # ---- 2) StrongSim Z reference (no statevector) ----
    print("[1/6] StrongSim ideal Z...", flush=True)
    z_ref = strongsim_all_Z(ideal_unit, ideal_unit.num_qubits, max_bond_dim=args.max_bond_dim, threshold=args.threshold)
    np.savetxt(out_root / "z_ref_ideal.txt", z_ref)

    # ---- 3) Single trajectory demo (find a seed that actually injects noise) ----
    print("[2/6] Single trajectory demo...", flush=True)
    seed_used = None
    st_used = None
    noisy_demo = None
    for s in range(int(args.seed), int(args.seed) + 2000):
        cand, st = apply_pauli_jump_after_each_gate(ideal_unit, gamma=float(args.gamma), seed=int(s), include_measurements=False)
        if int(getattr(st, "n_noise_ops", 0)) > 0:
            seed_used, st_used, noisy_demo = int(s), st, cand
            break
    if noisy_demo is None:
        raise RuntimeError("Could not find a noisy trajectory seed (unexpected).")

    demo_dir = out_root / "singletraj_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    save_circuit_png_or_txt(noisy_demo, demo_dir, "noisy_demo_unitary")
    z_demo = strongsim_all_Z(noisy_demo, ideal_unit.num_qubits, max_bond_dim=args.max_bond_dim, threshold=args.threshold)
    np.savetxt(demo_dir / "z_demo.txt", z_demo)
    (demo_dir / "demo_stats.json").write_text(
        json.dumps(
            {
                "seed_used": seed_used,
                "gamma": float(args.gamma),
                "n_gates_seen": int(getattr(st_used, "n_gates_seen", -1)),
                "n_noise_ops": int(getattr(st_used, "n_noise_ops", -1)),
                "z_match_score": z_match_score(z_ref, z_demo),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- 4) YAQS EquiCheck sanity (ideal vs noisy should usually be False) ----
    print("[3/6] YAQS EquiCheck (ideal vs noisy_demo)...", flush=True)
    eq_report = {"ideal_vs_noisy_demo": None, "err": None}
    try:
        eq_report["ideal_vs_noisy_demo"] = yaqs_equiv_run(ideal_unit, noisy_demo)
    except Exception as e:
        eq_report["err"] = f"{type(e).__name__}: {e}"
    (out_root / "equicheck_sanity.json").write_text(json.dumps(eq_report, indent=2), encoding="utf-8")

    # ---- 5) EGCS: build candidates, rank by (a) proxy sites, (b) trajectory StrongSim score ----
    print("[4/6] Build candidates (compilation variants)...", flush=True)
    cands = make_candidates(ideal_unit, n_cands=int(args.n_candidates), seed=int(args.seed) + 12345)

    cand_rows = []
    print(f"  candidates = {len(cands)}", flush=True)
    for i, c in enumerate(cands):
        sites = count_injection_sites(c)
        cand_rows.append(
            {
                "cid": i,
                "sites": sites,
                "depth": int(c.depth()),
                "size": int(c.size()),
            }
        )

    # Trajectory ranking (expensive): only for a subset to keep runtime sane
    # Strategy: take the lowest-sites ~min(15, n_candidates) and evaluate by StrongSim-MC
    cand_rows_sorted = sorted(cand_rows, key=lambda r: (r["sites"], r["depth"], r["size"]))
    shortlist = [r["cid"] for r in cand_rows_sorted[: min(15, len(cand_rows_sorted))]]

    ranked = []
    print(f"  shortlist for MC ranking = {len(shortlist)}", flush=True)
    for cid in shortlist:
        c = cands[cid]
        stats = estimate_candidate(
            c,
            z_ref,
            gamma=float(args.gamma),
            traj=int(args.traj_rank),
            seed0=int(args.seed) + 999,
            workers=int(args.workers),
            chunk=int(args.chunk),
            max_bond_dim=int(args.max_bond_dim),
            threshold=float(args.threshold),
        )
        ranked.append(
            {
                "cid": int(cid),
                "sites": count_injection_sites(c),
                "depth": int(c.depth()),
                "size": int(c.size()),
                **stats,
            }
        )
        print(
            f"    cid={cid:02d} sites={ranked[-1]['sites']}  score={ranked[-1]['mean_score']:.4f}  p0={ranked[-1]['p0']:.4f}  wall={ranked[-1]['wall_s']:.1f}s",
            flush=True,
        )

    ranked_sorted = sorted(ranked, key=lambda r: (-r["mean_score"], r["sites"], r["depth"]))
    (out_root / "egcs_ranked.json").write_text(json.dumps(ranked_sorted, indent=2), encoding="utf-8")

    # Choose best candidate (if shortlist empty, fallback to base)
    best_cid = int(ranked_sorted[0]["cid"]) if ranked_sorted else 0
    best = cands[best_cid]
    save_circuit_png_or_txt(best, out_root, f"best_candidate_cid{best_cid:02d}")

    # EquiCheck on top-k best-ranked
    topk = ranked_sorted[: int(args.topk_verify)]
    eq_topk = []
    print("[5/6] YAQS EquiCheck on top candidates vs base...", flush=True)
    for r in topk:
        cid = int(r["cid"])
        try:
            b = yaqs_equiv_run(ideal_unit, cands[cid])
        except Exception as e:
            b = f"ERROR {type(e).__name__}: {e}"
        eq_topk.append({"cid": cid, "equicheck": b})
        print(f"    cid={cid:02d} equicheck={b}", flush=True)
    (out_root / "egcs_equicheck_topk.json").write_text(json.dumps(eq_topk, indent=2), encoding="utf-8")

    # ---- 6) Paired experiment: base vs best under gamma sweep (same seeds) ----
    print("[6/6] Paired gamma sweep (base vs best)...", flush=True)
    gammas = [float(x) for x in args.gamma_sweep.split(",") if x.strip()]
    paired_rows = []

    for g in gammas:
        base_stats = estimate_candidate(
            ideal_unit,
            z_ref,
            gamma=float(g),
            traj=int(args.traj_paired),
            seed0=int(args.seed) + 2025,   # fixed seeds list for fairness
            workers=int(args.workers),
            chunk=int(args.chunk),
            max_bond_dim=int(args.max_bond_dim),
            threshold=float(args.threshold),
        )
        best_stats = estimate_candidate(
            best,
            z_ref,
            gamma=float(g),
            traj=int(args.traj_paired),
            seed0=int(args.seed) + 2025,   # SAME seed0 => paired comparison
            workers=int(args.workers),
            chunk=int(args.chunk),
            max_bond_dim=int(args.max_bond_dim),
            threshold=float(args.threshold),
        )
        paired_rows.append(
            {
                "gamma": float(g),
                "base_mean": float(base_stats["mean_score"]),
                "base_std": float(base_stats["std_score"]),
                "base_p0": float(base_stats["p0"]),
                "best_cid": int(best_cid),
                "best_mean": float(best_stats["mean_score"]),
                "best_std": float(best_stats["std_score"]),
                "best_p0": float(best_stats["p0"]),
                "delta_mean": float(best_stats["mean_score"] - base_stats["mean_score"]),
            }
        )
        print(
            f"  gamma={g:.1e}  base={base_stats['mean_score']:.4f}  best={best_stats['mean_score']:.4f}  Δ={best_stats['mean_score']-base_stats['mean_score']:+.4f}",
            flush=True,
        )

    (out_root / "paired_gamma_sweep.json").write_text(json.dumps(paired_rows, indent=2), encoding="utf-8")

    # Plot
    g = np.array([r["gamma"] for r in paired_rows], dtype=float)
    yb = np.array([r["base_mean"] for r in paired_rows], dtype=float)
    yt = np.array([r["best_mean"] for r in paired_rows], dtype=float)

    plt.figure()
    plt.xscale("log")
    plt.plot(g, yb, marker="o", label="base (Z-match score)")
    plt.plot(g, yt, marker="o", label=f"best cid={best_cid} (Z-match score)")
    plt.xlabel("gamma (log)")
    plt.ylabel("Z-match score (1 - mean|ΔZ|/2)")
    plt.title(f"Paired comparison (traj={args.traj_paired}, workers={args.workers})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "paired_gamma_sweep.png", dpi=220)
    plt.close()

    # Quick bar for Δ at nominal gamma
    plt.figure()
    plt.xscale("log")
    plt.plot(g, (yt - yb), marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("gamma (log)")
    plt.ylabel("Δ score (best - base)")
    plt.title("Improvement curve")
    plt.tight_layout()
    plt.savefig(out_root / "paired_gamma_improvement.png", dpi=220)
    plt.close()

    print("\nDONE.")
    print("OUTPUT DIR =", out_root.resolve(), flush=True)
    print("FIG 1 =", (out_root / "paired_gamma_sweep.png").resolve(), flush=True)
    print("FIG 2 =", (out_root / "paired_gamma_improvement.png").resolve(), flush=True)


if __name__ == "__main__":
    main()
