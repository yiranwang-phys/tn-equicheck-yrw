from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate

# ---- repo src-layout import (Heisenberg builder) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from qem_yrw_project.circuits.heisenberg import HeisenbergParams, build_heisenberg_trotter

# ---- YAQS ----
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z, XX, YY, ZZ


@dataclass(frozen=True)
class IsingParams:
    dt: float = 0.20
    Jzz: float = 1.0
    hx: float = 0.0


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def symsinh_deltas(xmin: float, xmax: float, n: int, linthresh: float) -> np.ndarray:
    """Log-like spacing supporting negative deltas; dense near 0."""
    umin = np.arcsinh(xmin / linthresh)
    umax = np.arcsinh(xmax / linthresh)
    u = np.linspace(umin, umax, n, dtype=float)
    return linthresh * np.sinh(u)


def build_pauli_xyz_noise(n: int, gamma: float) -> NoiseModel:
    """Random Pauli XYZ jump with total rate gamma (split equally). Independent of delta."""
    g = float(gamma) / 3.0
    processes = []
    for i in range(n):
        processes.append({"name": "pauli_x", "sites": [i], "strength": g})
        processes.append({"name": "pauli_y", "sites": [i], "strength": g})
        processes.append({"name": "pauli_z", "sites": [i], "strength": g})
    return NoiseModel(processes)


def build_ising_trotter(n: int, depth: int, p: IsingParams, label: str = "ising") -> QuantumCircuit:
    """
    1D transverse-field Ising (Trotter):
        H = sum_<i,i+1> Jzz Z_i Z_{i+1} + sum_i hx X_i

    Gate convention:
        RZZ(theta) = exp(-i theta/2 Z⊗Z)  -> theta = 2*Jzz*dt
        RX(theta)  = exp(-i theta/2 X)    -> theta = 2*hx*dt
    """
    qc = QuantumCircuit(n, name=label)
    dt = float(p.dt)
    th_zz = 2.0 * float(p.Jzz) * dt
    th_x = 2.0 * float(p.hx) * dt

    even_pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_pairs = [(i, i + 1) for i in range(1, n - 1, 2)]

    for _ in range(int(depth)):
        for bonds in (even_pairs, odd_pairs):
            for (i, j) in bonds:
                if th_zz != 0.0:
                    qc.append(RZZGate(th_zz), [i, j])
        if th_x != 0.0:
            for q in range(n):
                qc.rx(th_x, q)

    return qc


def make_observables(n: int) -> Tuple[List[Tuple[str, Observable]], List[str]]:
    """Same observable set for both models -> fair comparison."""
    obs_specs: List[Tuple[str, Observable]] = []
    for i in range(n):
        obs_specs.append((f"z[{i}]", Observable(Z(), i)))
    for i in range(n - 1):
        obs_specs.append((f"xx[{i},{i+1}]", Observable(XX(), [i, i + 1])))
        obs_specs.append((f"yy[{i},{i+1}]", Observable(YY(), [i, i + 1])))
        obs_specs.append((f"zz[{i},{i+1}]", Observable(ZZ(), [i, i + 1])))
    labels = [k for k, _ in obs_specs]
    return obs_specs, labels


def clone_obs(o: Observable) -> Observable:
    sites = getattr(o, "sites", None)
    if sites is None:
        raise AttributeError("YAQS Observable expected attribute 'sites'.")
    if isinstance(sites, (list, tuple)):
        sites = list(sites)
    return Observable(o.gate, sites)


def vec_from_dict(d: Dict[str, float], labels: List[str]) -> np.ndarray:
    return np.array([float(d[k]) for k in labels], dtype=float)


def _qid(q) -> int:
    return int(getattr(q, "index", getattr(q, "_index")))


def apply_global_shift(qc: QuantumCircuit, delta: float) -> Tuple[QuantumCircuit, int]:
    """Add +delta to every 1-parameter rotation gate among rx/ry/rz/rxx/ryy/rzz."""
    delta = float(delta)
    q2 = QuantumCircuit(qc.num_qubits, name=f"{qc.name}_shift_global")
    n_shifted = 0

    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        cargs = ci.clbits
        name = inst.name.lower()

        # keep barriers/measures if any
        if name == "barrier":
            q2.barrier(*[q2.qubits[_qid(q)] for q in qargs], label=getattr(inst, "label", None))
            continue
        if name == "measure":
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)
            continue

        if name in ("rx", "ry", "rz", "rxx", "ryy", "rzz") and len(inst.params) == 1:
            n_shifted += 1
            th = float(inst.params[0]) + delta
            if name == "rx":
                q2.rx(th, _qid(qargs[0]))
            elif name == "ry":
                q2.ry(th, _qid(qargs[0]))
            elif name == "rz":
                q2.rz(th, _qid(qargs[0]))
            elif name == "rxx":
                q2.rxx(th, _qid(qargs[0]), _qid(qargs[1]))
            elif name == "ryy":
                q2.ryy(th, _qid(qargs[0]), _qid(qargs[1]))
            elif name == "rzz":
                q2.rzz(th, _qid(qargs[0]), _qid(qargs[1]))
            else:
                q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)
        else:
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)

    return q2, n_shifted


def run_noisy_observables(
    qc: QuantumCircuit,
    obs_specs: List[Tuple[str, Observable]],
    noise: NoiseModel,
    traj: int,
    max_bond_dim: int,
    threshold: float,
    parallel: bool,
    show_progress: bool,
) -> Dict[str, float]:
    obs_list = [clone_obs(o) for _, o in obs_specs]
    labels = [k for k, _ in obs_specs]

    sp = StrongSimParams(
        observables=obs_list,
        num_traj=int(traj),
        max_bond_dim=int(max_bond_dim),
        threshold=float(threshold),
        get_state=False,
        sample_layers=False,
        show_progress=bool(show_progress),
    )
    st = MPS(qc.num_qubits, state="zeros", pad=2)
    simulator.run(st, qc, sp, noise, parallel=parallel)

    out = {}
    for lab, obs in zip(labels, obs_list):
        if obs.results is None or len(obs.results) == 0:
            raise RuntimeError(f"Empty results for {lab}")
        out[lab] = float(np.real(obs.results[0]))
    return out


def metrics_from_obs(obs: Dict[str, float], n: int, model: str, Jx: float, Jy: float, Jz: float, Jzz: float) -> Dict[str, float]:
    z = [obs[f"z[{i}]"] for i in range(n)]
    mz = float(np.mean(z))

    def bond_avg(prefix: str) -> float:
        vals = [obs[f"{prefix}[{i},{i+1}]"] for i in range(n - 1)]
        return float(np.mean(vals))

    xx = bond_avg("xx")
    yy = bond_avg("yy")
    zz = bond_avg("zz")

    if model == "heisenberg":
        e = float(Jx * xx + Jy * yy + Jz * zz)
    else:
        e = float(Jzz * zz)

    return {"Mz": mz, "XX_bond_avg": xx, "YY_bond_avg": yy, "ZZ_bond_avg": zz, "E_bond": e}


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--depth", type=int, default=5)

    # shared dt
    ap.add_argument("--dt", type=float, default=0.20)

    # Heisenberg params
    ap.add_argument("--Jx", type=float, default=1.0)
    ap.add_argument("--Jy", type=float, default=1.0)
    ap.add_argument("--Jz", type=float, default=1.0)
    ap.add_argument("--hx", type=float, default=0.0)
    ap.add_argument("--hy", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=0.0)

    # Ising params
    ap.add_argument("--Jzz", type=float, default=1.0)
    ap.add_argument("--ising_hx", type=float, default=0.0)

    # noise
    ap.add_argument("--gamma", type=float, default=0.01)

    # traj
    ap.add_argument("--traj", type=int, default=1000)
    ap.add_argument("--traj_ref", type=int, default=10000)

    # delta sweep
    ap.add_argument("--delta_min", type=float, default=-2.0 * math.pi)
    ap.add_argument("--delta_max", type=float, default=+2.0 * math.pi)
    ap.add_argument("--delta_points", type=int, default=201)
    ap.add_argument("--linthresh", type=float, default=0.05)

    ap.add_argument("--max_bond_dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-9)

    ap.add_argument("--parallel", action="store_true", default=True)
    ap.add_argument("--no_parallel", action="store_true", default=False)
    ap.add_argument("--no_progress", action="store_true", default=False)

    ap.add_argument("--outdir", type=str, default="outputs/2026_1_01_global_compare")
    args = ap.parse_args()

    parallel = bool(args.parallel and (not args.no_parallel))
    show_progress = not args.no_progress

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    noise = build_pauli_xyz_noise(args.n, args.gamma)
    obs_specs, labels = make_observables(args.n)

    deltas = symsinh_deltas(args.delta_min, args.delta_max, args.delta_points, args.linthresh)

    # build both base circuits
    heis_p = HeisenbergParams(dt=args.dt, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz, hx=args.hx, hy=args.hy, hz=args.hz)
    base_heis = build_heisenberg_trotter(args.n, args.depth, heis_p, periodic=False, label="heisenberg")

    ising_p = IsingParams(dt=args.dt, Jzz=args.Jzz, hx=args.ising_hx)
    base_ising = build_ising_trotter(args.n, args.depth, ising_p, label="ising")

    circuits = {"heisenberg": base_heis, "ising": base_ising}

    # ref = noisy, delta=0, traj_ref
    ref_obs: Dict[str, Dict[str, float]] = {}
    ref_vec: Dict[str, np.ndarray] = {}
    ref_metrics: Dict[str, Dict[str, float]] = {}

    for model, base in circuits.items():
        ref = run_noisy_observables(
            base,
            obs_specs,
            noise,
            traj=args.traj_ref,
            max_bond_dim=args.max_bond_dim,
            threshold=args.threshold,
            parallel=parallel,
            show_progress=show_progress,
        )
        ref_obs[model] = ref
        ref_vec[model] = vec_from_dict(ref, labels)
        ref_metrics[model] = metrics_from_obs(ref, args.n, model, args.Jx, args.Jy, args.Jz, args.Jzz)

    rows: List[Dict[str, float]] = []
    loss_curves: Dict[str, List[float]] = {"heisenberg": [], "ising": []}
    metrics_series: Dict[str, Dict[str, List[float]]] = {
        "heisenberg": {k: [] for k in ref_metrics["heisenberg"].keys()},
        "ising": {k: [] for k in ref_metrics["ising"].keys()},
    }

    best = {m: (None, float("inf")) for m in circuits.keys()}

    t0 = time.time()
    for d in deltas:
        for model, base in circuits.items():
            shifted, n_shifted = apply_global_shift(base, d)

            obs = run_noisy_observables(
                shifted,
                obs_specs,
                noise,
                traj=args.traj,
                max_bond_dim=args.max_bond_dim,
                threshold=args.threshold,
                parallel=parallel,
                show_progress=show_progress,
            )
            v = vec_from_dict(obs, labels)
            loss = float(np.mean((v - ref_vec[model]) ** 2))
            loss_curves[model].append(loss)
            if loss < best[model][1]:
                best[model] = (float(d), loss)

            mets = metrics_from_obs(obs, args.n, model, args.Jx, args.Jy, args.Jz, args.Jzz)
            for k in mets:
                metrics_series[model][k].append(float(mets[k]))

            row = {"model": model, "delta": float(d), "loss_mse_vs_ref": loss, "n_shifted_gates": float(n_shifted)}
            # store a few metrics
            for k in mets:
                row[f"value_{k}"] = float(mets[k])
                row[f"ref_{k}"] = float(ref_metrics[model][k])
            rows.append(row)

    runtime = time.time() - t0

    # save
    with (outdir / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "time": ts(),
        "args": vars(args),
        "best": {m: {"delta": best[m][0], "loss": best[m][1]} for m in best},
        "runtime_sec": runtime,
        "ref": {"traj_ref": args.traj_ref, "definition": "no-rotation noisy ref"},
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # plot loss compare
    plt.figure()
    for model in ("heisenberg", "ising"):
        plt.plot(deltas, np.array(loss_curves[model], dtype=float), label=f"{model} (best δ={best[model][0]:.4g})")
    plt.xscale("symlog", linthresh=args.linthresh)
    plt.yscale("log")
    plt.xlabel("delta (shift of gate angles)")
    plt.ylabel("MSE loss vs noisy ref (delta=0, traj_ref)")
    plt.title(f"Global shift sweep | same noise pauli_xyz gamma={args.gamma} | traj={args.traj} ref={args.traj_ref}")
    plt.grid(True)
    plt.legend()
    plt.savefig(outdir / "loss_vs_delta_compare.png", dpi=200, bbox_inches="tight")
    plt.close()

    # metric plots per model (with noisy ref line)
    for model in ("heisenberg", "ising"):
        for k in metrics_series[model].keys():
            plt.figure()
            plt.plot(deltas, np.array(metrics_series[model][k], dtype=float), label=f"{model} noisy(δ)")
            plt.axhline(ref_metrics[model][k], linestyle="--", label=f"{model} ref (δ=0, traj_ref)")
            plt.xscale("symlog", linthresh=args.linthresh)
            plt.xlabel("delta")
            plt.ylabel(k)
            plt.title(f"{model} | {k} vs delta | noise gamma={args.gamma}")
            plt.grid(True)
            plt.legend()
            plt.savefig(outdir / f"{model}_{k}_vs_delta.png", dpi=200, bbox_inches="tight")
            plt.close()

    print("\n=== DONE 2026_1_01_global_compare ===")
    print(f"outdir: {outdir.resolve()}")
    for model in best:
        print(f"best[{model}]: delta={best[model][0]} loss={best[model][1]}")
    print(f"runtime_sec: {runtime:.2f}")


if __name__ == "__main__":
    main()
