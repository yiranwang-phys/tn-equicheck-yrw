from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit

# --- repo src-layout import ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from qem_yrw_project.circuits.heisenberg import HeisenbergParams, build_heisenberg_trotter

# --- YAQS ---
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import X, Y, Z, XX, YY, ZZ


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def symsinh_deltas(xmin: float, xmax: float, n: int, linthresh: float) -> np.ndarray:
    """Symlog-style sampling (supports negative). Dense near 0, log-like tails."""
    if n < 2:
        return np.array([0.0], dtype=float)
    umin = np.arcsinh(xmin / linthresh)
    umax = np.arcsinh(xmax / linthresh)
    u = np.linspace(umin, umax, n, dtype=float)
    x = linthresh * np.sinh(u)
    return x


def build_noise_model(n: int, gamma: float, kind: str) -> NoiseModel | None:
    kind = kind.lower()
    if kind in ("none", "no", "0"):
        return None
    if kind != "pauli_xyz":
        raise ValueError("This script supports noise: none | pauli_xyz")

    g = float(gamma) / 3.0
    processes = []
    for i in range(n):
        processes.append({"name": "pauli_x", "sites": [i], "strength": g})
        processes.append({"name": "pauli_y", "sites": [i], "strength": g})
        processes.append({"name": "pauli_z", "sites": [i], "strength": g})
    return NoiseModel(processes)


def make_observables(n: int, obs_csv: str) -> Tuple[List[Tuple[str, Observable]], List[str]]:
    want = [s.strip().lower() for s in obs_csv.split(",") if s.strip()]
    obs_specs: List[Tuple[str, Observable]] = []

    if "x" in want:
        for i in range(n):
            obs_specs.append((f"x[{i}]", Observable(X(), i)))
    if "y" in want:
        for i in range(n):
            obs_specs.append((f"y[{i}]", Observable(Y(), i)))
    if "z" in want:
        for i in range(n):
            obs_specs.append((f"z[{i}]", Observable(Z(), i)))

    if "xx" in want:
        for i in range(n - 1):
            obs_specs.append((f"xx[{i},{i+1}]", Observable(XX(), [i, i + 1])))
    if "yy" in want:
        for i in range(n - 1):
            obs_specs.append((f"yy[{i},{i+1}]", Observable(YY(), [i, i + 1])))
    if "zz" in want:
        for i in range(n - 1):
            obs_specs.append((f"zz[{i},{i+1}]", Observable(ZZ(), [i, i + 1])))

    labels = [k for (k, _) in obs_specs]
    return obs_specs, labels


def clone_obs(o: Observable) -> Observable:
    sites = getattr(o, "sites", None)
    if sites is None:
        raise AttributeError("Observable has no attribute 'sites' (YAQS API mismatch).")
    if isinstance(sites, (list, tuple)):
        sites = list(sites)
    return Observable(o.gate, sites)


def vec_from_dict(d: Dict[str, float], labels: List[str]) -> np.ndarray:
    return np.array([d[k] for k in labels], dtype=float)


def _qid(q) -> int:
    return int(getattr(q, "index", getattr(q, "_index")))


def apply_shift(qc: QuantumCircuit, delta: float, mode: str) -> Tuple[QuantumCircuit, int]:
    """Shift angles by +delta for selected gates. Return (shifted_circuit, num_shifted_gates)."""
    mode = mode.lower()
    delta = float(delta)

    def want_gate(name: str) -> bool:
        if mode == "global":
            return name in ("rx", "ry", "rz", "rxx", "ryy", "rzz")
        raise ValueError("This script is for global shift only.")

    q2 = QuantumCircuit(qc.num_qubits, qc.num_clbits, name=f"{qc.name}_shift_{mode}")
    n_shifted = 0

    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        cargs = ci.clbits
        name = inst.name.lower()

        if name == "barrier":
            qs = [q2.qubits[_qid(q)] for q in qargs]
            q2.barrier(*qs, label=getattr(inst, "label", None))
            continue
        if name == "measure":
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)
            continue

        if want_gate(name):
            n_shifted += 1
            theta = float(inst.params[0]) + delta
            if name == "rx":
                q2.rx(theta, _qid(qargs[0]))
            elif name == "ry":
                q2.ry(theta, _qid(qargs[0]))
            elif name == "rz":
                q2.rz(theta, _qid(qargs[0]))
            elif name == "rxx":
                q2.rxx(theta, _qid(qargs[0]), _qid(qargs[1]))
            elif name == "ryy":
                q2.ryy(theta, _qid(qargs[0]), _qid(qargs[1]))
            elif name == "rzz":
                q2.rzz(theta, _qid(qargs[0]), _qid(qargs[1]))
            else:
                q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)
        else:
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)

    return q2, n_shifted


def run_strong_observables(
    qc: QuantumCircuit,
    obs_specs: List[Tuple[str, Observable]],
    noise_model: NoiseModel | None,
    traj: int,
    max_bond_dim: int,
    threshold: float,
    parallel: bool,
    show_progress: bool,
) -> Dict[str, float]:
    obs_list = [clone_obs(o) for _, o in obs_specs]
    labels = [k for (k, _) in obs_specs]

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
    simulator.run(st, qc, sp, noise_model, parallel=parallel)

    out: Dict[str, float] = {}
    for lab, obs in zip(labels, obs_list):
        if obs.results is None or len(obs.results) == 0:
            raise RuntimeError(f"Observable {lab} empty results.")
        out[lab] = float(np.real(obs.results[0]))
    return out


def compute_metrics(obs: Dict[str, float], n: int, Jx: float, Jy: float, Jz: float) -> Dict[str, float]:
    z_list = [obs.get(f"z[{i}]", np.nan) for i in range(n)]
    mz = float(np.mean(z_list))

    def bond_avg(prefix: str) -> float:
        vals = [obs.get(f"{prefix}[{i},{i+1}]", np.nan) for i in range(n - 1)]
        return float(np.mean(vals))

    avg_xx = bond_avg("xx")
    avg_yy = bond_avg("yy")
    avg_zz = bond_avg("zz")
    energy = float(Jx * avg_xx + Jy * avg_yy + Jz * avg_zz)
    return {"Mz": mz, "XX_bond_avg": avg_xx, "YY_bond_avg": avg_yy, "ZZ_bond_avg": avg_zz, "E_bond": energy}


def plot_loss_vs_delta(outdir: Path, deltas: np.ndarray, losses: np.ndarray, linthresh: float, title: str) -> None:
    plt.figure()
    plt.plot(deltas, losses)
    plt.xscale("symlog", linthresh=linthresh)
    plt.yscale("log")
    plt.xlabel("delta")
    plt.ylabel("MSE loss vs ideal")
    plt.title(title)
    plt.grid(True)
    plt.savefig(outdir / "loss_vs_delta.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_vs_delta(
    outdir: Path,
    deltas: np.ndarray,
    values: np.ndarray,
    ideal_value: float,
    linthresh: float,
    name: str,
    title: str,
) -> None:
    plt.figure()
    plt.plot(deltas, values, label="noisy(δ)")
    plt.axhline(ideal_value, linestyle="--", label="ideal ref")
    plt.xscale("symlog", linthresh=linthresh)
    plt.xlabel("delta")
    plt.ylabel(name)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(outdir / f"{name}_vs_delta.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--depth", type=int, default=5)

    ap.add_argument("--dt", type=float, default=0.20)
    ap.add_argument("--Jx", type=float, default=1.0)
    ap.add_argument("--Jy", type=float, default=1.0)
    ap.add_argument("--Jz", type=float, default=1.0)
    ap.add_argument("--hx", type=float, default=0.0)
    ap.add_argument("--hy", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=0.0)

    ap.add_argument("--noise", type=str, default="pauli_xyz")
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=1000)

    ap.add_argument("--obs", type=str, default="z,xx,yy,zz")

    ap.add_argument("--delta_min", type=float, default=-2.0 * math.pi)
    ap.add_argument("--delta_max", type=float, default=+2.0 * math.pi)
    ap.add_argument("--delta_points", type=int, default=201)
    ap.add_argument("--linthresh", type=float, default=0.05)

    ap.add_argument("--max_bond_dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-9)
    ap.add_argument("--parallel", action="store_true", default=True)
    ap.add_argument("--no_parallel", action="store_true", default=False)
    ap.add_argument("--no_progress", action="store_true", default=False)

    ap.add_argument("--outdir", type=str, default="outputs/2026_1_01_global")
    args = ap.parse_args()

    parallel = bool(args.parallel and (not args.no_parallel))
    show_progress = not args.no_progress

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hp = HeisenbergParams(dt=args.dt, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz, hx=args.hx, hy=args.hy, hz=args.hz)
    base = build_heisenberg_trotter(args.n, args.depth, hp, periodic=False, label="heisenberg")

    obs_specs, labels = make_observables(args.n, args.obs)

    # Ideal reference (noise-free, traj=1)
    ideal_obs = run_strong_observables(
        base, obs_specs, None, traj=1, max_bond_dim=args.max_bond_dim, threshold=args.threshold, parallel=False, show_progress=False
    )
    ideal_vec = vec_from_dict(ideal_obs, labels)
    ideal_metrics = compute_metrics(ideal_obs, args.n, args.Jx, args.Jy, args.Jz)

    noise_model = build_noise_model(args.n, args.gamma, args.noise)

    deltas = symsinh_deltas(args.delta_min, args.delta_max, args.delta_points, args.linthresh)

    rows = []
    losses = []
    metric_series = {k: [] for k in ideal_metrics.keys()}

    best = (None, float("inf"))

    t0 = time.time()
    shifted_gate_count_at_0 = None

    for d in deltas:
        shifted, n_shifted = apply_shift(base, d, mode="global")
        if shifted_gate_count_at_0 is None and abs(d) < 1e-12:
            shifted_gate_count_at_0 = n_shifted

        noisy_obs = run_strong_observables(
            shifted,
            obs_specs,
            noise_model,
            traj=args.traj,
            max_bond_dim=args.max_bond_dim,
            threshold=args.threshold,
            parallel=parallel,
            show_progress=show_progress,
        )
        noisy_vec = vec_from_dict(noisy_obs, labels)
        loss = float(np.mean((noisy_vec - ideal_vec) ** 2))

        losses.append(loss)
        if loss < best[1]:
            best = (float(d), loss)

        noisy_metrics = compute_metrics(noisy_obs, args.n, args.Jx, args.Jy, args.Jz)
        for k in metric_series:
            metric_series[k].append(float(noisy_metrics[k]))

        row = {"delta": float(d), "loss_mse": loss}
        for lab in labels:
            row[f"ideal_{lab}"] = float(ideal_obs.get(lab, np.nan))
            row[f"noisy_{lab}"] = float(noisy_obs.get(lab, np.nan))
        for mk in noisy_metrics:
            row[f"ideal_{mk}"] = float(ideal_metrics[mk])
            row[f"noisy_{mk}"] = float(noisy_metrics[mk])
        rows.append(row)

    runtime = time.time() - t0
    losses_arr = np.array(losses, dtype=float)

    meta = {
        "script": Path(__file__).name,
        "time": timestamp(),
        "args": vars(args),
        "parallel": parallel,
        "shifted_gates_example_at_delta0": shifted_gate_count_at_0,
        "best_delta": best[0],
        "best_loss_mse": best[1],
        "runtime_sec": runtime,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with (outdir / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (outdir / "sweep.pkl").open("wb") as f:
        pickle.dump({"meta": meta, "labels": labels, "rows": rows}, f)

    plot_loss_vs_delta(
        outdir,
        deltas,
        losses_arr,
        args.linthresh,
        title=f"heisenberg | shift_mode=global | noise={args.noise} gamma={args.gamma} | n={args.n} depth={args.depth}",
    )

    # Key observable/metric plots with ideal ref
    for name, series in metric_series.items():
        plot_metric_vs_delta(
            outdir,
            deltas,
            np.array(series, dtype=float),
            ideal_metrics[name],
            args.linthresh,
            name,
            title=f"{name} vs delta | heisenberg global shift | gamma={args.gamma} traj={args.traj}",
        )

    print("\n=== DONE 2026_1_01_shift_sweep_heisenberg_global_YAQS ===")
    print(f"outdir: {outdir.resolve()}")
    print(f"best_delta: {best[0]}   best_loss_mse: {best[1]}")
    print(f"runtime_sec: {runtime:.2f}")


if __name__ == "__main__":
    main()
