from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from qem_yrw_project.circuits.heisenberg import HeisenbergParams, build_heisenberg_trotter

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z, XX, YY, ZZ


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def symsinh_deltas(xmin: float, xmax: float, n: int, linthresh: float) -> np.ndarray:
    umin = np.arcsinh(xmin / linthresh)
    umax = np.arcsinh(xmax / linthresh)
    u = np.linspace(umin, umax, n, dtype=float)
    return linthresh * np.sinh(u)


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


def make_observables(n: int) -> Tuple[List[Tuple[str, Observable]], List[str]]:
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
        raise AttributeError("Observable has no attribute 'sites'.")
    if isinstance(sites, (list, tuple)):
        sites = list(sites)
    return Observable(o.gate, sites)


def vec_from_dict(d: Dict[str, float], labels: List[str]) -> np.ndarray:
    return np.array([d[k] for k in labels], dtype=float)


def _qid(q) -> int:
    return int(getattr(q, "index", getattr(q, "_index")))


def apply_shift(qc: QuantumCircuit, delta: float, mode: str) -> QuantumCircuit:
    mode = mode.lower()
    delta = float(delta)

    def want_gate(name: str) -> bool:
        if mode == "xx":
            return name == "rxx"
        if mode == "yy":
            return name == "ryy"
        if mode == "zz":
            return name == "rzz"
        raise ValueError("mode must be xx|yy|zz")

    q2 = QuantumCircuit(qc.num_qubits, name=f"{qc.name}_shift_{mode}")
    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        cargs = ci.clbits
        name = inst.name.lower()

        if name in ("barrier", "measure"):
            continue

        if want_gate(name):
            theta = float(inst.params[0]) + delta
            if name == "rxx":
                q2.rxx(theta, _qid(qargs[0]), _qid(qargs[1]))
            elif name == "ryy":
                q2.ryy(theta, _qid(qargs[0]), _qid(qargs[1]))
            elif name == "rzz":
                q2.rzz(theta, _qid(qargs[0]), _qid(qargs[1]))
            else:
                q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)
        else:
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)

    return q2


def run_obs(
    qc: QuantumCircuit,
    obs_specs: List[Tuple[str, Observable]],
    noise: NoiseModel | None,
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
    simulator.run(st, qc, sp, noise, parallel=parallel)

    out = {}
    for lab, obs in zip(labels, obs_list):
        if obs.results is None or len(obs.results) == 0:
            raise RuntimeError(f"Observable {lab} empty.")
        out[lab] = float(np.real(obs.results[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--depth", type=int, default=5)

    ap.add_argument("--dt", type=float, default=0.20)
    ap.add_argument("--Jx", type=float, default=1.0)
    ap.add_argument("--Jy", type=float, default=1.0)
    ap.add_argument("--Jz", type=float, default=1.0)

    ap.add_argument("--noise", type=str, default="pauli_xyz")
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=1000)

    ap.add_argument("--delta_min", type=float, default=-2.0 * math.pi)
    ap.add_argument("--delta_max", type=float, default=+2.0 * math.pi)
    ap.add_argument("--delta_points", type=int, default=201)
    ap.add_argument("--linthresh", type=float, default=0.05)

    ap.add_argument("--modes", type=str, default="xx,yy,zz")

    ap.add_argument("--max_bond_dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-9)
    ap.add_argument("--parallel", action="store_true", default=True)
    ap.add_argument("--no_parallel", action="store_true", default=False)
    ap.add_argument("--no_progress", action="store_true", default=False)

    ap.add_argument("--outdir", type=str, default="outputs/2026_1_02_termwise")
    args = ap.parse_args()

    parallel = bool(args.parallel and (not args.no_parallel))
    show_progress = not args.no_progress

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hp = HeisenbergParams(dt=args.dt, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz)
    base = build_heisenberg_trotter(args.n, args.depth, hp, periodic=False, label="heisenberg")

    obs_specs, labels = make_observables(args.n)

    ideal = run_obs(base, obs_specs, None, traj=1, max_bond_dim=args.max_bond_dim, threshold=args.threshold, parallel=False, show_progress=False)
    ideal_vec = vec_from_dict(ideal, labels)

    noise = build_noise_model(args.n, args.gamma, args.noise)

    deltas = symsinh_deltas(args.delta_min, args.delta_max, args.delta_points, args.linthresh)

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    all_losses: Dict[str, np.ndarray] = {}
    bests: Dict[str, Tuple[float, float]] = {}

    rows = []

    t0 = time.time()
    for mode in modes:
        losses = []
        best = (None, float("inf"))
        for d in deltas:
            qc = apply_shift(base, d, mode)
            noisy = run_obs(qc, obs_specs, noise, traj=args.traj, max_bond_dim=args.max_bond_dim, threshold=args.threshold, parallel=parallel, show_progress=show_progress)
            v = vec_from_dict(noisy, labels)
            loss = float(np.mean((v - ideal_vec) ** 2))
            losses.append(loss)
            if loss < best[1]:
                best = (float(d), loss)

            rows.append({"mode": mode, "delta": float(d), "loss_mse": loss})

        all_losses[mode] = np.array(losses, dtype=float)
        bests[mode] = (best[0], best[1])

    runtime = time.time() - t0

    meta = {"script": Path(__file__).name, "time": timestamp(), "args": vars(args), "bests": bests, "runtime_sec": runtime}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with (outdir / "termwise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (outdir / "termwise.pkl").open("wb") as f:
        pickle.dump({"meta": meta, "deltas": deltas, "all_losses": all_losses, "bests": bests}, f)

    # plot combined
    plt.figure()
    for mode in modes:
        plt.plot(deltas, all_losses[mode], label=f"{mode} (best δ={bests[mode][0]:.4g})")
    plt.xscale("symlog", linthresh=args.linthresh)
    plt.yscale("log")
    plt.xlabel("delta")
    plt.ylabel("MSE loss vs ideal")
    plt.title(f"heisenberg | term-wise shift | noise={args.noise} gamma={args.gamma} | n={args.n} depth={args.depth}")
    plt.grid(True)
    plt.legend()
    plt.savefig(outdir / "termwise_loss_vs_delta.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n=== DONE 2026_1_02_shift_sweep_heisenberg_termwise_YAQS ===")
    print(f"outdir: {outdir.resolve()}")
    for mode in modes:
        print(f"best[{mode}]: delta={bests[mode][0]} loss={bests[mode][1]}")
    print(f"runtime_sec: {runtime:.2f}")


if __name__ == "__main__":
    main()
