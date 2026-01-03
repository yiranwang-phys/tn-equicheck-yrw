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


@dataclass(frozen=True)
class IsingParams:
    dt: float = 0.20
    Jzz: float = 1.0
    hx: float = 0.0


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def symsinh_deltas(xmin: float, xmax: float, n: int, linthresh: float) -> np.ndarray:
    umin = np.arcsinh(xmin / linthresh)
    umax = np.arcsinh(xmax / linthresh)
    u = np.linspace(umin, umax, n, dtype=float)
    return linthresh * np.sinh(u)


def build_pauli_xyz_noise(n: int, gamma: float) -> NoiseModel:
    g = float(gamma) / 3.0
    processes = []
    for i in range(n):
        processes.append({"name": "pauli_x", "sites": [i], "strength": g})
        processes.append({"name": "pauli_y", "sites": [i], "strength": g})
        processes.append({"name": "pauli_z", "sites": [i], "strength": g})
    return NoiseModel(processes)


def build_ising_trotter(n: int, depth: int, p: IsingParams, label: str = "ising") -> QuantumCircuit:
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


def apply_term_shift(qc: QuantumCircuit, delta: float, mode: str) -> Tuple[QuantumCircuit, int]:
    """Shift only rxx / ryy / rzz gates depending on mode. (delta only affects gate angles)"""
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
    n_shifted = 0
    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        name = inst.name.lower()

        if name in ("barrier", "measure"):
            continue

        if want_gate(name) and len(inst.params) == 1:
            n_shifted += 1
            th = float(inst.params[0]) + delta
            if name == "rxx":
                q2.rxx(th, _qid(qargs[0]), _qid(qargs[1]))
            elif name == "ryy":
                q2.ryy(th, _qid(qargs[0]), _qid(qargs[1]))
            elif name == "rzz":
                q2.rzz(th, _qid(qargs[0]), _qid(qargs[1]))
        else:
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], [])
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


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--dt", type=float, default=0.20)

    ap.add_argument("--Jx", type=float, default=1.0)
    ap.add_argument("--Jy", type=float, default=1.0)
    ap.add_argument("--Jz", type=float, default=1.0)

    ap.add_argument("--Jzz", type=float, default=1.0)
    ap.add_argument("--ising_hx", type=float, default=0.0)

    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--traj", type=int, default=1000)
    ap.add_argument("--traj_ref", type=int, default=10000)

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

    ap.add_argument("--outdir", type=str, default="outputs/2026_1_02_termwise_compare")
    args = ap.parse_args()

    parallel = bool(args.parallel and (not args.no_parallel))
    show_progress = not args.no_progress

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    noise = build_pauli_xyz_noise(args.n, args.gamma)
    obs_specs, labels = make_observables(args.n)
    deltas = symsinh_deltas(args.delta_min, args.delta_max, args.delta_points, args.linthresh)

    heis_p = HeisenbergParams(dt=args.dt, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz)
    base_heis = build_heisenberg_trotter(args.n, args.depth, heis_p, periodic=False, label="heisenberg")

    ising_p = IsingParams(dt=args.dt, Jzz=args.Jzz, hx=args.ising_hx)
    base_ising = build_ising_trotter(args.n, args.depth, ising_p, label="ising")

    circuits = {"heisenberg": base_heis, "ising": base_ising}

    # ref = noisy delta=0 no-rotation
    ref_vec: Dict[str, np.ndarray] = {}
    for model, base in circuits.items():
        ref = run_noisy_observables(
            base, obs_specs, noise, traj=args.traj_ref,
            max_bond_dim=args.max_bond_dim, threshold=args.threshold,
            parallel=parallel, show_progress=show_progress,
        )
        ref_vec[model] = vec_from_dict(ref, labels)

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]

    rows: List[Dict[str, float]] = []
    best: Dict[str, Dict[str, Tuple[float | None, float]]] = {mode: {m: (None, float("inf")) for m in circuits} for mode in modes}

    t0 = time.time()
    for mode in modes:
        loss_curve = {m: [] for m in circuits.keys()}
        for d in deltas:
            for model, base in circuits.items():
                shifted, n_shifted = apply_term_shift(base, d, mode)
                obs = run_noisy_observables(
                    shifted, obs_specs, noise,
                    traj=args.traj, max_bond_dim=args.max_bond_dim, threshold=args.threshold,
                    parallel=parallel, show_progress=show_progress,
                )
                v = vec_from_dict(obs, labels)
                loss = float(np.mean((v - ref_vec[model]) ** 2))
                loss_curve[model].append(loss)

                if loss < best[mode][model][1]:
                    best[mode][model] = (float(d), loss)

                rows.append({
                    "mode": mode,
                    "model": model,
                    "delta": float(d),
                    "loss_mse_vs_ref": loss,
                    "n_shifted_gates": float(n_shifted),
                })

        # plot per mode (two models)
        plt.figure()
        for model in ("heisenberg", "ising"):
            plt.plot(deltas, np.array(loss_curve[model], dtype=float), label=f"{model} (best δ={best[mode][model][0]:.4g})")
        plt.xscale("symlog", linthresh=args.linthresh)
        plt.yscale("log")
        plt.xlabel("delta")
        plt.ylabel("MSE loss vs noisy ref (δ=0, traj_ref)")
        plt.title(f"Term-wise shift mode={mode} | noise pauli_xyz gamma={args.gamma} | traj={args.traj} ref={args.traj_ref}")
        plt.grid(True)
        plt.legend()
        plt.savefig(outdir / f"loss_vs_delta_mode_{mode}.png", dpi=200, bbox_inches="tight")
        plt.close()

    runtime = time.time() - t0

    with (outdir / "termwise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "time": ts(),
        "args": vars(args),
        "best": {mode: {m: {"delta": best[mode][m][0], "loss": best[mode][m][1]} for m in circuits} for mode in modes},
        "runtime_sec": runtime,
        "ref": {"traj_ref": args.traj_ref, "definition": "no-rotation noisy ref"},
        "note": "If a model has no gates of a mode (e.g., Ising has no rxx/ryy), n_shifted_gates=0 and the curve is just sampling noise around ref.",
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== DONE 2026_1_02_termwise_compare ===")
    print(f"outdir: {outdir.resolve()}")
    print(json.dumps(meta["best"], indent=2))
    print(f"runtime_sec: {runtime:.2f}")


if __name__ == "__main__":
    main()
