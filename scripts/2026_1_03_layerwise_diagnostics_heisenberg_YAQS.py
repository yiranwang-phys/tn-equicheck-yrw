from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from qem_yrw_project.circuits.heisenberg import HeisenbergParams

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z, XX, YY, ZZ


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_noise_model(n: int, gamma: float) -> NoiseModel:
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


def _qid(q) -> int:
    return int(getattr(q, "index", getattr(q, "_index")))


def apply_global_shift(qc: QuantumCircuit, delta: float) -> QuantumCircuit:
    delta = float(delta)
    q2 = QuantumCircuit(qc.num_qubits, name=f"{qc.name}_shift_global")

    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        name = inst.name.lower()

        if name == "barrier":
            qs = [q2.qubits[_qid(q)] for q in qargs]
            q2.barrier(*qs, label=getattr(inst, "label", None))
            continue

        if name == "rx":
            q2.rx(float(inst.params[0]) + delta, _qid(qargs[0]))
        elif name == "ry":
            q2.ry(float(inst.params[0]) + delta, _qid(qargs[0]))
        elif name == "rz":
            q2.rz(float(inst.params[0]) + delta, _qid(qargs[0]))
        elif name == "rxx":
            q2.rxx(float(inst.params[0]) + delta, _qid(qargs[0]), _qid(qargs[1]))
        elif name == "ryy":
            q2.ryy(float(inst.params[0]) + delta, _qid(qargs[0]), _qid(qargs[1]))
        elif name == "rzz":
            q2.rzz(float(inst.params[0]) + delta, _qid(qargs[0]), _qid(qargs[1]))
        else:
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], ci.clbits)

    return q2


def build_heisenberg_sampled(n: int, depth: int, p: HeisenbergParams) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="heisenberg_sampled")

    dt = float(p.dt)
    th_xx = 2.0 * float(p.Jx) * dt
    th_yy = 2.0 * float(p.Jy) * dt
    th_zz = 2.0 * float(p.Jz) * dt

    th_x = 2.0 * float(p.hx) * dt
    th_y = 2.0 * float(p.hy) * dt
    th_z = 2.0 * float(p.hz) * dt

    even_pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_pairs = [(i, i + 1) for i in range(1, n - 1, 2)]

    for _layer in range(int(depth)):
        for bond_list in (even_pairs, odd_pairs):
            for (i, j) in bond_list:
                if th_xx != 0.0:
                    qc.append(RXXGate(th_xx), [i, j])
                if th_yy != 0.0:
                    qc.append(RYYGate(th_yy), [i, j])
                if th_zz != 0.0:
                    qc.append(RZZGate(th_zz), [i, j])

        for q in range(n):
            if th_x != 0.0:
                qc.rx(th_x, q)
            if th_y != 0.0:
                qc.ry(th_y, q)
            if th_z != 0.0:
                qc.rz(th_z, q)

        qc.barrier(label="SAMPLE_OBSERVABLES")

    return qc


def extract_layer_series(labels: List[str], obs_list: List[Observable]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for lab, obs in zip(labels, obs_list):
        if obs.results is None:
            raise RuntimeError(f"Observable {lab} has no results.")
        out[lab] = np.real(np.array(obs.results, dtype=float))
    return out


def layerwise_metrics(layer_vals: Dict[str, np.ndarray], n: int, Jx: float, Jy: float, Jz: float) -> Dict[str, np.ndarray]:
    # Mz
    zs = [layer_vals[f"z[{i}]"] for i in range(n)]
    mz = np.mean(np.stack(zs, axis=0), axis=0)

    # bond avgs
    def bavg(prefix: str) -> np.ndarray:
        terms = [layer_vals[f"{prefix}[{i},{i+1}]"] for i in range(n - 1)]
        return np.mean(np.stack(terms, axis=0), axis=0)

    xx = bavg("xx")
    yy = bavg("yy")
    zz = bavg("zz")
    energy = Jx * xx + Jy * yy + Jz * zz

    return {"Mz": mz, "XX_bond_avg": xx, "YY_bond_avg": yy, "ZZ_bond_avg": zz, "E_bond": energy}


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

    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=1000)
    ap.add_argument("--deltas", type=str, default="0.0,0.2,-0.2")

    ap.add_argument("--max_bond_dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-9)
    ap.add_argument("--parallel", action="store_true", default=True)
    ap.add_argument("--no_parallel", action="store_true", default=False)

    ap.add_argument("--outdir", type=str, default="outputs/2026_1_03_layerwise")
    args = ap.parse_args()

    parallel = bool(args.parallel and (not args.no_parallel))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hp = HeisenbergParams(
        dt=args.dt, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz, hx=args.hx, hy=args.hy, hz=args.hz
    )
    base = build_heisenberg_sampled(args.n, args.depth, hp)
    noise = build_noise_model(args.n, args.gamma)

    obs_specs, labels = make_observables(args.n)
    deltas = [float(s.strip()) for s in args.deltas.split(",") if s.strip()]

    results_all: Dict[str, Dict[str, np.ndarray]] = {}

    t0 = time.time()
    for d in deltas:
        obs_list = [clone_obs(o) for _, o in obs_specs]
        sp = StrongSimParams(
            observables=obs_list,
            num_traj=int(args.traj),
            max_bond_dim=int(args.max_bond_dim),
            threshold=float(args.threshold),
            get_state=False,
            sample_layers=True,
            show_progress=True,
        )
        qc = apply_global_shift(base, d)
        st = MPS(args.n, state="zeros", pad=2)
        simulator.run(st, qc, sp, noise, parallel=parallel)
        layer_vals = extract_layer_series(labels, obs_list)
        results_all[str(d)] = layerwise_metrics(layer_vals, args.n, args.Jx, args.Jy, args.Jz)

    runtime = time.time() - t0

    # plot each metric
    x = None
    for metric_name in ["Mz", "E_bond", "XX_bond_avg", "YY_bond_avg", "ZZ_bond_avg"]:
        plt.figure()
        for d in deltas:
            y = results_all[str(d)][metric_name]
            if x is None:
                x = np.arange(len(y), dtype=int)
            plt.plot(x, y, label=f"delta={d}")
        plt.xlabel("layer index (includes t=0 and final)")
        plt.ylabel(metric_name)
        plt.title(f"Layerwise {metric_name} | shift=global | gamma={args.gamma} traj={args.traj}")
        plt.grid(True)
        plt.legend()
        plt.savefig(outdir / f"layerwise_{metric_name}.png", dpi=200, bbox_inches="tight")
        plt.close()

    import pickle
    with (outdir / "layerwise.pkl").open("wb") as f:
        pickle.dump({"args": vars(args), "deltas": deltas, "results": results_all, "runtime_sec": runtime}, f)

    (outdir / "meta.json").write_text(
        json.dumps({"time": timestamp(), "runtime_sec": runtime, "args": vars(args)}, indent=2),
        encoding="utf-8",
    )

    print("\n=== DONE 2026_1_03_layerwise_diagnostics_heisenberg_YAQS ===")
    print(f"outdir: {outdir.resolve()}")
    print(f"runtime_sec: {runtime:.2f}")


if __name__ == "__main__":
    main()
