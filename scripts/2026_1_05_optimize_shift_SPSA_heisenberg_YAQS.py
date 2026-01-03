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


def vec_from_dict(d: Dict[str, float], labels: List[str]) -> np.ndarray:
    return np.array([d[k] for k in labels], dtype=float)


def _qid(q) -> int:
    return int(getattr(q, "index", getattr(q, "_index")))


def apply_shift_params(qc: QuantumCircuit, theta: np.ndarray, mode: str) -> QuantumCircuit:
    mode = mode.lower()
    if mode == "global":
        d_rx = d_ry = d_rz = d_xx = d_yy = d_zz = float(theta[0])
    elif mode == "xyz":
        d_xx, d_yy, d_zz = map(float, theta[:3])
        d_rx = d_ry = d_rz = 0.0
    else:
        raise ValueError("mode must be global | xyz")

    q2 = QuantumCircuit(qc.num_qubits, name=f"{qc.name}_shift_{mode}")
    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        cargs = ci.clbits
        name = inst.name.lower()

        if name in ("barrier", "measure"):
            continue

        if name == "rx":
            q2.rx(float(inst.params[0]) + d_rx, _qid(qargs[0]))
        elif name == "ry":
            q2.ry(float(inst.params[0]) + d_ry, _qid(qargs[0]))
        elif name == "rz":
            q2.rz(float(inst.params[0]) + d_rz, _qid(qargs[0]))
        elif name == "rxx":
            q2.rxx(float(inst.params[0]) + d_xx, _qid(qargs[0]), _qid(qargs[1]))
        elif name == "ryy":
            q2.ryy(float(inst.params[0]) + d_yy, _qid(qargs[0]), _qid(qargs[1]))
        elif name == "rzz":
            q2.rzz(float(inst.params[0]) + d_zz, _qid(qargs[0]), _qid(qargs[1]))
        else:
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], cargs)

    return q2


def run_observables(
    qc: QuantumCircuit,
    obs_specs: List[Tuple[str, Observable]],
    noise: NoiseModel | None,
    traj: int,
    max_bond_dim: int,
    threshold: float,
    parallel: bool,
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
        show_progress=True,
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

    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--traj", type=int, default=1000)

    ap.add_argument("--mode", choices=["global", "xyz"], default="global")

    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--a", type=float, default=0.15)
    ap.add_argument("--c", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.602)
    ap.add_argument("--gamma_exp", type=float, default=0.101)

    ap.add_argument("--max_bond_dim", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=1e-9)
    ap.add_argument("--parallel", action="store_true", default=True)
    ap.add_argument("--no_parallel", action="store_true", default=False)

    ap.add_argument("--outdir", type=str, default="outputs/2026_1_05_spsa")
    args = ap.parse_args()

    parallel = bool(args.parallel and (not args.no_parallel))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hp = HeisenbergParams(dt=args.dt, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz)
    base = build_heisenberg_trotter(args.n, args.depth, hp, periodic=False, label="heisenberg")

    obs_specs, labels = make_observables(args.n)

    ideal = run_observables(base, obs_specs, None, traj=1, max_bond_dim=args.max_bond_dim, threshold=args.threshold, parallel=False)
    v_ideal = vec_from_dict(ideal, labels)

    noise = build_noise_model(args.n, args.gamma)

    dim = 1 if args.mode == "global" else 3
    theta = np.zeros(dim, dtype=float)

    def cost(th: np.ndarray) -> float:
        qc = apply_shift_params(base, th, args.mode)
        noisy = run_observables(qc, obs_specs, noise, args.traj, args.max_bond_dim, args.threshold, parallel=parallel)
        v = vec_from_dict(noisy, labels)
        return float(np.mean((v - v_ideal) ** 2))

    # baseline (theta=0)
    baseline = cost(theta.copy())

    hist = []
    best_y = float("inf")
    best_theta = theta.copy()

    t0 = time.time()
    for k in range(1, args.iters + 1):
        ak = args.a / (k ** args.alpha)
        ck = args.c / (k ** args.gamma_exp)
        delta = np.random.choice([-1.0, 1.0], size=dim)

        y_plus = cost(theta + ck * delta)
        y_minus = cost(theta - ck * delta)
        ghat = (y_plus - y_minus) / (2.0 * ck) * delta

        theta = theta - ak * ghat
        y = cost(theta)

        if y < best_y:
            best_y = y
            best_theta = theta.copy()

        hist.append(
            {
                "k": k,
                "ak": ak,
                "ck": ck,
                "y_plus": y_plus,
                "y_minus": y_minus,
                "y": y,
                "best_y": best_y,
                "theta": theta.copy(),
                "best_theta": best_theta.copy(),
            }
        )
        print(f"[SPSA] k={k:03d}  y={y:.6g}  best={best_y:.6g}  theta={theta}  best_theta={best_theta}")

    runtime = time.time() - t0

    ys = np.array([h["y"] for h in hist], dtype=float)
    bests = np.array([h["best_y"] for h in hist], dtype=float)
    ks = np.arange(1, len(ys) + 1, dtype=int)

    plt.figure()
    plt.plot(ks, ys, label="current loss")
    plt.plot(ks, bests, label="running best")
    plt.axhline(baseline, linestyle="--", label="baseline (theta=0)")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("MSE loss vs ideal")
    plt.title(f"SPSA optimize shift | mode={args.mode} gamma={args.gamma} traj={args.traj} | n={args.n} depth={args.depth}")
    plt.grid(True)
    plt.legend()
    plt.savefig(outdir / "spsa_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    import pickle
    with (outdir / "spsa.pkl").open("wb") as f:
        pickle.dump(
            {
                "args": vars(args),
                "labels": labels,
                "ideal": ideal,
                "baseline": baseline,
                "hist": hist,
                "best_theta": best_theta,
                "best_y": best_y,
                "runtime_sec": runtime,
            },
            f,
        )

    meta = {
        "time": timestamp(),
        "runtime_sec": runtime,
        "baseline": baseline,
        "best_theta": best_theta.tolist(),
        "best_y": best_y,
        "args": vars(args),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== DONE 2026_1_05_optimize_shift_SPSA_heisenberg_YAQS ===")
    print(f"outdir: {outdir.resolve()}")
    print(f"baseline: {baseline}")
    print(f"best_theta: {best_theta}  best_y: {best_y}")
    print(f"runtime_sec: {runtime:.2f}")


if __name__ == "__main__":
    main()
