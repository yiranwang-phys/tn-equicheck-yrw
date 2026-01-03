# scripts/2026_1_common_shift_tools.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    RXGate,
    RYGate,
    RZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    XGate,
    YGate,
    ZGate,
    HGate,
)
from qiskit.quantum_info import Statevector, SparsePauliOp


# -----------------------------
# Paths (src-layout friendly)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


# -----------------------------
# Circuit builders (import if exists; fallback otherwise)
# -----------------------------
def build_heisenberg(n: int, depth: int) -> QuantumCircuit:
    if SRC_DIR.exists():
        import sys

        sys.path.insert(0, str(SRC_DIR))
    from qem_yrw_project.circuits.heisenberg import (  # type: ignore
        HeisenbergParams,
        build_heisenberg_trotter,
    )

    # IMPORTANT: keep hx/hy/hz=0 here for the noisy sweep baseline.
    # (phase script will set a symmetry-breaking prep or field separately)
    params = HeisenbergParams(dt=0.20, Jx=1.0, Jy=1.0, Jz=1.0, hx=0.0, hy=0.0, hz=0.0)
    qc = build_heisenberg_trotter(
        n_qubits=n,
        depth=depth,
        params=params,
        add_sample_barriers=False,
        add_measurements=False,
    )
    return qc


def build_ising(n: int, depth: int) -> QuantumCircuit:
    # Try your project builder if it exists; otherwise fallback to a minimal Ising(Trotter) circuit
    if SRC_DIR.exists():
        import sys

        sys.path.insert(0, str(SRC_DIR))
        try:
            from qem_yrw_project.circuits.ising import build_ising_trotter  # type: ignore

            return build_ising_trotter(n_qubits=n, depth=depth)
        except Exception:
            pass

    # Fallback: brickwork ZZ couplings + RX field each layer
    dt = 0.20
    Jz = 1.0
    hx = 1.0
    th_zz = 2.0 * Jz * dt
    th_x = 2.0 * hx * dt

    qc = QuantumCircuit(n, name="ising_trotter_fallback")
    bonds = [(i, i + 1) for i in range(n - 1)]
    for _ in range(depth):
        for (i, j) in bonds:
            qc.append(RZZGate(th_zz), [i, j])
        for q in range(n):
            qc.append(RXGate(th_x), [q])
    return qc


# -----------------------------
# Delta grid: [-2pi,2pi], log-style (symlog)
# -----------------------------
def make_symlog_deltas(
    *,
    delta_max: float,
    points: int,
    min_abs: float = 1e-4,
) -> np.ndarray:
    """
    Create symmetric log-spaced deltas in [-delta_max, delta_max] with dense sampling near 0.

    points must be odd so that 0 is included.
    """
    if points < 3:
        raise ValueError("points must be >= 3")
    if points % 2 == 0:
        points += 1

    half = (points - 1) // 2
    mags = np.logspace(np.log10(min_abs), np.log10(delta_max), half)
    deltas = np.concatenate([-mags[::-1], np.array([0.0]), mags])
    # Ensure exact endpoints (helpful for sanity)
    deltas[0] = -delta_max
    deltas[-1] = delta_max
    return deltas


# -----------------------------
# Shift logic
# -----------------------------
ShiftMode = Literal["global", "rzz", "xx", "yy", "zz"]

SHIFTABLE = {"rx", "ry", "rz", "rxx", "ryy", "rzz"}


def should_shift(op_name: str, mode: ShiftMode) -> bool:
    name = op_name.lower()
    if mode == "global":
        return name in SHIFTABLE
    if mode == "rzz":
        return name == "rzz"
    if mode == "xx":
        return name == "rxx"
    if mode == "yy":
        return name == "ryy"
    if mode == "zz":
        return name == "rzz"
    return False


def count_shift_targets(qc: QuantumCircuit, mode: ShiftMode) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for inst in qc.data:
        op = inst.operation
        name = op.name.lower()
        if name in ("barrier", "measure"):
            continue
        if should_shift(name, mode):
            counts[name] = counts.get(name, 0) + 1
    return counts


# -----------------------------
# Observables
# -----------------------------
@dataclass(frozen=True)
class ObsSet:
    names: List[str]
    # mapping name -> list of (SparsePauliOp, weight)
    terms: Dict[str, List[Tuple[SparsePauliOp, float]]]
    # mapping name -> normalization factor (divide by this)
    norm: Dict[str, float]


def _pauli_label(n: int, ops: Dict[int, str]) -> str:
    """
    Qiskit Pauli string convention: rightmost char is qubit 0.
    """
    label = ["I"] * n
    for q, ch in ops.items():
        label[n - 1 - q] = ch
    return "".join(label)


def build_observables(n: int, model: Literal["ising", "heisenberg"]) -> ObsSet:
    bonds = [(i, i + 1) for i in range(n - 1)]

    terms: Dict[str, List[Tuple[SparsePauliOp, float]]] = {}
    norm: Dict[str, float] = {}
    names: List[str] = []

    # Mz: average Z
    mz_terms = []
    for i in range(n):
        lab = _pauli_label(n, {i: "Z"})
        mz_terms.append((SparsePauliOp.from_list([(lab, 1.0)]), 1.0))
    terms["Mz"] = mz_terms
    norm["Mz"] = float(n)
    names.append("Mz")

    # ZZ/XX/YY bond averages
    for nm, ch in [("ZZ_bond_avg", "Z"), ("XX_bond_avg", "X"), ("YY_bond_avg", "Y")]:
        t = []
        for (i, j) in bonds:
            lab = _pauli_label(n, {i: ch, j: ch})
            t.append((SparsePauliOp.from_list([(lab, 1.0)]), 1.0))
        terms[nm] = t
        norm[nm] = float(len(bonds))
        names.append(nm)

    # E_bond (coupling energy per bond)
    # - for heisenberg: <XX> + <YY> + <ZZ> per bond (Jx=Jy=Jz=1 here)
    # - for ising: just <ZZ> per bond (coupling part)
    if model == "heisenberg":
        e_terms = []
        for (i, j) in bonds:
            for ch in ("X", "Y", "Z"):
                lab = _pauli_label(n, {i: ch, j: ch})
                e_terms.append((SparsePauliOp.from_list([(lab, 1.0)]), 1.0))
        terms["E_bond"] = e_terms
        norm["E_bond"] = float(len(bonds))  # note: weights sum to 3 per bond; we keep scale as-is
        names.insert(0, "E_bond")
    else:
        # Ising coupling energy proxy = ZZ bond average
        terms["E_bond"] = terms["ZZ_bond_avg"]
        norm["E_bond"] = norm["ZZ_bond_avg"]
        names.insert(0, "E_bond")

    return ObsSet(names=names, terms=terms, norm=norm)


def eval_obs(sv: Statevector, obs: ObsSet) -> np.ndarray:
    out = np.zeros(len(obs.names), dtype=float)
    for k, name in enumerate(obs.names):
        acc = 0.0
        for (op, w) in obs.terms[name]:
            acc += w * float(np.real(sv.expectation_value(op)))
        out[k] = acc / obs.norm[name]
    return out


# -----------------------------
# Prep states (for phase experiment symmetry-breaking)
# -----------------------------
def apply_prep(qc: QuantumCircuit, prep: Literal["zero", "plus", "neel", "random"], seed: int) -> None:
    n = qc.num_qubits
    if prep == "zero":
        return
    if prep == "plus":
        for q in range(n):
            qc.append(HGate(), [q])
        return
    if prep == "neel":
        # |0101...>
        for q in range(n):
            if q % 2 == 1:
                qc.append(XGate(), [q])
        return
    if prep == "random":
        rng = np.random.default_rng(seed)
        for q in range(n):
            qc.append(RYGate(float(rng.uniform(-np.pi, np.pi))), [q])
            qc.append(RZGate(float(rng.uniform(-np.pi, np.pi))), [q])
        return
    raise ValueError(f"Unknown prep={prep}")


# -----------------------------
# Monte-Carlo Pauli jump noise
# -----------------------------
def _apply_gate_step(sv: Statevector, name: str, theta: float | None, qargs: Tuple[int, ...]) -> Statevector:
    nm = name.lower()
    if nm == "rx":
        return sv.evolve(RXGate(float(theta)), qargs=list(qargs))
    if nm == "ry":
        return sv.evolve(RYGate(float(theta)), qargs=list(qargs))
    if nm == "rz":
        return sv.evolve(RZGate(float(theta)), qargs=list(qargs))
    if nm == "rzz":
        return sv.evolve(RZZGate(float(theta)), qargs=list(qargs))
    if nm == "rxx":
        return sv.evolve(RXXGate(float(theta)), qargs=list(qargs))
    if nm == "ryy":
        return sv.evolve(RYYGate(float(theta)), qargs=list(qargs))
    if nm == "x":
        return sv.evolve(XGate(), qargs=list(qargs))
    if nm == "y":
        return sv.evolve(YGate(), qargs=list(qargs))
    if nm == "z":
        return sv.evolve(ZGate(), qargs=list(qargs))
    raise ValueError(f"Unsupported gate name={name!r} (add it to _apply_gate_step)")


def extract_ops(qc: QuantumCircuit) -> List[Tuple[str, float | None, Tuple[int, ...]]]:
    qindex = {qb: i for i, qb in enumerate(qc.qubits)}
    ops: List[Tuple[str, float | None, Tuple[int, ...]]] = []
    for inst in qc.data:
        op = inst.operation
        nm = op.name.lower()
        if nm in ("barrier", "measure"):
            continue
        qargs = tuple(qindex[q] for q in inst.qubits)
        theta = None
        if getattr(op, "params", None):
            if len(op.params) == 1:
                theta = float(op.params[0])
        ops.append((nm, theta, qargs))
    return ops


def simulate_one_trajectory(
    *,
    n: int,
    base_ops: List[Tuple[str, float | None, Tuple[int, ...]]],
    delta: float,
    shift_mode: ShiftMode,
    gamma: float,
    seed: int,
    prep: Literal["zero", "plus", "neel", "random"],
    prep_seed: int,
    obs: ObsSet,
) -> np.ndarray:
    # Prepare initial state via a prep circuit (safer than hand-encoding)
    qc_prep = QuantumCircuit(n)
    apply_prep(qc_prep, prep=prep, seed=prep_seed)
    sv = Statevector.from_label("0" * n).evolve(qc_prep)

    rng = np.random.default_rng(seed)

    for (nm, theta, qargs) in base_ops:
        th = theta
        if th is not None and should_shift(nm, shift_mode):
            th = float(th + delta)

        if th is None:
            sv = _apply_gate_step(sv, nm, None, qargs)
        else:
            sv = _apply_gate_step(sv, nm, float(th), qargs)

        # Pauli jump noise: for each involved qubit, with prob gamma apply random {X,Y,Z}
        if gamma > 0.0:
            for q in qargs:
                if rng.random() < gamma:
                    r = int(rng.integers(0, 3))
                    if r == 0:
                        sv = _apply_gate_step(sv, "x", None, (q,))
                    elif r == 1:
                        sv = _apply_gate_step(sv, "y", None, (q,))
                    else:
                        sv = _apply_gate_step(sv, "z", None, (q,))

    return eval_obs(sv, obs)


def _worker_batch(args: dict) -> Tuple[np.ndarray, int]:
    n = int(args["n"])
    base_ops = args["base_ops"]
    delta = float(args["delta"])
    shift_mode = args["shift_mode"]
    gamma = float(args["gamma"])
    start_seed = int(args["start_seed"])
    batch = int(args["batch"])
    prep = args["prep"]
    prep_seed = int(args["prep_seed"])
    obs = args["obs"]

    acc = np.zeros(len(obs.names), dtype=float)
    for t in range(batch):
        acc += simulate_one_trajectory(
            n=n,
            base_ops=base_ops,
            delta=delta,
            shift_mode=shift_mode,
            gamma=gamma,
            seed=start_seed + t,
            prep=prep,
            prep_seed=prep_seed,
            obs=obs,
        )
    return acc, batch


def estimate_noisy_means(
    *,
    n: int,
    base_ops: List[Tuple[str, float | None, Tuple[int, ...]]],
    deltas: np.ndarray,
    shift_mode: ShiftMode,
    gamma: float,
    traj: int,
    workers: int,
    seed0: int,
    prep: Literal["zero", "plus", "neel", "random"],
    prep_seed: int,
    obs: ObsSet,
) -> np.ndarray:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    means = np.zeros((len(deltas), len(obs.names)), dtype=float)

    # chunk trajectories into batches for each worker task (avoid 1000 futures)
    batches = [traj // workers] * workers
    for i in range(traj % workers):
        batches[i] += 1
    batches = [b for b in batches if b > 0]

    for di, delta in enumerate(deltas):
        tasks = []
        next_seed = seed0 + di * 1000003  # separate seed stream per delta
        with ProcessPoolExecutor(max_workers=workers) as ex:
            s = next_seed
            for b in batches:
                payload = dict(
                    n=n,
                    base_ops=base_ops,
                    delta=float(delta),
                    shift_mode=shift_mode,
                    gamma=float(gamma),
                    start_seed=int(s),
                    batch=int(b),
                    prep=prep,
                    prep_seed=int(prep_seed),
                    obs=obs,
                )
                tasks.append(ex.submit(_worker_batch, payload))
                s += b

            acc = np.zeros(len(obs.names), dtype=float)
            tot = 0
            for fut in as_completed(tasks):
                a, k = fut.result()
                acc += a
                tot += k
        means[di, :] = acc / float(tot)

    return means


# -----------------------------
# Pickle cache
# -----------------------------
def stable_hash(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:12]


def cache_paths(outroot: Path) -> Tuple[Path, Path]:
    figdir = outroot / "figs"
    pkldir = outroot / "pickle"
    figdir.mkdir(parents=True, exist_ok=True)
    pkldir.mkdir(parents=True, exist_ok=True)
    return figdir, pkldir


def save_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# Metrics + plotting
# -----------------------------
def mse_vs_ref(means: np.ndarray, ref: np.ndarray) -> np.ndarray:
    diff = means - ref[None, :]
    return np.mean(diff * diff, axis=1)


def plot_mse(
    *,
    title: str,
    deltas: np.ndarray,
    mse: np.ndarray,
    figpath: Path,
    floor: float | None = None,
    best_delta: float | None = None,
) -> None:
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(deltas, mse, label="MSE loss")

    if floor is not None:
        ax.axhline(float(floor), linestyle="--", label="MC floor (δ=0, traj vs traj_ref)")
    if best_delta is not None:
        ax.axvline(float(best_delta), linestyle="--", label=f"best δ={best_delta:.4g}")

    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_yscale("log")
    ax.set_xlabel("delta (shift of gate angles)")
    ax.set_ylabel("MSE loss vs noisy ref (δ=0)")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figpath, dpi=160)
    plt.close(fig)


def plot_obs_vs_delta(
    *,
    title: str,
    deltas: np.ndarray,
    obs_name: str,
    obs_vals: np.ndarray,
    ref_val: float | None,
    figpath: Path,
) -> None:
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(deltas, obs_vals, label=f"{obs_name}(δ)")
    if ref_val is not None:
        ax.axhline(float(ref_val), linestyle="--", label=f"ref δ=0 (traj_ref)")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("delta")
    ax.set_ylabel(obs_name)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figpath, dpi=160)
    plt.close(fig)


def plot_fidelity(
    *,
    title: str,
    deltas: np.ndarray,
    fid: np.ndarray,
    figpath: Path,
) -> None:
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(deltas, fid, label="fidelity(δ) vs δ=0")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("delta")
    ax.set_ylabel("fidelity (0..1)")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figpath, dpi=160)
    plt.close(fig)


def noiseless_fidelity_and_obs(
    *,
    n: int,
    base_ops: List[Tuple[str, float | None, Tuple[int, ...]]],
    deltas: np.ndarray,
    shift_mode: ShiftMode,
    prep: Literal["zero", "plus", "neel", "random"],
    prep_seed: int,
    obs: ObsSet,
    obs_pick: str,
) -> Tuple[np.ndarray, np.ndarray]:
    # baseline state (delta=0)
    qc_prep = QuantumCircuit(n)
    apply_prep(qc_prep, prep=prep, seed=prep_seed)
    sv0 = Statevector.from_label("0" * n).evolve(qc_prep)

    def evolve(delta: float) -> Statevector:
        sv = sv0
        for (nm, theta, qargs) in base_ops:
            th = theta
            if th is not None and should_shift(nm, shift_mode):
                th = float(th + delta)
            sv = _apply_gate_step(sv, nm, th, qargs)
        return sv

    psi_ref = evolve(0.0)
    fid = np.zeros(len(deltas), dtype=float)
    obs_arr = np.zeros(len(deltas), dtype=float)

    k_obs = obs.names.index(obs_pick)

    for i, d in enumerate(deltas):
        psi = evolve(float(d))
        amp = np.vdot(psi_ref.data, psi.data)
        fid[i] = float(np.real(amp * np.conjugate(amp)))
        obs_arr[i] = float(eval_obs(psi, obs)[k_obs])

    return fid, obs_arr
