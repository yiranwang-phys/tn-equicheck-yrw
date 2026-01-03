from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from qem_yrw_project.circuits.heisenberg import HeisenbergParams, build_heisenberg_trotter


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def symsinh_deltas(xmin: float, xmax: float, n: int, linthresh: float) -> np.ndarray:
    umin = np.arcsinh(xmin / linthresh)
    umax = np.arcsinh(xmax / linthresh)
    u = np.linspace(umin, umax, n, dtype=float)
    return linthresh * np.sinh(u)


def _qid(q) -> int:
    return int(getattr(q, "index", getattr(q, "_index")))


def apply_shift(qc: QuantumCircuit, delta: float, mode: str) -> Tuple[QuantumCircuit, int]:
    mode = mode.lower()
    delta = float(delta)

    def want_gate(name: str) -> bool:
        if mode == "global":
            return name in ("rx", "ry", "rz", "rxx", "ryy", "rzz")
        if mode == "xx":
            return name == "rxx"
        if mode == "yy":
            return name == "ryy"
        if mode == "zz":
            return name == "rzz"
        raise ValueError("mode must be global|xx|yy|zz")

    q2 = QuantumCircuit(qc.num_qubits, name=f"{qc.name}_shift_{mode}")
    n_shifted = 0

    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        name = inst.name.lower()

        if name in ("barrier", "measure"):
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
                q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], [])
        else:
            q2.append(inst, [q2.qubits[_qid(q)] for q in qargs], [])

    return q2, n_shifted


def fidelity(a: Statevector, b: Statevector) -> float:
    # |<a|b>|^2
    amp = np.vdot(a.data, b.data)
    return float(np.real(amp * np.conjugate(amp)))


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

    ap.add_argument("--shift_mode", type=str, default="global")

    ap.add_argument("--delta_min", type=float, default=-2.0 * math.pi)
    ap.add_argument("--delta_max", type=float, default=+2.0 * math.pi)
    ap.add_argument("--delta_points", type=int, default=401)
    ap.add_argument("--linthresh", type=float, default=0.05)

    ap.add_argument("--outdir", type=str, default="outputs/2026_1_04_fidelity_phase")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hp = HeisenbergParams(dt=args.dt, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz, hx=args.hx, hy=args.hy, hz=args.hz)
    base = build_heisenberg_trotter(args.n, args.depth, hp, periodic=False, label="heisenberg")

    psi0 = Statevector.from_instruction(base)

    deltas = symsinh_deltas(args.delta_min, args.delta_max, args.delta_points, args.linthresh)
    F = np.zeros_like(deltas)

    # sanity-check: count shifted gates for a nonzero delta
    qc_test, n_shifted_test = apply_shift(base, 0.123, args.shift_mode)

    for i, d in enumerate(deltas):
        qc, _ = apply_shift(base, float(d), args.shift_mode)
        psi = Statevector.from_instruction(qc)
        F[i] = fidelity(psi0, psi)

    meta = {
        "script": Path(__file__).name,
        "time": timestamp(),
        "args": vars(args),
        "shifted_gates_count_example": n_shifted_test,
        "minF": float(np.min(F)),
        "maxF": float(np.max(F)),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Plot (symlog x)
    plt.figure()
    plt.plot(deltas, F)
    plt.xscale("symlog", linthresh=args.linthresh)
    plt.xlabel("delta")
    plt.ylabel("fidelity")
    plt.ylim(-0.02, 1.02)
    plt.title(f"heisenberg | shift_mode={args.shift_mode} | fidelity (noiseless)")
    plt.grid(True)
    plt.savefig(outdir / "fidelity_vs_delta_symlog.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Also save linear-x plot (helps you看周期结构，不违背你“delta用log”的主图)
    plt.figure()
    plt.plot(deltas, F)
    plt.xlabel("delta")
    plt.ylabel("fidelity")
    plt.ylim(-0.02, 1.02)
    plt.title(f"heisenberg | shift_mode={args.shift_mode} | fidelity (noiseless) [linear x]")
    plt.grid(True)
    plt.savefig(outdir / "fidelity_vs_delta_linear.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save data
    np.save(outdir / "deltas.npy", deltas)
    np.save(outdir / "fidelity.npy", F)

    print("\n=== DONE 2026_1_04_fidelity_phase_sweep_statevector ===")
    print(f"outdir: {outdir.resolve()}")
    print(f"shifted_gates(example delta=0.123): {n_shifted_test}")
    print(f"minF={meta['minF']}  maxF={meta['maxF']}")


if __name__ == "__main__":
    main()
