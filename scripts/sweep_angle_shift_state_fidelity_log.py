import argparse
import json
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

from qem_yrw_project.circuits.twolocal import build_twolocal
from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def circuit_no_measurements(circ):
    from qiskit import QuantumCircuit

    out = QuantumCircuit(circ.num_qubits)
    qmap = {qb: i for i, qb in enumerate(circ.qubits)}
    for ci in circ.data:
        op = ci.operation
        if op.name == "measure":
            continue
        qargs = [out.qubits[qmap[q]] for q in ci.qubits]
        out.append(op, qargs, [])
    return out


def shift_all_rotation_params(circ, delta: float):
    from qiskit import QuantumCircuit

    out = QuantumCircuit(circ.num_qubits)
    qmap = {qb: i for i, qb in enumerate(circ.qubits)}

    for ci in circ.data:
        op = ci.operation
        qargs = [out.qubits[qmap[q]] for q in ci.qubits]

        if hasattr(op, "params") and len(op.params) > 0:
            op2 = op.copy()
            new_params = []
            for p in op2.params:
                try:
                    new_params.append(float(p) + float(delta))
                except Exception:
                    new_params.append(p)
            op2.params = new_params
            out.append(op2, qargs, [])
        else:
            out.append(op, qargs, [])
    return out


def state_fidelity(psi_ref: Statevector, circ) -> float:
    psi = Statevector.from_instruction(circ)
    ov = np.vdot(psi_ref.data, psi.data)
    return float(np.abs(ov) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--delta-min", type=float, default=1e-6)
    ap.add_argument("--delta-max", type=float, default=2 * np.pi)
    ap.add_argument("--num", type=int, default=120)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--out-root", type=str, default="outputs/teacher_angle_shift")
    args = ap.parse_args()

    deltas_pos = np.logspace(np.log10(args.delta_min), np.log10(args.delta_max), args.num)
    deltas = np.concatenate(([0.0], deltas_pos))

    out_dir = _ensure_dir(os.path.join(args.out_root, f"twolocal_n{args.n}_seed{args.seed}", _timestamp()))
    print("OUTPUT DIR =", os.path.abspath(out_dir), flush=True)

    ideal = build_twolocal(n=args.n, reps=args.reps, seed=args.seed, add_measurements=True)
    ideal_u = circuit_no_measurements(ideal)
    psi_ref = Statevector.from_instruction(ideal_u)

    means = np.empty(len(deltas), dtype=float)
    stds = np.empty(len(deltas), dtype=float)

    for i, d in enumerate(deltas):
        shifted = shift_all_rotation_params(ideal_u, float(d))
        vals = np.empty(args.shots, dtype=float)
        for s in range(args.shots):
            noisy, _stats = apply_pauli_jump_after_each_gate(shifted, args.gamma, seed=args.seed * 10_000_000 + i * 1_000 + s)
            vals[s] = state_fidelity(psi_ref, noisy)
        means[i] = float(vals.mean())
        stds[i] = float(vals.std(ddof=1))
        if (i + 1) % max(1, len(deltas) // 20) == 0:
            print(f"delta sweep {i+1}/{len(deltas)}", flush=True)

    best_i = int(np.argmax(means))
    best_delta = float(deltas[best_i])
    best_mean = float(means[best_i])

    fig = plt.figure()
    plt.xscale("log")
    plt.xlabel("delta (log scale)")
    plt.ylabel("mean state fidelity")
    plt.title(f"Angle-shift sweep (log delta), gamma={args.gamma:g}, shots={args.shots}, n={args.n}")
    plt.plot(deltas_pos, means[1:])
    fig.savefig(os.path.join(out_dir, "angle_shift_fidelity_logx.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.xlabel("delta")
    plt.ylabel("mean state fidelity")
    plt.title(f"Angle-shift sweep (linear), gamma={args.gamma:g}, shots={args.shots}, n={args.n}")
    plt.plot(deltas, means)
    fig.savefig(os.path.join(out_dir, "angle_shift_fidelity_linear.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(
            {
                "meta": vars(args),
                "deltas": [float(x) for x in deltas],
                "mean_fidelity": [float(x) for x in means],
                "std_fidelity": [float(x) for x in stds],
                "best_delta": best_delta,
                "best_mean": best_mean,
            },
            f,
            indent=2,
        )

    print("BEST delta =", best_delta, "mean fidelity =", best_mean, flush=True)
    print("WROTE:", os.path.abspath(out_dir), flush=True)


if __name__ == "__main__":
    main()
