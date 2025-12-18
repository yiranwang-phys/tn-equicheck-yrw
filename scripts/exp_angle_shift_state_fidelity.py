import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qem_yrw_project.circuits.twolocal import build_twolocal
from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _sci(x: float) -> str:
    return f"{x:.0e}".replace("+", "")


def _pure_state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def _prep_basis_state(num_qubits: int, bitstring: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i, b in enumerate(reversed(bitstring)):
        if b == "1":
            qc.x(i)
    return qc


def _apply_angle_shift(circ: QuantumCircuit, delta: float) -> QuantumCircuit:
    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    for ci in circ.data:
        op = ci.operation
        q_idx = [circ.find_bit(q).index for q in ci.qubits]
        c_idx = [circ.find_bit(c).index for c in ci.clbits]
        if getattr(op, "params", None) and len(op.params) > 0:
            new_op = op.copy()
            new_op.params = [float(p) + float(delta) for p in op.params]
        else:
            new_op = op
        out.append(new_op, [out.qubits[i] for i in q_idx], [out.clbits[i] for i in c_idx])
    return out


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / float(w)
    sm = np.convolve(y, kernel, mode="same")
    return sm


@dataclass
class Result:
    num_qubits: int
    depth: int
    seed: int
    gamma: float
    shots: int
    num_inputs: int
    delta_min: float
    delta_max: float
    num_delta: int
    deltas: list
    mean_fidelity: list
    std_fidelity: list
    no_noise_mismatch: list
    baseline_no_shift: float
    best_delta: float
    best_mean_fidelity: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--num-inputs", type=int, default=8)
    ap.add_argument("--delta-min", type=float, default=1e-6)
    ap.add_argument("--delta-max", type=float, default=float(2.0 * np.pi))
    ap.add_argument("--num-delta", type=int, default=120)
    ap.add_argument("--smooth-window", type=int, default=21)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    ideal = build_twolocal(
        num_qubits=args.num_qubits,
        depth=args.depth,
        seed=args.seed,
        add_measurements=False,
    )

    inputs = []
    for _ in range(int(args.num_inputs)):
        bits = rng.integers(0, 2, size=args.num_qubits)
        bitstring = "".join(str(int(b)) for b in bits)
        inputs.append(bitstring)

    prep_circs = [_prep_basis_state(args.num_qubits, s) for s in inputs]
    ideal_states = []
    for prep in prep_circs:
        circ = prep.compose(ideal, inplace=False)
        st = Statevector.from_instruction(circ).data
        ideal_states.append(st)

    deltas = [0.0] + list(np.logspace(np.log10(args.delta_min), np.log10(args.delta_max), int(args.num_delta)))

    meanF = []
    stdF = []
    mismatch_no_noise = []

    for k, delta in enumerate(deltas, start=1):
        shifted = _apply_angle_shift(ideal, delta)

        mm = []
        for prep, psi_ideal in zip(prep_circs, ideal_states):
            st_shift = Statevector.from_instruction(prep.compose(shifted, inplace=False)).data
            mm.append(_pure_state_fidelity(psi_ideal, st_shift))
        mismatch_no_noise.append(float(np.mean(mm)))

        vals = []
        for _ in range(int(args.shots)):
            noisy_shifted, _stats = apply_pauli_jump_after_each_gate(
                shifted, gamma=float(args.gamma), seed=int(rng.integers(0, 2**31 - 1)), include_measurements=False
            )
            for prep, psi_ideal in zip(prep_circs, ideal_states):
                st_noisy = Statevector.from_instruction(prep.compose(noisy_shifted, inplace=False)).data
                vals.append(_pure_state_fidelity(psi_ideal, st_noisy))

        vals = np.array(vals, dtype=float)
        meanF.append(float(np.mean(vals)))
        stdF.append(float(np.std(vals)))

        if k == 1 or k % 20 == 0 or k == len(deltas):
            print(f"progress {k}/{len(deltas)}  delta={delta:.3e}  meanF={meanF[-1]:.6f}", flush=True)

    baseline = float(meanF[0])
    best_idx = int(np.argmax(meanF))
    best_delta = float(deltas[best_idx])
    best_mean = float(meanF[best_idx])

    out_dir = Path("outputs") / "experiments" / "angle_shift_state_fidelity" / f"twolocal_n{args.num_qubits}_seed{args.seed}" / f"gamma{_sci(args.gamma)}" / _ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    res = Result(
        num_qubits=int(args.num_qubits),
        depth=int(args.depth),
        seed=int(args.seed),
        gamma=float(args.gamma),
        shots=int(args.shots),
        num_inputs=int(args.num_inputs),
        delta_min=float(args.delta_min),
        delta_max=float(args.delta_max),
        num_delta=int(args.num_delta),
        deltas=[float(x) for x in deltas],
        mean_fidelity=[float(x) for x in meanF],
        std_fidelity=[float(x) for x in stdF],
        no_noise_mismatch=[float(x) for x in mismatch_no_noise],
        baseline_no_shift=baseline,
        best_delta=best_delta,
        best_mean_fidelity=best_mean,
    )
    (out_dir / "results.json").write_text(json.dumps(asdict(res), indent=2))

    x = np.array(deltas, dtype=float)
    y = np.array(meanF, dtype=float)
    y_sm = _rolling_mean(y, int(args.smooth_window))
    mm = np.array(mismatch_no_noise, dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=1.0)
    ax.axhline(baseline, linestyle=":", linewidth=2.0, label="baseline (no angle shift)")
    ax.plot(x, mm, linestyle="--", linewidth=2.0, label="no-noise mismatch")
    ax.set_xscale("symlog", linthresh=float(args.delta_min))
    ax.set_xlabel("delta (rad, symlog)")
    ax.set_ylabel("state fidelity")
    ax.set_title(f"Angle shift state fidelity (points), gamma={_sci(args.gamma)}, shots={args.shots}, inputs={args.num_inputs}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "state_fidelity_vs_delta_points.png", dpi=200)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y_sm, linewidth=2.0, label=f"rolling mean (window={int(args.smooth_window)})")
    ax.axhline(baseline, linestyle=":", linewidth=2.0, label="baseline (no angle shift)")
    ax.plot(x, mm, linestyle="--", linewidth=2.0, label="no-noise mismatch")
    ax.set_xscale("symlog", linthresh=float(args.delta_min))
    ax.set_xlabel("delta (rad, symlog)")
    ax.set_ylabel("state fidelity")
    ax.set_title(f"Angle shift state fidelity (smooth), gamma={_sci(args.gamma)}, shots={args.shots}, inputs={args.num_inputs}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "state_fidelity_vs_delta_curve.png", dpi=200)
    plt.close(fig)

    x_lin = np.linspace(0.0, float(args.delta_max), 401)
    y_lin = []
    for delta in x_lin:
        shifted = _apply_angle_shift(ideal, float(delta))
        vals = []
        for _ in range(int(args.shots)):
            noisy_shifted, _stats = apply_pauli_jump_after_each_gate(
                shifted, gamma=float(args.gamma), seed=int(rng.integers(0, 2**31 - 1)), include_measurements=False
            )
            for prep, psi_ideal in zip(prep_circs, ideal_states):
                st_noisy = Statevector.from_instruction(prep.compose(noisy_shifted, inplace=False)).data
                vals.append(_pure_state_fidelity(psi_ideal, st_noisy))
        y_lin.append(float(np.mean(vals)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_lin, np.array(y_lin, dtype=float), linewidth=2.0)
    ax.axhline(baseline, linestyle=":", linewidth=2.0, label="baseline (no angle shift)")
    ax.set_xlabel("delta (rad, linear)")
    ax.set_ylabel("state fidelity")
    ax.set_title(f"Angle shift state fidelity (linear sweep), gamma={_sci(args.gamma)}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "state_fidelity_vs_delta_linear.png", dpi=200)
    plt.close(fig)

    print(f"WROTE: {out_dir}", flush=True)
    print(f"FIG1 : {out_dir / 'state_fidelity_vs_delta_points.png'}", flush=True)
    print(f"FIG2 : {out_dir / 'state_fidelity_vs_delta_curve.png'}", flush=True)
    print(f"FIG3 : {out_dir / 'state_fidelity_vs_delta_linear.png'}", flush=True)
    print(f"BEST : delta={best_delta:.12g}, mean_fidelity={best_mean:.6f}, baseline={baseline:.6f}", flush=True)


if __name__ == "__main__":
    main()
