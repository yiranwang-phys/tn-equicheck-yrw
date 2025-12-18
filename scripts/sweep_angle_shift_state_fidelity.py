import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector, state_fidelity

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate

ROT_NAMES_DEFAULT = {
    "rx", "ry", "rz",
    "rxx", "ryy", "rzz", "rzx",
    "crx", "cry", "crz",
}

def ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def load_qpy(path: Path) -> QuantumCircuit:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("rb") as f:
        return qpy.load(f)[0]

def strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(circ.num_qubits)
    for ci in circ.data:
        op = ci.operation
        if op.name == "measure":
            continue
        out.append(op, ci.qubits, ci.clbits)
    return out

def shifted_circuit(circ: QuantumCircuit, delta: float, rot_names: set[str]) -> QuantumCircuit:
    out = QuantumCircuit(circ.num_qubits)
    for ci in circ.data:
        op = ci.operation
        if op.name in rot_names and hasattr(op, "params") and len(op.params) >= 1:
            new_op = op.copy()
            new_params = []
            for p in new_op.params:
                if isinstance(p, (int, float, np.floating)):
                    new_params.append(float(p) + float(delta))
                else:
                    new_params.append(p)
            new_op.params = new_params
            out.append(new_op, ci.qubits, ci.clbits)
        else:
            out.append(op, ci.qubits, ci.clbits)
    return out

def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=float) / float(w)
    ypad = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(ypad, k, mode="valid")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--delta-min", type=float, default=1e-6)
    ap.add_argument("--delta-max", type=float, default=2 * np.pi)
    ap.add_argument("--num-delta", type=int, default=120)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-inputs", type=int, default=8)
    ap.add_argument("--smooth-window", type=int, default=21)
    ap.add_argument("--ideal-qpy", type=str, default="outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")
    ap.add_argument("--rot-names", type=str, default=",".join(sorted(ROT_NAMES_DEFAULT)))
    ap.add_argument("--out-root", type=str, default="outputs/experiments/angle_shift_state_fidelity")
    return ap.parse_args()

def main():
    args = parse_args()

    ideal = strip_measurements(load_qpy(Path(args.ideal_qpy)))
    n = ideal.num_qubits
    rot_names = {s.strip() for s in args.rot_names.split(",") if s.strip()}

    deltas_pos = np.geomspace(args.delta_min, args.delta_max, args.num_delta)
    deltas = np.concatenate(([0.0], deltas_pos))

    rng = np.random.default_rng(args.seed)
    inputs = []
    for _ in range(max(1, args.num_inputs)):
        bits = rng.integers(0, 2, size=n, dtype=int)
        inputs.append("".join(str(b) for b in bits))

    psi_in_list = [Statevector.from_label(b) for b in inputs]
    psi_ideal_out = [psi_in.evolve(ideal) for psi_in in psi_in_list]

    outdir = Path(args.out_root) / f"twolocal_n{n}_seed{args.seed}" / f"gamma{args.gamma:.0e}" / ts()
    outdir.mkdir(parents=True, exist_ok=True)

    mean_fids = []
    std_fids = []
    mismatch_no_noise = []

    for i, d in enumerate(deltas):
        compiled = strip_measurements(shifted_circuit(ideal, float(d), rot_names))

        mf = []
        for psi_in, psi_ref in zip(psi_in_list, psi_ideal_out):
            psi_comp = psi_in.evolve(compiled)
            mf.append(state_fidelity(psi_ref, psi_comp))
        mismatch_no_noise.append(float(np.mean(mf)))

        shot_vals = []
        for s in range(args.shots):
            noisy, _stats = apply_pauli_jump_after_each_gate(
                compiled,
                float(args.gamma),
                seed=int(args.seed + 100000 * i + s),
                include_measurements=False,
            )
            noisy = strip_measurements(noisy)
            f_inp = []
            for psi_in, psi_ref in zip(psi_in_list, psi_ideal_out):
                psi_noisy = psi_in.evolve(noisy)
                f_inp.append(state_fidelity(psi_ref, psi_noisy))
            shot_vals.append(float(np.mean(f_inp)))

        shot_vals = np.asarray(shot_vals, dtype=float)
        mean_fids.append(float(np.mean(shot_vals)))
        std_fids.append(float(np.std(shot_vals)))

        if (i % max(1, len(deltas) // 10)) == 0 or i == len(deltas) - 1:
            print(f"progress {i+1}/{len(deltas)}  delta={d:.3e}  meanF={mean_fids[-1]:.6f}", flush=True)

    mean_fids = np.asarray(mean_fids, dtype=float)
    std_fids = np.asarray(std_fids, dtype=float)
    mismatch_no_noise = np.asarray(mismatch_no_noise, dtype=float)

    smooth = rolling_mean(mean_fids, args.smooth_window)
    base_mean = float(mean_fids[0])

    best_idx = int(np.argmax(mean_fids))
    best = {
        "delta": float(deltas[best_idx]),
        "mean_fidelity": float(mean_fids[best_idx]),
        "std_fidelity": float(std_fids[best_idx]),
        "mismatch_fidelity_no_noise": float(mismatch_no_noise[best_idx]),
    }

    payload = {
        "gamma": float(args.gamma),
        "n_qubits": int(n),
        "rot_names": sorted(list(rot_names)),
        "inputs": inputs,
        "deltas": [float(x) for x in deltas],
        "mean_fidelity": [float(x) for x in mean_fids],
        "std_fidelity": [float(x) for x in std_fids],
        "mismatch_fidelity_no_noise": [float(x) for x in mismatch_no_noise],
        "baseline_delta0": {
            "delta": 0.0,
            "mean_fidelity": float(mean_fids[0]),
            "std_fidelity": float(std_fids[0]),
            "mismatch_fidelity_no_noise": float(mismatch_no_noise[0]),
        },
        "best": best,
        "args": vars(args),
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    x = np.asarray(deltas, dtype=float)
    linthresh = max(args.delta_min, 1e-12)

    plt.figure()
    plt.title(f"Angle-shift state fidelity (points), gamma={args.gamma:.0e}, shots={args.shots}, inputs={len(inputs)}")
    plt.xscale("symlog", linthresh=linthresh)
    plt.plot(x, mean_fids, marker="o", linestyle="none", label="mean over shots")
    plt.axhline(base_mean, linestyle="--", label="baseline (no angle shift)")
    plt.xlabel("delta (rad, symlog)")
    plt.ylabel("state fidelity")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    p1 = outdir / "state_fidelity_vs_delta_points.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.title(f"Angle-shift state fidelity (smooth), gamma={args.gamma:.0e}, shots={args.shots}, inputs={len(inputs)}")
    plt.xscale("symlog", linthresh=linthresh)
    plt.plot(x, smooth, label=f"rolling mean (window={args.smooth_window})")
    plt.plot(x, mismatch_no_noise, linestyle="--", label="no-noise mismatch")
    plt.axhline(base_mean, linestyle=":", label="baseline (no angle shift)")
    plt.xlabel("delta (rad, symlog)")
    plt.ylabel("state fidelity")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    p2 = outdir / "state_fidelity_vs_delta_curve.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    print("WROTE:", str(outdir), flush=True)
    print("FIG1 :", str(p1), flush=True)
    print("FIG2 :", str(p2), flush=True)
    print("BASE :", json.dumps(payload["baseline_delta0"]), flush=True)
    print("BEST :", json.dumps(best), flush=True)

if __name__ == "__main__":
    main()
