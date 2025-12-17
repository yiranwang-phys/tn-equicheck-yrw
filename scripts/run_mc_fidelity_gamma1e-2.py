from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.quantum_info import Statevector

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


GAMMA = 1e-2
N_TRAJ = 200
SEED0 = 0

IDEAL_QPY = Path("outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")
OUT_BASE = Path("outputs/fidelity") / f"pauli_jump_gamma{GAMMA:.0e}" / "twolocal_n6_seed0"


def load_qpy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("rb") as f:
        return qpy.load(f)[0]


def state_fidelity_unitary(c1, c2) -> float:
    # Compare pure states from unitary circuits (no final measurements)
    u1 = c1.remove_final_measurements(inplace=False)
    u2 = c2.remove_final_measurements(inplace=False)
    psi1 = Statevector.from_instruction(u1)
    psi2 = Statevector.from_instruction(u2)
    return float(np.abs(np.vdot(psi1.data, psi2.data)) ** 2)


def main():
    print(">>> run_mc_fidelity_gamma1e-2.py")

    ideal_meas = load_qpy(IDEAL_QPY)
    ideal = ideal_meas.remove_final_measurements(inplace=False)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fidelities = np.zeros((N_TRAJ,), dtype=float)
    noise_ops = np.zeros((N_TRAJ,), dtype=int)

    for i in range(N_TRAJ):
        seed = SEED0 + i
        noisy, st = apply_pauli_jump_after_each_gate(ideal, GAMMA, seed, include_measurements=False)

        fidelities[i] = state_fidelity_unitary(ideal, noisy)
        noise_ops[i] = st.n_noise_ops

        if (i + 1) % 25 == 0:
            print(f"  done {i+1}/{N_TRAJ}")

    # Running mean + stderr (MC)
    running_mean = np.cumsum(fidelities) / np.arange(1, N_TRAJ + 1)
    running_var = np.array([fidelities[:k].var(ddof=1) if k > 1 else 0.0 for k in range(1, N_TRAJ + 1)])
    running_stderr = np.sqrt(running_var) / np.sqrt(np.arange(1, N_TRAJ + 1))

    np.savetxt(out_dir / "fidelity_per_traj.txt", fidelities)
    np.savetxt(out_dir / "noise_ops_per_traj.txt", noise_ops, fmt="%d")
    np.savetxt(out_dir / "fidelity_running_mean.txt", running_mean)
    np.savetxt(out_dir / "fidelity_running_stderr.txt", running_stderr)

    (out_dir / "meta.txt").write_text(
        "tag=MC_FIDELITY\n"
        f"gamma={GAMMA}\nN_TRAJ={N_TRAJ}\nSEED0={SEED0}\n",
        encoding="utf-8",
    )

    # Plot: running mean fidelity vs N
    plt.figure()
    plt.plot(np.arange(1, N_TRAJ + 1), running_mean)
    plt.xscale("log")
    plt.xlabel("Number of trajectories (N)")
    plt.ylabel("Running mean state fidelity")
    plt.title(f"MC fidelity convergence (gamma={GAMMA:.0e})")
    plt.tight_layout()
    plt.savefig(out_dir / "mc_fidelity_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Plot: histogram of fidelities
    plt.figure()
    plt.hist(fidelities, bins=30)
    plt.xlabel("State fidelity")
    plt.ylabel("Count")
    plt.title(f"Fidelity distribution (gamma={GAMMA:.0e}, N={N_TRAJ})")
    plt.tight_layout()
    plt.savefig(out_dir / "mc_fidelity_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("mean fidelity =", float(fidelities.mean()))
    print("stderr fidelity =", float(fidelities.std(ddof=1) / np.sqrt(N_TRAJ)))
    print("OUTPUT DIR =", out_dir.resolve())


if __name__ == "__main__":
    main()
