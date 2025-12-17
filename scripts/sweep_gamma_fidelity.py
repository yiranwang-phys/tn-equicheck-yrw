from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.quantum_info import Statevector

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


N_TRAJ = 100
SEED0 = 0

GAMMAS = np.logspace(-4, -1, 13)  # 1e-4 ... 1e-1 (edit as you like)

IDEAL_QPY = Path("outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")
OUT_BASE = Path("outputs/fidelity_sweep") / "twolocal_n6_seed0"


def load_qpy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("rb") as f:
        return qpy.load(f)[0]


def state_fidelity_unitary(c1, c2) -> float:
    u1 = c1.remove_final_measurements(inplace=False)
    u2 = c2.remove_final_measurements(inplace=False)
    psi1 = Statevector.from_instruction(u1)
    psi2 = Statevector.from_instruction(u2)
    return float(np.abs(np.vdot(psi1.data, psi2.data)) ** 2)


def main():
    print(">>> sweep_gamma_fidelity.py")

    ideal_meas = load_qpy(IDEAL_QPY)
    ideal = ideal_meas.remove_final_measurements(inplace=False)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    means = np.zeros((len(GAMMAS),), dtype=float)
    stderrs = np.zeros((len(GAMMAS),), dtype=float)

    for gi, gamma in enumerate(GAMMAS):
        fid = np.zeros((N_TRAJ,), dtype=float)
        for i in range(N_TRAJ):
            seed = SEED0 + i
            noisy, _ = apply_pauli_jump_after_each_gate(ideal, float(gamma), seed, include_measurements=False)
            fid[i] = state_fidelity_unitary(ideal, noisy)

        means[gi] = fid.mean()
        stderrs[gi] = fid.std(ddof=1) / np.sqrt(N_TRAJ)
        print(f"  gamma={gamma:.1e}: mean={means[gi]:.6f}, stderr={stderrs[gi]:.3e}")

    np.savetxt(out_dir / "gammas.txt", GAMMAS)
    np.savetxt(out_dir / "fidelity_mean.txt", means)
    np.savetxt(out_dir / "fidelity_stderr.txt", stderrs)

    (out_dir / "meta.txt").write_text(
        "tag=FIDELITY_SWEEP\n"
        f"N_TRAJ={N_TRAJ}\nSEED0={SEED0}\n"
        f"GAMMAS={GAMMAS.tolist()}\n",
        encoding="utf-8",
    )

    plt.figure()
    plt.errorbar(GAMMAS, means, yerr=stderrs, fmt="o-")
    plt.xscale("log")
    plt.ylim(0.0, 1.01)
    plt.xlabel("gamma")
    plt.ylabel("Mean state fidelity")
    plt.title(f"Fidelity vs gamma (N={N_TRAJ})")
    plt.tight_layout()
    plt.savefig(out_dir / "fidelity_vs_gamma.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("OUTPUT DIR =", out_dir.resolve())


if __name__ == "__main__":
    main()
