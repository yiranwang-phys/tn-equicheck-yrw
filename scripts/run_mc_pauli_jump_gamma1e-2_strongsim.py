from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from qiskit import qpy

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


GAMMA = 1e-2
N_TRAJ = 200
SEED0 = 0

MAX_BOND_DIM = 64
THRESHOLD = 1e-10

IDEAL_QPY = Path("outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")
OUT_BASE = Path("outputs/mc") / f"pauli_jump_gamma{GAMMA:.0e}" / "twolocal_n6_seed0"


def load_qpy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("rb") as f:
        return qpy.load(f)[0]


def save_circuit_png_or_txt(circ, out_dir: Path, stem: str):
    try:
        fig = circ.draw(output="mpl", fold=120)
        fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    except Exception as e:
        (out_dir / f"{stem}.txt").write_text(str(circ.draw(output="text")), encoding="utf-8")
        (out_dir / f"{stem}.draw_error.txt").write_text(str(e), encoding="utf-8")


def strongsim_all_Z(circ, num_qubits: int) -> np.ndarray:
    state = MPS(num_qubits, state="zeros")
    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        num_traj=1,
        max_bond_dim=MAX_BOND_DIM,
        threshold=THRESHOLD,
    )
    simulator.run(state, circ, sim_params, noise_model=None, parallel=False)
    return np.array([obs.results[0] for obs in sim_params.observables], dtype=float)


def main():
    print(">>> run_mc_pauli_jump_gamma1e-2_strongsim.py")

    ideal_meas = load_qpy(IDEAL_QPY)
    ideal = ideal_meas.remove_final_measurements(inplace=False)
    n = ideal.num_qubits

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "circuit_ideal_unitary.qpy").open("wb") as f:
        qpy.dump(ideal, f)
    save_circuit_png_or_txt(ideal, out_dir, "circuit_ideal_unitary")

    z_ideal = strongsim_all_Z(ideal, n)

    z_traj = np.zeros((N_TRAJ, n), dtype=float)
    noise_ops = np.zeros((N_TRAJ,), dtype=int)

    keep_examples = 5
    kept = 0

    for i in range(N_TRAJ):
        seed = SEED0 + i
        noisy, st = apply_pauli_jump_after_each_gate(ideal, GAMMA, seed, include_measurements=False)

        noise_ops[i] = st.n_noise_ops
        z_traj[i, :] = strongsim_all_Z(noisy, n)

        if kept < keep_examples:
            with (out_dir / f"circuit_noisy_seed{seed}.qpy").open("wb") as f:
                qpy.dump(noisy, f)
            save_circuit_png_or_txt(noisy, out_dir, f"circuit_noisy_seed{seed}")
            (out_dir / f"noise_stats_seed{seed}.txt").write_text(
                f"gamma={GAMMA}\nseed={seed}\n"
                f"n_noise_ops={st.n_noise_ops}\n"
                f"n_x={st.n_x}\nn_y={st.n_y}\nn_z={st.n_z}\n",
                encoding="utf-8",
            )
            kept += 1

        if (i + 1) % 25 == 0:
            print(f"  done {i+1}/{N_TRAJ}")

    z_mean = z_traj.mean(axis=0)
    z_std = z_traj.std(axis=0, ddof=1)
    z_stderr = z_std / np.sqrt(N_TRAJ)

    running = np.zeros((N_TRAJ,), dtype=float)
    for k in range(1, N_TRAJ + 1):
        z_k = z_traj[:k, :].mean(axis=0)
        running[k - 1] = float(np.abs(z_k - z_ideal).mean())

    np.savetxt(out_dir / "z_ideal.txt", z_ideal)
    np.savetxt(out_dir / "z_mc_mean.txt", z_mean)
    np.savetxt(out_dir / "z_mc_stderr.txt", z_stderr)
    np.savetxt(out_dir / "mc_running_mean_absdiff.txt", running)
    np.savetxt(out_dir / "noise_ops_per_traj.txt", noise_ops, fmt="%d")

    (out_dir / "meta.txt").write_text(
        "tag=MC_PAULI_JUMP_STRONGSIM\n"
        f"gamma={GAMMA}\n"
        f"N_TRAJ={N_TRAJ}\n"
        f"SEED0={SEED0}\n"
        f"MAX_BOND_DIM={MAX_BOND_DIM}\n"
        f"THRESHOLD={THRESHOLD}\n",
        encoding="utf-8",
    )

    plt.figure()
    plt.plot(np.arange(1, N_TRAJ + 1), running)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of trajectories (N)")
    plt.ylabel("Mean |<Z>_MC(N) - <Z>_ideal|")
    plt.title(f"MC convergence (gamma={GAMMA:.0e})")
    plt.tight_layout()
    plt.savefig(out_dir / "mc_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("OUTPUT DIR =", out_dir.resolve())


if __name__ == "__main__":
    main()
