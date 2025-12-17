from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
from qiskit import qpy

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


GAMMA = 1e-2
SEED_START = 0
SEED_TRIES = 500

IDEAL_QPY = Path("outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")
OUT_BASE = Path("outputs/noisy") / f"pauli_jump_gamma{GAMMA:.0e}" / "twolocal_n6_seed0"


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


def strongsim_all_Z(circ, n: int) -> np.ndarray:
    state = MPS(n, state="zeros")
    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(n)],
        num_traj=1,
        max_bond_dim=64,
        threshold=1e-10,
    )
    simulator.run(state, circ, sim_params, noise_model=None, parallel=False)
    return np.array([obs.results[0] for obs in sim_params.observables], dtype=float)


def main():
    print(">>> run_singletraj_pauli_jump_gamma1e-2_strongsim.py")

    ideal_meas = load_qpy(IDEAL_QPY)
    ideal = ideal_meas.remove_final_measurements(inplace=False)
    n = ideal.num_qubits

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    noisy = None
    stats = None
    seed_used = None

    for s in range(SEED_START, SEED_START + SEED_TRIES):
        cand, st = apply_pauli_jump_after_each_gate(ideal, GAMMA, s, include_measurements=False)
        if st.n_noise_ops > 0:
            noisy, stats, seed_used = cand, st, s
            break

    if noisy is None:
        raise RuntimeError("No noise inserted after many seeds (unexpected).")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_BASE / f"seed_{seed_used}" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save circuits
    with (out_dir / "circuit_ideal_unitary.qpy").open("wb") as f:
        qpy.dump(ideal, f)
    with (out_dir / "circuit_noisy_unitary.qpy").open("wb") as f:
        qpy.dump(noisy, f)

    # Save drawings
    save_circuit_png_or_txt(ideal, out_dir, "circuit_ideal_unitary")
    save_circuit_png_or_txt(noisy, out_dir, "circuit_noisy_unitary")

    # StrongSim compare
    z_ideal = strongsim_all_Z(ideal, n)
    z_noisy = strongsim_all_Z(noisy, n)
    diff = np.abs(z_noisy - z_ideal)

    np.savetxt(out_dir / "z_ideal.txt", z_ideal)
    np.savetxt(out_dir / "z_noisy.txt", z_noisy)
    np.savetxt(out_dir / "z_absdiff.txt", diff)

    (out_dir / "meta.txt").write_text(
        "tag=PAULI_JUMP_STRONGSIM_SINGLETRAJ\n"
        f"gamma={GAMMA}\n"
        f"seed_used={seed_used}\n"
        f"n_gates_seen={stats.n_gates_seen}\n"
        f"n_noise_ops={stats.n_noise_ops}\n",
        encoding="utf-8",
    )

    print("OUTPUT DIR =", out_dir.resolve())


if __name__ == "__main__":
    main()

