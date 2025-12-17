from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit


@dataclass(frozen=True)
class PauliJumpStats:
    gamma: float
    seed: int
    n_noise_ops: int


def apply_pauli_jump_after_each_gate(
    circuit: QuantumCircuit,
    gamma: float,
    seed: int,
    *,
    paulis: tuple[Literal["X", "Y", "Z"], ...] = ("X", "Y", "Z"),
    skip_ops: tuple[str, ...] = ("measure", "barrier"),
) -> tuple[QuantumCircuit, PauliJumpStats]:
    """
    Stochastic Pauli-jump model:
      - After each gate, for each involved qubit independently:
          with prob=gamma, append a random Pauli in `paulis`.
      - Skips operations with names in `skip_ops` (e.g., measure, barrier).
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1].")

    rng = np.random.default_rng(seed)

    noisy = QuantumCircuit(circuit.num_qubits, circuit.num_clbits, name=f"{circuit.name}_pauli_jump")
    noisy.global_phase = circuit.global_phase

    n_noise = 0

    for inst, qargs, cargs in circuit.data:
        # map bits by index (robust across registers)
        nq = [noisy.qubits[circuit.find_bit(q).index] for q in qargs]
        nc = [noisy.clbits[circuit.find_bit(c).index] for c in cargs]
        noisy.append(inst, nq, nc)

        if inst.name in skip_ops:
            continue

        # per-qubit independent jump after this gate
        for q in qargs:
            if rng.random() < gamma:
                qidx = circuit.find_bit(q).index
                p = paulis[int(rng.integers(len(paulis)))]
                if p == "X":
                    noisy.x(qidx)
                elif p == "Y":
                    noisy.y(qidx)
                elif p == "Z":
                    noisy.z(qidx)
                else:
                    raise RuntimeError("Invalid Pauli label.")
                n_noise += 1

    return noisy, PauliJumpStats(gamma=gamma, seed=seed, n_noise_ops=n_noise)
