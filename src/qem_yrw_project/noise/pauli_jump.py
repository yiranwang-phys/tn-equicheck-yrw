from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class PauliJumpStats:
    gamma: float
    seed: int
    n_gates_seen: int = 0
    n_noise_ops: int = 0


def apply_pauli_jump_after_each_gate(
    circuit: QuantumCircuit,
    gamma: float,
    seed: int,
    *,
    include_measurements: bool = False,
) -> Tuple[QuantumCircuit, PauliJumpStats]:
    """
    Insert stochastic Pauli X/Y/Z gates AFTER each gate.

    - gamma is the jump probability.
    - For each instruction (excluding barrier/measure/reset), for each involved qubit:
        with prob gamma apply a random Pauli {X,Y,Z} AFTER the gate.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0,1], got {gamma}")

    rng = np.random.default_rng(seed)
    stats = PauliJumpStats(gamma=gamma, seed=seed)

    # Keep registers if they exist
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs) if (circuit.qregs or circuit.cregs) else QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

    for inst, qargs, cargs in circuit.data:
        name = inst.name.lower()

        # StrongSim / EquiCheck usually wants unitary circuits
        if name == "measure" and (not include_measurements):
            continue

        out.append(inst, qargs, cargs)

        # Do not add noise after these
        if name in ("barrier", "measure", "reset"):
            continue

        stats.n_gates_seen += 1

        # Apply noise AFTER the gate on each involved qubit independently
        for q in qargs:
            if rng.random() < gamma:
                r = int(rng.integers(0, 3))
                if r == 0:
                    out.x(q)
                elif r == 1:
                    out.y(q)
                else:
                    out.z(q)
                stats.n_noise_ops += 1

    return out, stats
