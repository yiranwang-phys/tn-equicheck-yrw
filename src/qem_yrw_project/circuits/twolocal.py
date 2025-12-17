from __future__ import annotations

import numpy as np
from qiskit.circuit.library import TwoLocal


def build_twolocal(
    num_qubits: int = 6,
    depth: int | None = None,
    seed: int = 0,
    add_measurements: bool = True,
):
    """
    Build a TwoLocal circuit similar to YAQS docs example:
      - rotation blocks: rx
      - entanglement blocks: rzz
      - linear entanglement
      - reps = depth (default: num_qubits)
      - random parameters in [-pi, pi]
    """
    if depth is None:
        depth = num_qubits

    circuit = TwoLocal(
        num_qubits,
        rotation_blocks=["rx"],
        entanglement_blocks=["rzz"],
        entanglement="linear",
        reps=depth,
    ).decompose()

    rng = np.random.default_rng(seed)
    values = rng.uniform(-np.pi, np.pi, size=len(circuit.parameters))
    circuit.assign_parameters(values, inplace=True)

    if add_measurements:
        circuit.measure_all()

    return circuit
