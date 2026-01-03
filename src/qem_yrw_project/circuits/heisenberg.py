# src/qem_yrw_project/circuits/heisenberg.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate


@dataclass(frozen=True)
class HeisenbergParams:
    dt: float = 0.20
    Jx: float = 1.0
    Jy: float = 1.0
    Jz: float = 1.0
    hx: float = 0.0
    hy: float = 0.0
    hz: float = 0.0


def _brickwork_pairs(
    n_qubits: int,
    *,
    topology: Literal["line"] = "line",
    periodic: bool = False,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if topology != "line":
        raise ValueError(f"Unsupported topology={topology!r} (only 'line' for now).")

    even_pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_pairs = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]

    if periodic:
        # 2-sublayer brickwork on a ring only works without collisions if n is even.
        if n_qubits % 2 != 0:
            raise ValueError("periodic=True requires an even n_qubits for 2-layer brickwork.")
        odd_pairs = odd_pairs + [(n_qubits - 1, 0)]

    return even_pairs, odd_pairs


def build_heisenberg_trotter(
    n_qubits: int,
    depth: int,
    params: HeisenbergParams,
    *,
    topology: Literal["line"] = "line",
    periodic: bool = False,
    label: str | None = None,
    add_sample_barriers: bool = False,
    add_measurements: bool = False,
) -> QuantumCircuit:
    """
    Brickwork Trotter circuit for 1D Heisenberg-type Hamiltonian:

      H = Σ_<i,i+1> (Jx X_iX_{i+1} + Jy Y_iY_{i+1} + Jz Z_iZ_{i+1})
        + Σ_i (hx X_i + hy Y_i + hz Z_i)

    Gate conventions (Qiskit):
      RXX(theta) = exp(-i * theta/2 * X⊗X), theta = 2*Jx*dt
      similarly for RYY, RZZ; RX(theta)=exp(-i*theta/2*X), theta=2*hx*dt, etc.

    Notes:
      - If add_sample_barriers=True, we insert barrier(label="SAMPLE_OBSERVABLES") after each Trotter layer
        (YAQS can sample observables at these points).  :contentReference[oaicite:6]{index=6}
      - Measurements are optional; YAQS sampling ignores measure ops anyway. :contentReference[oaicite:7]{index=7}
    """
    if n_qubits < 2:
        raise ValueError("n_qubits must be >= 2")
    if depth < 1:
        raise ValueError("depth must be >= 1")

    qc = QuantumCircuit(n_qubits, name=label or "heisenberg_trotter")

    dt = float(params.dt)

    th_xx = 2.0 * float(params.Jx) * dt
    th_yy = 2.0 * float(params.Jy) * dt
    th_zz = 2.0 * float(params.Jz) * dt

    th_x = 2.0 * float(params.hx) * dt
    th_y = 2.0 * float(params.hy) * dt
    th_z = 2.0 * float(params.hz) * dt

    even_pairs, odd_pairs = _brickwork_pairs(n_qubits, topology=topology, periodic=periodic)

    for _layer in range(int(depth)):
        for bond_list in (even_pairs, odd_pairs):
            for (i, j) in bond_list:
                if th_xx != 0.0:
                    qc.append(RXXGate(th_xx), [i, j])
                if th_yy != 0.0:
                    qc.append(RYYGate(th_yy), [i, j])
                if th_zz != 0.0:
                    qc.append(RZZGate(th_zz), [i, j])

        for q in range(n_qubits):
            if th_x != 0.0:
                qc.rx(th_x, q)
            if th_y != 0.0:
                qc.ry(th_y, q)
            if th_z != 0.0:
                qc.rz(th_z, q)

        if add_sample_barriers:
            qc.barrier(label="SAMPLE_OBSERVABLES")

    if add_measurements:
        qc.measure_all()

    return qc
