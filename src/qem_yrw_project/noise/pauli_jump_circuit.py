from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator
from qiskit import QuantumCircuit


# YAQS-style process spec:
#   {"name": "pauli_x", "sites": [i], "strength": p}
#   {"name": "crosstalk_xx", "sites": [i, i+1], "strength": p}
ProcessSpec = Dict[str, Any]


@dataclass
class PauliJumpStats:
    """
    Debug counters.

    n_gates: number of (non-measure/barrier/reset) gates traversed
    n_sites: total number of "site checks" (sum over gates of |touched qubits|)
    n_noise_events: how many noise events triggered (one process firing = one event)
    n_noise_ops: how many Pauli gates inserted total (XX counts as 2 ops)
    """
    n_gates: int = 0
    n_sites: int = 0
    n_noise_events: int = 0
    n_noise_ops: int = 0


def make_rng(*, seed: Optional[int] = None, rng: Optional[Generator] = None) -> Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(None if seed is None else int(seed))


def _map_qubits(c_in: QuantumCircuit, c_out: QuantumCircuit, qargs) -> List:
    return [c_out.qubits[c_in.find_bit(q).index] for q in qargs]


def _map_clbits(c_in: QuantumCircuit, c_out: QuantumCircuit, cargs) -> List:
    return [c_out.clbits[c_in.find_bit(c).index] for c in cargs]


def apply_pauli_jump_after_each_gate(
    circ: QuantumCircuit,
    gamma: float,
    *,
    seed: Optional[int] = None,
    rng: Optional[Generator] = None,
    skip_ops: Tuple[str, ...] = ("measure", "barrier", "reset"),
) -> Tuple[QuantumCircuit, PauliJumpStats]:
    """
   导师版本（最直接、最清楚）：
      - 对每个 gate 的每个被作用 qubit，独立地以概率 gamma 加一个随机 Pauli (X/Y/Z)
      - 只在 gate 之后加（不做 before+after）
    """
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError("gamma must be in [0, 1].")

    rng = make_rng(seed=seed, rng=rng)

    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    out.global_phase = getattr(circ, "global_phase", 0.0)

    stats = PauliJumpStats()

    for inst, qargs, cargs in circ.data:
        name = inst.name.lower()

        # copy the original instruction (mapped onto out circuit bits)
        nq = _map_qubits(circ, out, qargs)
        nc = _map_clbits(circ, out, cargs)
        out.append(inst, nq, nc)

        if name in skip_ops:
            continue

        stats.n_gates += 1

        # after-gate noise: per touched qubit
        for q in nq:
            stats.n_sites += 1
            if rng.random() < g:
                r = int(rng.integers(0, 3))
                if r == 0:
                    out.x(q); stats.n_noise_ops += 1
                elif r == 1:
                    out.y(q); stats.n_noise_ops += 1
                else:
                    out.z(q); stats.n_noise_ops += 1
                stats.n_noise_events += 1

    return out, stats


def apply_processes_after_each_gate(
    circ: QuantumCircuit,
    processes: Sequence[ProcessSpec],
    *,
    seed: Optional[int] = None,
    rng: Optional[Generator] = None,
    restrict_to_touched_sites: bool = True,
    skip_ops: Tuple[str, ...] = ("measure", "barrier", "reset"),
) -> Tuple[QuantumCircuit, PauliJumpStats]:
    """
   更“科研可扩展”的版本：你用 YAQS NoiseModel 的 process list 来描述噪声，
   然后我们在每个 gate 之后按 strength=probability 来触发相应的 Pauli 事件。

   支持的 name（小写）：
      - pauli_x / pauli_y / pauli_z
      - xx / yy / zz
      - crosstalk_xx  (作为 xx 的别名，兼容 YAQS 文档示例)
    """
    rng = make_rng(seed=seed, rng=rng)

    # normalize processes once
    norm: List[Tuple[str, Tuple[int, ...], float]] = []
    for p in processes:
        name = str(p.get("name", "")).strip().lower()
        sites_raw = p.get("sites", [])
        if not isinstance(sites_raw, (list, tuple)):
            raise TypeError("process['sites'] must be list/tuple of ints")
        sites = tuple(int(x) for x in sites_raw)
        strength = float(p.get("strength", 0.0))
        if strength < 0.0:
            raise ValueError("process strength must be non-negative")
        if name == "crosstalk_xx":
            name = "xx"
        norm.append((name, sites, strength))

    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    out.global_phase = getattr(circ, "global_phase", 0.0)

    stats = PauliJumpStats()

    for inst, qargs, cargs in circ.data:
        inst_name = inst.name.lower()

        nq = _map_qubits(circ, out, qargs)
        nc = _map_clbits(circ, out, cargs)
        out.append(inst, nq, nc)

        if inst_name in skip_ops:
            continue

        stats.n_gates += 1

        touched = {circ.find_bit(q).index for q in qargs}
        stats.n_sites += len(touched)

        def q_by_index(i: int):
            return out.qubits[int(i)]

        # After-gate: iterate all processes, fire with prob=strength
        for (name, sites, strength) in norm:
            if restrict_to_touched_sites and not set(sites).issubset(touched):
                continue

            if rng.random() >= strength:
                continue

            stats.n_noise_events += 1

            if name == "pauli_x":
                out.x(q_by_index(sites[0])); stats.n_noise_ops += 1
            elif name == "pauli_y":
                out.y(q_by_index(sites[0])); stats.n_noise_ops += 1
            elif name == "pauli_z":
                out.z(q_by_index(sites[0])); stats.n_noise_ops += 1
            elif name in ("xx", "yy", "zz"):
                if len(sites) != 2:
                    raise ValueError(f"{name} expects 2 sites, got {sites}")
                a, b = sites
                if name == "xx":
                    out.x(q_by_index(a)); out.x(q_by_index(b)); stats.n_noise_ops += 2
                elif name == "yy":
                    out.y(q_by_index(a)); out.y(q_by_index(b)); stats.n_noise_ops += 2
                else:
                    out.z(q_by_index(a)); out.z(q_by_index(b)); stats.n_noise_ops += 2
            else:
                raise ValueError(
                    f"Unknown process name '{name}'. Supported: pauli_x/pauli_y/pauli_z, xx/yy/zz "
                    f"(crosstalk_xx is accepted as alias for xx)."
                )

    return out, stats


__all__ = [
    "ProcessSpec",
    "PauliJumpStats",
    "make_rng",
    "apply_pauli_jump_after_each_gate",
    "apply_processes_after_each_gate",
]
