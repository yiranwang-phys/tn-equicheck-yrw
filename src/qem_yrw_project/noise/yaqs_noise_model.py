from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

ProcessSpec = Dict[str, Any]


@dataclass
class YaQsNoiseMeta:
    num_qubits: int
    gamma: float
    per_pauli_strength: float
    include_xx_crosstalk: bool
    xx_strength: float
    two_qubit_name_used: str


def yaqs_example_processes(num_qubits: int, noise_factor: float) -> List[ProcessSpec]:
    """
    Mirror YAQS docs example:
      pauli_x on each site + crosstalk_xx on neighbors
    """
    n = int(num_qubits)
    nf = float(noise_factor)
    procs: List[ProcessSpec] = []
    procs += [{"name": "pauli_x", "sites": [i], "strength": nf} for i in range(n)]
    procs += [{"name": "crosstalk_xx", "sites": [i, i + 1], "strength": nf} for i in range(n - 1)]
    return procs


def depolarizing_xyz_processes(
    num_qubits: int,
    gamma: float,
    *,
    include_xx_crosstalk: bool = False,
    xx_strength: Optional[float] = None,
) -> List[ProcessSpec]:
    """
    Depolarizing-style (discrete-time) process list:
      X_i, Y_i, Z_i each with strength gamma/3.

    NOTE:
      - Here "strength" is used as a probability per gate-step in our circuit-sampling model.
      - In YAQS native NoiseModel, strength can be interpreted as rate/strength depending on algorithm.
    """
    n = int(num_qubits)
    g = float(gamma)
    if not math.isfinite(g) or g < 0.0:
        raise ValueError("gamma must be a finite non-negative float")

    per = g / 3.0
    procs: List[ProcessSpec] = []
    for i in range(n):
        procs.append({"name": "pauli_x", "sites": [i], "strength": per})
        procs.append({"name": "pauli_y", "sites": [i], "strength": per})
        procs.append({"name": "pauli_z", "sites": [i], "strength": per})

    if include_xx_crosstalk:
        s = float(per if xx_strength is None else xx_strength)
        for i in range(n - 1):
            procs.append({"name": "crosstalk_xx", "sites": [i, i + 1], "strength": s})

    return procs


def _pick_supported_two_qubit_name(preferred: str = "crosstalk_xx") -> str:
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel  # lazy import

    candidates = [preferred, "xx", "yy", "zz"]
    for name in candidates:
        try:
            _ = NoiseModel([{"name": name, "sites": [0, 1], "strength": 1e-12}])
            return name
        except Exception:
            continue
    return "xx"


def build_yaqs_noise_model(processes: Sequence[ProcessSpec]):
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel

    # YAQS expects list[dict]
    return NoiseModel(list(processes))


def build_depolarizing_xyz_noise_model(
    num_qubits: int,
    gamma: float,
    *,
    include_xx_crosstalk: bool = False,
    xx_strength: Optional[float] = None,
    two_qubit_name: str = "crosstalk_xx",
) -> Tuple[Any, YaQsNoiseMeta]:
    """
    Convenience: returns (NoiseModel, meta) for depolarizing XYZ (+ optional XX).
    Also auto-picks a supported 2-qubit name to reduce YAQS-version headaches.
    """
    n = int(num_qubits)
    g = float(gamma)
    per = g / 3.0

    procs = depolarizing_xyz_processes(
        n,
        g,
        include_xx_crosstalk=include_xx_crosstalk,
        xx_strength=xx_strength,
    )

    used = ""
    if include_xx_crosstalk:
        chosen = _pick_supported_two_qubit_name(preferred=str(two_qubit_name))
        for p in procs:
            if str(p.get("name", "")).strip().lower() == "crosstalk_xx":
                p["name"] = chosen
        used = chosen

    nm = build_yaqs_noise_model(procs)

    meta = YaQsNoiseMeta(
        num_qubits=n,
        gamma=g,
        per_pauli_strength=per,
        include_xx_crosstalk=bool(include_xx_crosstalk),
        xx_strength=float(per if xx_strength is None else xx_strength),
        two_qubit_name_used=used,
    )
    return nm, meta


__all__ = [
    "ProcessSpec",
    "YaQsNoiseMeta",
    "yaqs_example_processes",
    "depolarizing_xyz_processes",
    "build_yaqs_noise_model",
    "build_depolarizing_xyz_noise_model",
]
