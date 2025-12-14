from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from qiskit.quantum_info import Statevector
from .base import RunMeta

class QiskitBackend:
    name = "qiskit"

    def ideal_expectation(self, circuit: Any, observable: Any):
        sv = Statevector.from_instruction(circuit)
        val = float(observable(sv))
        return val, RunMeta(backend=self.name, extra={})

    def noisy_mc_expectations(
        self,
        circuit: Any,
        observable: Any,
        noise_cfg: Dict[str, Any],
        n_traj: int,
        seed: Optional[int] = None,
    ):
        raise NotImplementedError("Implement later (Aer or your MC).")
