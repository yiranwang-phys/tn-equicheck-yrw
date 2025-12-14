from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from .base import RunMeta

class YaqsBackend:
    name = "yaqs"

    def ideal_expectation(self, circuit: Any, observable: Any):
        raise NotImplementedError("Implement via YAQS later.")

    def noisy_mc_expectations(
        self,
        circuit: Any,
        observable: Any,
        noise_cfg: Dict[str, Any],
        n_traj: int,
        seed: Optional[int] = None,
    ):
        raise NotImplementedError("Implement via YAQS later.")
