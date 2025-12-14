from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any, Dict, Optional
import numpy as np

@dataclass(frozen=True)
class RunMeta:
    backend: str
    seed: Optional[int] = None
    extra: Dict[str, Any] | None = None

class Backend(Protocol):
    name: str

    def ideal_expectation(self, circuit: Any, observable: Any) -> tuple[float, RunMeta]:
        ...

    def noisy_mc_expectations(
        self,
        circuit: Any,
        observable: Any,
        noise_cfg: Dict[str, Any],
        n_traj: int,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, RunMeta]:
        ...
