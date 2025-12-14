from __future__ import annotations
from .qiskit_backend import QiskitBackend
from .yaqs_backend import YaqsBackend

def make_backend(name: str):
    name = name.lower().strip()
    if name == "qiskit":
        return QiskitBackend()
    if name == "yaqs":
        return YaqsBackend()
    raise ValueError(f"Unknown backend: {name}")
