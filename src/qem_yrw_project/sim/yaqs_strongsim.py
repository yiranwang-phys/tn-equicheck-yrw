from __future__ import annotations

import io
import inspect
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, qpy


# -----------------------------
# Basic utils
# -----------------------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    # robust: only remove final measurements; leaves unitary body intact
    return circ.remove_final_measurements(inplace=False)


def qpy_bytes(circ: QuantumCircuit) -> bytes:
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    return buf.getvalue()


def qpy_load_bytes(b: bytes) -> QuantumCircuit:
    buf = io.BytesIO(b)
    return qpy.load(buf)[0]


def fidelity_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    amp = np.vdot(psi, phi)
    f = float(np.real(amp * np.conjugate(amp)))
    if f < 0.0:
        f = 0.0
    if f > 1.0 + 1e-9:
        f = 1.0
    return f


# -----------------------------
# CPU / worker helpers
# -----------------------------
def set_thread_env(blas_threads: int) -> None:
    n = str(int(max(1, blas_threads)))
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[k] = n


def try_set_high_priority_windows() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        HIGH_PRIORITY_CLASS = 0x00000080
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, HIGH_PRIORITY_CLASS)
    except Exception:
        pass


def worker_init(blas_threads: int) -> None:
    set_thread_env(int(blas_threads))
    try_set_high_priority_windows()


def shot_seed(seed_base: int, global_index: int) -> int:
    # deterministic paired seeds across sweeps
    return int((int(seed_base) + 1000003 * int(global_index)) % (2**31 - 1))


# -----------------------------
# YAQS StrongSim: MPS -> dense statevector
# -----------------------------
def mps_to_statevector(mps) -> np.ndarray:
    """
    Convert YAQS MPS (list of tensors) to dense statevector.
    Works for the "MPS of pure state" that StrongSim returns.
    """
    tensors = mps.tensors
    psi = np.asarray(tensors[0])
    # expected shape (2, chiL, chiR); for first site chiL=1
    psi = psi[:, 0, :]  # (2, chi)

    for t in tensors[1:]:
        t = np.asarray(t)  # (2, chi_prev, chi_next)
        psi = np.tensordot(psi, t, axes=([1], [1]))  # (2^k, 2, chi_next)
        psi = psi.reshape(psi.shape[0] * psi.shape[1], psi.shape[2])  # (2^(k+1), chi_next)

    # last bond dimension should be 1
    psi = psi[:, 0]
    return psi.astype(np.complex128, copy=False)


def _make_mps(n: int):
    from mqt.yaqs.core.data_structures.networks import MPS

    # Different YAQS versions differ slightly; try the common style first
    try:
        return MPS(length=int(n), state="zeros")
    except TypeError:
        return MPS(int(n))


def _make_strongsim_params(max_bond_dim: int, threshold: float):
    from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams

    sig = inspect.signature(StrongSimParams)
    params = sig.parameters
    kwargs = {}

    # max bond dimension name
    if "max_bond_dim" in params:
        kwargs["max_bond_dim"] = int(max_bond_dim)
    elif "maxBondDim" in params:
        kwargs["maxBondDim"] = int(max_bond_dim)

    # truncation / threshold name (many versions use 'threshold')
    if "threshold" in params:
        kwargs["threshold"] = float(threshold)
    elif "svd_cut" in params:
        kwargs["svd_cut"] = float(threshold)
    elif "svdCut" in params:
        kwargs["svdCut"] = float(threshold)

    # ask YAQS to keep the output state if supported
    if "get_state" in params:
        kwargs["get_state"] = True

    # some versions have num_traj, show_progress; keep them quiet
    if "num_traj" in params:
        kwargs["num_traj"] = 1
    if "show_progress" in params:
        kwargs["show_progress"] = False

    return StrongSimParams(**kwargs)


def run_strongsim_statevector(
    circ: QuantumCircuit,
    *,
    n: int,
    max_bond_dim: int,
    threshold: float,
    noise_model=None,
) -> np.ndarray:
    """
    Run YAQS strong simulation and return final statevector.

    IMPORTANT:
      - If you want Monte Carlo over stochastic noise, the most robust way is:
        sample a noisy circuit per trajectory (explicit Pauli gates), then call this with noise_model=None.
      - You *can* pass YAQS-native noise_model (NoiseModel), but whether it yields a pure-state MPS
        depends on your YAQS version / algorithm. If it returns a mixed object, mps_to_statevector
        will not apply.
    """
    from mqt.yaqs import simulator

    state = _make_mps(int(n))
    sim_params = _make_strongsim_params(int(max_bond_dim), float(threshold))

    # run signature differs by YAQS version; try both
    try:
        simulator.run(state, circ, sim_params, noise_model=noise_model, parallel=False)
    except TypeError:
        simulator.run(state, circ, sim_params, noise_model=noise_model)

    out_state = getattr(sim_params, "output_state", None)
    if out_state is None:
        out_state = state

    return mps_to_statevector(out_state)
