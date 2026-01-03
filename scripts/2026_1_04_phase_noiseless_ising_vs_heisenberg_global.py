# scripts/2026_1_04_phase_noiseless_ising_vs_heisenberg_global.py
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate
from qiskit.quantum_info import Statevector, SparsePauliOp

# -----------------------------
# Repo + src import (src-layout)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Import project circuits
# -----------------------------
try:
    from qem_yrw_project.circuits.heisenberg import (
        build_heisenberg_trotter,
        HeisenbergParams,
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import qem_yrw_project.circuits.heisenberg. "
        "Make sure src/ is on PYTHONPATH and the file exists."
    ) from e

# Optional: import your ising builder if it exists; otherwise fallback.
try:
    from qem_yrw_project.circuits.ising import build_ising_trotter  # type: ignore
except Exception:
    build_ising_trotter = None  # type: ignore[assignment]


# -----------------------------
# Helpers: delta grid
# -----------------------------
def symsinh_grid(delta_min: float, delta_max: float, num: int, *, scale: float = 1.0) -> np.ndarray:
    """
    Log-like spacing around 0 while still covering [delta_min, delta_max].
    Uses a symmetric sinh mapping to cluster points near 0.
    """
    # map u in [-1,1] -> sinh(k*u)/sinh(k)
    u = np.linspace(-1.0, 1.0, int(num))
    k = float(scale)
    s = np.sinh(k * u) / np.sinh(k)
    # affine map to [min,max]
    return 0.5 * (delta_max + delta_min) + 0.5 * (delta_max - delta_min) * s


# -----------------------------
# Helpers: shift gates (GLOBAL)
# -----------------------------
@dataclass
class ShiftStats:
    total: int = 0
    by_name: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.by_name is None:
            self.by_name = {}


def shift_circuit_global_angles(qc: QuantumCircuit, delta: float) -> Tuple[QuantumCircuit, ShiftStats]:
    """
    Return a NEW circuit where we add `delta` to angles of supported rotation gates:
      rx, ry, rz, rxx, ryy, rzz
    Works directly on the native (untranspiled) circuit.
    """
    out = QuantumCircuit(qc.num_qubits, name=(qc.name or "qc") + f"_shift_{delta:+.6g}")
    stats = ShiftStats()

    for inst in qc.data:
        op = inst.operation
        qargs = inst.qubits
        cargs = inst.clbits
        name = op.name

        new_op = op

        # Single-qubit rotations
        if name == "rx" and len(op.params) == 1:
            new_op = RXGate(float(op.params[0]) + delta)
        elif name == "ry" and len(op.params) == 1:
            new_op = RYGate(float(op.params[0]) + delta)
        elif name == "rz" and len(op.params) == 1:
            new_op = RZGate(float(op.params[0]) + delta)

        # Two-qubit rotations (Heisenberg core!)
        elif name == "rxx" and len(op.params) == 1:
            new_op = RXXGate(float(op.params[0]) + delta)
        elif name == "ryy" and len(op.params) == 1:
            new_op = RYYGate(float(op.params[0]) + delta)
        elif name == "rzz" and len(op.params) == 1:
            new_op = RZZGate(float(op.params[0]) + delta)

        # Count shifts
        if new_op is not op:
            stats.total += 1
            stats.by_name[name] = stats.by_name.get(name, 0) + 1

        out.append(new_op, qargs, cargs)

    return out, stats


# -----------------------------
# Helpers: Statevector + fidelity + observable
# -----------------------------
def build_plus_prep(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="prep_plus")
    qc.h(range(n))
    return qc


def statevector_from_circuit(qc: QuantumCircuit) -> Statevector:
    sv0 = Statevector.from_label("0" * qc.num_qubits)
    return sv0.evolve(qc)


def fidelity(psi: Statevector, phi: Statevector) -> float:
    ov = np.vdot(psi.data, phi.data)
    return float(np.abs(ov) ** 2)


def zz_bond_avg_op(n: int) -> SparsePauliOp:
    """
    (1/(n-1)) sum_{i=0}^{n-2} Z_i Z_{i+1}
    """
    terms = []
    coeffs = []
    for i in range(n - 1):
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        label[n - 1 - (i + 1)] = "Z"
        terms.append("".join(label))
        coeffs.append(1.0 / (n - 1))
    return SparsePauliOp(terms, coeffs=np.array(coeffs, dtype=float))


def expval(state: Statevector, op: SparsePauliOp) -> float:
    val = state.expectation_value(op)
    return float(np.real_if_close(val))


# -----------------------------
# Ising fallback (if your project import missing)
# -----------------------------
def _fallback_build_ising_trotter(n: int, depth: int, *, dt: float = 0.2, Jz: float = 1.0, hx: float = 1.0) -> QuantumCircuit:
    """
    Very simple Ising-like trotter:
      layers of RZZ(2*Jz*dt) on even/odd bonds + RX(2*hx*dt) on all sites
    """
    qc = QuantumCircuit(n, name="ising_trotter_fallback")
    th_zz = 2.0 * Jz * dt
    th_x = 2.0 * hx * dt

    even_pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_pairs = [(i, i + 1) for i in range(1, n - 1, 2)]

    for _ in range(depth):
        for (i, j) in even_pairs:
            qc.append(RZZGate(th_zz), [i, j])
        for (i, j) in odd_pairs:
            qc.append(RZZGate(th_zz), [i, j])
        for q in range(n):
            qc.append(RXGate(th_x), [q])
    return qc


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--depth", type=int, default=5)

    # delta sweep
    ap.add_argument("--delta_min", type=float, default=-2.0 * math.pi)
    ap.add_argument("--delta_max", type=float, default=+2.0 * math.pi)
    ap.add_argument("--delta_points", type=int, default=401)
    ap.add_argument("--delta_symsinh_scale", type=float, default=3.0, help="bigger -> more points near 0")

    # plotting
    ap.add_argument("--xscale", type=str, default="symlog", choices=["symlog", "linear"])
    ap.add_argument("--linthresh", type=float, default=1e-3, help="symlog linear threshold")

    ap.add_argument("--outdir", type=str, default=str(REPO_ROOT / "outputs" / "2026_1_04_phase_noiseless"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = int(args.n)
    depth = int(args.depth)

    # Build circuits (NO MEASUREMENTS, NO TRANSPILATION)
    heis = build_heisenberg_trotter(
        n,
        depth,
        HeisenbergParams(),
        add_sample_barriers=False,
        add_measurements=False,
        label="heisenberg_trotter",
    )

    if build_ising_trotter is not None:
        ising = build_ising_trotter(n_qubits=n, depth=depth)  # type: ignore[misc]
        # If your build_ising_trotter adds measurements by default, strip them:
        if any(inst.operation.name == "measure" for inst in ising.data):
            ising = ising.remove_final_measurements(inplace=False)
    else:
        ising = _fallback_build_ising_trotter(n, depth)

    # delta grid (log-like)
    deltas = symsinh_grid(float(args.delta_min), float(args.delta_max), int(args.delta_points), scale=float(args.delta_symsinh_scale))

    # prep |+>^n
    prep = build_plus_prep(n)

    # reference states at delta=0
    heis0 = QuantumCircuit(n, name="heis0")
    heis0.compose(prep, inplace=True)
    heis0.compose(heis, inplace=True)
    psi_heis0 = statevector_from_circuit(heis0)

    ising0 = QuantumCircuit(n, name="ising0")
    ising0.compose(prep, inplace=True)
    ising0.compose(ising, inplace=True)
    psi_ising0 = statevector_from_circuit(ising0)

    # observable (same for both, easy to compare)
    op_zz = zz_bond_avg_op(n)
    ref_heis_zz = expval(psi_heis0, op_zz)
    ref_ising_zz = expval(psi_ising0, op_zz)

    # sweep
    heis_fids: List[float] = []
    ising_fids: List[float] = []
    heis_zzs: List[float] = []
    ising_zzs: List[float] = []

    # one-time gate-shift sanity check at a nonzero delta
    test_delta = float(deltas[len(deltas) // 3])
    _, heis_stats = shift_circuit_global_angles(heis, test_delta)
    _, ising_stats = shift_circuit_global_angles(ising, test_delta)

    print("[SANITY] shift gate counts (heisenberg):", heis_stats.by_name, "total=", heis_stats.total)
    print("[SANITY] shift gate counts (ising):", ising_stats.by_name, "total=", ising_stats.total)
    if heis_stats.total == 0:
        print("WARNING: heisenberg shift applied to 0 gates. Fidelity will likely be identically 1. Fix shift matcher.")
    if ising_stats.total == 0:
        print("WARNING: ising shift applied to 0 gates. Check your ising circuit gate names (rx/rzz expected).")

    for d in deltas:
        d = float(d)

        # heisenberg shifted
        heis_shifted, _ = shift_circuit_global_angles(heis, d)
        qc_h = QuantumCircuit(n, name=f"heis_d{d:+.6g}")
        qc_h.compose(prep, inplace=True)
        qc_h.compose(heis_shifted, inplace=True)
        psi_h = statevector_from_circuit(qc_h)

        heis_fids.append(fidelity(psi_h, psi_heis0))
        heis_zzs.append(expval(psi_h, op_zz))

        # ising shifted
        ising_shifted, _ = shift_circuit_global_angles(ising, d)
        qc_i = QuantumCircuit(n, name=f"ising_d{d:+.6g}")
        qc_i.compose(prep, inplace=True)
        qc_i.compose(ising_shifted, inplace=True)
        psi_i = statevector_from_circuit(qc_i)

        ising_fids.append(fidelity(psi_i, psi_ising0))
        ising_zzs.append(expval(psi_i, op_zz))

    # Save data
    payload = {
        "n": n,
        "depth": depth,
        "delta_min": float(args.delta_min),
        "delta_max": float(args.delta_max),
        "delta_points": int(args.delta_points),
        "delta_symsinh_scale": float(args.delta_symsinh_scale),
        "heisenberg": {
            "fidelity": heis_fids,
            "zz_bond_avg": heis_zzs,
            "zz_ref_delta0": ref_heis_zz,
        },
        "ising": {
            "fidelity": ising_fids,
            "zz_bond_avg": ising_zzs,
            "zz_ref_delta0": ref_ising_zz,
        },
    }
    (outdir / "phase_noiseless_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Plot 1: fidelity vs delta (noiseless)
    plt.figure()
    plt.plot(deltas, heis_fids, label="heisenberg fidelity(δ) vs δ=0")
    plt.plot(deltas, ising_fids, label="ising fidelity(δ) vs δ=0")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("delta (shift of gate angles)")
    plt.ylabel("fidelity (0..1)")
    plt.title("Noiseless phase response: fidelity vs delta (global shift)")
    plt.grid(True)
    if args.xscale == "symlog":
        plt.xscale("symlog", linthresh=float(args.linthresh))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "phase_noiseless_fidelity_vs_delta.png", dpi=200)
    plt.close()

    # Plot 2: ZZ bond avg vs delta (noiseless) + ref lines
    plt.figure()
    plt.plot(deltas, heis_zzs, label="heisenberg ZZ_bond_avg(δ)")
    plt.plot(deltas, ising_zzs, label="ising ZZ_bond_avg(δ)")
    plt.axhline(ref_heis_zz, linestyle="--", linewidth=1.5, label="heisenberg ref (δ=0)")
    plt.axhline(ref_ising_zz, linestyle="--", linewidth=1.5, label="ising ref (δ=0)")
    plt.xlabel("delta (shift of gate angles)")
    plt.ylabel("ZZ_bond_avg")
    plt.title("Noiseless observable vs delta (global shift)")
    plt.grid(True)
    if args.xscale == "symlog":
        plt.xscale("symlog", linthresh=float(args.linthresh))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "phase_noiseless_ZZ_bond_avg_vs_delta.png", dpi=200)
    plt.close()

    print("=== DONE 2026_1_04_phase_noiseless_ising_vs_heisenberg_global ===")
    print("outdir:", outdir)
    print("NOTE: If heisenberg fidelity is still identically 1, check the printed shift gate counts. "
          "You should see nonzero counts for rxx/ryy/rzz.")

if __name__ == "__main__":
    main()
