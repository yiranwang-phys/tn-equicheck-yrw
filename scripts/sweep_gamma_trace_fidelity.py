# scripts/sweep_gamma_trace_fidelity.py

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from qiskit import qpy
from qiskit.quantum_info import Operator

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate
from qem_yrw_project.utils.circuit import strip_measurements  # 如果你没有这个，就用下面的 fallback

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs" / "figures" / "sweep_gamma_trace_fidelity"
IDEAL_QPY = REPO / "outputs" / "ideal" / "twolocal_n6_seed0" / "circuit_ideal.qpy"

N_QUBITS = 6

def load_qpy(path: Path):
    with path.open("rb") as f:
        return qpy.load(f)[0]

def strip_meas_fallback(circ):
    # 如果你项目里没有 strip_measurements，就用这个
    # （简单粗暴：过滤掉 measure 指令）
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(circ.num_qubits)
    for inst, qargs, cargs in circ.data:
        if inst.name == "measure":
            continue
        qc.append(inst, qargs, cargs)
    return qc

def trace_fidelity(U, V) -> float:
    # F = |Tr(U† V)| / 2^n  (paper’s trace-based check)
    d = 2 ** N_QUBITS
    return float(np.abs(np.trace(U.conj().T @ V)) / d)

def ensure_ideal_exists():
    if IDEAL_QPY.exists():
        return
    raise FileNotFoundError(
        f"Missing ideal reference: {IDEAL_QPY}\n"
        "Run: python scripts/make_ideal_twolocal_n6.py"
    )

def main():
    print(">>> sweep_gamma_trace_fidelity.py")
    ensure_ideal_exists()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ideal = load_qpy(IDEAL_QPY)
    try:
        ideal_u = strip_measurements(ideal)
    except Exception:
        ideal_u = strip_meas_fallback(ideal)

    U = Operator(ideal_u).data

    gammas = np.geomspace(0.8, 1e-3, 60)   # 0.8 -> 0.001, log-spaced, points=60
    seeds = list(range(10))                # 每个 gamma 平均 10 个 noisy 实例（你可以改大）

    meanF, stdF = [], []
    rows = []

    for g in gammas:
        Fs = []
        for s in seeds:
            noisy, stats = apply_pauli_jump_after_each_gate(ideal_u, gamma=float(g), seed=int(s))
            V = Operator(noisy).data
            F = trace_fidelity(U, V)
            Fs.append(F)
            rows.append((g, s, F, getattr(stats, "n_noise_ops", None)))
        meanF.append(float(np.mean(Fs)))
        stdF.append(float(np.std(Fs)))
        print(f"gamma={g:.6g}  F={meanF[-1]:.6g} ± {stdF[-1]:.2g}")

    # save csv
    csv_path = OUT_DIR / "trace_fidelity_vs_gamma.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("gamma,seed,trace_fidelity,n_noise_ops\n")
        for (g, s, F, nops) in rows:
            f.write(f"{g},{s},{F},{'' if nops is None else nops}\n")
    print("Saved:", csv_path)

    # plot
    plt.figure()
    plt.xscale("log")
    plt.errorbar(gammas, meanF, yerr=stdF, fmt="o", capsize=2)
    plt.xlabel("gamma (log scale)")
    plt.ylabel(r"Trace fidelity  $|Tr(U^\dagger V)|/2^n$")
    plt.title("Trace fidelity vs noise strength (Pauli-jump)")
    plt.tight_layout()

    fig_path = OUT_DIR / "trace_fidelity_vs_gamma.png"
    plt.savefig(fig_path, dpi=200)
    print("Saved:", fig_path)

if __name__ == "__main__":
    main()
