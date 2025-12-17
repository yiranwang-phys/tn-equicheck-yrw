import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy

from mqt.yaqs.digital.equivalence_checker import run as yaqs_equiv_run

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


def load_qpy(path: Path):
    with path.open("rb") as f:
        return qpy.load(f)[0]


def strip_measurements(circ):
    # 只保留 unitary 部分（等价检查默认针对 unitary）
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(circ.num_qubits)
    for inst, qargs, cargs in circ.data:
        if inst.name == "measure":
            continue
        qc.append(inst, qargs, [])
    return qc


def main():
    repo = Path(__file__).resolve().parents[1]
    ideal_dir = repo / "outputs" / "ideal" / "twolocal_n6_seed0"
    ideal_qpy = ideal_dir / "circuit_ideal.qpy"
    ideal = strip_measurements(load_qpy(ideal_qpy))

    # gamma: 0.001 -> 0.8
    gammas = np.logspace(np.log10(1e-3), np.log10(8e-1), 16)

    n_traj = 30
    eps = 1e-13  # 论文同量级的 trace-fidelity tolerance :contentReference[oaicite:3]{index=3}
    seed0 = 0

    out_dir = repo / "outputs" / "figures" / "sweep_gamma_fidelity" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_F = []
    std_F = []

    for i, g in enumerate(gammas):
        Fs = []
        for t in range(n_traj):
            noisy, meta = apply_pauli_jump_after_each_gate(ideal, gamma=float(g), seed=seed0 + 1000*i + t)

            # YAQS EquiCheck: 同时会算出内部 trace-based fidelity（你也可以在 apply/meta 里自己算）
            # 这里假设你在 yaqs_equivcheck_latest.py 里已经能拿到 fidelity（你截图里 report 有 fidelity 字段）
            # 若你的 yaqs_equiv_run 只返回 equivalent，你就用你脚本里同样的“report写法”把 fidelity 拿出来即可。
            equiv, fidelity = yaqs_equiv_run(ideal, noisy, threshold=eps)  # 若 API 不是这样，照你当前可工作的 wrapper 调整
            Fs.append(float(fidelity))

        mean_F.append(float(np.mean(Fs)))
        std_F.append(float(np.std(Fs)))

    # plot
    plt.figure()
    plt.xscale("log")
    plt.errorbar(gammas, mean_F, yerr=std_F, fmt="o-")
    plt.xlabel("gamma")
    plt.ylabel("trace fidelity (mean ± std)")
    plt.title(f"EquiCheck trace fidelity vs gamma (n_traj={n_traj})")
    fig_path = out_dir / "fidelity_vs_gamma.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")

    (out_dir / "data.json").write_text(json.dumps({
        "gammas": gammas.tolist(),
        "mean_fidelity": mean_F,
        "std_fidelity": std_F,
        "n_traj": n_traj,
        "eps": eps,
    }, indent=2), encoding="utf-8")

    print("Saved:", fig_path)
    print("Dir:", out_dir)


if __name__ == "__main__":
    main()
