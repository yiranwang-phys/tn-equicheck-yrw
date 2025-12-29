from __future__ import annotations

from pathlib import Path
from datetime import datetime

from qiskit import qpy

from qem_yrw_project.circuits.twolocal import build_twolocal


def main() -> None:
    out_dir = Path("outputs/checks/twolocal") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 你可以改这些参数做不同检查
    num_qubits = 6
    depth = 6
    seed = 0

    qc = build_twolocal(num_qubits=num_qubits, depth=depth, seed=seed, add_measurements=False)

    # 1) 文本电路图（最稳，永远能输出）
    txt = qc.draw(output="text").single_string()
    (out_dir / "circuit.txt").write_text(txt, encoding="utf-8")

    # 2) PNG 电路图（依赖 matplotlib）
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        fig = qc.draw(output="mpl")
        fig.savefig(out_dir / "circuit.png", dpi=200, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        (out_dir / "circuit_png_error.txt").write_text(str(e), encoding="utf-8")

    # 3) OpenQASM（便于人读/ diff）
    try:
        (out_dir / "circuit.qasm").write_text(qc.qasm(), encoding="utf-8")
    except Exception as e:
        (out_dir / "circuit_qasm_error.txt").write_text(str(e), encoding="utf-8")

    # 4) QPY（最稳的 Qiskit 原生序列化格式，推荐用于复现）
    with open(out_dir / "circuit.qpy", "wb") as f:
        qpy.dump(qc, f)

    # 记录参数，避免以后看不懂
    meta = (
        f"num_qubits={num_qubits}\n"
        f"depth={depth}\n"
        f"seed={seed}\n"
        f"num_params={qc.num_parameters}\n"
        f"num_ops={len(qc.data)}\n"
    )
    (out_dir / "meta.txt").write_text(meta, encoding="utf-8")

    print(f"WROTE: {out_dir}")


if __name__ == "__main__":
    main()
