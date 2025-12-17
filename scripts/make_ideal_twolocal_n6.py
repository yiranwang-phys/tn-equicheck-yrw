from __future__ import annotations

from pathlib import Path

from qiskit import qpy

from qem_yrw_project.circuits.twolocal import build_twolocal


def main():
    print(">>> building IDEAL TwoLocal circuit (n=6)")

    # --- config (ideal reference) ---
    num_qubits = 6
    depth = num_qubits
    seed = 0
    add_measurements = True

    # --- fixed output location for ideal reference ---
    out_dir = Path("outputs/ideal") / f"twolocal_n{num_qubits}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- build ideal circuit ---
    circuit_ideal = build_twolocal(
        num_qubits=num_qubits,
        depth=depth,
        seed=seed,
        add_measurements=add_measurements,
    )

    # --- save circuit (most robust: QPY) ---
    qpy_path = out_dir / "circuit_ideal.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(circuit_ideal, f)

    # --- try save qasm3 (optional, depends on qiskit version) ---
    try:
        from qiskit.qasm3 import dumps as qasm3_dumps  # type: ignore

        (out_dir / "circuit_ideal.qasm3").write_text(qasm3_dumps(circuit_ideal), encoding="utf-8")
        print("Saved: circuit_ideal.qasm3")
    except Exception as e:
        print(f"Skip qasm3 export: {e}")

    # --- save figure (mpl), fallback to text ---
    try:
        fig = circuit_ideal.draw(output="mpl", fold=120)
        fig.savefig(out_dir / "circuit_ideal.png", dpi=200, bbox_inches="tight")
        print("Saved: circuit_ideal.png")
    except Exception as e:
        (out_dir / "circuit_ideal.txt").write_text(str(circuit_ideal.draw(output="text")), encoding="utf-8")
        print(f"Skip mpl drawing, saved circuit_ideal.txt instead: {e}")

    # --- meta info ---
    (out_dir / "meta.txt").write_text(
        "tag=IDEAL_REFERENCE\n"
        f"num_qubits={num_qubits}\n"
        f"depth={depth}\n"
        f"seed={seed}\n"
        f"add_measurements={add_measurements}\n",
        encoding="utf-8",
    )

    print("Saved ideal reference to:", out_dir.resolve())
    print("QPY:", qpy_path.resolve())


if __name__ == "__main__":
    main()
