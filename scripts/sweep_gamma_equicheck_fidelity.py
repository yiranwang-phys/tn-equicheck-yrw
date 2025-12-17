from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import inspect

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.quantum_info import Operator

from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


IDEAL_QPY = Path("outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")

GAMMA_MAX = 8e-1
GAMMA_MIN = 1e-3
N_GAMMAS = 60

SEED = 0  # fixed seed per gamma for now (deterministic)
YAQS_DO_CHECK = True

OUT_BASE = Path("outputs/sweeps") / "gamma_equicheck_tracefidelity" / "twolocal_n6_seed0"


def load_qpy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run: python scripts/make_ideal_twolocal_n6.py")
    with path.open("rb") as f:
        return qpy.load(f)[0]


def strip_measurements(circ):
    return circ.remove_final_measurements(inplace=False)


def save_circuit_png_or_txt(circ, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig = circ.draw(output="mpl", fold=120)
        fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (out_dir / f"{stem}.txt").write_text(str(circ.draw(output="text")), encoding="utf-8")
        (out_dir / f"{stem}.err.txt").write_text(str(e), encoding="utf-8")


def trace_fidelity_unitaries(ideal, noisy) -> float:
    """Paper-style: |Tr(G * G'^†)| / 2^n  (global phase insensitive via abs)"""
    u_ideal = Operator(ideal).data
    u_noisy = Operator(noisy).data
    d = u_ideal.shape[0]
    val = np.trace(u_ideal @ u_noisy.conj().T) / d
    return float(np.abs(val))


def yaqs_equivcheck_bool(ideal, noisy) -> bool | None:
    """
    Calls mqt.yaqs.digital.equivalence_checker.run in a signature-tolerant way.
    Returns True/False if available, else None.
    """
    try:
        from mqt.yaqs.digital.equivalence_checker import run as yaqs_run
    except Exception:
        return None

    sig = inspect.signature(yaqs_run)
    kwargs = {}
    # pass only supported kwargs (keep minimal)
    for k in ["parallel", "max_bond_dim", "threshold", "eps", "epsilon", "tol", "tolerance"]:
        if k in sig.parameters:
            if k in {"parallel"}:
                kwargs[k] = False
            elif k in {"max_bond_dim"}:
                kwargs[k] = 64
            elif k in {"threshold"}:
                kwargs[k] = 1e-10
            else:
                kwargs[k] = 1e-6

    try:
        res = yaqs_run(ideal, noisy, **kwargs)
    except TypeError:
        res = yaqs_run(ideal, noisy)

    # normalize return
    if isinstance(res, bool):
        return res
    if isinstance(res, dict):
        for key in ["equivalent", "is_equivalent", "result"]:
            if key in res and isinstance(res[key], bool):
                return res[key]
    if isinstance(res, tuple) or isinstance(res, list):
        for x in res:
            if isinstance(x, bool):
                return x
    return None


def main():
    print(">>> sweep_gamma_equicheck_fidelity.py")

    ideal_meas = load_qpy(IDEAL_QPY)
    ideal = strip_measurements(ideal_meas)
    n = ideal.num_qubits

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save ideal circuit once
    save_circuit_png_or_txt(ideal, out_dir, "circuit_ideal_unitary")

    gammas = np.logspace(np.log10(GAMMA_MAX), np.log10(GAMMA_MIN), N_GAMMAS)
    rows_fid = []
    rows_bool = []

    # Save only a few representative noisy circuit drawings
    save_idx = {0, len(gammas)//2, len(gammas)-1}

    for i, g in enumerate(gammas):
        noisy, st = apply_pauli_jump_after_each_gate(ideal, float(g), SEED)
        fid = trace_fidelity_unitaries(ideal, noisy)

        eq_bool = None
        if YAQS_DO_CHECK:
            eq_bool = yaqs_equivcheck_bool(ideal, noisy)

        rows_fid.append((float(g), fid, st.n_noise_ops))
        rows_bool.append((float(g), eq_bool))

        if i in save_idx:
            save_circuit_png_or_txt(noisy, out_dir, f"circuit_noisy_gamma_{g:.2e}")

        print(f"gamma={g:.3e}  fid={fid:.6f}  n_noise={st.n_noise_ops}  yaqs_eq={eq_bool}")

    # Write CSVs
    with (out_dir / "gamma_fidelity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "trace_fidelity_absTr_over_2n", "n_noise_ops"])
        w.writerows(rows_fid)

    with (out_dir / "yaqs_equivcheck_bool.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "yaqs_equivalent_bool_or_None"])
        w.writerows(rows_bool)

    # Plot
    gam = np.array([r[0] for r in rows_fid], dtype=float)
    fid = np.array([r[1] for r in rows_fid], dtype=float)

    plt.figure()
    plt.plot(gam, fid)
    plt.xscale("log")
    plt.ylim(0.0, 1.02)
    plt.xlabel("gamma (log scale)")
    plt.ylabel(r"trace fidelity  $|\mathrm{Tr}(G G'^\dagger)|/2^n$")
    plt.title(f"TwoLocal n={n}, seed={SEED}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "fidelity_vs_gamma.png", dpi=200)
    plt.close()

    # Report which gammas are False (if YAQS returned bools)
    false_gammas = [g for g, b in rows_bool if b is False]
    (out_dir / "yaqs_false_gammas.txt").write_text(
        "\n".join([f"{x:.8e}" for x in false_gammas]) if false_gammas else "No False values (or YAQS returned None).",
        encoding="utf-8",
    )

    print("OUTPUT DIR =", out_dir.resolve())
    print("Saved: fidelity_vs_gamma.png, gamma_fidelity.csv, yaqs_equivcheck_bool.csv")


if __name__ == "__main__":
    main()
