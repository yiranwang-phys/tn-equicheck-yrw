
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import inspect
from qiskit import qpy

from mqt.yaqs.digital.equivalence_checker import run as equiv_run
from qem_yrw_project.noise.pauli_jump import apply_pauli_jump_after_each_gate


GAMMA = 1e-2
SEED_START = 0
SEED_TRIES = 500

IDEAL_QPY = Path("outputs/ideal/twolocal_n6_seed0/circuit_ideal.qpy")
OUT_BASE = Path("outputs/equivcheck") / f"pauli_jump_gamma{GAMMA:.0e}" / "twolocal_n6_seed0"


def load_qpy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("rb") as f:
        return qpy.load(f)[0]


def save_circuit_png_or_txt(circ, out_dir: Path, stem: str):
    try:
        fig = circ.draw(output="mpl", fold=120)
        fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    except Exception as e:
        (out_dir / f"{stem}.txt").write_text(str(circ.draw(output="text")), encoding="utf-8")
        (out_dir / f"{stem}.draw_error.txt").write_text(str(e), encoding="utf-8")


def main():
    print(">>> yaqs_equivcheck_latest.py")

    ideal_meas = load_qpy(IDEAL_QPY)
    ideal = ideal_meas.remove_final_measurements(inplace=False)

    noisy = None
    stats = None
    seed_used = None

    for s in range(SEED_START, SEED_START + SEED_TRIES):
        cand, st = apply_pauli_jump_after_each_gate(ideal, GAMMA, s, include_measurements=False)
        if st.n_noise_ops > 0:
            noisy, stats, seed_used = cand, st, s
            break
    if noisy is None:
        raise RuntimeError("No noise inserted after many seeds (unexpected).")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_BASE / f"seed_{seed_used}" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    save_circuit_png_or_txt(ideal, out_dir, "circuit_ideal_unitary")
    save_circuit_png_or_txt(noisy, out_dir, "circuit_noisy_unitary")

    sig = str(inspect.signature(equiv_run))
    result = None
    err = None

    try:
        # most common usage
        result = equiv_run(ideal, noisy)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    report = []
    report.append("tag=YAQS_EQUIVCHECK\n")
    report.append(f"gamma={GAMMA}\nseed_used={seed_used}\n")
    report.append(f"n_gates_seen={stats.n_gates_seen}\n")
    report.append(f"n_noise_ops={stats.n_noise_ops}\n")
    report.append(f"equiv_run_signature={sig}\n")
    report.append(f"result={result}\n")
    if err is not None:
        report.append("\nERROR:\n" + err + "\n")

    (out_dir / "yaqs_equivcheck_report.txt").write_text("".join(report), encoding="utf-8")
    print("OUTPUT DIR =", out_dir.resolve())


if __name__ == "__main__":
    main()
