import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd):
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--python", type=str, default="")
    args = ap.parse_args()

    py = Path(args.python).expanduser() if args.python.strip() else Path(sys.executable)
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"

    if args.clean:
        if outputs.exists():
            shutil.rmtree(outputs)
        outputs.mkdir(parents=True, exist_ok=True)

    _run([py, "-u", root / "scripts" / "make_ideal_twolocal_n6.py"])
    _run([py, "-u", root / "scripts" / "run_singletraj_pauli_jump_gamma1e-2_strongsim.py"])
    _run([py, "-u", root / "scripts" / "yaqs_equivcheck_latest.py"])

    _run([
        py, "-u", root / "scripts" / "sweep_gamma_trace_fidelity_dense.py",
        "--gamma-min", "1e-4", "--gamma-max", "1", "--num", "500",
        "--shots", "200", "--seed", "0",
        "--smooth-window", "31",
        "--mode", "proxy"
    ])

    _run([py, "-u", root / "scripts" / "exp_error_vs_gamma_log.py"])
    _run([py, "-u", root / "scripts" / "exp_angle_shift_state_fidelity.py"])
    _run([py, "-u", root / "scripts" / "exp_mc_convergence.py"])

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
