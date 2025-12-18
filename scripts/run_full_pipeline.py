import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def _run(cmd):
    s = " ".join(str(x) for x in cmd)
    print(s, flush=True)
    subprocess.run([str(x) for x in cmd], check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--gamma_min", type=float, default=1e-4)
    ap.add_argument("--gamma_max", type=float, default=1.0)
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--smooth_window", type=int, default=31)
    ap.add_argument("--mode", type=str, default="proxy", choices=["proxy"])
    ap.add_argument("--tmax", type=int, default=10000)
    ap.add_argument("--gamma_list", type=str, default="1e-3,1e-2,1e-1")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / "outputs"

    if args.clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    _run([py, "-u", root / "scripts" / "make_ideal_twolocal_n6.py"])
    _run([py, "-u", root / "scripts" / "run_singletraj_pauli_jump_gamma1e-2_strongsim.py"])
    _run([py, "-u", root / "scripts" / "yaqs_equivcheck_latest.py"])

    _run([
        py, "-u", root / "scripts" / "sweep_gamma_trace_fidelity_dense.py",
        "--gamma-min", str(args.gamma_min),
        "--gamma-max", str(args.gamma_max),
        "--num", str(args.num),
        "--shots", str(args.shots),
        "--seed", str(args.seed),
        "--smooth-window", str(args.smooth_window),
        "--mode", args.mode
    ])

    if args.teacher:
        _run([
            py, "-u", root / "scripts" / "exp_error_vs_gamma_log.py",
            "--shots", str(args.shots),
            "--seed", str(args.seed)
        ])
        _run([
            py, "-u", root / "scripts" / "sweep_angle_shift_state_fidelity_log.py",
            "--gamma", "1e-2",
            "--delta-min", "1e-6",
            "--delta-max", "6.283185307179586",
            "--num", "120",
            "--shots", str(args.shots),
            "--seed", str(args.seed)
        ])
        _run([
            py, "-u", root / "scripts" / "exp_mc_convergence.py",
            "--tmax", str(args.tmax),
            "--gamma-list", args.gamma_list,
            "--n-batches", "50",
            "--shots", str(args.shots),
            "--seed", str(args.seed)
        ])

    print("DONE. Check outputs/.", flush=True)

if __name__ == "__main__":
    main()
