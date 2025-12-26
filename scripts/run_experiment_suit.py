import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _make_env(workers: int, blas_threads: int) -> dict:
    env = os.environ.copy()
    env["TN_WORKERS"] = str(workers)

    # BLAS/OpenMP threads used by numpy/scipy/yaqs backend (if any)
    env["OMP_NUM_THREADS"] = str(blas_threads)
    env["MKL_NUM_THREADS"] = str(blas_threads)
    env["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    env["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    env["BLIS_NUM_THREADS"] = str(blas_threads)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(cmd, cwd: Path, env: dict, high_priority: bool = True):
    cmd = [str(x) for x in cmd]
    print("\n[RUN]", " ".join(cmd), flush=True)

    creationflags = 0
    if os.name == "nt" and high_priority:
        # Windows-only: run child process at high priority
        creationflags = subprocess.HIGH_PRIORITY_CLASS

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        creationflags=creationflags,
    )

    if p.stdout:
        print(p.stdout, end="", flush=True)

    if p.returncode != 0:
        if p.stderr:
            print("\n[STDERR]\n" + p.stderr, flush=True)
        raise SystemExit(p.returncode)

    if p.stderr:
        # keep warnings visible
        print("\n[STDERR]\n" + p.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--python", type=str, default="")
    ap.add_argument("--workers", type=int, default=0)        # 0 => all cores
    ap.add_argument("--blas-threads", type=int, default=1)   # usually 1 is best if child scripts use multiprocessing
    ap.add_argument("--no-high-priority", action="store_true")
    args = ap.parse_args()

    py = Path(args.python).expanduser() if args.python.strip() else Path(sys.executable)
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1

    env = _make_env(workers=workers, blas_threads=int(args.blas_threads))
    hp = (not args.no_high_priority)

    print(f"[INFO] workers={workers} blas_threads={args.blas_threads} high_priority={hp}", flush=True)
    print(f"[INFO] python={py}", flush=True)

    if args.clean:
        if outputs.exists():
            shutil.rmtree(outputs)
        outputs.mkdir(parents=True, exist_ok=True)

    _run([py, "-u", root / "scripts" / "make_ideal_twolocal_n6.py"], cwd=root, env=env, high_priority=hp)
    _run([py, "-u", root / "scripts" / "run_singletraj_pauli_jump_gamma1e-2_strongsim.py"], cwd=root, env=env, high_priority=hp)
    _run([py, "-u", root / "scripts" / "yaqs_equivcheck_latest.py"], cwd=root, env=env, high_priority=hp)

    _run([
        py, "-u", root / "scripts" / "sweep_gamma_trace_fidelity_dense.py",
        "--gamma-min", "1e-4", "--gamma-max", "1", "--num", "500",
        "--shots", "200", "--seed", "0",
        "--smooth-window", "31",
        "--mode", "proxy"
    ], cwd=root, env=env, high_priority=hp)

    _run([py, "-u", root / "scripts" / "exp_error_vs_gamma_log.py"], cwd=root, env=env, high_priority=hp)
    _run([py, "-u", root / "scripts" / "exp_angle_shift_state_fidelity.py"], cwd=root, env=env, high_priority=hp)
    _run([py, "-u", root / "scripts" / "exp_mc_convergence.py"], cwd=root, env=env, high_priority=hp)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
