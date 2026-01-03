from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

IGNORE_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints",
}

INCLUDE_EXTS = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt"}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def pack_tree(base_dir: Path, output_file: Path) -> None:
    base_dir = base_dir.resolve()
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    files = []
    for p in base_dir.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(base_dir).parts
        if any(part in IGNORE_DIRS for part in rel_parts[:-1]):
            continue
        if p.suffix.lower() not in INCLUDE_EXTS:
            continue
        if p.resolve() == output_file:
            continue
        files.append(p)

    files.sort(key=lambda x: x.relative_to(base_dir).as_posix().lower())

    with output_file.open("w", encoding="utf-8") as out:
        out.write(f"PACK_BASE: {base_dir.as_posix()}\n")
        out.write(f"PACK_TIME: {_ts()}\n")
        out.write(f"FILE_COUNT: {len(files)}\n")
        out.write("=" * 80 + "\n\n")

        for p in files:
            rel = p.relative_to(base_dir).as_posix()
            out.write("\n" + "=" * 30 + "\n")
            out.write(f"FILE: {rel}\n")
            out.write("=" * 30 + "\n\n")
            try:
                out.write(p.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                out.write(p.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                out.write(f"[ERROR reading file] {e}\n")
            out.write("\n")

    print(f"Packed {len(files)} files into: {output_file}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="src", help="Src directory (default: src)")
    ap.add_argument("--out", type=str, default="", help="Output txt path (default: dist/src_context_<ts>.txt)")
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_path = Path(args.out) if args.out else Path("dist") / f"src_context_{_ts()}.txt"
    pack_tree(src_dir, out_path)


if __name__ == "__main__":
    main()
