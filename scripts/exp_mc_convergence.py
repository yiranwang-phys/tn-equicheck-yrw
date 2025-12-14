import argparse
import json
import time
from pathlib import Path

import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    run_dir = Path("outputs") / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config_used.yaml").write_text(cfg_path.read_text())
    (run_dir / "status.json").write_text(json.dumps({"ok": True, "cfg": cfg}, indent=2))

    print(f"[OK] wrote: {run_dir}")

if __name__ == "__main__":
    main()
