#!/usr/bin/env python3
"""Run the synthetic pilot: scenarios x policies x seeds -> results.jsonl."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experience_learning.evaluation import run_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "pilot")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    path = run_experiment(cfg, args.out)
    n = sum(1 for _ in path.open())
    print(f"wrote {n} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
