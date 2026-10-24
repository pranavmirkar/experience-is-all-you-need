#!/usr/bin/env python3
"""Summarise results.jsonl: per-policy mean/std and paired differences vs baselines."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLS = [
    ("final_test_pass_rate", "final test"),
    ("heldout_family_final_test", "heldout fam"),
    ("average_forgetting", "forgetting"),
    ("average_accuracy", "avg acc"),
    ("forward_transfer", "fwd transfer"),
    ("regression_rate", "regression"),
    ("update_tokens", "update tokens"),
]
PAIRED_AGAINST = ["uniform", "loss", "novelty", "codeit_priority"]
REFERENCE_ONLY = {"oracle_gain"}  # privileged; shown, never used as a comparator


def fmt(xs, digits=3):
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f}"
    return f"{statistics.mean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "pilot" / "results.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.results.open()]
    by = defaultdict(dict)  # (scenario, policy) -> seed -> metrics
    for r in rows:
        by[(r["scenario"], r["policy"])][r["seed"]] = r["metrics"]
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    seeds = sorted({r["seed"] for r in rows})

    out = [
        "# Pilot summary (synthetic harness)\n",
        f"{len(seeds)} seeds. Mean ± std across seeds. Synthetic data: this checks plumbing, "
        "not whether any method helps a real model.\n",
    ]
    for sc in scenarios:
        out.append(f"\n## Scenario: {sc}\n")
        out.append("| policy | " + " | ".join(c[1] for c in COLS) + " |")
        out.append("|---|" + "---|" * len(COLS))
        for pol in policies:
            m = by[(sc, pol)]
            cells = []
            for key, _ in COLS:
                vals = [m[s][key] for s in seeds if m[s][key] is not None]
                cells.append(fmt(vals, 1 if key == "update_tokens" else 3) if vals else "-")
            label = f"{pol} (privileged reference)" if pol in REFERENCE_ONLY else pol
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("\nPaired difference in final test pass rate (policy minus baseline), per-seed pairs:\n")
        out.append("| policy | " + " | ".join(f"vs {b}" for b in PAIRED_AGAINST) + " |")
        out.append("|---|" + "---|" * len(PAIRED_AGAINST))
        for pol in policies:
            if pol in ("frozen",) or pol in PAIRED_AGAINST or pol in REFERENCE_ONLY:
                continue
            cells = []
            for b in PAIRED_AGAINST:
                if (sc, b) not in by:
                    cells.append("-")
                    continue
                d = [by[(sc, pol)][s]["final_test_pass_rate"] - by[(sc, b)][s]["final_test_pass_rate"] for s in seeds]
                wins = sum(x > 0 for x in d)
                cells.append(f"{fmt(d, 4)} ({wins}/{len(d)} seeds >0)")
            out.append(f"| {pol} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    dest = args.out or args.results.parent / "summary.md"
    dest.write_text(text)
    print(text)
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
