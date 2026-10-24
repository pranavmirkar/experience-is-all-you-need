#!/usr/bin/env python3
"""Report sandbox isolation and verify the task bank end to end. Downloads nothing."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experience_learning.schemas import Split  # noqa: E402
from experience_learning.stage2 import collect, sandbox, tasks  # noqa: E402


def main() -> int:
    rep = sandbox.detect_isolation(refresh=True)
    print("isolation strong:", rep.strong)
    print("problems:", list(rep.problems) or "none")
    print("observed:", json.dumps(rep.facts, sort_keys=True))
    if not rep.strong:
        print("Sandbox unavailable: no model-generated code may be executed on this machine.")
        return 2
    t0 = time.time()
    bad = []
    families = tasks.STREAM_FAMILIES + tasks.HELDOUT_FAMILIES
    names = [(f, n) for f in families for n in tasks.templates_of(f)]
    for fam, name in names:
        t = tasks.build_task(fam, name, 0, 0, Split.TRAIN)
        if not collect.verify(t, t.reference_source).passed:
            bad.append((fam, name, "reference fails"))
        if collect.verify(t, "def solve(*a):\n    return None").passed:
            bad.append((fam, name, "constant-None passes"))
    print(f"{len(names)} templates checked in {time.time() - t0:.1f}s; problems: {bad or 'none'}")
    bank = tasks.TaskBank(0)
    for split in (Split.TRAIN, Split.DEV, Split.FINAL_TEST):
        print(split.value, "templates:", len(bank.templates(split)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
