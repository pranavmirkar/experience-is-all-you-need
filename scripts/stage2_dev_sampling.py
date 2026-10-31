"""Stage 2 dev diagnostic: sample k completions per DEV task and report pass@1 (mean) and pass@k.

DEV tasks only. Same prompt format and sandbox verification as ``stage2_smoke.py``. Needs torch and
transformers (not repo dependencies).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experience_learning.schemas import Split  # noqa: E402
from experience_learning.stage2 import collect, sandbox  # noqa: E402
from experience_learning.stage2.tasks import TaskBank  # noqa: E402
from stage2_smoke import STOP, clean, format_prompt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--per-template", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot", action="store_true", help="prefix two worked examples (see stage2_smoke.SHOTS)")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--out", default="results/stage2_dev_sampling")
    args = ap.parse_args()

    sandbox.require_isolation()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16).eval()
    bank = TaskBank(0)
    dev = []
    for fam in ("strings", "lists", "dicts", "math", "dp", "parsing"):
        dev += bank.tasks(Split.DEV, fam, args.per_template)
    assert all(t.split is Split.DEV for t in dev)

    rows = []
    t0 = time.time()
    for ti, task in enumerate(dev):
        torch.manual_seed(args.seed * 100003 + ti)
        ids = tok(format_prompt(task, args.fewshot), return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new, do_sample=True, temperature=args.temperature, top_p=1.0,
                num_return_sequences=args.k, stop_strings=STOP, tokenizer=tok, pad_token_id=tok.eos_token_id,
            )
        n_prompt = ids["input_ids"].shape[1]
        outcomes = []
        for seq in out:
            src = "def solve(" + clean(tok.decode(seq[n_prompt:], skip_special_tokens=True))
            o = collect.verify(task, src)
            outcomes.append({"passed": o.passed, "n_passed": o.n_passed, "source": src})
        rows.append({
            "task_id": task.task_id, "family": task.family, "template": task.template, "split": task.split.value,
            "k": args.k, "n_pass": sum(o["passed"] for o in outcomes),
            "mean_test_fraction": sum(o["n_passed"] for o in outcomes) / (args.k * len(task.tests)),
            "attempts": outcomes,
        })
        r = rows[-1]
        print(f"{task.task_id:40s} {r['n_pass']}/{args.k} full-pass  tests {r['mean_test_fraction']:.2f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    per_tpl: dict[str, list] = {}
    for r in rows:
        per_tpl.setdefault(f"{r['family']}/{r['template']}", []).append(r)
    summary = {
        "model": args.model, "dtype": args.dtype, "fewshot": args.fewshot, "k": args.k, "temperature": args.temperature, "seed": args.seed,
        "dev_tasks": len(rows), "pass_at_1_mean": sum(r["n_pass"] for r in rows) / (len(rows) * args.k),
        "pass_at_k": sum(r["n_pass"] > 0 for r in rows) / len(rows),
        "mean_test_fraction": sum(r["mean_test_fraction"] for r in rows) / len(rows),
        "per_template": {
            t: {"pass_at_1": sum(r["n_pass"] for r in rs) / (len(rs) * args.k),
                f"pass_at_{args.k}": sum(r["n_pass"] > 0 for r in rs) / len(rs),
                "mean_test_fraction": sum(r["mean_test_fraction"] for r in rs) / len(rs)}
            for t, rs in sorted(per_tpl.items())
        },
        "wall_s": round(time.time() - t0, 1),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "attempts.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
