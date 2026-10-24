"""Stage 2 smoke test: model load, greedy generation on DEV tasks, sandbox verification, one LoRA step.

Reads TRAIN and DEV tasks only. The FINAL_TEST split is never touched. Needs torch, transformers and peft
(not repo dependencies). Run with the system python3 first on PATH, so the sandbox can find an interpreter
outside the venv:

    HF_HOME=... /path/to/venv/bin/python scripts/stage2_smoke.py --model Qwen/Qwen2.5-Coder-0.5B
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experience_learning.schemas import Split  # noqa: E402
from experience_learning.stage2 import collect, sandbox  # noqa: E402
from experience_learning.stage2.tasks import TaskBank  # noqa: E402

STOP = ["\n\n\n", "\nif __name__", "\nprint(", "\nassert", "\n# ", "\nclass ", "\ndef test"]


def format_prompt(task) -> str:
    lines = [f"# {ln}" if ln.strip() else "#" for ln in task.prompt.strip().splitlines()]
    return "\n".join(lines) + "\n\ndef solve("


def clean(completion: str) -> str:
    for s in STOP:
        i = completion.find(s)
        if i != -1:
            completion = completion[:i]
    return completion.rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--per-template", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--out", default="results/stage2_smoke")
    args = ap.parse_args()

    sandbox.require_isolation()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report = {"model": args.model, "torch": torch.__version__, "threads": torch.get_num_threads()}
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    report["load_s"] = round(time.time() - t0, 1)
    report["params"] = sum(p.numel() for p in model.parameters())

    bank = TaskBank(0)
    dev = []
    for fam in ("strings", "lists", "dicts", "math", "dp", "parsing"):
        dev += bank.tasks(Split.DEV, fam, args.per_template)
    assert all(t.split is Split.DEV for t in dev)

    rows, gen_tokens, gen_s = [], 0, 0.0
    for task in dev:
        prompt = format_prompt(task)
        ids = tok(prompt, return_tensors="pt")
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new, do_sample=False, stop_strings=STOP, tokenizer=tok,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.time() - t1
        new = out[0, ids["input_ids"].shape[1]:]
        completion = tok.decode(new, skip_special_tokens=True)
        source = "def solve(" + clean(completion)
        o = collect.verify(task, source)
        gen_tokens += len(new)
        gen_s += dt
        rows.append({
            "task_id": task.task_id, "family": task.family, "template": task.template, "split": task.split.value,
            "prompt_tokens": int(ids["input_ids"].shape[1]), "new_tokens": int(len(new)), "gen_s": round(dt, 1),
            "passed": o.passed, "n_passed": o.n_passed, "total": o.total, "summary": o.summary, "source": source,
        })
        print(f"{task.task_id:40s} {o.n_passed}/{o.total} {len(new):3d} tok {dt:5.1f}s", flush=True)

    by_fam: dict[str, list[bool]] = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r["passed"])
    report["dev_tasks"] = len(rows)
    report["dev_pass_rate"] = sum(r["passed"] for r in rows) / len(rows)
    report["dev_test_pass_fraction"] = sum(r["n_passed"] for r in rows) / sum(r["total"] for r in rows)
    report["dev_pass_by_family"] = {f: sum(v) / len(v) for f, v in by_fam.items()}
    report["gen_tokens_per_s"] = round(gen_tokens / gen_s, 2)
    report["gen_s_per_task"] = round(gen_s / len(rows), 1)

    # One LoRA step on a TRAIN task with its reference solution.
    train = bank.tasks(Split.TRAIN, "strings", 1)[0]
    assert train.split is Split.TRAIN
    text = format_prompt(train) + train.reference_source.split("def solve(", 1)[1] + tok.eos_token
    batch = tok(text, return_tensors="pt")
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.train()
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-4)
    step_s = []
    for _ in range(3):
        t2 = time.time()
        loss = pm(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        step_s.append(round(time.time() - t2, 1))
    report["lora"] = {
        "trainable_params": sum(p.numel() for p in pm.parameters() if p.requires_grad),
        "seq_tokens": int(batch["input_ids"].shape[1]), "step_s": step_s, "last_loss": float(loss),
    }
    report["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out_dir / "dev_attempts.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(json.dumps({k: v for k, v in report.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
