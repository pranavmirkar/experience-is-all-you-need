"""Stage 2 timing: LoRA update cost (mode ``train``) and adapter save/load/merge/generation cost (mode ``gen``).

One configuration per process, so peak RSS (a process-wide maximum) belongs to that configuration. Uses TRAIN
tasks only (prompt + reference solution as training text); the content is irrelevant for timing. Needs torch,
transformers, peft (not repo dependencies). Says nothing about learning.

An 8-sequence update that does not fit in memory is run as micro-batches with gradient accumulation; the
reported ``update_s_for_8_seqs`` scales the measured micro-batch step time to 8 sequences.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experience_learning.schemas import Split  # noqa: E402
from experience_learning.stage2.tasks import TaskBank  # noqa: E402
from stage2_smoke import format_prompt  # noqa: E402

TARGETS = {
    "qv": ["q_proj", "v_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "gen"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--micro", type=int, default=1, help="sequences per forward/backward")
    ap.add_argument("--len", type=int, default=300, help="padded sequence length in tokens")
    ap.add_argument("--targets", choices=list(TARGETS), default="qv")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report: dict = {"mode": args.mode, "model": args.model, "dtype": args.dtype, "torch": torch.__version__,
                    "threads": torch.get_num_threads()}
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16)
    report["load_s"] = round(time.time() - t0, 1)
    report["rss_after_load_gb"] = rss_gb()

    bank = TaskBank(0)
    train = []
    for fam in ("strings", "lists", "dicts", "math", "dp"):
        train += bank.tasks(Split.TRAIN, fam, 1)
    assert all(t.split is Split.TRAIN for t in train)
    texts = [format_prompt(t) + t.reference_source.split("def solve(", 1)[1] + tok.eos_token for t in train]
    natural = [len(tok(x)["input_ids"]) for x in texts]
    report["natural_seq_tokens"] = {"min": min(natural), "mean": round(sum(natural) / len(natural), 1), "max": max(natural)}

    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=TARGETS[args.targets], task_type="CAUSAL_LM")

    if args.mode == "train":
        if args.ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        pm = get_peft_model(model, cfg)
        pm.train()
        params = [p for p in pm.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        rows = [texts[i % len(texts)] for i in range(args.micro)]
        enc = tok(rows, return_tensors="pt", padding="max_length", truncation=True, max_length=args.len)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        times = []
        for _ in range(args.steps + 1):  # first step is warm-up
            t1 = time.time()
            loss = pm(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            times.append(round(time.time() - t1, 2))
        mean = sum(times[1:]) / len(times[1:])
        report.update({
            "config": {"targets": args.targets, "r": 8, "micro": args.micro, "len": args.len, "grad_ckpt": args.ckpt},
            "trainable_params": sum(p.numel() for p in params), "warmup_s": times[0], "step_s": times[1:],
            "mean_micro_step_s": round(mean, 2), "tokens_per_s": round(args.micro * args.len / mean, 1),
            "update_s_for_8_seqs": round(mean * 8 / args.micro, 1), "peak_rss_gb": rss_gb(),
            "last_loss_finite": bool(torch.isfinite(loss)),
        })
    else:
        pm = get_peft_model(model, cfg)
        with tempfile.TemporaryDirectory() as d:
            t2 = time.time()
            pm.save_pretrained(d)
            report["adapter_save_s"] = round(time.time() - t2, 2)
            report["adapter_mb"] = round(sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 1e6, 2)
            base = pm.unload()
            t3 = time.time()
            loaded = PeftModel.from_pretrained(base, d)
            report["adapter_load_s"] = round(time.time() - t3, 2)
        loaded.eval()
        ids = tok(format_prompt(train[0]), return_tensors="pt")

        def gen_s_per_token(m, n=3):
            out = []
            for _ in range(n):
                t = time.time()
                with torch.no_grad():
                    m.generate(**ids, max_new_tokens=60, min_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
                out.append(time.time() - t)
            return round(sum(out) / n / 60, 4)

        report["gen_s_per_token"] = {"unmerged_adapter": gen_s_per_token(loaded)}
        t4 = time.time()
        merged = loaded.merge_and_unload()
        report["merge_s"] = round(time.time() - t4, 2)
        report["gen_s_per_token"]["merged_adapter"] = gen_s_per_token(merged)
        report["peak_rss_gb"] = rss_gb()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
