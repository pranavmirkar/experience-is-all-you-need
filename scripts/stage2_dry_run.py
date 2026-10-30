"""Stage 2 plumbing dry run on a real model: collect, select, LoRA-update, evaluate on DEV.

Purpose: exercise the whole path (sandbox verification -> Episode -> TRAIN buffer -> policy selection ->
LoRA update -> DEV evaluation) and measure its cost. It is **not** an experiment: tiny sample, one seed, no
tuning. Reads TRAIN tasks (collection, training) and DEV tasks (evaluation) only; FINAL_TEST is never touched.

Mirrors ``evaluation.run_stream`` (the Stage 1 runner) with the simulator replaced by the model.
Needs torch, transformers and peft (not repo dependencies). All settings below are fixed in advance.

    HF_HOME=... <venv>/bin/python scripts/stage2_dry_run.py --out results/stage2_dry_run
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experience_learning.buffer import ExperienceBuffer  # noqa: E402
from experience_learning.evaluation import _sample_to_budget  # noqa: E402
from experience_learning.schemas import Episode, Split  # noqa: E402
from experience_learning.selection import SelectionContext, build_policies  # noqa: E402
from experience_learning.stage2 import collect, sandbox  # noqa: E402
from experience_learning.stage2.tasks import HELDOUT_FAMILIES, STREAM_FAMILIES, TaskBank  # noqa: E402
from stage2_smoke import STOP, _comment_block, clean  # noqa: E402

HEAD = "def solve("
CRASH_TEST = False
# Fixed in advance (see docs/stage2_plan.md section 10).
SETTINGS = dict(
    collect_instances=2, k_collect=8, k_eval=4, temperature=0.8, max_new=200, steps_per_phase=3,
    new_tokens_per_step=600, replay_tokens_per_step=600, buffer_capacity=200, lr=1e-4, lora_r=8, lora_alpha=16,
    grad_clip=1.0, eval_instances=1,
)


def prompt_text(task_prompt: str) -> str:
    return _comment_block(task_prompt) + "\n\n" + HEAD


def rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


class Runner:
    def __init__(self, model_name: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
        self.model.eval()
        self.peft = None
        self.opt = None

    # ---------------------------------------------------------------- adapter lifecycle
    def new_adapter(self, seed: int) -> None:
        from peft import LoraConfig, get_peft_model

        torch = self.torch
        if self.peft is not None:
            self.model = self.peft.unload()
        torch.manual_seed(seed)
        cfg = LoraConfig(r=SETTINGS["lora_r"], lora_alpha=SETTINGS["lora_alpha"], lora_dropout=0.0,
                         target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        self.peft = get_peft_model(self.model, cfg)
        params = [p for p in self.peft.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=SETTINGS["lr"])
        self.params = params

    # ---------------------------------------------------------------- generation
    def sample(self, task_prompt: str, k: int, seed: int) -> list[str]:
        torch = self.torch
        m = self.peft if self.peft is not None else self.model
        m.eval()
        torch.manual_seed(seed)
        ids = self.tok(prompt_text(task_prompt), return_tensors="pt")
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=SETTINGS["max_new"], do_sample=True, temperature=SETTINGS["temperature"],
                             top_p=1.0, num_return_sequences=k, stop_strings=STOP, tokenizer=self.tok,
                             pad_token_id=self.tok.eos_token_id)
        n = ids["input_ids"].shape[1]
        return [HEAD + clean(self.tok.decode(s[n:], skip_special_tokens=True)) for s in out]

    # ---------------------------------------------------------------- loss
    def _encode(self, task_prompt: str, attempt: str):
        torch = self.torch
        p_ids = self.tok(prompt_text(task_prompt))["input_ids"]
        target = attempt[len(HEAD):].rstrip("\n") + "\n" + self.tok.eos_token
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + t_ids])
        labels = ids.clone()
        labels[0, : len(p_ids)] = -100  # train and score on the completion only
        return ids, labels

    def nll(self, task_prompt: str, attempt: str) -> float:
        """Mean token NLL of the completion under the current model (adapter included if present)."""
        torch = self.torch
        m = self.peft if self.peft is not None else self.model
        m.eval()
        ids, labels = self._encode(task_prompt, attempt)
        with torch.no_grad():
            return float(m(input_ids=ids, labels=labels).loss)

    def train_step(self, episodes: list[Episode]) -> float:
        """One optimiser step on the given episodes (micro-batches of 1, gradients accumulated)."""
        torch = self.torch
        self.peft.train()
        total = 0.0
        for e in episodes:
            ids, labels = self._encode(e.prompt, e.correction if e.correction is not None else e.attempt)
            loss = self.peft(input_ids=ids, labels=labels).loss / len(episodes)
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(self.params, SETTINGS["grad_clip"])
        self.opt.step()
        self.opt.zero_grad()
        return total


def collect_stream(r: Runner, bank: TaskBank, model_name: str) -> list[list[Episode]]:
    """Base-model attempts on TRAIN tasks, one phase per stream family, verified in the sandbox."""
    phases: list[list[Episode]] = []
    idx = 0
    for p, fam in enumerate(STREAM_FAMILIES):
        tasks = bank.tasks(Split.TRAIN, fam, SETTINGS["collect_instances"])
        assert tasks and all(t.split is Split.TRAIN for t in tasks)
        eps: list[Episode] = []
        t0 = time.time()
        for ti, task in enumerate(tasks):
            for src in r.sample(task.prompt, SETTINGS["k_collect"], seed=10_000 + 100 * p + ti):
                idx += 1
                if CRASH_TEST and ti == 0:
                    src = task.reference_source  # crash test only: guarantees a verified episode so training runs
                e = collect.collect_episode(task, src, p * 1000 + idx % 1000, model_checkpoint=model_name)
                eps.append(replace(e, loss=r.nll(e.prompt, e.attempt)))
        phases.append(eps)
        print(f"collect phase {p} {fam}: {len(eps)} attempts, {sum(e.verified for e in eps)} verified, {time.time() - t0:.0f}s", flush=True)
    return phases


def evaluate(r: Runner, dev_tasks: list) -> dict:
    """Sampled DEV evaluation: per-family full-pass rate and test fraction."""
    fam_pass: dict[str, list[float]] = defaultdict(list)
    fam_frac: dict[str, list[float]] = defaultdict(list)
    for ti, task in enumerate(dev_tasks):
        for src in r.sample(task.prompt, SETTINGS["k_eval"], seed=7919 + ti):
            o = collect.verify(task, src)
            fam_pass[task.family].append(float(o.passed))
            fam_frac[task.family].append(o.n_passed / o.total)
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    return {
        "pass_by_family": {f: mean(v) for f, v in fam_pass.items()},
        "test_fraction_by_family": {f: mean(v) for f, v in fam_frac.items()},
        "pass_overall": mean([x for v in fam_pass.values() for x in v]),
        "test_fraction_overall": mean([x for v in fam_frac.values() for x in v]),
    }


def run_policy(r: Runner, policy, phases: list[list[Episode]], dev_tasks: list, seed: int) -> dict:
    gate = policy.gate_verified
    buffer = ExperienceBuffer(SETTINGS["buffer_capacity"], gate, random.Random(f"admission:{seed}"))
    replay_counts: dict[str, int] = defaultdict(int)
    policy.reset()
    r.new_adapter(seed)
    arrival_step: dict[str, int] = {}
    scoring_passes = gstep = 0
    timing = defaultdict(float)
    steps_log, rows = [], []
    for p, episodes in enumerate(phases):
        pool = [e for e in episodes if e.verified or not gate]
        phase_start = episodes[0].stream_index
        for e in episodes:
            arrival_step[e.episode_id] = gstep
        if policy.scope != "none" and pool:
            for step in range(SETTINGS["steps_per_phase"]):
                new = _sample_to_budget(pool, SETTINGS["new_tokens_per_step"], random.Random(f"batch:{seed}:{p}:{step}"))
                cl = None
                if policy.needs_current_loss:

                    def cl(e, _r=r):
                        nonlocal scoring_passes
                        scoring_passes += 1
                        return _r.nll(e.prompt, e.attempt)

                ctx = SelectionContext(p, (), None, dict(replay_counts), None, gstep, arrival_step, cl)
                t0 = time.time()
                if policy.scope == "past":
                    candidates = buffer.eligible(phase_start)
                else:
                    taken = {e.episode_id for e in new}
                    candidates = [e for e in pool if e.episode_id not in taken]
                chosen = policy.select(candidates, SETTINGS["replay_tokens_per_step"], ctx, random.Random(f"policy:{seed}:{p}:{step}"))
                timing["select_s"] += time.time() - t0
                if policy.scope == "past":
                    for e in chosen:
                        replay_counts[e.family] += 1
                batch = new + chosen
                t0 = time.time()
                loss = r.train_step(batch) if batch else None
                timing["train_s"] += time.time() - t0
                steps_log.append({"phase": p, "step": step, "new": len(new), "replay": [e.episode_id for e in chosen],
                                  "replay_families": [e.family for e in chosen], "loss": loss})
                gstep += 1
        for e in episodes:
            buffer.add(e)
        t0 = time.time()
        row = evaluate(r, dev_tasks)
        timing["eval_s"] += time.time() - t0
        rows.append(row)
        print(f"  [{policy.name}] after phase {p} ({STREAM_FAMILIES[p]}): dev pass {row['pass_overall']:.3f} "
              f"tests {row['test_fraction_overall']:.3f}  buffer={len(buffer)} rejected_unverified={buffer.rejected_unverified}", flush=True)
    return {"policy": policy.name, "rows": rows, "steps": steps_log, "replay_counts": dict(replay_counts),
            "scoring_passes": scoring_passes, "timing_s": dict(timing), "buffer_size": len(buffer),
            "rejected_unverified": buffer.rejected_unverified}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--policies", nargs="+", default=["sequential", "uniform", "loss_fresh"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stage2_dry_run")
    ap.add_argument("--crash-test", action="store_true", help="tiny settings to check the code path; results are meaningless")
    args = ap.parse_args()
    if args.crash_test:
        global CRASH_TEST
        CRASH_TEST = True
        SETTINGS.update(collect_instances=1, k_collect=1, k_eval=1, steps_per_phase=1, max_new=40)

    sandbox.require_isolation()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = TaskBank(0)
    dev_tasks = []
    for fam in STREAM_FAMILIES + HELDOUT_FAMILIES:
        dev_tasks += bank.tasks(Split.DEV, fam, SETTINGS["eval_instances"])
    assert all(t.split is Split.DEV for t in dev_tasks)

    t0 = time.time()
    r = Runner(args.model, args.dtype)
    print(f"model loaded in {time.time() - t0:.0f}s ({args.dtype})", flush=True)

    t0 = time.time()
    coll_file = out / "collection.jsonl"
    phases = collect_stream(r, bank, args.model)
    coll_file.write_text("".join(json.dumps(e.to_dict(), sort_keys=True) + "\n" for ph in phases for e in ph))
    collect_s = time.time() - t0

    t0 = time.time()
    base_row = evaluate(r, dev_tasks)  # no adapter yet: the plain base model
    base_eval_s = time.time() - t0
    print(f"base model dev: pass {base_row['pass_overall']:.3f} tests {base_row['test_fraction_overall']:.3f} ({base_eval_s:.0f}s)", flush=True)

    policies = build_policies()
    results = []
    for name in args.policies:
        t0 = time.time()
        res = run_policy(r, policies[name], phases, dev_tasks, args.seed)
        res["wall_s"] = round(time.time() - t0, 1)
        results.append(res)
        (out / f"run_{name}.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")

    summary = {
        "model": args.model, "dtype": args.dtype, "seed": args.seed, "settings": SETTINGS, "dev_tasks": len(dev_tasks),
        "collection": {"attempts": sum(len(p) for p in phases), "verified": sum(e.verified for p in phases for e in p),
                       "verified_by_phase": [sum(e.verified for e in p) for p in phases], "wall_s": round(collect_s, 1)},
        "base_dev": base_row, "base_eval_s": round(base_eval_s, 1),
        "runs": {x["policy"]: {"final_dev": x["rows"][-1], "replay_counts": x["replay_counts"], "scoring_passes": x["scoring_passes"],
                               "timing_s": x["timing_s"], "wall_s": x["wall_s"]} for x in results},
        "peak_rss_gb": rss_gb(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
