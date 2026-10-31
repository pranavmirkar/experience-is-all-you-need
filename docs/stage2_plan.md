# Stage 2 plan: real code tasks, real model (status: smoke test run, see section 6)

Stage 2 moves from the simulator to executable coding tasks and a real small code model.
This document says what exists, what was measured, what is only estimated, and what needs
approval before it can proceed. After approval, the packages and `Qwen/Qwen2.5-Coder-0.5B` were
installed and a smoke test was run (section 6). No paid API has been called. The final-test split
has not been read.

Nothing here supports a claim about VTAR, novelty or improvement. Stage 1 results are
synthetic, and Stage 2 has produced no model results yet.

## 1. Environment audit (measured)

| Item | Value |
|---|---|
| CPU | 4 vCPU (Intel Xeon), no GPU (`nvidia-smi` absent) |
| RAM | 15 GB |
| Free disk | about 30 GB |
| Python | 3.13; `torch`, `transformers`, `peft` not installed |
| Network | HTTPS through a proxy; `curl` works |

Consequence: any model work is CPU-only here. Section 5 shows what that implies.

## 2. Sandbox for generated code (implemented, measured)

`src/experience_learning/stage2/sandbox.py`. Generated code never runs in the harness process.

Isolation per run:

- `unshare` with net, pid, mount, ipc and uts namespaces, so no network, no view of host processes
- private tmpfs over `/tmp` and `/home`, so no host files are readable or writable
- `setpriv` to uid/gid 65534 with `no_new_privs`
- empty environment (`env -i`)
- `prlimit` caps: CPU 15 s, memory 512 MiB, file size 1 MiB
- per-call timeout 2 s (itimer inside the runner), 20 s total, hard kill of the process group from outside
- only inputs go in. Expected outputs stay outside and are compared outside, so a candidate cannot print a forged "pass"

`detect_isolation()` runs a real probe on every new machine (uid, network, host-file reads,
visible PIDs, env). If isolation is not strong, `run_candidate` raises `SandboxUnavailable`
rather than running with weaker guarantees. Tests skip in that case.

Measured on this machine: isolation strong, no problems (`python3 scripts/stage2_selfcheck.py`).
Hostile candidates tested: infinite loop, a loop that swallows the alarm (killed by the outer
timeout), memory bomb, fork bomb, network connect, host file read and write, forged result
lines, `os._exit`, syntax error, unserialisable return value. All were contained.

Limits: this is not a defence against kernel exploits. It is process-level isolation suitable
for small, untrusted function-level snippets, not for hostile actors with kernel exploits.

## 3. Task bank and split rules (implemented; bank v2)

`src/experience_learning/stage2/tasks.py`, `templates_v2.py`, `collect.py`.

- 52 templates in 6 families: 5 stream families (strings, lists, dicts, math, dp) with 9 templates each,
  and the held-out family `parsing` with 7. Vintage 1 is the original 30; vintage 2 added 22.
- Each template generates instances from a seed. Each instance has 12 tests: 3 visible in the prompt,
  9 hidden. No single expected value may fill more than 6 of the 12 (hand-written edge cases excepted).
- Splits are assigned per template, never per instance, and per vintage, so adding templates cannot move
  one that was already seen in dev (pinned by `tests/data/stage2_v1_splits.json`). Seed 0 totals:
  25 train, 13 dev, 14 final_test templates. Each stream family has 5 train / 2 dev / 2 final-test;
  `parsing` has 0 train / 3 dev / 4 final-test.
- Every reference solution passes its own tests through the sandbox (self-check). Constant-`None` and
  constant-`0` candidates are rejected, and a test asserts no constant answer averages more than 55% of tests.
- `collect_episode` produces an `Episode`. It is `verified` only if the solution it relies on
  passed every test in the sandbox. Otherwise it carries `Verification(False, "model_assertion")`,
  which the TRAIN buffer's gate rejects.
- Revision log, bias disclosure and the pre-declared drop rule are in `docs/experiment_protocol.md`
  ("Stage 2 task bank: revision log").

Known weaknesses:

- The templates are small and hand-written. They are not a recognised benchmark. A pass rate here says
  little about general coding.
- "Interference between families" is only meaningful if the real model shows measurable cross-family
  effects, and that is untested.
- Token counts use chars/4 as a placeholder. Real cost accounting needs the model tokenizer.
- The final-test templates exist but are locked: nothing scores them until the protocol is pre-registered.
- The answer-balance cap changed the tests of nearly all vintage-1 templates after the smoke test and the
  first dev diagnostic were run. Sections 6 and 6b are therefore superseded for comparison purposes by 6c.

## 4. Tests

`python3 -m pytest -q` runs 276 tests (89 from Stage 1, 187 for Stage 2). Stage 2 tests skip if the
sandbox is not strong. They include a polling test that no sandboxed process outlives its run.

## 5. Model candidates (metadata from the Hugging Face API; no weights downloaded)

| Model | Licence | Params | Weights size | Notes |
|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-0.5B | apache-2.0 | 494M | 0.99 GB safetensors | Not gated. Base (not instruct) model. |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | apache-2.0 | 494M | 0.99 GB | Not gated. |
| Qwen/Qwen2.5-Coder-1.5B | apache-2.0 | 1544M | 3.09 GB | Larger; slower on CPU. |
| deepseek-ai/deepseek-coder-1.3b-base | "other" (custom licence) | 1.3B | 2.69 GB, `.bin` | Licence needs review. |
| Salesforce/codegen-350M-mono | bsd-3-clause | 350M | 0.80 GB, `.bin` | Older; Python only. |
| bigcode/tiny_starcoder_py | bigcode-openrail-m | 164M | 0.66 GB | Use restrictions in licence. |
| HuggingFaceTB/SmolLM2-360M, -135M | apache-2.0 | 360M / 135M | 0.72 / 0.27 GB | General-purpose, not code-specific. |

Original recommendation was `Qwen/Qwen2.5-Coder-0.5B` (base); the dev diagnostics (section 6) found it too weak on this bank, and `Qwen/Qwen2.5-Coder-1.5B` (6e) is now the working choice. Original text: It is the smallest permissively licensed
model with real code training, and it fits in RAM with room for LoRA. It might be too weak or
too strong for the task bank; section 7 gates on that. The Instruct variant is the fallback if
the base model does not follow the prompt format.

## 6. Smoke test (measured, 2026-10-24)

Command (venv in the session scratchpad, not committed; versions torch 2.14.1+cpu,
transformers 5.20.2, peft 0.21.2):

```bash
HF_HOME=<scratchpad>/hf HF_HUB_OFFLINE=1 <venv>/bin/python scripts/stage2_smoke.py --out results/stage2_smoke
```

It loads the model in fp32, greedily generates one solution per DEV task (7 dev templates x 2
instances = 14 tasks), verifies each in the sandbox against all 12 tests, then runs 3 LoRA
steps (r=8, q_proj/v_proj, 540,672 trainable parameters) on one TRAIN example. It reads only
TRAIN and DEV tasks. Outputs: `results/stage2_smoke/report.json`, `dev_attempts.jsonl`.

| Measurement | Value |
|---|---|
| Download + installs | model 954 MB on disk; venv 1.2 GB |
| Model load | 2.6 s |
| Generation speed | 10.3 tokens/s; 5.8 s per attempt (26 to 106 new tokens) |
| LoRA step (94-token sequence, batch 1) | 0.6 to 0.9 s |
| Peak RSS | 4.3 GB |
| Dev tasks fully passed (all 12 tests) | 2 of 14 (14%) |
| Dev tests passed (fraction of all tests) | 45.8% |

Dev full-pass by family (2 instances each): strings 0, lists 0, dicts 0, math 0, dp 0, parsing 0.5
(`brackets` 2/2, `kv_parse` 0/2). Per task: second_largest 11/12 (misses one edge case), brackets 12/12,
vowel_positions 6 to 8/12, digit_root_base 2 to 6/12, merge_filter 3/12, lis_gap 1/12, kv_parse 0 to 1/12.

I read the failing sources for six tasks. The failures are wrong logic (ignoring the spec, unhandled
edges), not prompt-format or extraction artefacts, so the pipeline measures the model and not the plumbing.

What the measurements change:

- My earlier estimates were too pessimistic by roughly 10x to 30x (I assumed 5 tokens/s generation and
  8 to 26 s per LoRA sequence; measured 10 tokens/s and under 1 s for short sequences). A CPU-only
  reduced pilot looks feasible. The LoRA figure is for 94 tokens and 3 repeated steps on one example;
  longer sequences, larger batches and adapter re-merging for evaluation are not yet timed.
- The smoke test is one greedy sample per task. It says nothing about variance.

What the measurements do not show, and the caveats:

- Sample size is thin. The two instances of a template are not independent, so this is 7 independent
  templates. "14%" is 1 template (`brackets`) passing, not a rate with a meaningful interval.
- All five stream families score 0 full-passes under greedy decoding. That leaves headroom for learning,
  but also risks a floor effect: if training seldom moves a task from fail to full-pass, binary pass rate
  will be insensitive and partial-test fraction would carry the signal. The brief defines the metric as
  pass rate, so I would not switch metrics after seeing this; I would instead report both and say which
  was pre-registered.
- The base model sees only the three visible examples in a comment prompt. A different prompt format could
  change these numbers; I tried one format.
- The LoRA step only shows that backward and optimiser run and fit in memory. It says nothing about learning.

### 6b. Dev sampling diagnostic (measured)

Command: `scripts/stage2_dev_sampling.py` (k=8 completions per dev task, temperature 0.8, top_p 1.0,
seed 0, same 14 DEV tasks, same prompt and sandbox verification; 360 s wall time). Outputs:
`results/stage2_dev_sampling/summary.json`, `attempts.jsonl`. DEV and nothing else.

| Template | pass@1 (mean of 16) | pass@8 (2 tasks) | mean test fraction |
|---|---|---|---|
| parsing/brackets | 0.31 | 1.0 | 0.55 |
| strings/vowel_positions | 0.25 | 0.5 | 0.56 |
| lists/second_largest | 0.125 | 0.5 | 0.66 |
| dp/lis_gap | 0.06 | 0.5 | 0.34 |
| dicts/merge_filter | 0 | 0 | 0.22 |
| math/digit_root_base | 0 | 0 | 0.23 |
| parsing/kv_parse | 0 | 0 | 0.08 |
| **all 14 tasks** | **0.107** | **0.357** | **0.378** |

What it shows:

- Sampling finds full solutions where greedy found none: `vowel_positions` was 0 of 2 under greedy
  decoding and has one instance at 4 of 8 when sampled; `lis_gap` and `second_largest` likewise. So the
  greedy 0% in the smoke test overstated how hard those templates are, and the failures are not all stable.
- Four of seven templates have some sampled successes (pass@1 between 0.06 and 0.31). Three have none in
  16 samples each (`merge_filter`, `digit_root_base`, `kv_parse`) and look out of reach for this model at
  this prompt. Templates that are never solved cannot show a gain from verified experience, so they add
  cost without information.
- Instances of one template differ a lot (`vowel_positions`: 0/8 vs 4/8; `second_largest`: 2/8 vs 0/8).
  With k=8 per instance and 7 templates, every number above has a wide uncertainty that I have not
  quantified. Do not read the ordering of templates as meaningful beyond "some solved, some not".
- Sampled pass@1 (0.107) is close to the greedy full-pass rate (0.143), but the two are different
  quantities on a handful of tasks.

### 6c. Dev sampling on bank v2 (measured)

Same script and settings as 6b (`--out results/stage2_dev_sampling_bank2`, k=8, T=0.8, seed 0), now on
all 13 dev templates x 2 instances = 26 tasks, with the rebalanced tests. 659 s wall time.
Dev only; final_test untouched.

| Template | vintage | pass@1 (of 16) | any pass in 16 | mean test fraction |
|---|---|---|---|---|
| strings/case_by_index | 2 | 0.38 | yes | 0.52 |
| strings/vowel_positions | 1 | 0.19 | yes | 0.43 |
| lists/second_largest | 1 | 0.06 | yes | 0.48 |
| dp/subset_sum_count | 2 | 0.06 | yes | 0.31 |
| parsing/brackets | 1 | 0.06 | yes | 0.42 |
| dicts/anagram_groups | 2 | 0 | no | 0.21 |
| dicts/merge_filter | 1 | 0 | no | 0.22 |
| dp/lis_gap | 1 | 0 | no | 0.17 |
| lists/unique_keep_last | 2 | 0 | no | 0.26 |
| math/digit_root_base | 1 | 0 | no | 0.13 |
| math/fib_prefix_mod | 2 | 0 | no | 0.16 |
| parsing/kv_parse | 1 | 0 | no | 0.08 |
| parsing/version_compare | 2 | 0 | no | 0.29 |
| **all 26 tasks** | | **0.058** | 5 of 13 templates | **0.283** |

- Widening did not raise the share of informative dev templates: 5 of 13 (38%) have any success, versus 4
  of 7 (57%) before. By the pre-declared rule, 8 of 13 dev templates are flagged "not informative for this
  model" (never fully passed in 16 samples). Nothing has been dropped.
- Of the 6 new dev templates, 2 are sometimes solved. Reading three failing outputs
  (`unique_keep_last`, `fib_prefix_mod`, `version_compare`): the model returns the first occurrence or a
  set, answers a different recursion, or misparses versions, so these are model errors and not broken
  templates. The prompts agree with the references.
- Numbers for vintage-1 templates fell against 6b (brackets pass@1 0.31 to 0.06, lis_gap 0.06 to 0,
  vowel_positions 0.25 to 0.19, and test fractions lower everywhere). The rebalanced tests are the likely
  cause for the drop in test fraction. I did not isolate the effect on full-pass rates, which could also
  reflect sampling noise at k=8 on 2 instances.
- The model is a base model prompted with one comment-style format. A few-shot prompt, the Instruct
  variant or a larger model could raise the solved share. None of these was tried.

### 6d. Few-shot prompt on bank v2 (measured)

Same 26 dev tasks, k=8, T=0.8, seed 0 as 6c. The only change is two worked examples in front of each prompt
(`stage2_smoke.SHOTS`: sum of squares of odd numbers, reverse words; neither is a bank template; fixed
before the run). Command: `scripts/stage2_dev_sampling.py --fewshot --out results/stage2_dev_sampling_bank2_fewshot`.
560 s. Dev only.

| | zero-shot (6c) | few-shot |
|---|---|---|
| Sampled pass@1 | 0.058 | 0.053 |
| Templates with any full pass in 16 samples | 5 of 13 | 4 of 13 |
| Mean test fraction | 0.283 | 0.250 |

- No improvement. Full passes per task moved up on 3 tasks, down on 3, unchanged on 20; with 1 to 3 passes per
  task these differences are noise. Test fraction fell on 17 of 26 tasks and rose on 7. The templates that
  were never solved stay never solved (the same 8 are flagged by the declared rule, plus `subset_sum_count`,
  which lost its single success).
- So the low solve rate is not mainly a prompt-format artefact, at least not for this kind of two-shot
  prefix. I tried one few-shot design; other designs (more shots, shots from the same family, an
  instruction-style wrapper) were not tried.
- Cost of the few-shot prefix: about 150 more prompt tokens per attempt; generation time was not slower.

### 6e. Qwen2.5-Coder-1.5B on bank v2 (measured)

Downloaded `Qwen/Qwen2.5-Coder-1.5B` (apache-2.0, 3,087,467,144-byte safetensors file; 17 s). Same 26 dev tasks,
same zero-shot prompt, k=8, T=0.8, seed 0 as 6c. Command:
`scripts/stage2_dev_sampling.py --model Qwen/Qwen2.5-Coder-1.5B --out results/stage2_dev_sampling_bank2_1p5b`.
Dev only; final_test untouched.

| | 0.5B (6c) | 1.5B |
|---|---|---|
| Sampled pass@1 | 0.058 | 0.212 |
| Templates with any full pass in 16 samples | 5 of 13 | 12 of 13 |
| Mean test fraction | 0.283 | 0.443 |
| Wall time for 26 tasks x 8 samples | 659 s | 1705 s (2.6x) |
| System memory in use mid-run (`free`) | not recorded | about 7 GB (fp32) |

Per template, 1.5B pass@1 (of 16 samples): case_by_index 0.75, second_largest 0.31, fib_prefix_mod 0.25,
vowel_positions 0.25, anagram_groups 0.19, digit_root_base 0.19, brackets 0.19, version_compare 0.19,
merge_filter 0.12, unique_keep_last 0.12, kv_parse 0.12, subset_sum_count 0.06, lis_gap 0. No template is
at or above 0.9; one is at 0.

- Per-task full passes were higher with 1.5B on 19 of 26 tasks, lower on none, equal on 7.
- Only `lis_gap` is flagged by the declared drop rule now (8 of 13 with 0.5B). Nothing is dropped.
- Caveat: most templates sit at 1 to 3 successes per 16 samples. The bank is no longer at the floor, but
  it is still hard for this model, and per-template rates have wide uncertainty that I have not quantified.
  A policy that raises pass rates would have room to show it, and the margin to detect is not yet known.
- Speed: 1.5B takes 2.6x as long per task as 0.5B, in line with my earlier estimate of about 3x.
  LoRA timing for the 1.5B model has not been measured.

### 6f. LoRA update timing for Qwen2.5-Coder-1.5B (measured)

Script `scripts/stage2_lora_timing.py`, one configuration per process, results in
`results/stage2_lora_timing/*.json`. LoRA r=8, alpha 16, AdamW. Training text is a TRAIN prompt plus its
reference solution, padded to 300 tokens (natural length 94 to 286, mean 169). Times are per optimiser step
on a micro-batch, scaled to 8 sequences. 3 timed steps after a warm-up step. Content is irrelevant for timing;
no learning is measured.

| Config | Micro-batch | Grad ckpt | Time for 8 seqs | Peak RSS | Tokens/s |
|---|---|---|---|---|---|
| fp32, q/v | 1 | no | 58.0 s (steady state about 39 s, see note) | 9.5 GB | 41 (about 62 steady) |
| fp32, q/v | 4 | yes | 42.6 s | 9.5 GB | 56 |
| fp32, all linear | 1 | no | 37.7 s | 10.1 GB | 64 |
| **bf16, q/v** | 1 | no | **9.8 s** | **5.2 GB** | 246 |
| bf16, q/v | 4 | yes | 10.6 s | 6.2 GB | 226 |

- Note on the 58 s: the three steps were 4.7, 5.0 and 12.1 s, so one outlier step inflates the mean. Steady
  state is about 4.8 s per sequence, consistent with the all-linear run. Timing noise on this machine is
  large with only 3 steps; read these as order-of-magnitude figures.
- Memory finding: a single batch of 8 sequences x 128 tokens in fp32 ran out of memory and was killed (cgroup
  OOM at about 14 GB resident) before I switched to micro-batches. fp32 training should use micro-batches of 1.
- bf16 is about 4x faster than fp32 and uses about half the memory. The CPU has AMX bf16 and AVX512-bf16.
  What I have not checked: whether bf16 training or bf16 generation changes solution quality. The 6e
  dev results are fp32.
- Adapter (r=8, q/v, 1.09 M parameters): 4.4 MB on disk, saved in 0.12 s, loaded in 0.05 s, merged in 0.04 s.
- Generation with an adapter, fp32, batch 1: 0.146 s/token unmerged, 0.136 s/token merged (about 7 tokens/s),
  so merging is a modest gain. bf16 generation speed was not measured.
- Trainable parameters: 1.09 M (q/v), 9.2 M (all linear).

Implication for a pilot: a bf16 update of about 8 sequences costs about 10 s, so updates are cheap compared
with evaluation. Evaluation (sampled generation on dev tasks) dominates the budget.

## 7. Decision gates

Stage 2 continues past the smoke test only if all hold:

1. The smoke test runs end to end: generation, sandbox verification and one LoRA step. **Met.**
2. Base-model pass rates on dev-split tasks are neither near 0% nor near 100%, so the benchmark can
   discriminate between policies. **Supported with 1.5B, with caveats.** Sampled pass@1 is 0.212 (6e);
   12 of 13 dev templates have some success and none is saturated. Most templates have only a few
   successes in 16 samples. With 0.5B the gate was weak (pass@1 0.058, 8 of 13 templates flagged).
3. Measured step times make a reduced pilot feasible within a budget the user approves. **Measured (6f):**
   an 8-sequence bf16 update is about 10 s; evaluation dominates cost. bf16 quality effects are unchecked.
4. A paired-variance pilot on dev tasks is run before the H1 margin is pre-registered. **Not started.**

If gate 2 fails after a fair check, the honest result is "this task bank does not discriminate for this
model", and I report that rather than tune the bank until it does. Any change to the bank must be
justified on dev data only and logged in the protocol revision log.

## 8. Not implemented yet

- A reusable training loop (the smoke script only runs 3 LoRA steps on one example; no adapter saving, batching or evaluation harness)
- Gradient and influence-based selectors (specified in the protocol; need a model)
- The retrieval baseline
- Real token accounting
- A pre-registered H1 margin and the locked-test run

## 9. Proposed next steps (need your go-ahead)

The 1.5B model gives a usable base (section 6e), so I propose using it as the Stage 2 model. Next, in order:

1. ~~LoRA timing for 1.5B~~ done (6f).
2. **Plumbing dry run on dev**: collect verified episodes from the model's own sampled successes, apply one
   selection policy, take a few LoRA steps, re-measure dev pass rate. Purpose is plumbing and timing, not results.
3. **Paired-variance pilot** to size the H1 margin before pre-registering it. The pilot design has to be
   cut to fit the CPU budget (1.5B generation is about 8 s per attempt in this setup; the number of tasks,
   samples and seeds per policy will be set from measured times).
4. Pre-register H1, then (only then) consider the locked final test.

Everything above touches dev and train data only. The final-test split stays locked.

## 10. Plumbing dry run (settings fixed before the run; results in 10b)

Script `scripts/stage2_dry_run.py`, output `results/stage2_dry_run/`. **Purpose: exercise the full path and measure
its cost. Not an experiment.** One seed, tiny samples, no tuning, no hypothesis test. Whatever the numbers
look like, they must not be read as evidence about any policy.

Path: base model (bf16) samples attempts on TRAIN tasks, one phase per stream family (strings, lists, dicts,
math, dp); the sandbox verifies them; attempts that pass all tests become verified `Episode`s (failed attempts
stay unverified and the buffer's gate rejects them); the TRAIN buffer feeds a policy that picks replay
episodes under a token budget; a LoRA update trains on new + replayed episodes (completion tokens only);
after each phase the adapter is evaluated on DEV. `evaluation.run_stream` from Stage 1 is mirrored, with the
simulator replaced by the model. Policies run: `sequential`, `uniform`, `loss_fresh` (the last exercises the
`current_loss` hook with a real forward pass).

Fixed settings: Qwen2.5-Coder-1.5B, bf16; collection: 2 instances per TRAIN template (10 tasks per family) x 8
samples at T=0.8; 3 optimiser steps per phase; each step trains on up to 600 tokens of new verified episodes
plus up to 600 tokens of replay (token counts are chars/4); LoRA r=8, alpha 16, q/v, AdamW lr 1e-4, grad clip
1.0, micro-batch 1 with accumulation; buffer capacity 200; adapter seed 0 for every policy. Evaluation: 13 DEV
tasks (1 instance of each of 2 dev templates per stream family, plus the 3 `parsing` dev templates) x 4
samples at T=0.8, after every phase and once for the base model. Expected cost about 1.5 to 2 hours of CPU
(estimated from 6e and the bf16 generation benchmark: bf16 sampling is 2.2x (k=4) to 2.9x (k=8) faster than fp32
in aggregate tokens/s; unmerged adapter overhead about 8%).

Known limitations (decided in advance):

- The 52 evaluation samples per point give a standard error of about 0.05 to 0.07 at a pass rate near 0.2.
  Differences between policies of that size are noise, and there is one seed.
- DEV evaluation instances are the first instance of each dev template, the same ones in every evaluation.
  A separate probe set (dev instances at a different offset) is not used because none of these three
  policies reads a probe.
- Verified episodes are the model's own successful samples, so training on them is self-distillation of
  successes. Failed attempts have no correction in this dry run.
- Replay candidates are only earlier phases' verified episodes, as in Stage 1.
- bf16 quality effects are not validated; the base-model DEV score here (bf16) can be compared with 6e (fp32)
  only loosely.
- A crash test on the 0.5B model with tiny settings and reference solutions injected as verified episodes
  confirmed the code path (collection, gate rejection, replay only from earlier phases, 10 loss scoring
  passes counted for `loss_fresh`, adapter reset between policies). Its numbers are meaningless and not kept.

### 10b. Dry-run results (measured; plumbing and cost, not evidence about policies)

Run on 2026-10-31 after a container restart killed the first attempt during model loading (nothing was saved
from it; the run was restarted with identical settings). Files: `results/stage2_dry_run/` (`collection.jsonl`,
`run_*.json`, `summary.json`). FINAL_TEST untouched.

**Plumbing worked end to end.** 400 base-model attempts on TRAIN tasks were verified in the sandbox; 99 passed
(by phase: strings 17, lists 34, dicts 25, math 16, dp 7) and became verified episodes. The buffer rejected all
301 unverified attempts. Replay drew only from earlier phases. Adapters were reset between policies (same
init seed). `loss_fresh` made 708 real forward-pass loss evaluations.

| Policy | DEV pass after each phase (strings, lists, dicts, math, dp) | Final pass | Final tests passed |
|---|---|---|---|
| base (no adapter) | 0.096 | - | 0.295 |
| sequential | 0.077, 0.077, 0.096, 0.135, 0.115 | 0.115 (6 of 52) | 0.346 |
| uniform | 0.096, 0.096, 0.096, 0.096, 0.154 | 0.154 (8 of 52) | 0.393 |
| loss_fresh | 0.096, 0.077, 0.077, 0.115, 0.115 | 0.115 (6 of 52) | 0.388 |

**These differences are noise.** Each point is 52 samples with a standard error of about 0.04 to 0.05, there is
one seed, and a single sample changes the rate by 0.019. 6 vs 8 of 52 says nothing about the policies. No
hypothesis was tested and none of these numbers is a claim.

What the run does show:

- **Evaluation dominates cost.** Per policy run: evaluation about 1,590 s, training 110 to 125 s (15 optimiser
  steps), versus a total of about 1,700 s. Collection took 1,457 s and the base evaluation 275 s. The whole
  run took about 2.1 hours. A 5-point policy run costs about 28 minutes at this evaluation size, so a pilot
  of P policies x S seeds costs roughly 0.5 x P x S hours plus one collection. For example 6 policies x 3
  seeds is about 9 hours. Evaluation size, not training, sets the budget.
- **Fresh-loss scoring is expensive.** `loss_fresh` spent 252 s scoring candidates (708 forward passes,
  about 0.36 s each) against 108 s for the training steps themselves, 2.3x the training cost. In Stage 1 this
  is the `scoring_passes` column, and it is a real cost that the equal-budget comparison has to count.
- **Replay follows pool size.** `uniform` and `loss_fresh` replayed mostly strings and lists (26 and 25;
  27 and 24), the largest earlier pools, and rarely dicts and math (5 and 5; 6 and 3). Unbalanced replay is
  expected when the buffer pool is imbalanced; a coverage-based policy would behave differently.
- **Phase pools are small and uneven.** Only 7 verified `dp` episodes exist, so any `dp` training is on very
  little data. Success rates of the base model on TRAIN tasks differed a lot by family (strings 21%, lists
  43%, dicts 31%, math 20%, dp 9%).
- Peak memory 8.2 GB (bf16).

**Open problem: bf16 may degrade sampling quality.** On the same 13 DEV tasks (first instance of each dev
template), the earlier fp32 diagnostic (6e) gave 25 of 104 full passes (0.24) and 0.47 of tests passed. The
bf16 base model here gave 5 of 52 (0.096) and 0.295 (about 2.5 standard errors apart on pass rate; different random
seeds and sample counts). This looks like a real quality drop from bf16, though the evidence is weak. If it holds,
bf16 evaluation and bf16 training would not be comparable with the fp32 numbers, and the 4x speed gain from bf16
costs accuracy. Resolve this before any pilot: run the bf16 base model on the same 26-task, k=8 protocol as 6e
(about 10 minutes) and compare directly, and consider fp32 evaluation with bf16 training only if the cost is
acceptable.

Other limitations of the dry run (as decided in advance, plus two found):

- Training episodes are the model's own verified successes without any correction, so the update is
  self-distillation; a "new data" effect on unseen families is not expected from it.
- Training on 15 steps and 99 episodes is tiny. A flat result here would be unsurprising and says nothing.
- `sequential` (scope: current phase) replays other current-phase episodes, so it is "extra steps on current
  data", not zero extra data; this follows the Stage 1 definition.

## 11. Proposed next steps (need your go-ahead)

1. **Resolve bf16 vs fp32** (about 10 minutes, no download): rerun the 6e protocol (26 dev tasks, k=8,
   T=0.8) with the bf16 base model and compare per template with the fp32 result.
2. **Decide evaluation size and budget.** The pilot is limited by evaluation. Options: fewer, larger evaluation
   points (for example only before and after the stream), larger k on fewer tasks, or running on a GPU
   (needs a provider and a budget; I have not chosen one). I would want your budget in hours of CPU time.
3. **Paired-variance pilot** to size the H1 margin, using the cheapest design that still has acceptable standard
   error, then pre-register H1 before any final-test use.
4. Add a probe split (dev instances at a separate offset) before running any policy that reads a probe.
