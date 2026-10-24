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

## 3. Task bank and split rules (implemented)

`src/experience_learning/stage2/tasks.py`, `collect.py`.

- 30 templates in 6 families. Stream families: strings, lists, dicts, math, dp (5 templates each).
  Held-out family: `parsing` (5 templates, never trained on, used to test transfer to an unseen family).
- Each template generates instances from a seed. Each instance has 12 tests: 3 visible in the prompt,
  9 hidden.
- Splits are assigned per template, not per instance, so near-duplicate instances cannot straddle
  train and test. For seed 0: train 15, dev 7, final_test 8 templates. Stream families have
  3 train / 1 dev / 1 final-test templates; `parsing` has 2 dev / 3 final-test / 0 train.
- A reference solution exists per template and passes all of its own tests through the sandbox
  (checked by the self-check). A constant-`None` candidate is rejected.
- `collect_episode` produces an `Episode`. It is `verified` only if the solution it relies on
  passed every test in the sandbox. Otherwise it carries `Verification(False, "model_assertion")`,
  which the TRAIN buffer's gate rejects.

Known weaknesses:

- The templates are small, hand-written and synthetic-in-spirit. They are not a recognised
  benchmark. A pass rate here says little about general coding.
- The task families are a design choice. "Interference between families" is only meaningful if
  the real model shows measurable cross-family effects, and that is untested.
- Token counts use chars/4 as a placeholder. Real cost accounting needs the model tokenizer.
- The final-test templates exist but are locked: nothing scores them until the protocol is
  pre-registered.

## 4. Tests

`python3 -m pytest -q` runs 177 tests (89 from Stage 1, 88 for Stage 2). Stage 2 tests skip if the
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

Recommendation: `Qwen/Qwen2.5-Coder-0.5B` (base). It is the smallest permissively licensed
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

## 7. Decision gates

Stage 2 continues past the smoke test only if all hold:

1. The smoke test runs end to end: generation, sandbox verification and one LoRA step. **Met.**
2. Base-model pass rates on dev-split tasks are neither near 0% nor near 100%, so the benchmark can
   discriminate between policies. **Partly supported, still thin.** Sampled pass@1 is 10.7% and pass@8 is
   35.7% on 14 dev tasks, so there is neither a ceiling nor a total floor. Four of seven templates are
   sometimes solved; three are never solved in 16 samples. Seven templates is too few to call this
   settled, and the unsolvable templates need replacing or a stronger model.
3. Measured step times make a reduced pilot feasible within a budget the user approves. **Plausible on
   CPU**, pending timing at realistic sequence lengths and batch sizes.
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

## 9. Proposed next steps (need your go-ahead; no new downloads)

1. Widen the bank. Dev has 7 templates, 3 of which the model never solves, which leaves 4 informative ones:
   too few for a probe set, a stream-eval set and a variance estimate. Add templates in the stream
   families, with splits assigned before looking at model outputs for them, and keep dropping templates
   based only on pre-declared rules (for example "never solved in 16 base samples on dev"), logged in the
   protocol revision log.
2. Time a realistic LoRA update (batch of 8 sequences of about 300 tokens) and adapter evaluation.
3. A tiny end-to-end dry run on dev: collect verified episodes from the base model's own sampled
   successes, apply one policy's selection, take a few LoRA steps, re-measure dev pass rate. Purpose is
   plumbing and timing, not results.
4. Only then design the reduced pilot and pre-register the H1 margin from paired variance.

Steps 1 to 3 touch dev and train data only. The final-test split stays locked.
