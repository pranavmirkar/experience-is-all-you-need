# Stage 2 plan: real code tasks, real model (status: foundation built, nothing downloaded)

Stage 2 moves from the simulator to executable coding tasks and a real small code model.
This document says what exists, what was measured, what is only estimated, and what needs
approval before it can proceed. No model, package or dataset has been downloaded or installed
for Stage 2. No paid API has been called.

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

## 6. Compute estimates (not measured)

These are back-of-envelope figures for planning. A smoke test replaces them with measurements.

- LoRA training costs about 4·N·T FLOPs per sequence (N ≈ 0.494e9 parameters, T ≈ 400 tokens),
  about 8e11 FLOPs. At an assumed 30–100 GFLOP/s on 4 CPU cores that is roughly 8–26 s per sequence.
- Generation on CPU is assumed to be about 5 tokens/s, so about 25 s for a 120-token attempt.
- A full pilot (several policies, several seeds, many attempts each) would therefore take
  tens of hours on this machine.

So the full pilot is not feasible here without cutting scale hard (fewer tasks, shorter
completions, fewer seeds, fewer policies) or moving to a GPU. A GPU run needs separate
approval and a cost estimate; I have not chosen a provider.

## 7. Decision gates

Stage 2 continues past the smoke test only if all hold:

1. The smoke test runs end to end: generation, sandbox verification and one LoRA step.
2. Base-model pass rates on dev-split tasks are neither near 0% nor near 100%. Otherwise the
   benchmark cannot discriminate between policies, and the task bank gets revised first
   (on dev data only).
3. Measured step times make a reduced pilot feasible within a budget the user approves.
4. A paired-variance pilot on dev tasks is run before the H1 margin is pre-registered.

If gate 2 fails, the honest result is "this task bank does not discriminate for this model",
and I report that rather than tune until it does.

## 8. Not implemented yet

- Model loading, generation, and LoRA training code
- Gradient and influence-based selectors (specified in the protocol; need a model)
- The retrieval baseline
- Real token accounting
- A pre-registered H1 margin and the locked-test run

## 9. Approval requested (nothing below has been done)

The brief requires approval before downloading weights or starting long runs. Proposed smoke test:

| Action | Detail | Cost |
|---|---|---|
| `pip install` CPU `torch`, `transformers`, `peft`, `accelerate` | CPU wheels, installed into a venv under the scratchpad, not globally | about 0.5–2 GB disk and network (not measured) |
| Download `Qwen/Qwen2.5-Coder-0.5B` | apache-2.0, 0.99 GB safetensors, not gated | 0.99 GB download and disk; roughly 2–4 GB RAM to load in fp32 (estimate) |
| Smoke test | generation speed; base pass rate on dev tasks; one LoRA step time | roughly 20 minutes of CPU; no paid API; no final-test tasks touched |

Total: about 1.5–3 GB of disk, within the 30 GB free. Nothing outside the session container is
modified. The smoke test never reads the final-test split.

Alternatives: a different model from the table (say which), a GPU run (needs a provider and budget),
or stopping at the current foundation.
