# Experience Is All You Need: harness (Stage 1 simulated; Stage 2 foundation)

A dependency-free (stdlib only), CPU-only harness for testing experience-selection
policies on a stream of tasks. **Everything is simulated.** It validates software
correctness and baseline plumbing. It makes no claim about real coding models, about
VTAR being better than anything, or about novelty. See `RESEARCH_PLAN.md` and
`docs/experiment_protocol.md`.

## Setup

```bash
python3 -m pip install pytest      # only test dependency; Python >= 3.10
```

## Commands

```bash
python3 -m pytest -q                       # full test suite
python3 scripts/run_pilot.py               # writes results/pilot/{config.json,results.jsonl}
python3 scripts/summarize_results.py       # writes results/pilot/summary.md
python3 scripts/stage2_selfcheck.py        # Stage 2: isolation probe + reference solutions through the sandbox
```

`run_pilot.py` takes `--config` and `--out`. Each result row records config hash, scenario,
policy, seed, metrics, accuracy matrix and a UTC timestamp. Metrics are identical across
reruns; only timestamps differ.

## Layout

- `src/experience_learning/schemas.py` episode records; "verified" requires an external method
- `buffer.py` TRAIN-only reservoir buffer with verification gate; `LeakageError`
- `selection.py` frozen, sequential, uniform, loss (stale), loss_fresh, loss_per, loss_per_decay, novelty, CodeIt-style priority (adaptation), VTAR and ablations, and `oracle_gain` (privileged reference, simulator only)
- `metrics.py` accuracy, forgetting, forward transfer, regression rate
- `simulator.py` seeded world, hidden dynamics, locked final test set
- `evaluation.py` sequential runner and experiment grid
- `tests/` unit tests including leakage, determinism and equal-candidate checks

## What is simulated

One skill value per task family. Training raises the family's skill, moves others through a
hidden transfer matrix, and unprotected families drift back (forgetting). Unverified
corrections may be bad. Constants were set by hand.

## What is not validated

Any behaviour on real models or real code; the VTAR transfer and interference proxies
(the pilot gives no support for them); the retrieval baseline (specified for Stage 2, not implemented); the
CodeIt-style baseline is an adaptation, not a reproduction; hypothesis thresholds H1 (not yet pre-registered); the literature gap claim.

## Stage 2 (foundation only)

`src/experience_learning/stage2/` holds a sandbox for generated code (namespaces, private tmpfs,
unprivileged uid, resource limits; refuses to run if isolation is not strong), a bank of 52
function-level task templates with per-template train/dev/final-test splits, and episode collection with
external verification. Qwen2.5-Coder-0.5B was downloaded for a smoke test (outside the repo); there are no training
results beyond a smoke test (`scripts/stage2_smoke.py`, needs torch/transformers/peft, which are not repo dependencies). Status and measurements are in `docs/stage2_plan.md`.
