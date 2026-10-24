# Experiment protocol (Stage 1: synthetic harness)

Status: Stage 1 only. Nothing here is evidence about real coding models. The
method name VTAR is provisional; novelty and efficacy are unknown.

## What the simulator is

A deterministic world with one scalar skill per task family. A model "passes" a
task if a fixed per-task threshold `u` is below the family's skill, so every
policy is scored on the same tasks (paired design). Training on an episode
raises the skill of its family, shifts other families through a hidden transfer
matrix, and unprotected families drift back toward base skill (forgetting).
Unverified episodes can carry a bad correction that lowers skill.

Hidden from policies: per-episode difficulty, bad-correction flags, the true
transfer matrix, all skills. Policies see `Episode` fields, dev-probe items and
their own replay counts.

## Task sets and splits

| set | purpose | visible to policies |
|---|---|---|
| train stream (`Split.TRAIN`) | training and replay | yes |
| probe (dev) | VTAR transfer / learning-potential proxies | yes (stream families only) |
| stream-eval (dev) | accuracy matrix for forgetting, transfer, regression | no |
| final test (`Split.FINAL_TEST`) | one scoring at the end of a run | no |

The last family is a held-out transfer family: never trained on, excluded from
the probe, scored on stream-eval and final test.

Leakage defences, in depth: `ExperienceBuffer.add`, `Selector.select` and
`Learner.step` each raise `LeakageError` on a non-TRAIN episode; `Episode` is
frozen; `LockedTestSet.evaluate_once` can be called once and exposes no tasks;
`tests/test_no_leakage.py` audits every candidate and chosen set.

## Fair-comparison rules as implemented

- One stream per `(config, seed)`, shared by all policies (logged offline from the base model).
- Buffer admission (reservoir) uses its own RNG, so buffer contents do not depend on the policy.
- Same per-step budgets: new-data tokens and replay tokens. Sequential fine-tuning spends the replay budget on extra current-phase episodes.
- Verification gating applies to new data and replay together and is a property of the policy (`gate_verified`). Only `vtar_no_gating` turns it off.
- Final test is never read before the end of a run, and nothing was tuned on it. VTAR weights are fixed defaults and have not been tuned at all.

## Metric definitions

`R[t][f]` = stream-eval pass rate on family f after phase t; `R_init` = before any training; P = number of stream families.

- Average accuracy: mean over t of the mean of `R[t][0..t]`.
- Average forgetting: mean over f < P-1 of `max_{f<=t<P-1} R[t][f] - R[P-1][f]`. Not clipped, so it can be negative.
- Forward transfer: mean over t >= 1 of `R[t-1][t] - R_init[t]`.
- Regression rate: pooled over phases, of tasks in earlier families that passed before the phase's update, the fraction failing after.
- Final test pass rate: all families; also reported for stream families and the held-out family separately.

## VTAR proxies (not measured causal quantities)

See `selection.py`. N is coverage, L mixes logged loss and probe failure rate,
`T_hat` is probe-weighted embedding similarity, `I_hat` is negative cosine to
the current-phase centroid (a stand-in for gradient conflict), C is relative
token cost. Weights are fixed at 0.3/0.3/0.4/0.4.

## Assumptions and decisions (simplest reversible choice)

1. JSON configs instead of YAML, so the harness has no dependencies. Only `configs/pilot.json` exists; `final.yaml` belongs to Stage 4.
2. The retrieval/memory-only baseline is not implemented. Simulating retrieval means assuming how much a stored solution helps, which would build the answer into the simulator. It needs a real model.
3. The literature matrix (`docs/literature_review.md`) has a first-pass reading of the 11 brief papers; the systematic search is open. Its section "Changes this suggests" lists protocol changes not yet applied.
4. Python `random` seeded with strings is used for all randomness; results are identical across runs and processes.
5. Seeds 0-4 for the pilot. No variance estimate has been used to fix thresholds; the H1 thresholds are untouched.

## Pilot-time revisions (before any scientific claim)

Recorded because they were made after seeing pilot output. They fix plumbing
problems, not VTAR.

1. **Novelty baseline.** The first version balanced cumulative replay counts and
   sent a whole batch to one lagging family, which left other families
   unprotected that step. It now round-robins within each call, with cumulative
   counts as tie-break.
2. **Token value.** The first simulator made an episode's learning signal and
   protective effect independent of its length. VTAR's cost normalisation then
   replayed ~5.8 episodes/step versus ~3.5 for uniform and won by construction.
   Gain now scales with `(tokens/350) ** token_value_exp` (default 1.0) and
   protection uses token share. The `flat_token_value` scenario sets the
   exponent to 0 and is kept to show that effect; VTAR's results there are
   expected to be inflated.
3. **vtar_random** originally used the same RNG stream and shuffle as uniform
   and was identical to it. It now has its own stream.

## Scenarios

| name | purpose |
|---|---|
| `aligned` | embeddings predict true transfer; the regime VTAR's proxies are meant for |
| `proxy_uninformative` | true transfer independent of embeddings; proxies are noise |
| `no_transfer_clean` | no transfer, no bad corrections; coverage and gating should not matter beyond retention |
| `flat_token_value` | value independent of length; cost-aware selection favoured by construction |

## Reading the pilot

Observed in `results/pilot/summary.md` (5 seeds, synthetic): sequential fine-tuning
forgets heavily; every replay policy cuts it sharply; dropping verification gating
hurts when bad corrections exist and not when they do not; VTAR and novelty are
within noise of each other and both ahead of uniform; the VTAR advantage over
uniform is about the same in `no_transfer_clean`, where the transfer and
interference proxies carry no signal, and `vtar_no_interference` scores at least as
well as `vtar`. So the pilot gives no support for the transfer or interference
terms. It shows the harness detects forgetting, gating effects and policy
differences. Absolute numbers depend on simulator constants chosen by hand.
