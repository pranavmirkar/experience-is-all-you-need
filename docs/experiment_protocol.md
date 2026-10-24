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
2. The retrieval/memory-only baseline is not implemented in the simulator. Simulating retrieval means assuming how much a stored solution helps, which would build the answer into the simulator. It is specified for Stage 2 above.
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

## Added in the literature-informed update

- `codeit_priority`: **adaptation** of CodeIt's prioritised replay. Priority is the fraction of the logged
  attempt's test outputs that passed (floor 0.05), sampled weighted without replacement. No hindsight
  relabelling or program execution. In this simulator the attempt outcome carries no information about an
  episode's value, so it is expected to behave like uniform replay; it is a plumbing check for the baseline.
- `oracle_gain`: **privileged reference**, not a method. Greedy by measured marginal gain per token, where
  gain is the change in mean stream-family skill when the candidate is added to the actual batch, computed
  through a hidden-dynamics hook (`Learner.counterfactual`). It is the simulator counterpart of AuPair-style
  measured-gain selection. Only selectors with `needs_counterfactual` receive the hook (tested). It must not
  be used as an H1 comparator; it shows how much headroom a perfect proxy has.

## Stage 2 protocol (specified, not implemented)

These came from the literature pass (`docs/literature_review.md`) and bind any real-model run.

Baselines, all under the fair-comparison rules above:
1. Frozen, sequential fine-tuning, uniform, loss-prioritised, novelty/coverage replay (as in Stage 1).
2. CodeIt-style priority (adaptation; document every difference from the paper).
3. Cross-task memory baseline: store verified solutions or reflections from earlier tasks and retrieve by
   similarity for later tasks, no weight update. This is **not** Reflexion's per-task retry loop. Add a
   full-context in-context baseline (all prior experiences in the prompt up to the context budget), since
   CL-Bench found full-context ICL beat dedicated memory systems.
4. Gradient or virtual-update selectors as the stronger version of VTAR's proxies: a LESS-style
   gradient-similarity selector, a DualSFT-style validation-improvement selector with a prior-capability
   penalty, and a MIR-style selector (replay the stored examples whose loss rises most under a virtual update
   on the incoming batch). Report their selection cost next to VTAR's.
5. A dev-probe measured-gain reference (AuPair/SPEE-style): score a candidate by the change in pass rate on a
   held-out dev probe after a trial update or in-context use. Dev probes only, never the final test.

Metric rules:
- Forgetting is defined in `metrics.py`. Do not reuse the SWE-Bench-CL code: its v1 forgetting and backward
  transfer take the same dict as initial and final state and are 0 by construction.
- Pair every adaptive run with a stateless run of the same model on the same instances (CL-Bench "gain")
  and run several task orders.

H6 content control: with equal token and update budget, compare (a) the selected experiences, (b) the same
experiences with content shuffled across tasks, (c) irrelevant experiences, (d) no experience. Interventions
follow Zhao et al.; applying them to weight updates is this project's adaptation.

H1 pre-registration: do not fix the 3-point margin until a pilot gives the paired standard error of the
difference between methods on the actual benchmark. Procedure: run the strongest simple baseline and a
second copy of itself with a different seed on at least 5 seeds and 3 task orders; take the paired SE of the
pass-rate difference; set the margin to a value that SE can resolve (at least 2 SE), write it into
`configs/final.json` before any final-test run, and do not change it afterwards. Self-play SWE-RL reports
about 2% paired SE on SWE-bench Verified, which is a warning that 3 points may not be resolvable.

Per-update reporting (from the evolving-agents survey, Zhou 2026): for each update report the acquisition rate
G_t = P[score on a case rises] and the regression rate R_t = P[score falls], against a matched frozen copy
that sees the same observations and budget; keep the roles of update data, selection data and final
evaluation data disjoint; separate the cost of evolving (experience processing, candidate construction,
selection) from the cost of using the result, and report amortised cost per use. Add Selective Experience Use
(PATH-Bench: admit experience estimated to help, filter experience estimated to interfere) to the memory
baselines, since it is the frozen-weight counterpart of VTAR's idea. Compare pre-commit gating against
post-hoc rollback of admitted experience (Shang 2026).

Sealed verification: the tests that decide whether an experience counts as verified, and the final test, must be
authored and held outside the model being trained, with only pass/fail returned (Guo 2026, SEAL; Reflexion's 16.3%
false-positive rate; SkillLearnBench's drift under self-feedback). Self-written tests may be used for generation
but not for admission.

Staleness: run the `loss` baseline in two forms, priority from the loss logged at generation time and priority
recomputed on the current model each step, and add an age-decayed variant (FreshPER). Report all three; a stale
baseline is not a fair comparator.

Single-model claims: because in-context baselines tie or win in several 2026 benchmarks and gains vary by model,
report every Stage 2 comparison for at least two base models and include the full-context in-context baseline.
Report token cost next to success (SEA-Eval finds up to 31x differences at equal success).

Estimator validity: report the correlation between each proxy (T_hat, I_hat, gradient or probe scores) and
realised gain at 1, 5, 20 and 50 update steps. Sweeney (2026) finds order-prediction accuracy falling from
93% at 1 step to 65% at 50 steps on Llama-3.2-1B, and Piontkovskaia & Nikolenko (2026) find pairwise effects
fragile inside the first-order validity window.

Measurement noise: AuPair-style and probe-based utilities are noisy for small probes (trial-to-trial spread
is large in Sarukkai et al., Fig. 1). Size the probe from the pilot SE and report the estimator's rank
correlation with realised gain before using it to select.

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
