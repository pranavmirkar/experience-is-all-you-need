# Literature matrix

Status 2026-10-24: **first pass complete for the 11 papers in the brief; not a systematic review.**
Gap claims stay off until the open items at the end are done.

## Provenance and scope

All 11 papers were downloaded as PDFs and read as extracted text (arXiv for 7, PMLR/GitHub for 4).
Title, authors and venue/ID match the brief in every case. Two version notes: CODESKILL is arXiv v2
(25 Sep 2026); the SWE-Exp PDF is dated February 2026 (the brief says 2025 preprint).

What I read: abstract, method, experiments, ablations and limitation/discussion sections of each. I did
not read appendices in full (CODESKILL, GSE, SWE-Exp prompts and extra tables; SSR appendices only for
compute and limitations), Reflexion's HotpotQA/ALFWorld sections, or Voyager's related work. "Not stated"
means not stated in the text I read. I did not check any released code.

Cells give section or table references where it helps. Numbers are the papers' own, not reproduced.

## Matrix

| # | Work | Learning signal | Experience representation | Selection / retention | Weights change? | Test setup | Compute | Code |
|---|---|---|---|---|---|---|---|---|
| 1 | Reflexion (Shinn 2023) | binary/scalar evaluator reward; for code, self-generated unit tests (§4.3) | verbal self-reflection text; memory capped at 1-3 items (§3) | most recent reflections (sliding window) | No | HumanEval/MBPP (Py, Rust), LeetcodeHardGym (40); GPT-4 | not stated | yes (repo named) |
| 2 | Voyager (Wang 2023) | env feedback, execution errors, GPT-4 self-verification (§2.3) | executable JS skills, indexed by description embedding (§2.2) | GPT-4 curriculum picks next task; top-5 skill retrieval | No (GPT-4 API) | Minecraft, 3 runs/method, 160 iterations (Tab. 1) | GPT-4 cost named as limitation (§4) | not stated in text |
| 3 | CodeIt (Butt 2024) | program execution; hindsight relabelling of goal to realised output (§2.2) | (program, inputs, realised outputs) triplets | prioritised replay, priority = fraction of demo outputs matched (§2.2) | Yes (CodeT5+ 220M, NLL, 1 epoch/meta-iteration) | ARC train to eval set, 400 tasks, 100 meta-iterations, 3 seeds (Fig. 4, 6) | 100k samples per learning stage (§3.1); GPU-hours not stated | yes |
| 4 | SWE-Bench-CL (Joshi 2025) | none yet; proposes RAG memory agent (§6) | FAISS vectors of task summaries | top-k semantic retrieval, same-sequence first | No | 8 repos, 273 tasks, chronological + difficulty order (§3); **no results reported** | not stated | yes (repo named) |
| 5 | AuPair (Mavalankar 2025) | unit-test score of fix vs guess (§2.1) | (guess, fix) pair + scores as 1-shot example | greedy submodular selection on a validation fix-quality matrix, with residual subtraction (Alg. 2) | No (inference-time ICL) | 7 competitive-programming sets, 5 LLMs, N=32 calls; in/out-of-dist and cross-model (§3) | pair generation + \|C\|x\|Dval\| LLM calls; total not tabulated | not stated in text |
| 6 | SWE-Exp (Chen 2025/26) | patch validated against ground truth; failure cause annotated (§3.2) | "perspective" + "modification" experiences, LLM-distilled (§3.3) | embed issue type, top-10 then rerank agent picks k=1 (§4.4) | No | SWE-Bench-Verified 500; DeepSeek-V3 and Claude-4-Sonnet (Tab. 1) | 203K vs 189K tokens/instance, +37.5 s retrieval (Tab. 3) | anonymous link |
| 7 | Metacognitive position (Liu 2025) | n/a (position paper) | n/a | argues agents should decide what/how to learn | n/a | no experiments | n/a | n/a |
| 8 | CODESKILL (Li 2026) | verifier gain of skill-conditioned rollout over no-skill baseline, gated by an alignment judge, plus rubric quality reward (§3.3.2) | markdown skills, task-level and event-driven | learned add/merge/drop policy (Qwen3.5-4B, SFT then GRPO); dense retrieval | Skill manager yes; coding policy frozen | EnvBench, SWE-Bench Verified (150 held out), Terminal-Bench 2; two frozen policies (Tab. 1) | GRPO ~200 steps (Fig. 3); GPU-hours not stated | not stated in text |
| 9 | GSE (Yang 2026) | task outcome; local replay of the originating failure (§3.2) | skills + typed relation graph (dependency, co-usage, conflict) | cluster consolidation; accept bank only if replay of relevant cases does not regress (§3.3.2) | No (DeepSeek-V4-Flash API) | bug-revealing tests (108 Java methods, 9 projects), FP filtering (500 Go reports); one-project-held-out (§4.1) | evolution 401.6K tokens/case; execution 593K (Tab. 4, §5.4) | not stated in text |
| 10 | Self-play SWE-RL (Wei 2026) | binary: all tests pass; bug reward 1-(1+a)s for 0<s<1, penalty otherwise (Eq. 1-2) | self-generated bug artifacts (inject/weaken diffs, test script, parser) | learnability by solve rate s; higher-order bugs from failed attempts | Yes (CWM-sft 32B, RL, shared policy) | SWE-bench Verified (500), SWE-Bench Pro (731); one attempt each (§3.1) | 512 H100s per run (App.) | not stated in text |
| 11 | Unfaithful self-evolvers (Zhao 2026) | n/a (evaluates ExpeL, Dynamic Cheatsheet, ReasoningBank, G-Memory) | raw trajectories vs condensed insights | retrieval as in each framework | No (frozen backbones) | 13 backbones, 9 environments; interventions: empty, shuffle, irrelevant, corrupt, filler (§3) | A800 GPUs, details App. C | yes (repo in abstract) |

## Per-paper notes (what matters for VTAR)

**1. Reflexion.** The loop in Algorithm 1 repeats trials on one task; the stored reflections serve later trials
of that task. It is not formulated as a cross-task stream, so a memory-only baseline for this project has to
be designed (store/retrieve across tasks), not copied. For code, the evaluator is the model's own unit tests:
Table 2 reports a 16.3% false-positive rate on MBPP (tests pass, solution wrong) and Reflexion is below the
GPT-4 baseline there (77.1 vs 80.1, Table 1). Direct support for the brief's rule that model-generated checks
are not verification.

**2. Voyager.** Self-verification by GPT-4 is the most important component in its ablation (-73% items,
§3.4), so "verified" there means model-judged. Skill library shows plateau without it (§3.4). Skills are code,
so the memory-vs-weights question is answered by construction (no weight updates). Three runs per method.

**3. CodeIt.** The closest replay baseline. Priority is the fraction of demonstration outputs matched, which
favours experiences that solve real tasks. It is a success-based score with no transfer, interference or cost
term. Ablation A3 (uniform instead of prioritised): policy-only performance 38/400 vs 49/400, cumulative 58 vs
59 (Tab. 2), which the authors read as less forgetting. There is no explicit forgetting metric, 3 seeds, and
one domain (ARC with a DSL, CodeT5+ 220M). Hyperparameters were tuned on a custom train/validation split
(§3), test examples never enter the buffer (§2.2). Takeaway: a CodeIt-style priority is a required baseline
(brief §5.4), as a documented adaptation, not a reproduction.

**4. SWE-Bench-CL.** A benchmark and metric proposal; its own text says full evaluation is ongoing. The
preliminary harness runs gave pass rates mostly below 8.5% with harness mismatch, so no usable learning
result exists (§5). Useful: low inter-task patch similarity (mean cosine 0.18, §4.1), which suggests transfer
signals from surface similarity may be weak, and a prompt-poisoning probe showing irrelevant context shifts
outputs (§4.2). Metric note: the printed forgetting formula takes `max_{1<=k<=j} a_{k,j}`, which as written
includes k before task j is learned; I read it as a typo for k>=j and the repo should be checked. My
`metrics.average_forgetting` uses `max_{f<=t<P-1}`.

**5. AuPair.** Selection by *measured marginal gain on a validation set*, greedy with residual subtraction.
That is a realised counterpart of the brief's ideal utility U(e) (gain on held-out problems per experience),
bought with a large number of LLM calls and applied in context, not to weights. Their random-pair ablation
shows selection matters in-context (12 AuPairs match 32 random pairs, §3.2). They also note AuPair can make
individual fixes worse (§3.10). This is the main alternative to a proxy score: if VTAR's proxies cannot beat
or track an AuPair-style measured-gain selector, the proxies add little. Not stated: total selection cost.

**6. SWE-Exp.** Experiences are LLM-distilled from trajectories of an earlier SWE-Search run on the same
benchmark. Leakage control excludes same-repository experiences posterior to the target (§4.4, §6.1; the two
passages word the rule differently). One retrieved experience is best: 42.0% vs 39.8% (two) and 39.0% (four)
(§6.2). The text compares against 37.2% in §6.2 but Table 1 lists SWE-Search at 35.4% for the same model, so the
baseline of record is unclear. One example experience cites the golden patch (§3.3.3); whether extraction uses
the patch for failed trajectories is not stated. Limitation they state: no mechanism to judge applicability.

**7. Metacognitive position paper.** Conceptual only. Relevant claims: "what to learn" is a planning decision
(§5.3.1); fixed human-designed loops limit scaling; replay is named as an existing remedy for forgetting.
Evidence is case studies of STaR, Voyager and Generative Agents, which the authors call preliminary.
Their suggestion to test whether a controller can improve from its own errors maps to the brief's
metacognition question and is not tested anywhere I read.

**8. CODESKILL.** Utility is measured, not proxied: skill-conditioned verifier score minus a pre-cached
no-skill average, multiplied by an alignment factor from an LLM judge (§3.3.2). That is a counterfactual
ΔA estimate per skill with a frozen policy. The skill bank is built in two passes over the *test* instances,
with provenance filtering to exclude a skill derived from the current instance (§4.1); the conclusion lists
strictly streaming settings as future work. So it is transductive rather than a stream protocol. Gains: +11.03
average pass rate over no-skill, +5.10 over the strongest baseline (abstract). Seeds and variance: not stated.
Bank maintenance reduced skills from 1252 to 676 with the same or better pass rate (Tab. 2).

**9. GSE.** Adds a retention gate: a candidate skill bank is accepted only if replaying relevant historical cases
maintains or improves performance (§3.3.2), plus a relation graph for conflicts. One-project-held-out protocol
(§4.1), temperature 0. Small evaluation sets (108 methods; 500 reports), and the largest gains come with low
baselines (e.g. OpenHands F1 0.08 base). Task type differs from coding-model pass rate (precision/recall/F1).

**10. Self-play SWE-RL.** Weights are updated by RL at large scale (512 H100s per run). The training-task
selection signal is learnability: reward peaks for solve rates strictly between 0 and 1. Their own ablation
finds solver feedback in the injector reward gave only a slight, "largely negligible" advantage over the
consistency-only reward (§3.5). Reports about 2% paired standard error on SWE-bench Verified, which is the
same order as the brief's 3-point H1 threshold. Reports training instability at scale (App. E).
Closest to "experience generation", which the brief deliberately leaves out of scope.

**11. Unfaithful self-evolvers.** Method for the H6 faithfulness test: intervene on experience content
(empty / shuffle / irrelevant / corrupt / filler) and measure behaviour change. Finding: agents depend on raw
trajectories but largely ignore condensed insights, even when condensed is the only experience (§4.2).
All frameworks are frozen-weight. For weight-updating learners the analogue is training on shuffled or
irrelevant experiences under an equal token and update budget; that analogue is mine, not the paper's.

## Cross-cutting observations for this project

1. **Frozen-weight dominates.** 8 of 11 papers do not change the model's weights (memory, skills, in-context
   pairs). Only CodeIt and SSR update weights, and neither studies selection by estimated transfer. Any
   fine-tuning study therefore needs a memory/skills baseline, as the brief says.
2. **Measured utility exists in-context.** AuPair (validation marginal gain) and CODESKILL (gain over no-skill)
   both measure what a stored item adds. I found no paper that does this for *weight updates* on stream data
   or approximates it cheaply. This is the only candidate gap so far, and it is based on a limited reading.
3. **Replay evidence is thin.** The only replay-vs-uniform ablation I read is CodeIt A3 (3 seeds, ARC).
   GSE's replay is a validation gate, not a training replay.
4. **Verification is mostly model-judged** (Reflexion tests, Voyager critic, CODESKILL alignment judge,
   SWE-Exp rerank). External verification is used in SWE-Exp, SSR, AuPair, GSE and CodeIt.
5. **Variance is under-reported.** Seeds are not stated for most; SSR states about 2% noise; Voyager and CodeIt
   use 3 runs. The H1 threshold of 3 points should be treated as untested until a pilot gives paired variance.
6. **Leakage handling differs** (repo/time exclusion in SWE-Exp, provenance filter over a shared test pool in
   CODESKILL, held-out project in GSE). The harness's split rules (family-level, locked final test) are stricter
   than any of these.

## Changes this suggests for the plan (not yet applied)

- Add a CodeIt-style priority (matched-fraction) policy to the simulator baselines, labelled an adaptation.
- Add an AuPair-style measured-marginal-gain selector as an oracle-ish reference in Stage 2, since it is the
  nearest thing to the ideal U(e).
- Keep the retrieval/memory baseline for Stage 2 and define it as cross-task retrieval, not Reflexion's per-task retry.
- Pre-register the H1 margin only after a paired-variance pilot; SSR's 2% noise is a warning.
- Add an experience-content control (shuffled / irrelevant experiences, equal budget) to the Stage 2 protocol for H6.
- Check SWE-Bench-CL's repo for the forgetting definition before adopting it.

## Open items (needed before any gap claim)

1. Systematic search beyond the brief. Candidates surfaced by two quick searches, **unread**:
   Possibility and Pass-rate Prioritized Experience Replay for code generation (arXiv 2410.12236);
   Snorkel AI "Continual Learning Bench" (June 2026, dates inconsistent across pages);
   SPEE (explicit experience evolution then policy optimisation); Self-Evolving GPT (ACL 2024);
   arXiv 2505.00234 (in-context experience selection for agents; title not checked).
2. Read appendices for compute and seeds (CODESKILL, GSE, SSR, SWE-Exp).
3. Check released code for each paper and for Snorkel's benchmark.
4. Check later versions of the 2026 preprints.
5. Search specifically for influence/gradient-based or counterfactual selection of replay data for LLM
   fine-tuning (my searches did not find any, which is weak evidence).
