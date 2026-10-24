# Literature matrix

Status 2026-10-24: **fourth pass.** 11 brief papers read, plus 21 further papers found by search (read at the depth marked in each table; abstract-only entries are flagged).
This is still not a systematic review (search log and open items at the end). Gap statements are
provisional and marked as such.

## Provenance and scope

Papers were downloaded as PDFs and read as extracted text. In every case the latest arXiv version was read
(checked against each paper's submission history). Title, authors and venue of the 11 brief papers match the brief.

What I read: abstract, method, experiments, ablations and limitation/discussion sections. Not read in full:
appendices (most papers), Reflexion HotpotQA/ALFWorld, Voyager related work, CL-Bench appendices, LESS
appendices. "Not stated" means not found in the text I read; for seeds and compute I also searched the
extracted text for "seed", "runs", "confidence interval", "GPU" and similar. Numbers are the papers' own.
Code status comes from the PDF text, the arXiv abstract page (comments field), or the README I fetched;
I did not run or inspect released code except the SWE-Bench-CL metric file noted below.

Version notes: CODESKILL v2 (25 Sep 2026); SWE-Exp v2 (2 Feb 2026; the brief says 2025);
DualSFT v2 (22 Oct 2026, "Accepted by NeurIPS 2026" per arXiv page). SWE-Bench-CL has only v1.

## A. Papers listed in the brief

| # | Work | Learning signal | Experience representation | Selection / retention | Weights change? | Test setup | Compute | Seeds / variance | Code |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Reflexion (Shinn 2023) | evaluator reward; for code, self-generated unit tests (§4.3) | verbal reflection, memory cap 1-3 (§3) | most recent reflections | No | HumanEval/MBPP (Py, Rust), LeetcodeHardGym (40), GPT-4 | not stated | not stated | yes (github.com/noahshinn024/reflexion) |
| 2 | Voyager (Wang 2023) | env feedback, errors, GPT-4 self-verification (§2.3) | executable JS skills, description-embedding index | GPT-4 curriculum; top-5 skill retrieval | No | Minecraft, 160 iterations | GPT-4 cost named a limitation | 3 runs/method (Tab. 1) | yes ("open-source codebase", arXiv comments) |
| 3 | CodeIt (Butt 2024) | execution; hindsight relabelling (§2.2) | (program, inputs, realised outputs) | prioritised replay, priority = matched fraction of demo outputs | Yes (CodeT5+ 220M) | ARC, 400 eval tasks, 100 meta-iterations | 100k samples/learning stage; GPU-hours not stated | 3 seeds (Fig. 4, 6) | yes (github.com/Qualcomm-AI-research/codeit) |
| 4 | SWE-Bench-CL (Joshi 2025) | none yet (RAG agent proposed) | FAISS vectors of task summaries | top-k semantic retrieval | No | 8 repos, 273 tasks; **no learning results** | not stated | n/a | yes (github.com/thomasjoshi/agents-never-forget) |
| 5 | AuPair (Mavalankar 2025) | unit-test score of fix vs guess | (guess, fix) pair as 1-shot example | greedy submodular on validation fix-quality matrix | No (in-context) | 7 contest sets, 5 LLMs, N=32 | pair gen + \|C\|x\|Dval\| calls; total not tabulated | not stated | no link found in PDF or PMLR page |
| 6 | SWE-Exp (Chen 2025/26) | patch validated vs ground truth | perspective + modification experiences (LLM-distilled) | embed top-10, rerank agent picks k=1 | No | SWE-Bench-Verified 500; DeepSeek-V3, Claude-4-Sonnet | 203K vs 189K tokens/instance | not stated | yes (github.com/YerbaPage/SWE-Exp, arXiv comments) |
| 7 | Metacognitive position (Liu 2025) | n/a | n/a | argues agents should choose what/how to learn | n/a | no experiments | n/a | n/a | n/a |
| 8 | CODESKILL (Li 2026) | skill-conditioned verifier gain over no-skill, gated by alignment judge, plus rubric reward | markdown skills (task-level, event-driven) | learned add/merge/drop policy | Skill manager yes; coding policy frozen | EnvBench, SWE-Bench Verified (150 held out), Terminal-Bench 2 | 4xH100 for training; ~200 GRPO steps | not stated | no link found |
| 9 | GSE (Yang 2026) | task outcome; replay of originating failure | skills + typed relation graph | accept bank only if replay of relevant cases does not regress | No (API LLM) | 108 Java methods, 500 Go reports; one-project-held-out | 401.6K tokens/case evolution | temperature 0; seeds not stated | no link found |
| 10 | Self-play SWE-RL (Wei 2026) | all-tests-pass; bug reward peaks at 0<solve rate<1 | self-generated bug artifacts | solve-rate learnability; higher-order bugs | Yes (CWM-sft 32B, RL) | SWE-bench Verified 500, SWE-Bench Pro 731 | 512 H100s per run | ~2% paired SE stated (App.) | no link found |
| 11 | Unfaithful self-evolvers (Zhao 2026) | n/a (evaluates 4 frameworks) | raw trajectories vs condensed insights | per framework | No | 13 backbones, 9 environments | A800 GPUs | not stated | yes (github.com/Dreamcatcher0622/Faithfulness) |

## B. Papers found by search (all read as above)

| # | Work | Relevance | What it does | Weights change? | Test setup | Seeds / variance | Code |
|---|---|---|---|---|---|---|---|
| 12 | PPER / BTP (Chen 2024, arXiv 2410.12236v2) | prioritised replay for code LLMs | priority = a*P(program) + (1-a)*test pass rate; rank-based sampling (§3.4) | Yes (GPT-2, GPT-Neo, WizardCoder-34B) | APPS, CodeContests, HumanEval | not stated; alpha tuned per dataset (App. B/C not read) | no link found |
| 13 | Self-generated in-context examples (Sarukkai 2025, 2505.00234v3) | measured utility of stored experiences | only successful trajectories kept; exemplar value Q(tau) = retrieval-weighted outcome (Eq. 1) | No | ALFWorld, Wordcraft, InterCode-SQL; gpt-4o-mini | 5 trials; large trial-to-trial variance with identical procedure (Fig. 1) | no link found |
| 14 | CL-Bench (Asawa 2026, 2606.05661, NeurIPS 2026) | benchmark + paired "gain" metric | gain g_t = reward stateful - reward of same system stateless on same instance (§4.2) | No (context-based systems only) | 6 domains incl. software engineering; Claude, GPT-5.4, Gemini | 5 rollouts in varied orders, 95% CIs | project website named; not checked |
| 15 | SPEE (Ren 2026, 2608.02139v2) | measured marginal utility gate, then weight updates | candidate experience kept if mean reward gain on a held-out probe set exceeds eps (Eq. 9-11), then distilled (OPSD) and RL (GRPO) | Yes | math only: Qwen3-1.7/4/8B-Base, DAPO-math-17k, 5 benchmarks | not stated | github.com/rrrsj/SPEE per a search result; not verified |
| 16 | DualSFT (Chen 2026, 2605.06166v2, NeurIPS 2026) | closest utility form to VTAR | data utility = local validation improvement of L_new + lambda*L_prior from gradients (Eq. 4-7); one-shot top-b after a 5% warmup | Yes (SFT) | Magicoder (code), MetaMathQA, Alpaca; Llama-3.2-3B, Gemma-3-4B, Qwen-3.5-9B; plasticity = HumanEval, stability = 4 out-of-domain benchmarks | 3 seeds, mean +/- std | stated available; link not seen |
| 17 | LESS (Xia 2024) | gradient-based influence selection | LoRA-gradient similarity of train examples to few-shot target examples, Adam-aware; top 5% | Yes | MMLU, TyDiQA, BBH; Llama-2-7B/13B, Mistral | not checked in detail | yes (github.com/princeton-nlp/LESS) |
| 18 | MIR (Aljundi 2019) | measured interference for replay | replay the stored samples whose loss rises most under a virtual SGD step on the incoming batch (§3.1) | Yes | MNIST splits, CIFAR-10, MiniImagenet | 15-20 runs | yes (github.com/optimass/Maximally_Interfered_Retrieval) |
| 19 | RHO-LOSS (Mindermann 2022) | holdout-loss data selection | training loss minus irreducible loss from a holdout-trained model | Yes | Clothing-1M, CIFAR, BERT; i.i.d., not continual | across seeds (Fig. 1) | not found on arXiv page |
| 20 | FOREVER (Feng 2026, 2601.03938v2, ACL 2026) | when to replay | schedule driven by optimiser update size; 2% of each task stored | Yes (LoRA) | 3 NLP CL benchmarks; Qwen3-0.6B, 4B | 3 runs / several task orders | yes (link in PDF) |
| 21 | Forgetting in LMs (Marek 2026, 2605.26097) | replay without stored data | KL penalty on model-generated samples nearly removes forgetting; capacity limits it | Yes | pretrained small LMs; Llama-3.2-1B-Instruct on Verilog | not checked | yes (github.com/martin-marek/forgetting) |

### B2. Replay and continual-learning papers added in the third pass

| # | Work | What it does | Weights change? | Test setup | Seeds / variance | Code |
|---|---|---|---|---|---|---|
| 22 | SuRe (Hazard 2025, 2511.22367v1) | replay buffer keeps the highest-NLL sequences with an equal per-task quota (Eq. 4); optional dual fast/slow LoRA | Yes (LoRA, T5) | Standard CL (4 text-classification sets) and 15-task LNT; buffer = 2% of data | means over 3 runs x 3 task orders (Tab. 4-5) | not checked |
| 23 | REPEAT (Gao 2023, ICSE, 2302.03482) | exemplars: keep the lowest-loss candidates, then pick diverse ones by TF-IDF clustering; plus adaptive parameter regularisation | Yes (CodeBERT, CodeT5) | code summarisation, vulnerability detection, clone detection; vs random exemplars (EMR) and EWC | not checked | not checked |
| 24 | InsCL (Wang 2024, NAACL, 2403.11435) | replay amount per earlier task by Wasserstein task similarity, data picked by an instruction-information score | Yes (LLaMA-7B) | 16 tasks, several orders; +3.0 relative gain over random replay, +27.96 over no replay (abstract) | 3 training orders averaged | not checked |
| 25 | CodeTask-CL (Yadav 2023, 2307.02435) | benchmark for continual learning on code generation/translation/summarisation/refinement; compares CL methods | Yes | CodeT5 + experience replay is a strong baseline; prompt pooling forgets; PP-TF fixes it | not checked | yes (github.com/amazon-science/codetask-cl-pptf) |
| 26 | GrADS (Liu 2025, 2511.08620v1) *(skimmed)* | gradient-magnitude data selection for domain SFT, with a forgetting claim | Yes | CMedQA, LawQA, FinQA; Qwen/ChatGLM/Llama; baselines Random, LESS, others | not checked | not checked |

### B3. Transfer-estimation and gradient-selection leads, followed up in the fourth pass

| # | Work | What it does | Weights change? | Test setup | Seeds / variance | Code |
|---|---|---|---|---|---|---|
| 27 | Lie-bracket order prediction (Sweeney 2026, 2606.24993, ICML 2026) | pairwise score from the commutator of two sources' gradient updates predicts whether A then B or B then A gives lower target loss; one Hessian-vector product per source | Yes (small-step SGD ground truth; last 8 layers of Llama-3.2-1B) | Qwen2.5-1.5B, Llama-3.2-1B, Llama-3.1-8B; instruction-SFT, DPO, Pile domains, 85 Stack languages for a Python target, 56 MMLU subjects | 4-5 seeds in the tables I read; 1,020 Llama trials in Tab. 3 | not checked |
| 28 | "First-order predictable but pairwise fragile" (Piontkovskaia & Nikolenko 2026, 2607.16821) | prospectively registered measurement of eight local-approximation properties around a multitask LoRA checkpoint | LoRA perturbations | nine transformers, 82M-7B, seven families | registered thresholds and dev/test split | not checked |
| 29 | HESIT (Chen 2024, ACL Findings, 2405.10992) | exemplar selection by hyper-gradient influence on validation performance, without a Hessian; for episodic replay | Yes (GPT-2) | task-oriented dialogue CL, 37 domains; vs random, IFORM, GSS, ARPER, A-GEM, LAMOL, EWC | not stated in the part read | not checked |
| 30 | HCTOS (Nguyen 2025, 2502.06544) | sequence-level forward/backward transferability measures (TFT, TRT) and a heuristic to choose task order | Yes | Split CIFAR-10, CIFAR-100, tiny-ImageNet; not language | not checked | not checked |
| 31 | Weyssow et al. (FSE 2023, 2305.04106) | stream of programs with new, unseen APIs; compares replay (buffer of 200) and EWC/SI/RWalk | Yes (GPT-2-style decoder, RoBERTa-style encoder) | API call and API usage prediction | not checked | not checked |
| 32 | For-Value (Deng 2026, 2508.10180v3) *(abstract only)* | forward-only data valuation for fine-tuning | n/a | LLMs and VLMs | not read | not checked |
| 33 | DataInf (Kwon 2024, ICLR, 2310.00902) *(search summary only, not read)* | closed-form influence approximation for LoRA-tuned models | n/a | not read | not read | not read |

## Notes that matter for VTAR

**SuRe (22), REPEAT (23).** They pull in opposite directions on loss. SuRe replays the *highest*-loss
sequences and beats reservoir replay on the 15-task benchmark (72.1 vs 69.1, Tab. 1). REPEAT filters to the
*lowest*-loss candidates before picking diverse ones, to avoid noisy samples. So a loss-prioritised baseline
can go either way, and both directions belong in Stage 2. SuRe's own ablation also shows that a
task-balanced random buffer updated after each task is close behind surprise: at replay ratio 1:2, 73.35
(random, after) vs 74.22 (surprise, after) (Tab. 5). Much of the gain over reservoir replay therefore comes
from balancing and timing, which my `novelty` baseline and the shared reservoir buffer partly reflect.
Neither paper has a held-out task family or code agents.

**CodeIt (3).** Priority is the matched fraction of demo outputs, success-based, with no transfer,
interference or cost term. Ablation A3 (uniform): policy-only 38/400 vs 49/400, cumulative 58 vs 59 (Tab. 2),
read by the authors as less forgetting. No explicit forgetting metric; 3 seeds; one domain; hyperparameters
tuned on a custom split (§3). Required baseline, as an adaptation.

**PPER (12).** The only other prioritised-replay-for-code paper found. Its priority mixes model likelihood
and pass rate. In the text I read there is no uniform-replay ablation, and the "APPS mixed" set is described
as used "to train and evaluate" without a stated split (§4.2), so I would not treat its gains as evidence of
prioritisation helping. Single-shot fine-tuning, not a stream.

**AuPair (5), CODESKILL (8), SPEE (15), exemplar curation (13).** Four papers estimate an experience's value
by *measuring* what it adds on held-out items: AuPair on a validation matrix, CODESKILL against a no-skill
baseline, SPEE on a probe set, (13) by retrieval-weighted outcomes. Three of them keep the model frozen.
Only SPEE then updates weights, and only on maths. In (13), identical procedures vary a lot across 5 trials
and some added trajectories lower performance, which bears on measurement noise for any such estimate.

**DualSFT (16), LESS (17), RHO-LOSS (19), MIR (18).** These are the closest prior work to VTAR's score.
DualSFT's utility is a first/second-order validation-improvement estimate with a prior-capability penalty,
the same form as U(e) = [dA - lambda*dF] / C without the cost term, applied to code SFT with plasticity
(HumanEval) and stability (four benchmarks). MIR measures interference by a virtual update instead of an
embedding cosine, and reports ER-MIR vs ER forgetting 7.0 vs 15.0 on MNIST-split over 20 runs (Tab. 1).
RHO-LOSS selects by holdout-loss reduction for i.i.d. training. None of these works on a stream of verified
coding experience with an equal-budget comparison to uniform, loss and CodeIt-style replay (as far as found).
VTAR's embedding-cosine T_hat and I_hat are cheaper and cruder stand-ins for quantities these papers compute
from gradients or virtual updates.

**SWE-Bench-CL (4).** No learning results. Its printed forgetting formula takes `max_{1<=k<=j}`, which I
read as a typo for k>=j. Checking `eval_v1/eval_procedure.py` at the main branch on 2026-10-24:
`calculate_forgetting` and `calculate_backward_transfer` are called with the same results dict as both
"initial" and "final" (lines ~1359-1367), so both are 0 by construction in that version; forward transfer is
the pass-rate difference on the next task between a memory-enabled and a no-memory run. I did not inspect
`eval_v2_agent` or `eval_v3_agent`. Do not adopt its code for metrics.

**CL-Bench (14).** Useful design: pair each stateful run with the same system run statelessly on the same
instance (gain), several rollouts in different orders, CIs. In its results full-context ICL is best
(normalised gain 25.4%) and a dedicated memory system (ACE) costs the most ($62.8/run) for less gain (8.6%).
Context-based systems only; parametric methods are listed as not evaluated. Supports a strong
"just keep it in context" baseline for Stage 2.

**SWE-Exp (6).** One retrieved experience is best (42.0% vs 39.8% with two, 39.0% with four, §6.2). Text
baseline 37.2% vs Table 1 SWE-Search 35.4%: unresolved. Leakage rule: exclude same-repository experiences
posterior to the target. A §3.3.3 example cites the golden patch; whether extraction uses it is not stated.

**Self-play SWE-RL (10).** Solver feedback in the injector reward gave a "slight and largely negligible"
advantage (§3.5). About 2% paired SE on SWE-bench Verified, comparable to the brief's 3-point H1 margin.

**Reflexion (1), Voyager (2), Zhao (11).** Reflexion's test-based evaluator had a 16.3% false-positive rate
on MBPP (Tab. 2) and its loop is per-task retry. Voyager's self-verification is model-judged and its most
important component (-73% items, §3.4). Zhao shows intervention controls (empty/shuffle/irrelevant/
corrupt/filler) for frozen agents; the weight-update analogue is mine.

**Lie-bracket (27), "pairwise fragile" (28).** These bear directly on whether a cheap local estimate of
transfer or interference can be trusted. In (27) the order prediction is made once at the starting weights
and then checked against actual training: Llama-3.2-1B pairwise accuracy falls from 93.0% at k=1 step per
domain to 81.5% at k=20 and 65.3% at k=50 (Tab. 3, 1,020 trials). It also ranks 85 Stack programming
languages for a Python target in the 99th sampled percentile at k=1 but 86th at N=10, k=5 (the table beside Tab. 3; table number not checked).
The unit being ordered is a whole source domain, not a single experience. (28), a prospectively registered
study, finds first-order predictions of single-direction loss changes hold across its grid, but that
order sensitivity between two updates sets in inside that window on over a third of (model, task pair)
combinations, and states that "sample-stable prediction of task effects remains open". Together they
argue that VTAR's T_hat and I_hat (and DualSFT's local surrogate) should be validated against realised
gain at the training horizon actually used, not only at one step.

**HESIT (29).** Hyper-gradient influence selection for GPT-2 dialogue CL beats random exemplars by 6.70%,
5.94%, 13.33% and 7.86% relative on four metrics (§5.1), but random exemplars with replay already reach
79.92 intent accuracy against 2.65 for vanilla sequential tuning (Tab. 1), and the other selection baselines
(IFORM 81.10, GSS 81.36) sit within about 1.5 points of random. Replay does most of the work; selection adds
a few points. Earlier in this review I listed HESIT as an instruction-tuning method; it is a dialogue-system method.

**Weyssow et al. (31).** Evidence that code models forget earlier APIs under plain fine-tuning and that
simple replay with a small buffer reduces it. Supports random replay as a strong default for code.

**HCTOS (30), For-Value (32), DataInf (33).** HCTOS chooses task order, in vision benchmarks, so it is
background only. For-Value and DataInf are cost-oriented influence estimators; I did not read their method
sections, so I make no claims beyond what their titles and abstracts say.

**CodeTask-CL (25), InsCL (24).** Show that replay with random exemplars is already a strong default for
code models (CodeT5 + ER) and that smarter allocation across earlier tasks (InsCL) adds a few points over
random replay in instruction tuning. Both use fixed task sequences, not verified agent experience.

**Liu (7).** Position paper; "what to learn" as planning; no experiments.

**GSE (9).** Retention gate by replaying relevant historical cases before accepting an update.

**FOREVER (20), Marek (21).** Address *when* to replay and *whether stored data is needed*, not which
examples to replay; Marek's code-adjacent result (Verilog fine-tuning of Llama-3.2-1B-Instruct) is a
replay-helps data point.

## Cross-cutting observations (provisional)

1. **Weights.** Of the 21 papers, weights are updated in CodeIt, PPER, SSR, SPEE, DualSFT, LESS, MIR,
   RHO-LOSS, FOREVER, Marek (and CODESKILL's manager); the rest are frozen-weight. Fine-tuning work on
   code selection exists (PPER, DualSFT) but is not a stream.
2. **Measured utility is common; proxies are the novelty-poor part.** Measuring marginal gain on held-out
   items appears in AuPair, CODESKILL, SPEE, (13), DualSFT, MIR. A cosine-similarity proxy is a weaker
   version of these.
3. **Continual learning for code models exists** (CodeTask-CL, REPEAT, API-evolution streams), with
   replay baselines, but on fixed task sequences for encoder or seq2seq models, not selection of verified
   agent experience.
4. **Narrowest candidate gap (unconfirmed).** Equal-budget comparison of transfer- and interference-aware
   *selection of verified coding experience for weight updates in a stream*, against uniform, loss-,
   novelty- and CodeIt-style replay, with forgetting and held-out-family transfer reported. I did not find
   this in 33 papers, but the search is limited (see log).
5. **Variance.** Most papers do not state seeds. Where stated: 3 (CodeIt, Voyager, DualSFT, FOREVER),
   5 (13, CL-Bench), 15-20 (MIR). SSR reports ~2% evaluation noise; (13) shows large trial-to-trial spread.
   The 3-point H1 margin remains untested.
6. **Verification.** Model-judged checks: Reflexion tests, Voyager critic, CODESKILL alignment judge,
   SWE-Exp rerank. External: SWE-Exp, SSR, AuPair, GSE, CodeIt, PPER (tests), SPEE (reward).
7. **Leakage handling differs** (repo/time exclusion in SWE-Exp; provenance filter over a shared test pool
   in CODESKILL; held-out project in GSE; PPER unclear). The harness split rules are stricter.

## Plan changes: status

| Change | Status |
|---|---|
| CodeIt-style priority baseline in the simulator, labelled an adaptation | **Applied** (`codeit_priority`, `selection.py`, tests) |
| Measured-marginal-gain reference selector (AuPair-style) | **Applied in the simulator** as `oracle_gain`, a privileged reference using hidden dynamics. The real-model version (dev-probe measured gain) is specified in `docs/experiment_protocol.md` for Stage 2 |
| Cross-task retrieval/memory baseline, defined as such, plus full-context ICL | Specified for Stage 2 in the protocol (not implementable honestly in the simulator) |
| Pre-register H1 margin only after a paired-variance pilot | Procedure written in the protocol |
| Shuffled / irrelevant experience control (H6) | Specified for Stage 2 in the protocol |
| Check SWE-Bench-CL's forgetting definition | **Done** (finding above); do not reuse its metric code |
| New, from this pass: gradient/virtual-update baselines (LESS-, DualSFT-, MIR-style) | Added to the Stage 2 baseline list in the protocol |

## Search log

Queries run (standard unless noted): continual learning coding agents replay selection benchmark;
learning which experiences to replay / influence-based selection for code; MIR; RHO-LOSS; LESS;
replay data selection for LLM fine-tuning (extended); Snorkel Continual Learning Bench; SPEE;
gradient-based replay selection in continual instruction tuning; continual learning for code models
(API evolution); data valuation / influence for self-generated samples; forward/backward transfer
prediction and task-order selection.
Fourth-pass queries: exact-title searches for the Lie-bracket paper, HESIT, HCTOS, the API-stream code paper,
and DataInf/For-Value/LoGra (LoGra returned nothing).
Results used as leads only; every paper in sections A, B, B2 and B3 was downloaded and read at the depth noted
(GrADS skimmed; For-Value abstract only; DataInf not read).
The data-valuation and self-improvement-selection query returned no coding-specific method; weak evidence
of absence only.

## Open items

1. Still unread or only skimmed: DataInf and For-Value methods, GrADS, LoGra (not found), Continual Learning
   From a Stream of APIs (2309.00023; a different setting, black-box APIs), "Optimal Task Order for Continual
   Learning of Multiple Tasks" (ICML 2025). Searches not done: curriculum or learnability selection for code RL
   beyond SSR; cited-by searches on CodeIt, AuPair, DualSFT, SWE-Exp, SuRe.
2. Appendices: PPER alpha tuning (does it use test data?), CODESKILL and GSE seeds, DualSFT code link,
   SWE-Exp extraction prompts (gold-patch use).
3. Check released code for papers marked "no link found" (author pages, PMLR/OpenReview supplements).
4. Check CL-Bench appendices for the software-engineering task details and cost accounting.
5. Re-run the arXiv version check before any submission; 2026 preprints are changing quickly.
