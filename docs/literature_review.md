# Literature matrix (scaffold, NOT yet completed)

Status as of 2026-10-24: **no paper has been read in full.** The cloud environment's network policy
blocks `arxiv.org` and `proceedings.mlr.press` (DNS/CONNECT 403), so methods and experiment sections
could not be fetched. Per the brief (section 3.1), no gap claim may be made until each close paper's
methods and experiments are read and this table is filled in.

Rules for filling this in:
- A cell is filled only from the paper's methods/experiments text. Put the section or page in the cell.
- "Relevance" below is copied from the brief and is unverified. "Not read" means exactly that.
- Dates, venues and arXiv IDs come from the brief and must be re-checked against the sources
  (the brief itself says to confirm details and check for newer versions). Several 2026 IDs are unconfirmed.

## Matrix

Columns to fill per paper: **Learning signal** | **Experience representation** | **Selection method** |
**Weights change?** | **Test setup** | **Compute budget** | **Reported limitations** | **Code released?** |
**Difference from VTAR**.

| # | Work (as cited in brief) | Relevance per brief (unverified) | Learning signal | Experience repr. | Selection | Weights change? | Test setup | Compute | Limitations | Code | Differs from VTAR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Reflexion, Shinn et al. 2023 (NeurIPS) | verbal feedback + episodic memory, no weight updates; memory-only baseline | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 2 | Voyager, Wang et al. 2023 / TMLR 2024 | executable skill library + curriculum, no fine-tuning | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 3 | CodeIt, Butt et al. ICML 2024 (PMLR 235) | sampling + hindsight relabeling + prioritized replay; closest replay baseline | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 4 | SWE-Bench-CL, Joshi et al. 2025 | sequential issues, continual-learning metrics | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 5 | AuPair, Mavalankar et al. ICML 2025 (PMLR 267) | complementary verified guess/fix pairs, inference-time | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 6 | SWE-Exp, Chen et al. 2025 | distilled experience from issue-resolution trajectories | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 7 | Intrinsic metacognitive learning position paper, Liu & Van der Schaar ICML 2025 | learning the learning process | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 8 | CODESKILL, Li et al. 2026 (ID per brief: 2605.25430) | skill extraction / evolution / maintenance | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 9 | Learning Globally Reusable Skills for Coding Agents, Yang et al. 2026 (ID per brief: 2608.06153) | skill relationships, consolidation, verification | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 10 | Self-play SWE-RL, Wei et al. ICML 2026 (PMLR 306) | sandboxed bug injection + repair, self-play RL | not read | not read | not read | not read | not read | not read | not read | not read | not read |
| 11 | LLM agents are not always faithful self-evolvers, Zhao et al. ICML 2026 (PMLR 306) | whether agents actually use accumulated experience | not read | not read | not read | not read | not read | not read | not read | not read | not read |

## Search-level note (not a reading)

A web search returned a summary of CodeIt (arXiv 2402.04858; code reported at `Qualcomm-AI-research/codeit`).
The summary says it targets ARC as programming-by-examples, uses hindsight relabeling, and credits prioritized
replay with reducing forgetting. That is secondary-source text and has not been checked against the paper.
If it holds, prioritized replay for self-improvement already exists and VTAR must be compared against it.

## Questions each read must answer for VTAR

1. Does the paper select experience by estimated future transfer, interference, or cost? Or by loss, recency, or reward only?
2. Are weights updated, or is the gain from memory/context only? (Sets which baseline it maps to.)
3. Is there a held-out task family and a forgetting measure?
4. Is the compute budget matched against baselines?
5. Is the benefit shown to depend on the content of the experience (shuffled/irrelevant controls)?

## To unblock

Allow `arxiv.org` and `proceedings.mlr.press` (plus `openreview.net`, `icml.cc` if wanted) under Network access
in the environment settings, then start a new session or ask me to continue. Alternatively upload the PDFs.
