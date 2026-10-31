# Research plan: Experience Is All You Need

Source: `EXPERIENCE_IS_ALL_YOU_NEED_RESEARCH_BRIEF.docx` (v1.0, 24 Oct 2026). The brief is
authoritative; this file records the implementation plan and assumptions for the first session.

## Research question

Can an experience-selection policy that estimates expected transfer to future tasks beat simple
replay policies under equal data and compute budgets, while limiting forgetting?
VTAR is a provisional name for a candidate policy. It is not claimed to be novel or better.

## Repository state at start

Empty directory, not a git repository. Python 3.13, no GPU, ~30 GB free disk, numpy available,
pytest installed with `pip install pytest` (user-level dev dependency only).

## Plan for this session (Stage 1 only)

1. Episode schema with enforced external-verification semantics.
2. Split-safe buffer (TRAIN only, optional verification gate, reservoir admission).
3. Policies: frozen, sequential, uniform, loss-prioritised, novelty, VTAR, five VTAR ablations.
4. Metrics with hand-calculated unit tests.
5. Deterministic simulator and sequential runner; pilot over four scenarios and five seeds.
6. Tests for leakage, determinism, shared candidate pools, budgets, gating.

## Out of scope until Stage 1 is accepted

Model downloads, LoRA fine-tuning, real coding tasks, SWE-Bench-CL, the retrieval baseline and the
gradient-based selectors (specified in `docs/experiment_protocol.md`, Stage 2), locked final study.

## Next smallest steps

1. Done in Stage 2: sandbox, 52-template task bank, model choice (Qwen2.5-Coder-1.5B), LoRA timing, plumbing dry run (`docs/stage2_plan.md` sections 6 to 10). No policy comparison has been made; the dry run is plumbing and cost only.
2. Done: bf16 vs fp32 on the matched protocol shows no detectable difference (pass@1 0.202 vs 0.212, bf16 2.1x faster; `docs/stage2_plan.md` 10c).
3. Decide the evaluation size and CPU budget: evaluation is about 90% of the cost (about 28 minutes per 5-point policy run).
4. Paired-variance pilot to size the H1 margin; pre-register H1 before touching the locked final test.
5. Literature: remaining open items are listed in `docs/literature_review.md` (53 entries beyond the brief).

See `docs/experiment_protocol.md` for definitions, assumptions and the revision log.
