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

1. Done in Stage 2 so far: sandboxed executor and a 52-template task bank (bank v2) (`docs/stage2_plan.md`).
2. Done: smoke test on Qwen2.5-Coder-0.5B (results in `docs/stage2_plan.md` section 6; 10 tok/s, LoRA step under 1 s, base dev full-pass 2 of 14).
3. Gate 2 (benchmark discriminates) was weak with 0.5B; with Qwen2.5-Coder-1.5B it is supported with caveats (dev pass@1 0.212, 12 of 13 templates sometimes solved). Earlier 0.5B finding: on bank v2 sampled dev pass@1 is 0.058 and 8 of 13 dev templates are never fully passed in 16 samples. A two-shot prompt did not help (pass@1 0.053). Next: LoRA timing for 1.5B, plumbing dry run, paired-variance pilot (`docs/stage2_plan.md` section 9).
4. Paired-variance pilot on dev tasks before pre-registering the H1 margin.
5. Literature: remaining open items are listed in `docs/literature_review.md` (53 entries beyond the brief).

See `docs/experiment_protocol.md` for definitions, assumptions and the revision log.
