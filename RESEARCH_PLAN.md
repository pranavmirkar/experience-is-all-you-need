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

1. Done in Stage 2 so far: sandboxed executor and a 30-template task bank (`docs/stage2_plan.md`).
2. Awaiting approval: install CPU torch/transformers/peft, download Qwen2.5-Coder-0.5B (0.99 GB), run a ~20 minute smoke test (generation speed, base pass rate on dev tasks, one LoRA step).
3. Decision gates after the smoke test: the benchmark discriminates (dev pass rate not near 0% or 100%); a reduced pilot is feasible at the measured speed.
4. Paired-variance pilot on dev tasks before pre-registering the H1 margin.
5. Literature: remaining open items are listed in `docs/literature_review.md` (53 entries beyond the brief).

See `docs/experiment_protocol.md` for definitions, assumptions and the revision log.
