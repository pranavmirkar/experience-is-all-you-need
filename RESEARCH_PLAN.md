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

Model downloads, LoRA fine-tuning, real coding tasks, SWE-Bench-CL, published-baseline
adaptation (CodeIt), the retrieval baseline, the literature matrix, locked final study.

## Next smallest steps

1. Finish the literature review: systematic search and open items in `docs/literature_review.md` (first pass on the 11 brief papers is done).
2. Decide on a small code model; report licence, size and VRAM before downloading anything.
3. Build a sandboxed executor for generated code (isolated dir, timeouts, no network) with
   function-level Python tasks grouped by family.
4. Pilot variance estimate before pre-registering the H1 thresholds.

See `docs/experiment_protocol.md` for definitions, assumptions and the revision log.
