"""Continual-learning metrics.

Conventions
-----------
``R`` is the accuracy matrix: ``R[t][f]`` is the pass rate on family ``f``
after training phase ``t`` (t = 0 .. P-1). Family ``f`` for ``f < P`` is
introduced in phase ``f``; columns ``>= P`` are never trained on directly.
``R_init[f]`` is the pass rate of the untrained model.
"""

from __future__ import annotations

from typing import Sequence

Matrix = Sequence[Sequence[float]]


def pass_rate(passed: Sequence[bool]) -> float:
    if not passed:
        raise ValueError("pass_rate of an empty list")
    return sum(passed) / len(passed)


def average_accuracy(R: Matrix, n_phases: int) -> float:
    """Mean over phases t of the mean accuracy on families seen so far (f <= t)."""
    per_phase = [sum(R[t][: t + 1]) / (t + 1) for t in range(n_phases)]
    return sum(per_phase) / n_phases


def average_forgetting(R: Matrix, n_phases: int) -> float:
    """Mean over f < P-1 of ``max_{f <= t < P-1} R[t][f] - R[P-1][f]``.

    The best earlier value minus the final value, not clipped at zero, so a
    family that kept improving contributes a negative term.
    """
    if n_phases < 2:
        raise ValueError("forgetting needs at least two phases")
    last = n_phases - 1
    terms = [max(R[t][f] for t in range(f, last)) - R[last][f] for f in range(last)]
    return sum(terms) / len(terms)


def forward_transfer(R: Matrix, R_init: Sequence[float], n_phases: int) -> float:
    """Mean over t >= 1 of ``R[t-1][t] - R_init[t]``: accuracy on the next
    family before training on it, relative to the untrained model."""
    if n_phases < 2:
        raise ValueError("forward transfer needs at least two phases")
    terms = [R[t - 1][t] - R_init[t] for t in range(1, n_phases)]
    return sum(terms) / len(terms)


def regression_counts(before: Sequence[bool], after: Sequence[bool]) -> tuple[int, int]:
    """(# previously passing tasks, # of those that now fail)."""
    if len(before) != len(after):
        raise ValueError("before/after must cover the same tasks")
    prior = sum(before)
    broken = sum(1 for b, a in zip(before, after) if b and not a)
    return prior, broken


def regression_rate(before: Sequence[bool], after: Sequence[bool]) -> float:
    """Fraction of previously passing tasks that fail after an update (0.0 if none passed)."""
    prior, broken = regression_counts(before, after)
    return broken / prior if prior else 0.0
