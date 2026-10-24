"""Verify candidate solutions and turn attempts into :class:`Episode` records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from ..schemas import Episode, Split, ToolOutput, Verification
from . import sandbox
from .tasks import Task

VERIFY_PROCEDURE = (
    "all tests of the task (visible and hidden) executed in the isolated sandbox; outputs compared "
    "with expected values held outside the sandbox"
)


@dataclass(frozen=True)
class Outcome:
    passed: bool  # every test passed
    n_passed: int
    total: int
    wall_s: float
    summary: str  # short, safe to store


def verify(task: Task, source: str, limits: sandbox.Limits = sandbox.Limits()) -> Outcome:
    """Run ``source`` against all tests of ``task`` in the sandbox."""
    run = sandbox.run_candidate(source, task.entry_point, [list(t.args) for t in task.tests], limits)
    n_ok = 0
    first_problem: Optional[str] = None
    for t, call in zip(task.tests, run.calls):
        if call.ok and call.value == t.expected:
            n_ok += 1
        elif first_problem is None:
            first_problem = call.error if not call.ok else "wrong answer"
    total = len(task.tests)
    summary = f"{n_ok}/{total} tests passed"
    if run.definition_error:
        summary += f"; {run.definition_error}"
    elif first_problem:
        summary += f"; first failure: {first_problem}"
    if run.killed:
        summary += "; killed on wall-clock limit"
    return Outcome(n_ok == total, n_ok, total, run.wall_s, summary[:300])


def approx_tokens(*texts: Optional[str]) -> int:
    """Rough token count (characters / 4). Replace with the model tokenizer in the model integration."""
    return max(1, sum(len(t) for t in texts if t) // 4)


def collect_episode(
    task: Task,
    attempt: str,
    stream_index: int,
    *,
    correction: Optional[str] = None,
    model_checkpoint: str = "unset",
    adapter_version: str = "none",
    created_at: str = "unset",
    limits: sandbox.Limits = sandbox.Limits(),
) -> Episode:
    """Build an episode from an attempt and, if the attempt failed, an optional correction.

    ``verified`` is True only when a solution this record relies on (the attempt if it
    passed, otherwise the correction) passed the sandboxed test run. The model asserting
    its own correctness never counts.
    """
    attempt_out = verify(task, attempt, limits)
    tool_outputs = [ToolOutput("test", attempt_out.passed, attempt_out.summary)]
    wall = attempt_out.wall_s
    corr_src: Optional[str] = None
    if attempt_out.passed:
        verified_ok = True
    elif correction is not None:
        corr_out = verify(task, correction, limits)
        tool_outputs.append(ToolOutput("test", corr_out.passed, "correction: " + corr_out.summary))
        wall += corr_out.wall_s
        corr_src = correction
        verified_ok = corr_out.passed
    else:
        verified_ok = False
    verification = (
        Verification(True, "hidden_tests", VERIFY_PROCEDURE, True)
        if verified_ok
        else Verification(False, "model_assertion")
    )
    return Episode(
        episode_id=f"{task.task_id}@{stream_index}",
        created_at=created_at,
        stream_index=stream_index,
        family=task.family,
        language="python",
        source="stage2-taskbank",
        split=task.split,
        prompt=task.prompt,
        attempt=attempt,
        tool_outputs=tuple(tool_outputs),
        correction=corr_src,
        verification=verification,
        model_checkpoint=model_checkpoint,
        adapter_version=adapter_version,
        tokens=approx_tokens(task.prompt, attempt, corr_src),
        wall_time_s=wall,
        license="CC0-1.0 (generated templates)",
    )
