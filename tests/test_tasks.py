import json
import random
from pathlib import Path
from collections import defaultdict

import pytest

from experience_learning.schemas import Split
from experience_learning.stage2 import collect as C
from experience_learning.stage2 import sandbox as sb
from experience_learning.stage2 import tasks as T

ALL = [(f, n) for f in T.STREAM_FAMILIES + T.HELDOUT_FAMILIES for n in T.templates_of(f)]
needs_sandbox = pytest.mark.skipif(not sb.detect_isolation().strong, reason="strong sandbox isolation unavailable here")


def test_inventory():
    assert len(ALL) == 52
    assert all(len(T.templates_of(f, 1)) == 5 for f in T.STREAM_FAMILIES + T.HELDOUT_FAMILIES)
    assert all(len(T.templates_of(f, 2)) == 4 for f in T.STREAM_FAMILIES)
    assert all(len(T.templates_of(f, 2)) == 2 for f in T.HELDOUT_FAMILIES)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_vintage1_split_assignment_is_frozen(seed):
    """Adding templates must not move any original template between splits (dev results were already seen)."""
    golden = json.loads((Path(__file__).parent / "data" / "stage2_v1_splits.json").read_text())[str(seed)]
    now = {f"{f}/{n}": s.value for (f, n), s in T.assign_splits(seed).items() if T._VINTAGE[(f, n)] == 1}
    assert now == golden


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_splits_are_by_template_and_cover_the_roles(seed):
    bank = T.TaskBank(seed)
    by_template = defaultdict(set)
    for (fam, name), split in bank.splits.items():
        by_template[(fam, name)].add(split)
    assert all(len(s) == 1 for s in by_template.values())  # no template straddles splits
    for fam in T.STREAM_FAMILIES:
        roles = sorted(s.value for (f, _), s in bank.splits.items() if f == fam)
        assert roles == ["dev"] * 2 + ["final_test"] * 2 + ["train"] * 5
        for vintage, n_train in ((1, 3), (2, 2)):  # each vintage covers every role
            v = sorted(s.value for (f, n), s in bank.splits.items() if f == fam and T._VINTAGE[(f, n)] == vintage)
            assert v.count("train") == n_train and v.count("dev") == 1 and v.count("final_test") == 1
    held = [s for (f, _), s in bank.splits.items() if f in T.HELDOUT_FAMILIES]
    assert Split.TRAIN not in held and held.count(Split.FINAL_TEST) == 4 and held.count(Split.DEV) == 3


def test_split_assignment_depends_on_seed_but_is_reproducible():
    assert T.assign_splits(0) == T.assign_splits(0)
    assert T.assign_splits(0) != T.assign_splits(1)


def test_tasks_are_deterministic_and_instances_differ():
    a = T.build_task("lists", "power_sum", 2, 0, Split.TRAIN)
    b = T.build_task("lists", "power_sum", 2, 0, Split.TRAIN)
    assert a == b
    prompts = {T.build_task("lists", "power_sum", i, 0, Split.TRAIN).prompt for i in range(12)}
    assert len(prompts) >= 5


@pytest.mark.parametrize("fam,name", ALL)
def test_every_template_builds_distinct_tests_and_hides_most_of_them(fam, name):
    t = T.build_task(fam, name, 0, 0, Split.TRAIN)
    keys = {json.dumps(c.args, sort_keys=True) for c in t.tests}
    assert len(t.tests) == T.N_TESTS and len(keys) == T.N_TESTS
    assert len(t.visible) == T.N_VISIBLE and len(t.hidden) == T.N_TESTS - T.N_VISIBLE
    assert "def solve" not in t.prompt  # the reference is not leaked into the prompt
    shown = t.prompt
    assert all(json.dumps(c.expected) in shown for c in t.visible)


@needs_sandbox
@pytest.mark.parametrize("fam,name", ALL)
def test_reference_passes_and_trivial_wrong_answers_fail(fam, name):
    t = T.build_task(fam, name, 1, 0, Split.TRAIN)
    assert C.verify(t, t.reference_source).passed
    assert not C.verify(t, "def solve(*a):\n    return None").passed
    assert not C.verify(t, "def solve(*a):\n    return 0").passed


@pytest.mark.parametrize("fam,name", ALL)
def test_no_constant_answer_scores_well_on_average(fam, name):
    """A constant candidate must not pass most tests, or partial credit would reward doing nothing."""
    from collections import Counter

    shares = []
    for i in range(10):
        t = T.build_task(fam, name, i, 0, Split.TRAIN)
        shares.append(Counter(json.dumps(c.expected, sort_keys=True) for c in t.tests).most_common(1)[0][1] / len(t.tests))
    assert sum(shares) / len(shares) <= 0.55
