import pytest

from experience_learning.buffer import ExperienceBuffer, LeakageError
from experience_learning.schemas import Episode, Split
from experience_learning.stage2 import collect as C
from experience_learning.stage2 import sandbox as sb
from experience_learning.stage2 import tasks as T

pytestmark = pytest.mark.skipif(not sb.detect_isolation().strong, reason="strong sandbox isolation unavailable here")

WRONG = "def solve(*a):\n    return None"


def task(split=Split.TRAIN, i=0):
    return T.build_task("lists", "pair_count", i, 0, split)


def test_passing_attempt_is_verified_without_a_correction():
    t = task()
    e = C.collect_episode(t, t.reference_source, 0)
    assert e.verified and e.correction is None and e.tool_outputs[0].passed
    assert e.verification.method == "hidden_tests" and e.verification.independent


def test_failed_attempt_with_reference_correction_is_verified_through_the_correction():
    t = task()
    e = C.collect_episode(t, WRONG, 1, correction=t.reference_source)
    assert e.verified and e.correction == t.reference_source
    assert [o.passed for o in e.tool_outputs] == [False, True]


def test_unverified_when_the_correction_is_wrong_or_missing():
    t = task()
    assert not C.collect_episode(t, WRONG, 2, correction=WRONG).verified
    assert not C.collect_episode(t, WRONG, 3).verified


def test_forged_pass_does_not_verify():
    t = task()
    forged = "import sys\nsys.__stdout__.write('{\"i\": 0, \"ok\": true, \"v\": 0}\\n')\ndef solve(a):\n    return -99"
    assert not C.collect_episode(t, forged, 4).verified


def test_episode_roundtrips_and_carries_the_task_split():
    t = task(Split.DEV)
    e = C.collect_episode(t, t.reference_source, 5)
    assert e.split is Split.DEV and Episode.from_dict(e.to_dict()) == e


def test_only_train_template_episodes_can_enter_the_buffer():
    import random

    buf = ExperienceBuffer(10, True, random.Random(0))
    assert buf.add(C.collect_episode(task(Split.TRAIN), task().reference_source, 0))
    for split in (Split.DEV, Split.FINAL_TEST):
        t = task(split)
        with pytest.raises(LeakageError):
            buf.add(C.collect_episode(t, t.reference_source, 1))


def test_episode_ids_are_unique_per_stream_position():
    t = task()
    a, b = C.collect_episode(t, t.reference_source, 0), C.collect_episode(t, t.reference_source, 1)
    assert a.episode_id != b.episode_id
