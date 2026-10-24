import random

import pytest

from experience_learning.buffer import ExperienceBuffer, LeakageError
from experience_learning.schemas import Split

from helpers import make_episode


def buf(cap=3, gate=True, seed=0):
    return ExperienceBuffer(cap, gate, random.Random(seed))


@pytest.mark.parametrize("split", [Split.DEV, Split.FINAL_TEST])
def test_rejects_non_train(split):
    with pytest.raises(LeakageError):
        buf().add(make_episode(split=split))


def test_gating_rejects_unverified():
    b = buf()
    assert not b.add(make_episode("u", verified=False))
    assert b.rejected_unverified == 1 and len(b) == 0
    assert buf(gate=False).add(make_episode("u", verified=False))


def test_capacity_and_determinism():
    def fill(seed):
        b = buf(cap=5, seed=seed)
        for i in range(50):
            b.add(make_episode(f"e{i:02d}", stream_index=i))
        return [e.episode_id for e in b.eligible(100)]

    assert len(fill(0)) == 5
    assert fill(0) == fill(0)


def test_eligible_filters_by_stream_index_and_duplicates_fail():
    b = buf(cap=10)
    for i in range(4):
        b.add(make_episode(f"e{i}", stream_index=i))
    assert [e.episode_id for e in b.eligible(2)] == ["e0", "e1"]
    with pytest.raises(ValueError):
        b.add(make_episode("e0", stream_index=9))
