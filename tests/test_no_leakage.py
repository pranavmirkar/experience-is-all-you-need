import pytest

from experience_learning.buffer import LeakageError
from experience_learning.evaluation import run_stream
from experience_learning.schemas import Split
from experience_learning.selection import build_policies
from experience_learning.simulator import Learner, World, WorldConfig

from helpers import make_episode

SMALL = WorldConfig(episodes_per_phase=20, steps_per_phase=3, n_stream_eval=20, n_final_test=20, n_probe=4)


def test_learner_refuses_non_train():
    learner = Learner(World(SMALL, 0))
    with pytest.raises(LeakageError):
        learner.step([make_episode(split=Split.FINAL_TEST)])
    with pytest.raises(LeakageError):
        learner.step([make_episode(split=Split.DEV)])


def test_whole_stream_is_train_and_runs_never_train_on_anything_else():
    world = World(SMALL, 1)
    assert all(e.split is Split.TRAIN for phase in world.stream for e in phase)
    for name, policy in build_policies().items():
        res = run_stream(world, policy)
        assert set(res["trained_ids"]) <= world.train_episode_ids, name


def test_final_test_is_scored_once_and_hidden_from_policies():
    world = World(SMALL, 2)
    locked = world.locked_test()
    locked.evaluate_once(world.base_skill)
    with pytest.raises(RuntimeError):
        locked.evaluate_once(world.base_skill)

    seen = []
    run_stream(world, build_policies()["vtar"], hook=lambda *a: seen.append(a))
    assert seen
    for _, _, _, candidates, chosen in seen:
        assert all(e.split is Split.TRAIN for e in list(candidates) + list(chosen))


def test_final_test_does_not_affect_training():
    """Different final-test tasks must not change any training or dev quantity."""
    a = World(SMALL, 3)
    b = World(WorldConfig(**{**SMALL.__dict__, "n_final_test": 33}), 3)
    ra, rb = run_stream(a, build_policies()["vtar"]), run_stream(b, build_policies()["vtar"])
    assert ra["trained_ids"] == rb["trained_ids"]
    assert ra["R"] == rb["R"]
