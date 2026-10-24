import json
import statistics

import pytest

from experience_learning.buffer import ExperienceBuffer
from experience_learning.evaluation import run_experiment, run_stream
from experience_learning.selection import build_policies
from experience_learning.simulator import World, WorldConfig

SMALL = WorldConfig(episodes_per_phase=30, steps_per_phase=4, n_stream_eval=40, n_final_test=40, n_probe=5)
POLICIES = build_policies()


def metrics_of(res):
    return {k: v for k, v in res.items() if k != "trained_ids"}


def test_same_seed_is_reproducible_and_different_seed_differs():
    a = metrics_of(run_stream(World(SMALL, 0), POLICIES["vtar"]))
    b = metrics_of(run_stream(World(SMALL, 0), POLICIES["vtar"]))
    c = metrics_of(run_stream(World(SMALL, 1), POLICIES["vtar"]))
    assert a == b
    assert a != c


def test_stream_is_identical_across_policies():
    w1, w2 = World(SMALL, 5), World(SMALL, 5)
    assert [[e.to_dict() for e in p] for p in w1.stream] == [[e.to_dict() for e in p] for p in w2.stream]


def test_all_policies_see_the_same_candidate_pool():
    """Candidates shown at (phase, step) are identical for every gated replay policy."""
    world = World(SMALL, 4)
    seen = {}
    for name in ("uniform", "loss", "novelty", "vtar", "vtar_random"):
        log = {}
        run_stream(world, POLICIES[name], hook=lambda n, p, s, cand, ch, log=log: log.__setitem__((p, s), [e.episode_id for e in cand]))
        seen[name] = log
    ref = seen["uniform"]
    assert all(seen[n] == ref for n in seen)


def test_frozen_does_no_updates():
    res = run_stream(World(SMALL, 0), POLICIES["frozen"])
    assert res["update_tokens"] == 0 and res["update_steps"] == 0
    assert res["average_forgetting"] == pytest.approx(0.0, abs=1e-12)
    assert res["R"][0] == res["R_init"]


def test_training_policies_use_the_same_budget():
    world = World(SMALL, 0)
    steps = {n: run_stream(world, POLICIES[n])["update_steps"] for n in ("sequential", "uniform", "vtar")}
    assert len(set(steps.values())) == 1
    cap = SMALL.n_stream_families * SMALL.steps_per_phase * (SMALL.new_tokens_per_step + SMALL.replay_tokens_per_step)
    for n in ("sequential", "uniform", "loss", "novelty", "vtar"):
        assert run_stream(world, POLICIES[n])["update_tokens"] <= cap


def test_harness_is_sensitive_to_forgetting():
    """Sanity: with no replay, forgetting is clearly worse than with uniform replay."""
    cfg = WorldConfig(**{**SMALL.__dict__, "transfer_mode": "none", "bad_correction_rate": 0.0})
    seq = statistics.mean(run_stream(World(cfg, s), POLICIES["sequential"])["average_forgetting"] for s in range(4))
    uni = statistics.mean(run_stream(World(cfg, s), POLICIES["uniform"])["average_forgetting"] for s in range(4))
    assert seq > uni


def test_gating_changes_buffer_contents():
    world = World(SMALL, 0)
    gated = run_stream(world, POLICIES["uniform"])
    ungated = run_stream(world, POLICIES["vtar_no_gating"])
    assert gated["rejected_unverified"] > 0
    assert ungated["rejected_unverified"] == 0


def test_run_experiment_writes_config_seed_policy_metrics_timestamp(tmp_path):
    cfg = {
        "name": "tiny",
        "seeds": [0, 1],
        "policies": ["frozen", "uniform", "vtar"],
        "world": {"episodes_per_phase": 20, "steps_per_phase": 2, "n_stream_eval": 20, "n_final_test": 20, "n_probe": 4},
        "scenarios": {"aligned": {}},
    }
    path = run_experiment(cfg, tmp_path)
    rows = [json.loads(l) for l in path.open()]
    assert len(rows) == 6
    assert (tmp_path / "config.json").exists()
    for r in rows:
        assert {"seed", "policy", "scenario", "metrics", "timestamp", "config_hash"} <= set(r)
        assert "final_test_pass_rate" in r["metrics"]
    with pytest.raises(ValueError):
        run_experiment({**cfg, "policies": ["nope"]}, tmp_path / "x")


def test_token_value_exp_controls_whether_length_matters():
    """With exp=1 a long episode teaches more than a short one; with exp=0 they are equal."""
    from dataclasses import replace

    from experience_learning.simulator import Learner

    def gain(exp, tokens):
        w = World(replace(SMALL, token_value_exp=exp, transfer_mode="none", bad_correction_rate=0.0, p_verified=1.0), 0)
        e = replace(w.stream[0][0], tokens=tokens)
        L = Learner(w)
        before = L.skills[0]
        L.step([e])
        return L.skills[0] - before

    assert gain(1.0, 700) > 1.5 * gain(1.0, 350)
    assert gain(0.0, 700) == pytest.approx(gain(0.0, 350))


def test_counterfactual_does_not_mutate_and_matches_real_step():
    from experience_learning.simulator import Learner

    world = World(SMALL, 0)
    L = Learner(world)
    batch = world.stream[0][:3]
    before = list(L.skills)
    cf = L.counterfactual(batch[:1])(batch[1:])
    assert L.skills == before and L.update_steps == 0
    L.step(batch)
    assert cf == pytest.approx(L.skills)


def test_only_privileged_selectors_receive_the_counterfactual_hook():
    from experience_learning.selection import UniformReplay

    seen = []

    class Spy(UniformReplay):
        name = "spy"

        def rank(self, candidates, ctx, rng):
            seen.append(ctx.counterfactual)
            return super().rank(candidates, ctx, rng)

    run_stream(World(SMALL, 0), Spy())
    assert seen and all(c is None for c in seen)
    assert POLICIES["oracle_gain"].needs_counterfactual
    assert not any(p.needs_counterfactual for n, p in POLICIES.items() if n != "oracle_gain")


def test_oracle_gain_runs_within_budget_and_beats_random_replay_on_average():
    cap = SMALL.n_stream_families * SMALL.steps_per_phase * (SMALL.new_tokens_per_step + SMALL.replay_tokens_per_step)
    res = [run_stream(World(SMALL, s), POLICIES["oracle_gain"]) for s in range(4)]
    assert all(r["update_tokens"] <= cap for r in res)
    uni = [run_stream(World(SMALL, s), POLICIES["uniform"]) for s in range(4)]
    assert statistics.mean(r["average_accuracy"] for r in res) > statistics.mean(r["average_accuracy"] for r in uni)
