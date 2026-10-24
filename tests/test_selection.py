import random

import pytest

from experience_learning.buffer import LeakageError
from experience_learning.schemas import Split
from experience_learning.selection import (
    ProbeItem,
    SelectionContext,
    VTARConfig,
    VTARSelector,
    build_policies,
)

from helpers import make_episode


def ctx(probe=(), centroid=None, counts=None, step=0, current_loss=lambda e: e.loss, arrival=None):
    return SelectionContext(0, tuple(probe), centroid, counts or {}, None, step, arrival or {}, current_loss)


def pool(n=20):
    fams = ["A", "B", "C"]
    return [
        make_episode(
            f"e{i:02d}",
            family=fams[i % 3],
            verified=(i % 4 != 0),
            tokens=80 + 10 * (i % 5),
            loss=(i % 7) / 7,
            embedding=((-1.0) ** i, 0.5 + i / 40),
            stream_index=i,
        )
        for i in range(n)
    ]


REPLAY_POLICIES = [n for n in build_policies() if n not in ("frozen", "sequential", "oracle_gain")]  # oracle needs the simulator hook


@pytest.mark.parametrize("name", REPLAY_POLICIES)
def test_choice_is_subset_within_budget_and_deterministic(name):
    p = build_policies()[name]
    cands = pool()
    probe = [ProbeItem("A", (1.0, 0.0), True), ProbeItem("B", (0.0, 1.0), False)]
    run = lambda: p.select(cands, 400, ctx(probe, (1.0, 0.0)), random.Random(3))
    a, b = run(), run()
    assert [e.episode_id for e in a] == [e.episode_id for e in b]
    assert {e.episode_id for e in a} <= {e.episode_id for e in cands}
    assert sum(e.tokens for e in a) <= 400
    assert len({e.episode_id for e in a}) == len(a)


@pytest.mark.parametrize("name", [n for n in REPLAY_POLICIES if n != "vtar_no_gating"])
def test_gating_excludes_unverified(name):
    chosen = build_policies()[name].select(pool(), 10_000, ctx((), (1.0, 0.0)), random.Random(0))
    assert chosen and all(e.verified for e in chosen)


def test_no_gating_ablation_can_pick_unverified():
    chosen = build_policies()["vtar_no_gating"].select(pool(), 10_000, ctx((), (1.0, 0.0)), random.Random(0))
    assert any(not e.verified for e in chosen)


@pytest.mark.parametrize("name", [n for n in build_policies() if n != "oracle_gain"])
def test_refuses_non_train_candidates(name):
    bad = pool(3) + [make_episode("t", split=Split.FINAL_TEST)]
    with pytest.raises(LeakageError):
        build_policies()[name].select(bad, 1000, ctx(), random.Random(0))


def test_loss_policy_orders_by_loss():
    cands = [make_episode("a", loss=0.1), make_episode("b", loss=0.9), make_episode("c", loss=0.5)]
    chosen = build_policies()["loss"].select(cands, 10_000, ctx(), random.Random(0))
    assert [e.episode_id for e in chosen] == ["b", "c", "a"]


def test_novelty_prefers_least_replayed_family():
    cands = [make_episode(f"a{i}", family="A") for i in range(3)] + [make_episode("b0", family="B")]
    chosen = build_policies()["novelty"].select(cands, 100, ctx(counts={"A": 5}), random.Random(0))
    assert [e.family for e in chosen] == ["B"]


def test_vtar_components_hand_calculated():
    e = make_episode("x", family="A", tokens=100, loss=0.4, embedding=(1.0, 0.0))
    probe = [ProbeItem("A", (1.0, 0.0), True), ProbeItem("B", (0.0, 1.0), False)]
    sel = VTARSelector(VTARConfig())
    (c,) = sel.components([e], ctx(probe, (-1.0, 0.0)))
    assert c["L"] == pytest.approx(0.5 * 0.4 + 0.5 * 1.0)  # 0.7
    assert c["T"] == pytest.approx(0.5)  # (cos 1 * failed + cos 0 * ok) / 2
    assert c["I"] == pytest.approx(1.0)  # centroid is opposite
    assert c["C"] == pytest.approx(1.0)
    # 0.3*1 + 0.3*0.7 + 0.4*0.5 - 0.4*1 = 0.31
    assert sel._score(c, 1.0) == pytest.approx(0.31, abs=1e-5)


def test_vtar_score_is_zero_for_unverified_when_gated_and_clamped_at_zero():
    sel = VTARSelector(VTARConfig())
    unverified = {"L": 1, "T": 1, "I": 0, "C": 1, "gate": 0.0}
    conflicting = {"L": 0, "T": 0, "I": 1, "C": 1, "gate": 1.0}
    assert sel._score(unverified, 1.0) == 0.0
    assert sel._score(conflicting, 1.0) == 0.0


def test_vtar_cost_normalisation():
    sel = VTARSelector(VTARConfig())
    cheap = {"L": 0.5, "T": 0.5, "I": 0, "C": 0.5, "gate": 1.0}
    dear = {**cheap, "C": 2.0}
    assert sel._score(cheap, 1.0) > sel._score(dear, 1.0)


def test_novelty_spreads_a_single_call_across_families():
    # family C is far behind in cumulative counts, but one call must still cover A, B and C
    cands = [make_episode(f"{f}{i}", family=f) for f in "ABC" for i in range(4)]
    chosen = build_policies()["novelty"].select(cands, 300, ctx(counts={"A": 9, "B": 9, "C": 0}), random.Random(0))
    assert sorted(e.family for e in chosen) == ["A", "B", "C"]


def test_vtar_random_is_not_a_copy_of_uniform():
    cands = pool(30)
    a = build_policies()["uniform"].select(cands, 10_000, ctx(), random.Random(1))
    b = build_policies()["vtar_random"].select(cands, 10_000, ctx(), random.Random(1))
    assert [e.episode_id for e in a] != [e.episode_id for e in b]


def test_codeit_priority_uses_attempt_pass_fraction_with_floor():
    from experience_learning.schemas import ToolOutput
    from experience_learning.selection import CodeItPriorityReplay
    from dataclasses import replace

    base = make_episode("x")
    passed = replace(base, tool_outputs=(ToolOutput("test", True), ToolOutput("test", True)))
    half = replace(base, tool_outputs=(ToolOutput("test", True), ToolOutput("test", False)))
    failed = replace(base, tool_outputs=(ToolOutput("test", False),))
    none = replace(base, tool_outputs=())
    assert CodeItPriorityReplay.priority(passed) == 1.0
    assert CodeItPriorityReplay.priority(half) == 0.5
    assert CodeItPriorityReplay.priority(failed) == CodeItPriorityReplay.floor
    assert CodeItPriorityReplay.priority(none) == CodeItPriorityReplay.floor


def test_codeit_priority_samples_high_priority_more_often():
    from dataclasses import replace
    from experience_learning.schemas import ToolOutput

    good = [replace(make_episode(f"g{i}", family="A", tokens=100), tool_outputs=(ToolOutput("test", True),)) for i in range(5)]
    bad = [replace(make_episode(f"b{i}", family="B", tokens=100), tool_outputs=(ToolOutput("test", False),)) for i in range(5)]
    sel = build_policies()["codeit_priority"]
    wins = sum(
        sum(e.family == "A" for e in sel.select(good + bad, 200, ctx(), random.Random(s))) for s in range(200)
    )
    assert wins > 2 * (200 * 2 - wins)  # far more A than B among 2 picks per call


def test_loss_fresh_ranks_by_current_not_logged_loss():
    cands = [make_episode("a", loss=0.9), make_episode("b", loss=0.1), make_episode("c", loss=0.5)]
    now = {"a": 0.05, "b": 0.8, "c": 0.4}
    chosen = build_policies()["loss_fresh"].select(
        cands, 10_000, ctx(current_loss=lambda e: now[e.episode_id]), random.Random(0)
    )
    assert [e.episode_id for e in chosen] == ["b", "c", "a"]


def test_per_refreshes_priority_of_replayed_episodes_only():
    from experience_learning.selection import PrioritizedLossReplay

    sel = PrioritizedLossReplay("t")
    a, b = make_episode("a", loss=0.9, tokens=100), make_episode("b", loss=0.9, tokens=100)
    # budget fits exactly one; find which is chosen, its priority must be overwritten
    chosen = sel.select([a, b], 100, ctx(step=5, current_loss=lambda e: 0.2), random.Random(1))
    assert len(chosen) == 1
    other = b if chosen[0] is a else a
    assert sel._prio[chosen[0].episode_id] == (0.2, 5)
    assert other.episode_id not in sel._prio
    sel.reset()
    assert sel._prio == {}


def test_age_decay_demotes_stale_priorities_but_plain_per_does_not():
    from experience_learning.selection import PrioritizedLossReplay

    fresh = make_episode("fresh", loss=0.3)
    stale = make_episode("stale", loss=0.9)  # high but measured long ago
    c = ctx(step=40, arrival={"stale": 0, "fresh": 40})
    decay = PrioritizedLossReplay("d", tau=5.0)
    plain = PrioritizedLossReplay("p")
    assert decay.weight(stale, c) < decay.weight(fresh, c)
    assert plain.weight(stale, c) > plain.weight(fresh, c)


def test_per_variants_are_reproducible_across_runs_because_state_is_reset():
    from experience_learning.simulator import World, WorldConfig
    from experience_learning.evaluation import run_stream

    small = WorldConfig(episodes_per_phase=30, steps_per_phase=4, n_stream_eval=40, n_final_test=40, n_probe=5)
    pol = build_policies()["loss_per_decay"]
    w = World(small, 0)
    a, b = run_stream(w, pol), run_stream(w, pol)
    assert a["trained_ids"] == b["trained_ids"] and a["R"] == b["R"]
