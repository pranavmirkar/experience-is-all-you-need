"""Sequential evaluation runner and experiment grid."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import metrics
from .buffer import ExperienceBuffer
from .selection import Selector, SelectionContext, VTARConfig, build_policies
from .simulator import Learner, World, WorldConfig

# Called as hook(selector_name, phase, step, candidates, chosen); used by tests
# to audit what each policy was shown.
SelectHook = Callable[[str, int, int, Sequence, Sequence], None]


def _centroid(embs: Sequence[Sequence[float]]) -> tuple[float, ...]:
    n = len(embs)
    return tuple(sum(e[i] for e in embs) / n for i in range(len(embs[0])))


def _sample_to_budget(pool: Sequence, budget: int, rng: random.Random) -> list:
    order = list(pool)
    rng.shuffle(order)
    out, used = [], 0
    for e in order:
        if used + e.tokens <= budget:
            out.append(e)
            used += e.tokens
    return out


def run_stream(world: World, policy: Selector, hook: Optional[SelectHook] = None) -> dict[str, Any]:
    """Train one policy over the world's stream and return raw results."""
    cfg, seed = world.cfg, world.seed
    P = cfg.n_stream_families
    learner = Learner(world)
    gate = policy.gate_verified
    buffer = ExperienceBuffer(cfg.buffer_capacity, gate, random.Random(f"admission:{seed}"))
    replay_counts: dict[str, int] = defaultdict(int)
    policy.reset()
    arrival_step: dict[str, int] = {}
    scoring_passes = 0
    gstep = 0

    def eval_row() -> tuple[list[float], list[list[bool]]]:
        flags = world.stream_eval(learner.skills)
        return [metrics.pass_rate(f) for f in flags], flags

    R_init, init_flags = eval_row()
    R: list[list[float]] = []
    prev_flags = init_flags
    reg_prior = reg_broken = 0

    for p, episodes in enumerate(world.stream):
        pool = [e for e in episodes if e.verified or not gate]
        phase_start = episodes[0].stream_index
        for e in episodes:
            arrival_step[e.episode_id] = gstep
        if policy.scope != "none":
            centroid = _centroid([e.embedding for e in episodes])
            for step in range(cfg.steps_per_phase):
                new = _sample_to_budget(pool, cfg.new_tokens_per_step, random.Random(f"batch:{seed}:{p}:{step}"))
                cf = learner.counterfactual(new) if policy.needs_counterfactual else None
                cl = None
                if policy.needs_current_loss:

                    def cl(e, _step=gstep):
                        nonlocal scoring_passes
                        scoring_passes += 1
                        return learner.observed_loss(e, _step)

                probe = world.probe(learner.skills)
                if policy.uses_probe:
                    scoring_passes += len(probe)
                ctx = SelectionContext(p, probe, centroid, dict(replay_counts), cf, gstep, arrival_step, cl)
                if policy.scope == "past":
                    candidates = buffer.eligible(phase_start)
                else:
                    taken = {e.episode_id for e in new}
                    candidates = [e for e in pool if e.episode_id not in taken]
                chosen = policy.select(
                    candidates, cfg.replay_tokens_per_step, ctx, random.Random(f"policy:{seed}:{p}:{step}")
                )
                if hook:
                    hook(policy.name, p, step, candidates, chosen)
                if policy.scope == "past":
                    for e in chosen:
                        replay_counts[e.family] += 1
                learner.step(new + chosen)
                gstep += 1
        for e in episodes:
            buffer.add(e)
        row, flags = eval_row()
        R.append(row)
        if p >= 1:
            for f in range(p):  # families seen before this phase
                prior, broken = metrics.regression_counts(prev_flags[f], flags[f])
                reg_prior += prior
                reg_broken += broken
        prev_flags = flags

    final_test = world.locked_test().evaluate_once(learner.skills)
    return {
        "R": R,
        "R_init": R_init,
        "final_test_by_family": final_test,
        "final_test_pass_rate": sum(final_test) / len(final_test),
        "final_test_stream_families": sum(final_test[:P]) / P,
        "heldout_family_final_test": sum(final_test[P:]) / (len(final_test) - P) if len(final_test) > P else None,
        "average_accuracy": metrics.average_accuracy(R, P),
        "average_forgetting": metrics.average_forgetting(R, P),
        "forward_transfer": metrics.forward_transfer(R, R_init, P),
        "regression_rate": reg_broken / reg_prior if reg_prior else 0.0,
        "update_tokens": learner.update_tokens,
        "update_steps": learner.update_steps,
        "scoring_passes": scoring_passes,
        "buffer_size": len(buffer),
        "rejected_unverified": buffer.rejected_unverified,
        "trained_ids": learner.trained_ids,
    }


SCALAR_METRICS = (
    "final_test_pass_rate",
    "final_test_stream_families",
    "heldout_family_final_test",
    "average_accuracy",
    "average_forgetting",
    "forward_transfer",
    "regression_rate",
    "update_tokens",
    "update_steps",
    "scoring_passes",
)


def config_hash(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def run_experiment(cfg: Mapping[str, Any], out_dir: Path) -> Path:
    """Run every scenario x policy x seed in ``cfg``; write config + raw rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    base_world = dict(cfg.get("world", {}))
    policies = build_policies(VTARConfig(**cfg.get("vtar", {})))
    names = cfg.get("policies") or list(policies)
    unknown = [n for n in names if n not in policies]
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    chash = config_hash(cfg)
    results_path = out_dir / "results.jsonl"
    with results_path.open("w") as fh:
        for scen_name, overrides in cfg["scenarios"].items():
            wcfg = WorldConfig.from_dict({**base_world, **overrides})
            for seed in cfg["seeds"]:
                world = World(wcfg, seed)
                for name in names:
                    res = run_stream(world, policies[name])
                    row = {
                        "config_name": cfg.get("name", "unnamed"),
                        "config_hash": chash,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "scenario": scen_name,
                        "policy": name,
                        "seed": seed,
                        "world": asdict(wcfg),
                        "metrics": {k: res[k] for k in SCALAR_METRICS},
                        "R": res["R"],
                        "R_init": res["R_init"],
                        "final_test_by_family": res["final_test_by_family"],
                        "rejected_unverified": res["rejected_unverified"],
                    }
                    fh.write(json.dumps(row) + "\n")
    return results_path
