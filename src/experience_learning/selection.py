"""Replay-selection policies.

Every policy implements ``rank`` (a full ordering of the candidates it is
given). The shared ``select`` enforces the rules common to all policies:
TRAIN-only candidates, optional verification gating, and a token budget.

The VTAR terms below are cheap *proxies* (embedding cosines, probe failure
rates). They are not measured causal transfer or interference.
"""

from __future__ import annotations

import math
import random
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional, Sequence

from .buffer import assert_trainable
from .schemas import Episode

Scope = Literal["past", "current", "none"]


@dataclass(frozen=True)
class ProbeItem:
    """One development-probe task as visible to selectors."""

    family: str
    embedding: tuple[float, ...]
    failed: bool


@dataclass(frozen=True)
class SelectionContext:
    phase: int
    probe: tuple[ProbeItem, ...]
    current_centroid: Optional[tuple[float, ...]]
    replay_counts: Mapping[str, int]
    # Privileged simulator hook, set only for selectors with needs_counterfactual:
    # maps extra episodes to the skills a step on (new batch + extra) would give.
    counterfactual: Optional[Callable[[Sequence[Episode]], Sequence[float]]] = None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class Selector(ABC):
    """Base class. ``scope`` says which pool the runner draws candidates from."""

    name: str
    scope: Scope = "past"
    gate_verified: bool = True
    needs_counterfactual: bool = False  # True only for privileged reference selectors

    @abstractmethod
    def rank(
        self, candidates: Sequence[Episode], ctx: SelectionContext, rng: random.Random
    ) -> list[Episode]:
        """Return all candidates, most preferred first."""

    def select(
        self,
        candidates: Sequence[Episode],
        budget_tokens: int,
        ctx: SelectionContext,
        rng: random.Random,
    ) -> list[Episode]:
        assert_trainable(candidates)
        pool = [e for e in candidates if e.verified or not self.gate_verified]
        pool.sort(key=lambda e: e.episode_id)
        ordered = self.rank(pool, ctx, rng)
        if sorted(e.episode_id for e in ordered) != [e.episode_id for e in pool]:
            raise RuntimeError(f"{self.name}.rank must return a permutation of its input")
        chosen: list[Episode] = []
        used = 0
        for e in ordered:
            if used + e.tokens <= budget_tokens:
                chosen.append(e)
                used += e.tokens
        return chosen


class FrozenPolicy(Selector):
    """No updates at all."""

    name = "frozen"
    scope: Scope = "none"

    def rank(self, candidates, ctx, rng):
        return []


class UniformReplay(Selector):
    name = "uniform"

    def rank(self, candidates, ctx, rng):
        out = list(candidates)
        rng.shuffle(out)
        return out


class SequentialFineTune(UniformReplay):
    """No replay: the extra update budget goes to the current phase's data."""

    name = "sequential"
    scope: Scope = "current"


class LossPrioritizedReplay(Selector):
    """Highest logged loss first."""

    name = "loss"

    def rank(self, candidates, ctx, rng):
        if any(e.loss is None for e in candidates):
            raise ValueError("loss policy needs a loss on every candidate")
        return sorted(candidates, key=lambda e: (-e.loss, e.episode_id))


class NoveltyReplay(Selector):
    """Coverage: spread each call across families, least-replayed families first.

    Within a call, round-robin over families; ties go to the family replayed
    the fewest times in earlier calls. (An earlier version balanced only the
    cumulative counts, which put a whole batch on one lagging family.)
    """

    name = "novelty"

    def rank(self, candidates, ctx, rng):
        by_family: dict[str, list[Episode]] = defaultdict(list)
        for e in candidates:
            by_family[e.family].append(e)
        for fam in by_family:
            rng.shuffle(by_family[fam])
        local: dict[str, int] = defaultdict(int)
        out: list[Episode] = []
        while by_family:
            fam = min(by_family, key=lambda f: (local[f], ctx.replay_counts.get(f, 0), f))
            out.append(by_family[fam].pop())
            local[fam] += 1
            if not by_family[fam]:
                del by_family[fam]
        return out


class CodeItPriorityReplay(Selector):
    """Adaptation of CodeIt's prioritised replay (Butt et al. 2024), not a reproduction.

    CodeIt samples experiences with priority proportional to the fraction of
    demonstration outputs the program matches. Here the priority is the fraction of
    the logged attempt's test outputs that passed, floored so failures stay
    reachable, and sampling is weighted without replacement. No hindsight
    relabelling, no iterative sampling stage, no program execution.
    """

    name = "codeit_priority"
    floor = 0.05

    @classmethod
    def priority(cls, e: Episode) -> float:
        flags = [t.passed for t in e.tool_outputs if t.passed is not None]
        frac = sum(flags) / len(flags) if flags else 0.0
        return max(cls.floor, frac)

    def rank(self, candidates, ctx, rng):
        # Efraimidis-Spirakis: key = u ** (1 / w), largest first
        keyed = [(rng.random() ** (1.0 / self.priority(e)), e.episode_id, e) for e in candidates]
        keyed.sort(key=lambda t: (-t[0], t[1]))
        return [e for _, _, e in keyed]


class MeasuredGainOracle(Selector):
    """Privileged reference: greedy by *measured* marginal gain per token.

    Counterpart of AuPair-style selection (Mavalankar et al. 2025): each candidate
    is scored by the change in mean stream-family skill when added to the current
    batch, divided by relative token cost, with the set grown greedily so later picks
    are scored given earlier ones. It reads true skills through a hidden-dynamics
    hook, so no real policy can use it. It is an upper-bound style reference and
    must not be compared against H1 as if it were a deployable method.
    """

    name = "oracle_gain"
    needs_counterfactual = True

    def rank(self, candidates, ctx, rng):
        raise NotImplementedError("oracle_gain overrides select")

    def select(self, candidates, budget_tokens, ctx, rng):
        assert_trainable(candidates)
        if ctx.counterfactual is None:
            raise ValueError("oracle_gain needs ctx.counterfactual (simulator only)")
        pool = sorted((e for e in candidates if e.verified or not self.gate_verified), key=lambda e: e.episode_id)
        if not pool:
            return []
        n_stream = len({p.family for p in ctx.probe}) or 1
        median_tokens = statistics.median(e.tokens for e in pool)
        chosen: list[Episode] = []
        used = 0
        base = ctx.counterfactual(chosen)
        while True:
            best, best_u, best_after = None, 0.0, None
            for e in pool:
                if e in chosen or used + e.tokens > budget_tokens:
                    continue
                after = ctx.counterfactual(chosen + [e])
                gain = sum(a - b for a, b in zip(after[:n_stream], base[:n_stream])) / n_stream
                u = gain / (e.tokens / median_tokens)
                if u > best_u:
                    best, best_u, best_after = e, u, after
            if best is None:
                return chosen
            chosen.append(best)
            used += best.tokens
            base = best_after


@dataclass(frozen=True)
class VTARConfig:
    """Provisional VTAR weights. Fixed, transparent, and not tuned."""

    alpha: float = 0.3  # novelty / coverage
    beta: float = 0.3  # residual learning difficulty
    gamma: float = 0.4  # estimated transfer (proxy)
    delta: float = 0.4  # estimated interference (proxy)
    epsilon: float = 1e-6
    use_gating: bool = True
    random_priorities: bool = False


class VTARSelector(Selector):
    """Verified, transfer-aware, interference-aware, cost-normalised selection.

    score(e) = verified(e) * max(0, a*N + b*L + g*T_hat - d*I_hat) / (C + eps)

    N      1 / (1 + times the family was replayed, including this call)
    L      0.5 * logged loss (clipped to [0,1]) + 0.5 * probe failure rate of the family
    T_hat  mean over probe items of max(0, cos(e, probe)) * failed(probe)
    I_hat  max(0, -cos(e, centroid of current-phase data))   (gradient-conflict proxy)
    C      tokens / median tokens of the candidates
    """

    def __init__(self, config: VTARConfig = VTARConfig(), name: str = "vtar"):
        self.config = config
        self.name = name
        self.gate_verified = config.use_gating

    def components(self, candidates: Sequence[Episode], ctx: SelectionContext) -> list[dict]:
        """Static (count-independent) components, in the order given."""
        if not candidates:
            return []
        for e in candidates:
            if e.embedding is None or e.loss is None:
                raise ValueError("VTAR needs loss and embedding on every candidate")
        median_tokens = statistics.median(e.tokens for e in candidates)
        fam_fail: dict[str, list[bool]] = defaultdict(list)
        for p in ctx.probe:
            fam_fail[p.family].append(p.failed)
        out = []
        for e in candidates:
            fails = fam_fail.get(e.family)
            fail_rate = sum(fails) / len(fails) if fails else 0.0
            L = 0.5 * min(1.0, e.loss) + 0.5 * fail_rate
            if ctx.probe:
                T = sum(max(0.0, cosine(e.embedding, p.embedding)) * p.failed for p in ctx.probe) / len(ctx.probe)
            else:
                T = 0.0
            I = max(0.0, -cosine(e.embedding, ctx.current_centroid)) if ctx.current_centroid else 0.0
            out.append(
                {
                    "L": L,
                    "T": T,
                    "I": I,
                    "C": e.tokens / median_tokens,
                    "gate": 1.0 if (e.verified or not self.config.use_gating) else 0.0,
                }
            )
        return out

    def _score(self, comp: dict, n_term: float) -> float:
        c = self.config
        raw = c.alpha * n_term + c.beta * comp["L"] + c.gamma * comp["T"] - c.delta * comp["I"]
        return comp["gate"] * max(0.0, raw) / (comp["C"] + c.epsilon)

    def rank(self, candidates, ctx, rng):
        cands = list(candidates)
        if self.config.random_priorities:
            # own stream, so this control is not identical to UniformReplay
            random.Random(f"vtar_random:{rng.random()}").shuffle(cands)
            return cands
        comps = self.components(cands, ctx)
        counts = defaultdict(int, ctx.replay_counts)
        remaining = list(range(len(cands)))
        out: list[Episode] = []
        while remaining:
            best = max(
                remaining,
                key=lambda i: (self._score(comps[i], 1.0 / (1 + counts[cands[i].family])), -i),
            )
            remaining.remove(best)
            out.append(cands[best])
            counts[cands[best].family] += 1
        return out


def build_policies(vtar: VTARConfig = VTARConfig()) -> dict[str, Selector]:
    """All policies by name, including the VTAR ablations."""
    from dataclasses import replace

    policies: list[Selector] = [
        FrozenPolicy(),
        SequentialFineTune(),
        UniformReplay(),
        LossPrioritizedReplay(),
        NoveltyReplay(),
        CodeItPriorityReplay(),
        VTARSelector(vtar, "vtar"),
        VTARSelector(replace(vtar, alpha=0.0), "vtar_no_novelty"),
        VTARSelector(replace(vtar, gamma=0.0), "vtar_no_transfer"),
        VTARSelector(replace(vtar, delta=0.0), "vtar_no_interference"),
        VTARSelector(replace(vtar, use_gating=False), "vtar_no_gating"),
        VTARSelector(replace(vtar, random_priorities=True), "vtar_random"),
        MeasuredGainOracle(),
    ]
    return {p.name: p for p in policies}
