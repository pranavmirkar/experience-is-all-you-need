"""Deterministic synthetic world for testing the harness.

This simulates *skill levels per task family*, not programming. It validates
software correctness and baseline plumbing only. Nothing here is evidence
about real coding models.

Hidden from policies: per-episode difficulty, whether an unverified
correction is bad, the true transfer matrix, and all skill values. Policies
only see ``Episode`` fields, probe items, and their own bookkeeping.

The stream is offline and logged: episodes are generated once per seed from
the base model, so every policy sees the same candidate experiences.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

from .buffer import assert_trainable
from .schemas import Episode, Split, ToolOutput, Verification
from .selection import ProbeItem


@dataclass(frozen=True)
class WorldConfig:
    n_stream_families: int = 6
    n_heldout_families: int = 1
    embedding_dim: int = 8
    episodes_per_phase: int = 60
    steps_per_phase: int = 6
    new_tokens_per_step: int = 1200
    replay_tokens_per_step: int = 1200
    buffer_capacity: int = 150
    p_verified: float = 0.7
    bad_correction_rate: float = 0.3  # of unverified corrections
    base_skill_low: float = 0.15
    base_skill_high: float = 0.35
    eta: float = 0.08  # learning rate on headroom
    bad_penalty: float = 0.3  # fraction of skill lost per bad correction
    forget_rate: float = 0.08  # drift back to base skill per step, if unprotected
    protect_share: float = 0.1  # batch share that fully protects a family
    transfer_mode: str = "aligned"  # aligned | independent | none
    transfer_scale: float = 0.25
    difficulty_gain_slope: float = 0.0
    # An episode's learning signal scales as (tokens / reference_tokens) ** token_value_exp.
    # 1.0: value proportional to length, so cost normalisation has no built-in edge.
    # 0.0: value independent of length, so cheap-first wins by construction.
    token_value_exp: float = 1.0
    reference_tokens: int = 350
    embedding_noise: float = 0.3
    n_probe: int = 10
    n_stream_eval: int = 100
    n_final_test: int = 200

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WorldConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown world config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.transfer_mode not in ("aligned", "independent", "none"):
            raise ValueError(f"bad transfer_mode {cfg.transfer_mode!r}")
        if cfg.n_stream_families < 2:
            raise ValueError("need at least two stream families")
        return cfg

    @property
    def n_families(self) -> int:
        return self.n_stream_families + self.n_heldout_families


@dataclass(frozen=True)
class _Latent:
    difficulty: float
    bad_correction: bool


def _unit(rng: random.Random, dim: int) -> list[float]:
    v = [rng.gauss(0, 1) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are unit vectors


class LockedTestSet:
    """Final test tasks. Policies never receive this object; it can be scored once."""

    def __init__(self, thresholds: list[list[float]]):
        self._u = thresholds
        self._used = False

    def evaluate_once(self, skills: Sequence[float]) -> list[float]:
        if self._used:
            raise RuntimeError("the final test set has already been evaluated")
        self._used = True
        return [sum(u < s for u in us) / len(us) for us, s in zip(self._u, skills)]


class World:
    """Everything seeded and fixed for one (config, seed)."""

    def __init__(self, cfg: WorldConfig, seed: int):
        self.cfg = cfg
        self.seed = seed
        F = cfg.n_families
        self.families = [f"fam{i}" for i in range(F)]
        self.stream_families = self.families[: cfg.n_stream_families]
        self.heldout_families = self.families[cfg.n_stream_families :]
        wr = random.Random(f"world:{seed}")
        z = [_unit(wr, cfg.embedding_dim) for _ in range(F)]
        z_true = z if cfg.transfer_mode == "aligned" else [_unit(wr, cfg.embedding_dim) for _ in range(F)]
        self.base_skill = [wr.uniform(cfg.base_skill_low, cfg.base_skill_high) for _ in range(F)]
        self.transfer = [
            [
                0.0 if (g == h or cfg.transfer_mode == "none") else cfg.transfer_scale * _cos(z_true[g], z_true[h])
                for h in range(F)
            ]
            for g in range(F)
        ]
        self._z = z
        self.latent: dict[str, _Latent] = {}
        self.stream: list[list[Episode]] = self._build_stream()
        self.stream_eval_u = self._thresholds("eval", cfg.n_stream_eval)
        self._probe_u = self._thresholds("probe", cfg.n_probe, only_stream=True)
        self._probe_emb = self._probe_embeddings()
        self._final_u = self._thresholds("final", cfg.n_final_test)
        self._check_disjoint_splits()

    # -- construction -------------------------------------------------------
    def _embed(self, rng: random.Random, fam_idx: int) -> tuple[float, ...]:
        v = [x + self.cfg.embedding_noise * rng.gauss(0, 1) for x in self._z[fam_idx]]
        n = math.sqrt(sum(x * x for x in v))
        return tuple(x / n for x in v)

    def _build_stream(self) -> list[list[Episode]]:
        cfg = self.cfg
        sr = random.Random(f"stream:{self.seed}")
        phases: list[list[Episode]] = []
        idx = 0
        for g, fam in enumerate(self.stream_families):
            phase: list[Episode] = []
            for i in range(cfg.episodes_per_phase):
                d = sr.random()
                verified = sr.random() < cfg.p_verified
                bad = (not verified) and sr.random() < cfg.bad_correction_rate
                attempt_passed = sr.random() < self.base_skill[g]
                loss = min(1.0, max(0.0, d + sr.gauss(0, 0.15)))
                tokens = int(min(1500, max(100, sr.lognormvariate(5.7, 0.4))))
                eid = f"train/{fam}/{i:04d}"
                ver = (
                    Verification(True, "hidden_tests", "synthetic hidden tests (simulated)", True)
                    if verified
                    else Verification(False, "model_assertion")
                )
                ep = Episode(
                    episode_id=eid,
                    created_at=f"2026-01-01T00:00:{idx % 60:02d}+00:00#{idx}",
                    stream_index=idx,
                    family=fam,
                    language="synthetic",
                    source="simulator",
                    split=Split.TRAIN,
                    prompt=f"synthetic task {fam} #{i}",
                    attempt="pass" if attempt_passed else "fail",
                    tool_outputs=(ToolOutput("test", attempt_passed),),
                    correction=None if attempt_passed else "synthetic correction",
                    verification=ver,
                    model_checkpoint="base",
                    adapter_version="none",
                    tokens=tokens,
                    wall_time_s=tokens / 1000.0,
                    license="CC0-1.0 (synthetic)",
                    loss=loss,
                    embedding=self._embed(sr, g),
                )
                self.latent[eid] = _Latent(d, bad)
                phase.append(ep)
                idx += 1
            phases.append(phase)
        return phases

    def _thresholds(self, kind: str, n: int, only_stream: bool = False) -> list[list[float]]:
        n_fam = self.cfg.n_stream_families if only_stream else self.cfg.n_families
        return [
            [random.Random(f"{kind}:{self.seed}:{f}:{j}").random() for j in range(n)]
            for f in range(n_fam)
        ]

    def _probe_embeddings(self) -> list[list[tuple[float, ...]]]:
        out = []
        for f in range(self.cfg.n_stream_families):
            r = random.Random(f"probe_emb:{self.seed}:{f}")
            out.append([self._embed(r, f) for _ in range(self.cfg.n_probe)])
        return out

    def _check_disjoint_splits(self) -> None:
        # Tasks are identified by (split, family, index); thresholds come from
        # differently-seeded streams, and train ids carry their own namespace.
        train_ids = set(self.latent)
        if len(train_ids) != self.cfg.episodes_per_phase * self.cfg.n_stream_families:
            raise RuntimeError("duplicate training episode ids")
        if any(not i.startswith("train/") for i in train_ids):
            raise RuntimeError("non-train id in the training stream")

    # -- views --------------------------------------------------------------
    @property
    def train_episode_ids(self) -> frozenset[str]:
        return frozenset(self.latent)

    def locked_test(self) -> LockedTestSet:
        return LockedTestSet(self._final_u)

    def stream_eval(self, skills: Sequence[float]) -> list[list[bool]]:
        """Per-family, per-task pass flags on the stream-evaluation dev tasks."""
        return [[u < s for u in us] for us, s in zip(self.stream_eval_u, skills)]

    def probe(self, skills: Sequence[float]) -> tuple[ProbeItem, ...]:
        """Dev probe tasks (stream families only) as policies may see them."""
        items = []
        for f, fam in enumerate(self.stream_families):
            for emb, u in zip(self._probe_emb[f], self._probe_u[f]):
                items.append(ProbeItem(fam, emb, not (u < skills[f])))
        return tuple(items)


class Learner:
    """The simulated model: one skill per family. Updates follow hidden dynamics."""

    def __init__(self, world: World):
        self.world = world
        self.skills = list(world.base_skill)
        self.update_tokens = 0
        self.update_steps = 0
        self.trained_ids: list[str] = []
        self._fam_idx = {f: i for i, f in enumerate(world.families)}

    def step(self, batch: Sequence[Episode]) -> None:
        """One optimiser step on ``batch``."""
        assert_trainable(batch)
        if not batch:
            return
        cfg, w, s = self.world.cfg, self.world, self.skills
        token_share = [0] * cfg.n_families
        for e in batch:
            g = self._fam_idx[e.family]
            token_share[g] += e.tokens
            lat = w.latent[e.episode_id]
            if lat.bad_correction:
                s[g] -= cfg.bad_penalty * s[g]
            else:
                value = (e.tokens / cfg.reference_tokens) ** cfg.token_value_exp
                strength = min(1.0, cfg.eta * value * (1 + cfg.difficulty_gain_slope * (lat.difficulty - 0.5)))
                gain = strength * (1 - s[g])
                s[g] += gain
                for h in range(cfg.n_families):
                    if h != g:
                        s[h] = min(1.0, max(0.0, s[h] + w.transfer[g][h] * gain))
            self.trained_ids.append(e.episode_id)
            self.update_tokens += e.tokens
        batch_tokens = sum(token_share)
        for h in range(cfg.n_families):
            protection = min(1.0, token_share[h] / batch_tokens / cfg.protect_share)
            s[h] -= cfg.forget_rate * max(0.0, s[h] - w.base_skill[h]) * (1 - protection)
        self.update_steps += 1
