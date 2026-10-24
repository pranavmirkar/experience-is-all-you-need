"""Versioned, immutable experience records.

An :class:`Episode` is one logged programming-task interaction. Records are
frozen so that split metadata cannot be changed after creation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

SCHEMA_VERSION = "1.0"

# Verification must come from an external procedure. A model asserting that its
# own correction is right ("model_assertion") never counts.
EXTERNAL_VERIFICATION_METHODS = frozenset(
    {"compiler", "unit_tests", "hidden_tests", "property_tests", "human_review"}
)


class SchemaError(ValueError):
    """Raised when a record violates the schema."""


class Split(str, enum.Enum):
    TRAIN = "train"
    DEV = "dev"
    FINAL_TEST = "final_test"


@dataclass(frozen=True)
class ToolOutput:
    """Output of one tool run (compiler, test runner, static analyser...)."""

    kind: str
    passed: Optional[bool]
    text: str = ""


@dataclass(frozen=True)
class Verification:
    """Evidence that a solution was checked by something other than the model."""

    verified: bool
    method: str
    procedure: str = ""
    independent: bool = False

    def __post_init__(self) -> None:
        if not self.verified:
            return
        if self.method not in EXTERNAL_VERIFICATION_METHODS:
            raise SchemaError(
                f"method {self.method!r} cannot establish verification; "
                f"use one of {sorted(EXTERNAL_VERIFICATION_METHODS)}"
            )
        if not self.independent:
            raise SchemaError("verified=True requires independent=True")
        if not self.procedure:
            raise SchemaError("verified=True requires the exact procedure")


@dataclass(frozen=True)
class Episode:
    """One experience record.

    ``loss`` and ``embedding`` are selection features visible to policies.
    They are optional because real episodes may not have them yet.
    """

    episode_id: str
    created_at: str
    stream_index: int
    family: str
    language: str
    source: str
    split: Split
    prompt: str
    attempt: str
    tool_outputs: tuple[ToolOutput, ...]
    correction: Optional[str]
    verification: Verification
    model_checkpoint: str
    adapter_version: str
    tokens: int
    wall_time_s: float
    license: str
    loss: Optional[float] = None
    embedding: Optional[tuple[float, ...]] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise SchemaError("episode_id must be non-empty")
        if not isinstance(self.split, Split):
            raise SchemaError(f"split must be a Split, got {self.split!r}")
        if self.tokens <= 0:
            raise SchemaError("tokens must be positive")
        if self.wall_time_s < 0:
            raise SchemaError("wall_time_s must be non-negative")
        if self.stream_index < 0:
            raise SchemaError("stream_index must be non-negative")
        if self.loss is not None and self.loss < 0:
            raise SchemaError("loss must be non-negative")

    @property
    def verified(self) -> bool:
        return self.verification.verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "created_at": self.created_at,
            "stream_index": self.stream_index,
            "family": self.family,
            "language": self.language,
            "source": self.source,
            "split": self.split.value,
            "prompt": self.prompt,
            "attempt": self.attempt,
            "tool_outputs": [vars(t) | {} for t in self.tool_outputs],
            "correction": self.correction,
            "verification": vars(self.verification) | {},
            "model_checkpoint": self.model_checkpoint,
            "adapter_version": self.adapter_version,
            "tokens": self.tokens,
            "wall_time_s": self.wall_time_s,
            "license": self.license,
            "loss": self.loss,
            "embedding": list(self.embedding) if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Episode":
        if d.get("schema_version") != SCHEMA_VERSION:
            raise SchemaError(f"unsupported schema_version {d.get('schema_version')!r}")
        emb = d.get("embedding")
        return cls(
            episode_id=d["episode_id"],
            created_at=d["created_at"],
            stream_index=d["stream_index"],
            family=d["family"],
            language=d["language"],
            source=d["source"],
            split=Split(d["split"]),
            prompt=d["prompt"],
            attempt=d["attempt"],
            tool_outputs=tuple(ToolOutput(**t) for t in d["tool_outputs"]),
            correction=d["correction"],
            verification=Verification(**d["verification"]),
            model_checkpoint=d["model_checkpoint"],
            adapter_version=d["adapter_version"],
            tokens=d["tokens"],
            wall_time_s=d["wall_time_s"],
            license=d["license"],
            loss=d.get("loss"),
            embedding=tuple(emb) if emb is not None else None,
        )
