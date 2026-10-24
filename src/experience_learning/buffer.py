"""Split-safe experience buffer.

Only TRAIN episodes can enter. DEV and FINAL_TEST episodes raise
:class:`LeakageError`. With ``require_verified`` set, unverified episodes are
rejected and counted.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from .schemas import Episode, Split


class LeakageError(RuntimeError):
    """A non-TRAIN episode reached a training or selection path."""


def assert_trainable(episodes: Iterable[Episode]) -> None:
    """Raise :class:`LeakageError` unless every episode is in the TRAIN split."""
    for e in episodes:
        if e.split is not Split.TRAIN:
            raise LeakageError(
                f"episode {e.episode_id} has split {e.split.value!r}; "
                "only TRAIN episodes may be trained on or selected"
            )


class ExperienceBuffer:
    """Fixed-capacity reservoir of TRAIN episodes.

    Admission uses reservoir sampling with its own seeded RNG, so the buffer
    contents depend only on the stream and the seed, never on the policy.
    """

    def __init__(self, capacity: int, require_verified: bool, rng: random.Random):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.require_verified = require_verified
        self._rng = rng
        self._items: list[Episode] = []
        self._ids: set[str] = set()
        self._seen = 0
        self.rejected_unverified = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, episode: Episode) -> bool:
        """Offer an episode. Returns True if it was admitted."""
        assert_trainable([episode])
        if episode.episode_id in self._ids:
            raise ValueError(f"duplicate episode_id {episode.episode_id}")
        if self.require_verified and not episode.verified:
            self.rejected_unverified += 1
            return False
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(episode)
            self._ids.add(episode.episode_id)
            return True
        j = self._rng.randrange(self._seen)
        if j < self.capacity:
            self._ids.discard(self._items[j].episode_id)
            self._items[j] = episode
            self._ids.add(episode.episode_id)
            return True
        return False

    def eligible(self, before_stream_index: int) -> Sequence[Episode]:
        """Buffered episodes from earlier in the stream, in a stable order."""
        out = [e for e in self._items if e.stream_index < before_stream_index]
        out.sort(key=lambda e: (e.stream_index, e.episode_id))
        return tuple(out)
