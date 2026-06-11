"""Source abstraction — the contract every ingestor implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from planetar_acoustic.bus.topics import HydrophoneSite
from planetar_acoustic.ingest.audio import Clip


@dataclass(frozen=True)
class SourceClip:
    """A `Clip` paired with the hydrophone it came from."""
    clip: Clip
    site: HydrophoneSite


class Source(ABC):
    """A long-running or finite stream of hydrophone clips."""

    sample_rate: int
    clip_seconds: float

    @abstractmethod
    def sites(self) -> list[HydrophoneSite]:
        """Hydrophone sites this source covers (1+). Used to emit acoustic.site once."""

    @abstractmethod
    def stream(self) -> Iterator[SourceClip]:
        """Yield SourceClips. Long-running sources block between yields; finite
        sources return when exhausted. Implementations must be cancel-safe —
        a SIGINT / KeyboardInterrupt in the caller should not leak file
        descriptors or subprocesses.
        """
