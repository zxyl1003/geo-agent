"""External experience memory for geolocation agents."""

from typing import Any

from geoagent.memory.models import (
    MemoryAgentReview,
    MemoryCandidate,
    MemoryItem,
    RetrievedMemory,
)

__all__ = [
    "MemoryAgentReview",
    "MemoryCandidate",
    "MemoryItem",
    "MemoryManager",
    "RetrievedMemory",
]


def __getattr__(name: str) -> Any:
    """Load the manager lazily so the memory agent can import model contracts."""

    if name == "MemoryManager":
        from geoagent.memory.manager import MemoryManager

        return MemoryManager
    raise AttributeError(name)
