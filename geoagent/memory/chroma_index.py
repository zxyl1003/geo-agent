"""Chroma index for semantic retrieval of memory items."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geoagent.memory.models import MemoryItem


class ChromaMemoryIndex:
    """Thin Chroma wrapper with SQLite remaining the authoritative store."""

    def __init__(self, path: str | Path, collection_name: str = "geolocation_experience_memory") -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            import chromadb  # type: ignore[import-not-found]

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:  # noqa: BLE001 - normalize dependency and initialization failures.
            raise RuntimeError(f"Chroma memory index initialization failed: {exc}") from exc

    def upsert(self, item: MemoryItem) -> None:
        if self._collection is None:
            raise RuntimeError("Chroma memory index is not initialized.")
        try:
            self._collection.upsert(
                ids=[item.memory_id],
                documents=[item.searchable_text()],
                metadatas=[
                    {
                        "memory_type": item.memory_type,
                        "confidence": item.confidence,
                        "merge_count": item.merge_count,
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Chroma memory index write failed: {exc}") from exc

    def rebuild(self, items: list[MemoryItem]) -> None:
        """Synchronize the index from authoritative SQLite rows."""

        if self._client is None:
            raise RuntimeError("Chroma memory index is not initialized.")
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            for item in items:
                self.upsert(item)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Chroma memory index rebuild failed: {exc}") from exc

    def count(self) -> int:
        if self._collection is None:
            raise RuntimeError("Chroma memory index is not initialized.")
        try:
            return int(self._collection.count())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Chroma memory index count failed: {exc}") from exc

    def query(self, query_text: str, top_k: int, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._collection is None:
            raise RuntimeError("Chroma memory index is not initialized.")
        try:
            result = self._collection.query(query_texts=[query_text], n_results=top_k, where=where)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Chroma memory index query failed: {exc}") from exc

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        hits: list[dict[str, Any]] = []
        for index, memory_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            # Chroma distances are metric-dependent and lower is better. This
            # bounded conversion is only used for ranking/display; SQLite stores
            # the authoritative memory content.
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
            hits.append(
                {
                    "memory_id": memory_id,
                    "score": max(0.0, min(1.0, score)),
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                }
            )
        return hits
