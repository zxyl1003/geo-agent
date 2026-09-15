import re

import pytest

from geoagent.memory.chroma_index import ChromaMemoryIndex


@pytest.fixture
def fake_chroma(monkeypatch):
    """Use an in-memory semantic-index stand-in for memory unit tests."""

    stores = {}

    def initialize(self):
        self._client = object()
        self._collection = object()
        self._items = stores.setdefault(str(self.path), {})

    def upsert(self, item):
        self._items[item.memory_id] = item

    def rebuild(self, items):
        self._items.clear()
        self._items.update({item.memory_id: item for item in items})

    def count(self):
        return len(self._items)

    def query(self, query_text, top_k, where=None):
        query_tokens = set(re.findall(r"[a-z0-9_]{2,}", query_text.lower()))
        hits = []
        for item in self._items.values():
            item_tokens = set(re.findall(r"[a-z0-9_]{2,}", item.searchable_text().lower()))
            score = len(query_tokens & item_tokens) / (len(query_tokens | item_tokens) or 1)
            hits.append(
                {
                    "memory_id": item.memory_id,
                    "score": score,
                    "metadata": {"memory_type": item.memory_type},
                }
            )
        return sorted(hits, key=lambda hit: hit["score"], reverse=True)[:top_k]

    monkeypatch.setattr(ChromaMemoryIndex, "_initialize", initialize)
    monkeypatch.setattr(ChromaMemoryIndex, "upsert", upsert)
    monkeypatch.setattr(ChromaMemoryIndex, "rebuild", rebuild)
    monkeypatch.setattr(ChromaMemoryIndex, "count", count)
    monkeypatch.setattr(ChromaMemoryIndex, "query", query)
