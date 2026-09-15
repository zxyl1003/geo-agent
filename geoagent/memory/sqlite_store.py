"""SQLite-backed authoritative store for geolocation experience memory."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from geoagent.core.schemas import utc_now
from geoagent.memory.models import MemoryCandidate, MemoryItem


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class SQLiteMemoryStore:
    """Small, reproducible SQLite store.

    SQLite is the source of truth. Vector/text indexes such as Chroma are treated
    as rebuildable accelerators.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def ensure_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    episode_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    user_query TEXT,
                    scene_type TEXT,
                    visual_conditions_json TEXT NOT NULL DEFAULT '{}',
                    visual_clues_json TEXT NOT NULL DEFAULT '[]',
                    observed_entities_json TEXT NOT NULL DEFAULT '[]',
                    ocr_results_json TEXT NOT NULL DEFAULT '[]',
                    hypotheses_json TEXT NOT NULL DEFAULT '[]',
                    final_answer_json TEXT,
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    tool_results_json TEXT NOT NULL DEFAULT '[]',
                    feedback_type TEXT NOT NULL DEFAULT 'none',
                    ground_truth_json TEXT,
                    outcome_json TEXT NOT NULL DEFAULT '{}',
                    success INTEGER,
                    error_distance_m REAL,
                    reasoning_trace_json TEXT NOT NULL DEFAULT '[]',
                    ground_truth_context_json TEXT NOT NULL DEFAULT '{}',
                    attribution_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    situation TEXT NOT NULL,
                    lesson TEXT NOT NULL,
                    action_policy_json TEXT NOT NULL DEFAULT '[]',
                    applicable_conditions_json TEXT NOT NULL DEFAULT '[]',
                    failure_conditions_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL,
                    merge_count INTEGER NOT NULL DEFAULT 0,
                    feedback_type TEXT NOT NULL DEFAULT 'none',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_evidence_links (
                    memory_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    diagnosis_confidence REAL,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, episode_id, relation),
                    FOREIGN KEY(memory_id) REFERENCES memory_items(memory_id)
                );

                CREATE TABLE IF NOT EXISTS memory_usage (
                    episode_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    retrieved_rank INTEGER,
                    retrieval_score REAL,
                    was_returned_to_brain INTEGER NOT NULL DEFAULT 1,
                    was_cited_by_brain INTEGER,
                    usage_role TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (episode_id, memory_id)
                );


                CREATE INDEX IF NOT EXISTS idx_memory_items_type ON memory_items(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memory_links_episode ON memory_evidence_links(episode_id);
                """
            )
    def insert_episode(self, episode: dict[str, Any]) -> str:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_episodes (
                    episode_id, task_id, image_path, user_query, scene_type,
                    visual_conditions_json, visual_clues_json, observed_entities_json,
                    ocr_results_json, hypotheses_json, final_answer_json, tool_calls_json,
                    tool_results_json, feedback_type, ground_truth_json, outcome_json, success,
                    error_distance_m, reasoning_trace_json, ground_truth_context_json, attribution_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode["episode_id"],
                    episode["task_id"],
                    episode["image_path"],
                    episode.get("user_query"),
                    episode.get("scene_type"),
                    _json_dump(episode.get("visual_conditions") or {}),
                    _json_dump(episode.get("visual_clues") or []),
                    _json_dump(episode.get("observed_entities") or []),
                    _json_dump(episode.get("ocr_results") or []),
                    _json_dump(episode.get("hypotheses") or []),
                    _json_dump(episode.get("final_answer")) if episode.get("final_answer") is not None else None,
                    _json_dump(episode.get("tool_calls") or []),
                    _json_dump(episode.get("tool_results") or []),
                    episode.get("feedback_type") or "none",
                    _json_dump(episode.get("ground_truth")) if episode.get("ground_truth") is not None else None,
                    _json_dump(episode.get("outcome") or {}),
                    None if episode.get("success") is None else int(bool(episode.get("success"))),
                    episode.get("error_distance_m"),
                    _json_dump(episode.get("reasoning_trace") or []),
                    _json_dump(episode.get("ground_truth_context") or {}),
                    _json_dump(episode.get("attribution") or {}),
                    str(episode.get("created_at") or utc_now()),
                ),
            )
        return str(episode["episode_id"])

    def update_episode_attribution(self, episode_id: str, attribution: dict[str, Any] | None) -> None:
        """Persist the reflection attribution even when no memory is written."""
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE memory_episodes SET attribution_json = ? WHERE episode_id = ?",
                (_json_dump(attribution or {}), str(episode_id)),
            )

    def insert_memory(self, candidate: MemoryCandidate) -> MemoryItem:
        now = utc_now()
        item = MemoryItem(
            memory_type=candidate.memory_type,
            situation=candidate.situation,
            lesson=candidate.lesson,
            action_policy=candidate.action_policy,
            applicable_conditions=candidate.applicable_conditions,
            failure_conditions=candidate.failure_conditions,
            confidence=candidate.confidence,
            feedback_type=candidate.feedback_type,
            created_at=now,
            updated_at=now,
            metadata=candidate.metadata,
        )
        self.upsert_memory(item)
        self.link_memory_episode(
            item.memory_id,
            candidate.source_episode_id,
            "source",
            diagnosis_confidence=candidate.diagnosis_confidence,
            note="initial memory write",
        )
        return item
    def upsert_memory(self, item: MemoryItem) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_items (
                    memory_id, memory_type, situation, lesson,
                    action_policy_json, applicable_conditions_json, failure_conditions_json,
                    confidence, merge_count, feedback_type, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    situation = excluded.situation,
                    lesson = excluded.lesson,
                    action_policy_json = excluded.action_policy_json,
                    applicable_conditions_json = excluded.applicable_conditions_json,
                    failure_conditions_json = excluded.failure_conditions_json,
                    confidence = excluded.confidence,
                    merge_count = excluded.merge_count,
                    feedback_type = excluded.feedback_type,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    item.memory_id,
                    item.memory_type,
                    item.situation,
                    item.lesson,
                    _json_dump(item.action_policy),
                    _json_dump(item.applicable_conditions),
                    _json_dump(item.failure_conditions),
                    item.confidence,
                    item.merge_count,
                    item.feedback_type,
                    _json_dump(item.metadata),
                    str(item.created_at),
                    str(item.updated_at),
                ),
            )
    def update_memory(self, item: MemoryItem, *, touch: bool = True) -> None:
        if touch:
            item.updated_at = utc_now()
        self.upsert_memory(item)

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM memory_items WHERE memory_id = ?", (memory_id,)).fetchone()
        return self._item_from_row(row) if row else None

    def list_memories(self, limit: int = 500) -> list[MemoryItem]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_items ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._item_from_row(row) for row in rows]
    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_episodes WHERE episode_id = ?", (str(episode_id),)
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def link_memory_episode(
        self,
        memory_id: str,
        episode_id: str,
        relation: str,
        diagnosis_confidence: float | None = None,
        note: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_evidence_links (
                    memory_id, episode_id, relation, diagnosis_confidence, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, episode_id, relation, diagnosis_confidence, note, str(utc_now())),
            )

    def record_usage(self, episode_id: str, usages: list[dict[str, Any]]) -> None:
        if not usages:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO memory_usage (
                    episode_id, memory_id, retrieved_rank, retrieval_score,
                    was_returned_to_brain, was_cited_by_brain, usage_role, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        episode_id,
                        usage.get("memory_id"),
                        usage.get("retrieved_rank"),
                        usage.get("retrieval_score"),
                        int(bool(usage.get("was_returned_to_brain", True))),
                        None
                        if usage.get("was_cited_by_brain") is None
                        else int(bool(usage.get("was_cited_by_brain"))),
                        usage.get("usage_role"),
                        str(utc_now()),
                    )
                    for usage in usages
                    if usage.get("memory_id")
                ],
            )

    def get_usage(self, episode_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_usage WHERE episode_id = ? ORDER BY retrieved_rank",
                (episode_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _item_from_row(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"],
            memory_type=row["memory_type"],
            situation=row["situation"],
            lesson=row["lesson"],
            action_policy=_json_load(row["action_policy_json"], []),
            applicable_conditions=_json_load(row["applicable_conditions_json"], []),
            failure_conditions=_json_load(row["failure_conditions_json"], []),
            confidence=float(row["confidence"]),
            merge_count=int(row["merge_count"]),
            feedback_type=row["feedback_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_json_load(row["metadata_json"], {}),
        )
