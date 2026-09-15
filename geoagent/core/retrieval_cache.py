"""Persistent cache for deterministic, billable retrieval tools."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from geoagent.core.config import AppConfig


CACHE_SCHEMA_VERSION = 1
_SCHEMA_LOCK = threading.Lock()


class RetrievalCache:
    """Store successful normalized retrieval responses in a local SQLite file."""

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._initialized = False

    @classmethod
    def from_config(cls, app_config: AppConfig) -> "RetrievalCache":
        config = app_config.system.get("search_cache") or {}
        path = Path(config.get("path") or "outputs/cache/web_cache/search_cache.sqlite3")
        if not path.is_absolute():
            path = Path(app_config.config_dir).resolve().parent / path
        return cls(
            path,
            enabled=bool(config.get("enabled", True)),
        )

    def key(self, tool_name: str, provider: str, request: dict[str, Any]) -> tuple[str, str]:
        canonical = json.dumps(_canonicalize(request), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        payload = f"{CACHE_SCHEMA_VERSION}\n{tool_name}\n{provider}\n{canonical}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), canonical

    def get(self, tool_name: str, provider: str, request: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            self._ensure_schema()
            cache_key, _ = self.key(tool_name, provider, request)
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT response_json FROM retrieval_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
                if row is None:
                    return None
                connection.execute(
                    "UPDATE retrieval_cache SET hit_count = hit_count + 1, last_accessed_at = ? WHERE cache_key = ?",
                    (_now(), cache_key),
                )
            data = json.loads(row[0])
            return data if isinstance(data, dict) else None
        except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            logger.warning("Retrieval cache read skipped for {}/{}: {}", tool_name, provider, exc)
            return None

    def put(self, tool_name: str, provider: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._ensure_schema()
            cache_key, request_json = self.key(tool_name, provider, request)
            response_json = json.dumps(response, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            now = _now()
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO retrieval_cache (
                        cache_key, schema_version, tool_name, provider, request_json,
                        response_json, created_at, last_accessed_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(cache_key) DO NOTHING
                    """,
                    (
                        cache_key,
                        CACHE_SCHEMA_VERSION,
                        tool_name,
                        provider,
                        request_json,
                        response_json,
                        now,
                        now,
                    ),
                )
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            logger.warning("Retrieval cache write skipped for {}/{}: {}", tool_name, provider, exc)

    def _ensure_schema(self) -> None:
        if self._initialized or not self.enabled:
            return
        with _SCHEMA_LOCK:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS retrieval_cache (
                        cache_key TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        tool_name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_accessed_at TEXT NOT NULL,
                        hit_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_retrieval_cache_tool_provider "
                    "ON retrieval_cache(tool_name, provider)"
                )
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
