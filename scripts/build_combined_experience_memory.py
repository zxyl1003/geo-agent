"""Build a deduplicated experience-memory database from three reviewed stores."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_DATABASES = {
    "gemma": REPO_ROOT / "outputs/memory/geoexp7k_learning/memory/memory.sqlite",
    "qwen": REPO_ROOT / "outputs/memory/geoexp7k_learning/memory_qwen/memory.sqlite",
    "legacy": REPO_ROOT / "outputs/memory/geoexp13k_mixed_offline_memory/memory/memory.sqlite",
}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/memory/geoexp7k_combined_curated"

# Each group was reviewed semantically. The canonical record supplies the concise
# retrieval text; provenance and evidence from every member are retained.
MERGE_GROUPS = [
    {
        "canonical": "gemma:mem_0f09a76a187c",
        "members": [
            "gemma:mem_0f09a76a187c",
            "gemma:mem_2989ef85a8b4",
            "gemma:mem_58028bfa751e",
            "gemma:mem_8b074fc3ee74",
        ],
        "reason": "Multiple independent local POIs should be localized through co-location.",
    },
    {
        "canonical": "legacy:mem_76f9a088461b",
        "members": [
            "legacy:mem_76f9a088461b",
            "gemma:mem_30b305439986",
            "gemma:mem_8726bd48820a",
        ],
        "reason": "Chain branches require branch-specific or co-located POI evidence.",
    },
    {
        "canonical": "legacy:mem_c2396cc0cba1",
        "members": [
            "legacy:mem_c2396cc0cba1",
            "gemma:mem_634d6cb74210",
            "gemma:mem_5d9c26c803da",
            "gemma:mem_b655e73e6687",
        ],
        "reason": "A specific entity plus phone, address, or website is one identifier strategy.",
    },
    {
        "canonical": "legacy:mem_4dd27408f34c",
        "members": [
            "legacy:mem_4dd27408f34c",
            "gemma:mem_608b683ea30f",
            "gemma:mem_84297850c199",
        ],
        "reason": "A specific named POI should be searched first and independently verified.",
    },
    {
        "canonical": "gemma:mem_9ec9dbce8e6b",
        "members": [
            "gemma:mem_9ec9dbce8e6b",
            "gemma:mem_066ff55b8b25",
        ],
        "reason": "Partial or ambiguous entity text should be searched with spelling variants.",
    },
    {
        "canonical": "qwen:mem_42d1eeb90583",
        "members": [
            "qwen:mem_42d1eeb90583",
            "qwen:mem_3f1bd26c993c",
            "qwen:mem_c0f835607022",
        ],
        "reason": "Generic branch signage is insufficient without a unique local visual anchor.",
    },
    {
        "canonical": "qwen:mem_5d3f75238063",
        "members": [
            "qwen:mem_5d3f75238063",
            "qwen:mem_0800470caf7f",
        ],
        "reason": "A transient tool failure should not discard a unique searchable POI.",
    },
    {
        "canonical": "qwen:mem_91b911ce2f21",
        "members": [
            "qwen:mem_91b911ce2f21",
            "qwen:mem_fefc60880fe5",
        ],
        "reason": "Co-occurring ubiquitous chains do not justify branch-level precision.",
    },
    {
        "canonical": "qwen:mem_cdbbcae99767",
        "members": [
            "qwen:mem_cdbbcae99767",
            "qwen:mem_5aa6c25b0f97",
        ],
        "reason": "Generic branch names must be disambiguated with contact details.",
    },
    {
        "canonical": "qwen:mem_b9b6a27cfa18",
        "members": [
            "qwen:mem_b9b6a27cfa18",
            "qwen:mem_4e7f608c9bcb",
        ],
        "reason": "Unique local names are stronger anchors than generic or common names.",
    },
    {
        "canonical": "qwen:mem_6e3377d80b4c",
        "members": [
            "qwen:mem_6e3377d80b4c",
            "qwen:mem_3eec9e5b1e37",
        ],
        "reason": "Generic storefront verification cannot independently disambiguate a city.",
    },
]

EPISODE_COLUMNS = [
    "episode_id",
    "task_id",
    "image_path",
    "user_query",
    "scene_type",
    "visual_conditions_json",
    "visual_clues_json",
    "observed_entities_json",
    "ocr_results_json",
    "hypotheses_json",
    "final_answer_json",
    "tool_calls_json",
    "tool_results_json",
    "feedback_type",
    "ground_truth_json",
    "outcome_json",
    "success",
    "error_distance_m",
    "reasoning_trace_json",
    "ground_truth_context_json",
    "attribution_json",
    "created_at",
]

SCHEMA_SQL = """
CREATE TABLE memory_episodes (
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
CREATE TABLE memory_items (
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
CREATE TABLE memory_evidence_links (
    memory_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    diagnosis_confidence REAL,
    note TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, episode_id, relation),
    FOREIGN KEY(memory_id) REFERENCES memory_items(memory_id)
);
CREATE TABLE memory_usage (
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
CREATE INDEX idx_memory_items_type ON memory_items(memory_type);
CREATE INDEX idx_memory_links_episode ON memory_evidence_links(episode_id);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="New output directory. Existing directories are rejected.",
    )
    return parser.parse_args()


def load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def memory_polarity(metadata: dict[str, Any]) -> str:
    if str(metadata.get("success_pattern") or "").strip():
        return "positive"
    if str(metadata.get("failure_type") or "").strip():
        return "negative"
    return "unspecified"


def fetch_rows(connection: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return connection.execute(f"SELECT * FROM {table}").fetchall()


def append_unique(values: list[Any], value: Any) -> None:
    if value not in (None, "", [], {}) and value not in values:
        values.append(value)


def rebuild_chroma_if_available(memory_dir: Path) -> tuple[bool, int | None, str | None]:
    """Build the optional index when the active Python has project dependencies."""

    try:
        import chromadb  # noqa: F401, PLC0415
        import pydantic  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:
        return False, None, f"{exc.name} is not installed in the active Python environment"

    from geoagent.memory.chroma_index import ChromaMemoryIndex  # noqa: PLC0415
    from geoagent.memory.sqlite_store import SQLiteMemoryStore  # noqa: PLC0415

    store = SQLiteMemoryStore(memory_dir / "memory.sqlite")
    items = store.list_memories(limit=1_000_000)
    index = ChromaMemoryIndex(memory_dir / "chroma")
    index.rebuild(items)
    return True, index.count(), None


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")

    for source_path in SOURCE_DATABASES.values():
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    source_connections: dict[str, sqlite3.Connection] = {}
    source_rows: dict[str, dict[str, sqlite3.Row]] = {}
    selected: dict[str, sqlite3.Row] = {}
    source_counts: dict[str, dict[str, int]] = {}

    try:
        for source, source_path in SOURCE_DATABASES.items():
            connection = sqlite3.connect(source_path)
            connection.row_factory = sqlite3.Row
            source_connections[source] = connection
            rows = fetch_rows(connection, "memory_items")
            source_rows[source] = {str(row["memory_id"]): row for row in rows}

            included = 0
            for row in rows:
                metadata = load_json(row["metadata_json"], {})
                if source == "gemma" and memory_polarity(metadata) != "positive":
                    continue
                key = f"{source}:{row['memory_id']}"
                selected[key] = row
                included += 1

            source_counts[source] = {"available": len(rows), "selected": included}

        expected_selection = {"gemma": 16, "qwen": 80, "legacy": 11}
        actual_selection = {source: counts["selected"] for source, counts in source_counts.items()}
        if actual_selection != expected_selection:
            raise ValueError(
                f"Source selection changed; expected {expected_selection}, got {actual_selection}. "
                "Review the merge plan before rebuilding."
            )

        target_by_source: dict[str, str] = {}
        group_by_canonical: dict[str, dict[str, Any]] = {}
        grouped_members: set[str] = set()
        for group in MERGE_GROUPS:
            canonical = str(group["canonical"])
            members = [str(member) for member in group["members"]]
            if canonical not in members:
                raise ValueError(f"Canonical record is not a member of its group: {canonical}")
            missing = [member for member in members if member not in selected]
            if missing:
                raise KeyError(f"Merge group refers to missing records: {missing}")
            overlap = grouped_members.intersection(members)
            if overlap:
                raise ValueError(f"Records appear in more than one merge group: {sorted(overlap)}")

            polarities = {
                memory_polarity(load_json(selected[member]["metadata_json"], {})) for member in members
            }
            if len(polarities) != 1:
                raise ValueError(f"Cannot merge different memory polarities in {members}: {polarities}")

            target_id = canonical.split(":", maxsplit=1)[1]
            for member in members:
                target_by_source[member] = target_id
            group_by_canonical[canonical] = group
            grouped_members.update(members)

        for key, row in selected.items():
            target_by_source.setdefault(key, str(row["memory_id"]))

        target_ids = set(target_by_source.values())
        expected_final_count = len(selected) - sum(len(group["members"]) - 1 for group in MERGE_GROUPS)
        if len(target_ids) != expected_final_count:
            raise ValueError("Canonical memory IDs collide across otherwise distinct records.")

        output_memory_dir = output_root / "memory"
        output_db = output_memory_dir / "memory.sqlite"
        output_memory_dir.mkdir(parents=True)
        with sqlite3.connect(output_db) as schema_connection:
            schema_connection.executescript(SCHEMA_SQL)

        members_by_target: dict[str, list[str]] = defaultdict(list)
        for source_key, target_id in target_by_source.items():
            members_by_target[target_id].append(source_key)

        evidence_rows: list[tuple[str, sqlite3.Row]] = []
        usage_rows: list[tuple[str, sqlite3.Row]] = []
        for source, connection in source_connections.items():
            for row in fetch_rows(connection, "memory_evidence_links"):
                source_key = f"{source}:{row['memory_id']}"
                if source_key in selected:
                    evidence_rows.append((source, row))
            for row in fetch_rows(connection, "memory_usage"):
                source_key = f"{source}:{row['memory_id']}"
                if source_key in selected:
                    usage_rows.append((source, row))

        episode_ids_by_target: dict[str, list[str]] = defaultdict(list)
        for source, row in evidence_rows:
            target_id = target_by_source[f"{source}:{row['memory_id']}"]
            append_unique(episode_ids_by_target[target_id], str(row["episode_id"]))

        now = datetime.now(timezone.utc)
        output_items: list[dict[str, Any]] = []
        for target_id, member_keys in sorted(members_by_target.items()):
            canonical_key = next(
                (
                    key
                    for key in member_keys
                    if key in group_by_canonical
                    and group_by_canonical[key]["canonical"].split(":", maxsplit=1)[1] == target_id
                ),
                member_keys[0],
            )
            canonical_row = selected[canonical_key]
            canonical_metadata = load_json(canonical_row["metadata_json"], {})

            source_records = []
            merged_from_ids: list[str] = []
            source_databases: list[str] = []
            cue_categories: list[str] = []
            tool_scenarios: list[str] = []
            confidences: list[float] = []
            merge_counts: list[int] = []
            created_times: list[str] = []
            for member_key in member_keys:
                source, memory_id = member_key.split(":", maxsplit=1)
                row = selected[member_key]
                metadata = load_json(row["metadata_json"], {})
                source_records.append(
                    {
                        "source": source,
                        "memory_id": memory_id,
                        "memory_type": row["memory_type"],
                        "confidence": float(row["confidence"]),
                    }
                )
                append_unique(source_databases, source)
                append_unique(merged_from_ids, memory_id)
                for prior_id in metadata.get("merged_from_memory_ids") or []:
                    append_unique(merged_from_ids, prior_id)
                append_unique(cue_categories, metadata.get("cue_category"))
                append_unique(tool_scenarios, metadata.get("tool_scenario"))
                confidences.append(float(row["confidence"]))
                merge_counts.append(int(row["merge_count"]))
                created_times.append(str(row["created_at"]))

            group = group_by_canonical.get(canonical_key)
            metadata = dict(canonical_metadata)
            metadata.update(
                {
                    "review_mode": "combined_memory_deduplication",
                    "source_databases": sorted(source_databases),
                    "source_memory_records": source_records,
                    "merged_from_memory_ids": merged_from_ids,
                    "source_episode_ids": episode_ids_by_target[target_id],
                    "merged_cue_categories": cue_categories,
                    "merged_tool_scenarios": tool_scenarios,
                    "deduplication": {
                        "method": "manual_semantic_groups_with_audited_embedding_support",
                        "group_size": len(member_keys),
                        "reason": group["reason"] if group else "No duplicate merged.",
                    },
                }
            )
            output_items.append(
                {
                    "memory_id": target_id,
                    "memory_type": canonical_row["memory_type"],
                    "situation": canonical_row["situation"],
                    "lesson": canonical_row["lesson"],
                    "action_policy_json": canonical_row["action_policy_json"],
                    "applicable_conditions_json": canonical_row["applicable_conditions_json"],
                    "failure_conditions_json": canonical_row["failure_conditions_json"],
                    "confidence": max(confidences),
                    "merge_count": sum(merge_counts) + len(member_keys) - 1,
                    "feedback_type": canonical_row["feedback_type"],
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                    "created_at": min(created_times),
                    "updated_at": now.isoformat(),
                }
            )

        required_episode_ids: set[str] = {str(row["episode_id"]) for _, row in evidence_rows}
        required_episode_ids.update(str(row["episode_id"]) for _, row in usage_rows)

        episode_candidates: dict[str, tuple[int, sqlite3.Row]] = {}
        source_priority = {"legacy": 1, "gemma": 2, "qwen": 3}
        for source, connection in source_connections.items():
            placeholders = ",".join("?" for _ in required_episode_ids)
            if not placeholders:
                continue
            rows = connection.execute(
                f"SELECT * FROM memory_episodes WHERE episode_id IN ({placeholders})",
                tuple(sorted(required_episode_ids)),
            ).fetchall()
            for row in rows:
                episode_id = str(row["episode_id"])
                candidate = (source_priority[source], row)
                if episode_id not in episode_candidates or candidate[0] > episode_candidates[episode_id][0]:
                    episode_candidates[episode_id] = candidate

        missing_episodes = required_episode_ids.difference(episode_candidates)
        if missing_episodes:
            raise KeyError(f"Evidence or usage references missing episodes: {sorted(missing_episodes)}")

        output_connection = sqlite3.connect(output_db)
        output_connection.execute("PRAGMA foreign_keys=ON")
        try:
            for item in output_items:
                output_connection.execute(
                    """
                    INSERT INTO memory_items (
                        memory_id, memory_type, situation, lesson,
                        action_policy_json, applicable_conditions_json, failure_conditions_json,
                        confidence, merge_count, feedback_type, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["memory_id"],
                        item["memory_type"],
                        item["situation"],
                        item["lesson"],
                        item["action_policy_json"],
                        item["applicable_conditions_json"],
                        item["failure_conditions_json"],
                        item["confidence"],
                        item["merge_count"],
                        item["feedback_type"],
                        item["metadata_json"],
                        item["created_at"],
                        item["updated_at"],
                    ),
                )

            episode_column_sql = ", ".join(EPISODE_COLUMNS)
            episode_placeholders = ", ".join("?" for _ in EPISODE_COLUMNS)
            for episode_id in sorted(episode_candidates):
                row = episode_candidates[episode_id][1]
                output_connection.execute(
                    f"INSERT INTO memory_episodes ({episode_column_sql}) VALUES ({episode_placeholders})",
                    tuple(row[column] for column in EPISODE_COLUMNS),
                )

            combined_links: dict[tuple[str, str, str], dict[str, Any]] = {}
            for source, row in evidence_rows:
                target_id = target_by_source[f"{source}:{row['memory_id']}"]
                key = (target_id, str(row["episode_id"]), str(row["relation"]))
                note = f"[{source}:{row['memory_id']}] {row['note'] or ''}".rstrip()
                if key not in combined_links:
                    combined_links[key] = {
                        "diagnosis_confidence": row["diagnosis_confidence"],
                        "notes": [note],
                        "created_at": row["created_at"],
                    }
                else:
                    existing = combined_links[key]
                    append_unique(existing["notes"], note)
                    scores = [
                        value
                        for value in (existing["diagnosis_confidence"], row["diagnosis_confidence"])
                        if value is not None
                    ]
                    existing["diagnosis_confidence"] = max(scores) if scores else None
                    existing["created_at"] = min(str(existing["created_at"]), str(row["created_at"]))

            for (memory_id, episode_id, relation), data in sorted(combined_links.items()):
                output_connection.execute(
                    """
                    INSERT INTO memory_evidence_links (
                        memory_id, episode_id, relation, diagnosis_confidence, note, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        episode_id,
                        relation,
                        data["diagnosis_confidence"],
                        " | ".join(data["notes"]),
                        data["created_at"],
                    ),
                )

            for source, row in usage_rows:
                target_id = target_by_source[f"{source}:{row['memory_id']}"]
                output_connection.execute(
                    """
                    INSERT OR IGNORE INTO memory_usage (
                        episode_id, memory_id, retrieved_rank, retrieval_score,
                        was_returned_to_brain, was_cited_by_brain, usage_role, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["episode_id"],
                        target_id,
                        row["retrieved_rank"],
                        row["retrieval_score"],
                        row["was_returned_to_brain"],
                        row["was_cited_by_brain"],
                        row["usage_role"],
                        row["created_at"],
                    ),
                )
            output_connection.commit()
        finally:
            output_connection.close()

        chroma_built, chroma_count, chroma_skip_reason = rebuild_chroma_if_available(output_memory_dir)

        with sqlite3.connect(output_db) as verification_connection:
            table_counts = {
                table: int(verification_connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "memory_items",
                    "memory_episodes",
                    "memory_evidence_links",
                    "memory_usage",
                )
            }
            orphan_links = int(
                verification_connection.execute(
                    """
                    SELECT COUNT(*) FROM memory_evidence_links AS links
                    LEFT JOIN memory_items AS memories ON memories.memory_id = links.memory_id
                    LEFT JOIN memory_episodes AS episodes ON episodes.episode_id = links.episode_id
                    WHERE memories.memory_id IS NULL OR episodes.episode_id IS NULL
                    """
                ).fetchone()[0]
            )

        if table_counts["memory_items"] != expected_final_count:
            raise ValueError(f"Expected {expected_final_count} memories, got {table_counts['memory_items']}")
        if chroma_built and chroma_count != expected_final_count:
            raise ValueError(f"Expected {expected_final_count} Chroma records, got {chroma_count}")
        if orphan_links:
            raise ValueError(f"Found {orphan_links} orphan evidence links")

        manifest = {
            "created_at": now.isoformat(),
            "output_database": str(output_db),
            "source_databases": {source: str(path) for source, path in SOURCE_DATABASES.items()},
            "selection_policy": {
                "gemma": "Only positive records: non-empty success_pattern and empty failure_type.",
                "qwen": "All reviewed Qwen records.",
                "legacy": "All manually curated geoexp13k records.",
            },
            "source_counts": source_counts,
            "selected_before_deduplication": len(selected),
            "merge_group_count": len(MERGE_GROUPS),
            "merged_away_count": len(selected) - expected_final_count,
            "final_memory_count": expected_final_count,
            "table_counts": table_counts,
            "chroma": {
                "built": chroma_built,
                "count": chroma_count,
                "skip_reason": chroma_skip_reason,
                "runtime_behavior": "MemoryManager rebuilds Chroma from SQLite when the index is absent.",
            },
            "orphan_evidence_links": orphan_links,
            "merge_groups": MERGE_GROUPS,
            "source_to_target_memory_id": dict(sorted(target_by_source.items())),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "merge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        for connection in source_connections.values():
            connection.close()


if __name__ == "__main__":
    main()
