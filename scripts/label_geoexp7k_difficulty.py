"""Add 10 km model-correctness labels and difficulty to GeoExp7k metadata."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "GeoExp7k"
RESULTS_DIR = ROOT / "outputs" / "results"
THRESHOLD_M = 10_000.0
LABEL_COLUMNS = ["GLM", "GPT-5.6-Luna", "Qwen", "Gemini", "Difficulty"]

SPLITS = {
    "test": {
        "source_metadata": DATASET_DIR / "test_metadata.csv",
        "output_metadata": DATASET_DIR / "test_metadata.csv",
        "results": {
            "GLM": RESULTS_DIR / "concurrent_glm53_flash_geoexp7k_test_low" / "results.csv",
            "GPT-5.6-Luna": RESULTS_DIR
            / "concurrent_gpt56_luna_openai_flex_geoexp7k_test_low"
            / "results.csv",
            "Qwen": RESULTS_DIR / "vllm_qwen36_27b_geoexp7k_test" / "results.csv",
            "Gemini": RESULTS_DIR
            / "concurrent_gemini38_flash_google_ai_studio_flex_geoexp7k_test_low"
            / "results.csv",
        },
    },
    "learning": {
        "source_metadata": DATASET_DIR / "learning_metadata.csv",
        "output_metadata": DATASET_DIR / "learning_metadata.csv",
        "results": {
            "GLM": RESULTS_DIR / "concurrent_glm53_flash_geoexp7k_train_low" / "results.csv",
            "GPT-5.6-Luna": RESULTS_DIR
            / "concurrent_gpt56_luna_openai_flex_geoexp7k_train_low"
            / "results.csv",
            "Qwen": RESULTS_DIR / "vllm_qwen36_27b_geoexp7k_train" / "results.csv",
            "Gemini": RESULTS_DIR
            / "concurrent_gemini38_flash_google_ai_studio_flex_geoexp7k_train_low"
            / "results.csv",
        },
    },
}


def load_result_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            item_id = str(row.get("item_id") or "").strip()
            if not item_id:
                raise ValueError(f"Missing item_id in {path}")
            if item_id in labels:
                raise ValueError(f"Duplicate item_id {item_id!r} in {path}")
            try:
                distance_m = float(str(row.get("pred_gt_distance_m") or "").strip())
            except ValueError:
                labels[item_id] = 0
            else:
                labels[item_id] = int(distance_m <= THRESHOLD_M)
    return labels


def label_split(config: dict) -> tuple[Counter, dict[str, int]]:
    source_path: Path = config["source_metadata"]
    output_path: Path = config["output_metadata"]
    result_labels = {
        model: load_result_labels(path)
        for model, path in config["results"].items()
    }

    with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        source_fields = list(reader.fieldnames or [])
        rows = list(reader)

    sample_ids = [str(row.get("sample_id") or "").strip() for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"Missing sample_id in {source_path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample_id in {source_path}")

    expected_ids = set(sample_ids)
    for model, labels in result_labels.items():
        missing = expected_ids - labels.keys()
        extra = labels.keys() - expected_ids
        if missing or extra:
            raise ValueError(
                f"{model} result IDs do not match {source_path.name}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )

    difficulty_counts: Counter = Counter()
    model_correct_counts = {model: 0 for model in result_labels}
    for row, sample_id in zip(rows, sample_ids, strict=True):
        for model, labels in result_labels.items():
            value = labels[sample_id]
            row[model] = str(value)
            model_correct_counts[model] += value

        baseline_count = int(row["GLM"]) + int(row["GPT-5.6-Luna"]) + int(row["Qwen"])
        if baseline_count >= 2:
            difficulty = "easy"
        else:
            difficulty = "hard"
        row["Difficulty"] = difficulty
        difficulty_counts[difficulty] += 1

    fieldnames = [field for field in source_fields if field not in LABEL_COLUMNS] + LABEL_COLUMNS
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_path)
    return difficulty_counts, model_correct_counts


def main() -> None:
    for split, config in SPLITS.items():
        difficulty_counts, model_counts = label_split(config)
        output_path = config["output_metadata"]
        print(f"{split}: {output_path}")
        print(
            "  correctness: "
            + ", ".join(f"{model}={count}" for model, count in model_counts.items())
        )
        print(
            "  difficulty: "
            + ", ".join(
                f"{level}={difficulty_counts[level]}" for level in ("easy", "hard")
            )
        )


if __name__ == "__main__":
    main()
