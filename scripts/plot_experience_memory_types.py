"""Plot the type distribution of the current curated experience memory."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
MEMORY_DB = (
    ROOT
    / "outputs"
    / "memory"
    / "geoexp7k_learning"
    / "memory"
    / "memory.sqlite"
)
OUTPUT_DIR = ROOT / "outputs" / "figures"

MEMORY_TYPES = (
    ("evidence_reliability", "Evidence reliability"),
    ("strategy_policy", "Strategy policy"),
    ("tool_policy", "Tool policy"),
    ("failure_pattern", "Failure pattern"),
    ("conflict_resolution", "Conflict resolution"),
)


def load_counts() -> Counter[str]:
    uri = f"file:{MEMORY_DB.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("SELECT memory_type FROM memory_items").fetchall()
    return Counter(row[0] for row in rows)


def main() -> None:
    counts = load_counts()
    labels = [label for _, label in MEMORY_TYPES]
    values = [counts[memory_type] for memory_type, _ in MEMORY_TYPES]
    total = sum(values)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    y_positions = range(len(labels))
    colors = ["#2878B5" if value else "#D9D9D9" for value in values]
    ax.barh(y_positions, values, height=0.58, color=colors, zorder=3)

    for y, value in enumerate(values):
        percentage = value * 100.0 / total if total else 0.0
        ax.text(
            value + 0.10,
            y,
            f"{value}  ({percentage:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#202020",
        )

    ax.set_yticks(list(y_positions), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values, default=0) + 1.0)
    ax.set_xlabel("Number of memories")
    ax.set_title(f"Experience Memory Type Distribution (n={total})", pad=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3, color="#666666")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["bottom"].set_linewidth(0.8)

    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.17, top=0.84)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "experience_memory_types"
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")

    for memory_type, label in MEMORY_TYPES:
        print(f"{label}: {counts[memory_type]}")


if __name__ == "__main__":
    main()
