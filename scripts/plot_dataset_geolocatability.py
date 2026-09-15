"""Plot a single 100% stacked bar chart of dataset geolocatability."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "figures"

CATEGORY_ORDER = (
    "No scene text",
    "Text, no usable named anchor",
    "Difficult single anchor",
    "Specific single anchor",
    "Multiple POIs",
)
CATEGORY_COLORS = (
    "#D9D9D9",
    "#8C8C8C",
    "#E69F00",
    "#56B4E9",
    "#009E73",
)


def _read_rows(path: Path, *, completed_only: bool = False) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if completed_only:
        rows = [row for row in rows if row.get("status") == "completed"]
    return rows


def _category(row: dict[str, str]) -> str:
    stratum = row["recommended_stratum"]
    if stratum == "reject":
        if row["has_scene_text"] == "0":
            return "No scene text"
        return "Text, no usable named anchor"
    if stratum == "multi_poi":
        return "Multiple POIs"
    if stratum in {"branch_or_generic_name", "partial_or_non_latin_text"}:
        return "Difficult single anchor"
    if stratum in {"specific_named_anchor", "road_address_or_phone"}:
        return "Specific single anchor"
    raise ValueError(f"Unsupported recommended_stratum: {stratum!r}")


def _summarize(rows: list[dict[str, str]]) -> tuple[list[int], list[float]]:
    counts = Counter(_category(row) for row in rows)
    ordered_counts = [counts[category] for category in CATEGORY_ORDER]
    total = sum(ordered_counts)
    return ordered_counts, [count * 100.0 / total for count in ordered_counts]


def main() -> None:
    datasets = (
        (
            "GeoExp7K",
            _read_rows(ROOT / "datasets" / "GeoExp7k" / "learning_metadata.csv")
            + _read_rows(ROOT / "datasets" / "GeoExp7k" / "test_metadata.csv"),
        ),
        (
            "IMAGEO-Bench dataset2",
            _read_rows(
                ROOT
                / "outputs"
                / "annotations"
                / "qwen37_flash_imageobench_dataset2"
                / "annotations.csv",
                completed_only=True,
            ),
        ),
        (
            "Im2GPS3K",
            _read_rows(
                ROOT
                / "outputs"
                / "annotations"
                / "qwen37_flash_im2gps3k"
                / "annotations.csv",
                completed_only=True,
            ),
        ),
    )

    summaries = [(name, rows, *_summarize(rows)) for name, rows in datasets]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 3.65), constrained_layout=False)

    for y, (_, _, _, percentages) in enumerate(summaries):
        left = 0.0
        for category, color, value in zip(
            CATEGORY_ORDER, CATEGORY_COLORS, percentages, strict=True
        ):
            ax.barh(
                y,
                value,
                left=left,
                height=0.58,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=category if y == 0 else None,
                zorder=3,
            )
            if value >= 5.0:
                text_color = "white" if color in {"#8C8C8C", "#009E73"} else "#202020"
                ax.text(
                    left + value / 2.0,
                    y,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.8,
                    fontweight="bold",
                    zorder=4,
                )
            left += value

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, len(summaries) - 0.45)
    ax.invert_yaxis()
    ax.set_yticks(
        range(len(summaries)),
        [f"{name}  (n={len(rows):,})" for name, rows, _, _ in summaries],
    )
    ax.set_xlabel("Share of valid images (%)", labelpad=7)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis="x", color="#D7D7D7", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3, color="#666666")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["bottom"].set_linewidth(0.8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.5,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.2, top=0.71)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "dataset_geolocatability_profile"
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")

    for name, rows, counts, percentages in summaries:
        values = ", ".join(
            f"{category}={count} ({percentage:.2f}%)"
            for category, count, percentage in zip(
                CATEGORY_ORDER, counts, percentages, strict=True
            )
        )
        print(f"{name} (n={len(rows)}): {values}")


if __name__ == "__main__":
    main()
