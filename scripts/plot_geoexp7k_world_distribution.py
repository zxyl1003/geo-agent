"""Plot the GeoExp7K learning/test samples on a world map."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "GeoExp7k"
OUTPUT_DIR = ROOT / "outputs" / "figures"
MAP_CACHE_DIR = OUTPUT_DIR / "_natural_earth"
WORLD_MAP_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)
WORLD_MAP_PATH = MAP_CACHE_DIR / "ne_110m_admin_0_countries.shp"
WORLD_CRS = "ESRI:54030"  # Robinson projection

LEARNING_COLOR = "#E67E22"
TEST_COLOR = "#2F6FBB"
OCEAN_COLOR = "#DDEFF5"
LAND_COLOR = "#F4F1EA"
BORDER_COLOR = "#AEB8C2"


def _world_map_path() -> Path:
    if WORLD_MAP_PATH.exists():
        return WORLD_MAP_PATH

    MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = MAP_CACHE_DIR / "ne_110m_admin_0_countries.zip"
    try:
        urlretrieve(WORLD_MAP_URL, archive_path)
    except OSError:
        import pyogrio

        packaged_map = (
            Path(pyogrio.__file__).resolve().parent
            / "tests"
            / "fixtures"
            / "naturalearth_lowres"
            / "naturalearth_lowres.shp"
        )
        if packaged_map.exists():
            return packaged_map
        raise
    with ZipFile(archive_path) as archive:
        archive.extractall(MAP_CACHE_DIR)
    return WORLD_MAP_PATH


def _read_split(path: Path, expected_split: str) -> gpd.GeoDataFrame:
    frame = pd.read_csv(
        path,
        usecols=["sample_id", "latitude", "longitude", "experiment_split"],
    )
    split_values = set(frame["experiment_split"].dropna().astype(str))
    if split_values != {expected_split}:
        raise ValueError(f"Unexpected split labels in {path}: {sorted(split_values)}")

    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    invalid = frame[
        frame[["latitude", "longitude"]].isna().any(axis=1)
        | ~frame["latitude"].between(-90, 90)
        | ~frame["longitude"].between(-180, 180)
    ]
    if not invalid.empty:
        raise ValueError(f"{path} contains {len(invalid)} rows without valid WGS84 coordinates.")

    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    return points.to_crs(WORLD_CRS)


def main() -> None:
    learning = _read_split(DATASET_DIR / "learning_metadata.csv", "learning")
    test = _read_split(DATASET_DIR / "test_metadata.csv", "test")

    world = gpd.read_file(_world_map_path()).to_crs(WORLD_CRS)
    for name_column in ("ADMIN", "name", "NAME"):
        if name_column in world.columns:
            world = world[world[name_column] != "Antarctica"]
            break

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 20,
        }
    )
    fig, ax = plt.subplots(figsize=(11.4, 5.7), facecolor=OCEAN_COLOR)
    ax.set_facecolor(OCEAN_COLOR)

    world.plot(
        ax=ax,
        color=LAND_COLOR,
        edgecolor=BORDER_COLOR,
        linewidth=0.35,
        zorder=1,
    )
    test.plot(
        ax=ax,
        color=TEST_COLOR,
        marker="o",
        markersize=5.0,
        alpha=0.48,
        linewidth=0,
        zorder=2,
    )
    learning.plot(
        ax=ax,
        color=LEARNING_COLOR,
        marker="^",
        markersize=9.0,
        alpha=0.88,
        linewidth=0,
        zorder=3,
    )

    min_x, min_y, max_x, max_y = world.total_bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max(min_y, -7_000_000), max_y)
    ax.set_axis_off()

    handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor=LEARNING_COLOR,
            markeredgecolor="none",
            markersize=7,
            label=f"Experience learning (n={len(learning):,})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=TEST_COLOR,
            markeredgecolor="none",
            markersize=6,
            label=f"Test (n={len(test):,})",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.13)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "geoexp7k_world_distribution"
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"learning={len(learning)}, test={len(test)}, total={len(learning) + len(test)}")
    print(output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
