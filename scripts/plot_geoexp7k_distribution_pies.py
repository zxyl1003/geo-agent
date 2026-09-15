"""Plot GeoExp7K continent and leading-country distributions."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pycountry


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "GeoExp7k"
OUTPUT_DIR = ROOT / "outputs" / "figures"
MAP_CACHE_DIR = OUTPUT_DIR / "_natural_earth"
WORLD_MAP_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)
WORLD_MAP_PATH = MAP_CACHE_DIR / "ne_110m_admin_0_countries.shp"
WORLD_CRS = "ESRI:54030"

CONTINENT_ORDER = (
    "Asia",
    "North America",
    "South America",
    "Europe",
    "Oceania",
    "Africa",
)
CONTINENT_COLORS = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#D55E00",
)
COUNTRY_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#BAB0AC",
)
COUNTRY_NAMES = {
    "BR": "Brazil",
    "RU": "Russia",
    "CN": "China",
    "US": "United States",
    "CA": "Canada",
}


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


def _read_splits() -> dict[str, pd.DataFrame]:
    splits = {
        split: pd.read_csv(DATASET_DIR / f"{split}_metadata.csv")
        for split in ("learning", "test")
    }
    for split, frame in splits.items():
        labels = set(frame["experiment_split"].dropna().astype(str))
        if labels != {split}:
            raise ValueError(f"Unexpected split labels in {split}_metadata.csv: {sorted(labels)}")
    return splits


def _derive_geography(frame: pd.DataFrame) -> pd.DataFrame:
    world = gpd.read_file(_world_map_path())
    continent_column = "CONTINENT" if "CONTINENT" in world.columns else "continent"
    if "ISO_A2" in world.columns:
        world["derived_country_code"] = world["ISO_A2"]
    else:
        world["derived_country_code"] = world["iso_a3"].map(
            lambda code: (
                country.alpha_2
                if code != "-99" and (country := pycountry.countries.get(alpha_3=code))
                else "-99"
            )
        )
    world = world[[continent_column, "derived_country_code", "geometry"]].rename(
        columns={continent_column: "derived_continent"}
    )
    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, world, predicate="within", how="left")
    if len(joined) != len(points):
        raise ValueError("A sample coordinate matched more than one continent polygon.")

    missing = joined["derived_continent"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            joined.loc[missing]
            .drop(columns=["index_right", "derived_continent", "derived_country_code"])
            .to_crs(WORLD_CRS),
            world.to_crs(WORLD_CRS),
            how="left",
        )
        joined.loc[nearest.index, ["derived_continent", "derived_country_code"]] = nearest[
            ["derived_continent", "derived_country_code"]
        ]

    if joined["derived_continent"].isna().any():
        raise ValueError("Some sample coordinates could not be assigned to a continent.")
    return joined[["derived_continent", "derived_country_code"]]


def _split_counts(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, dict[str, int]]]:
    dimensions = {
        "Difficulty": ("easy", "hard"),
        "searchability": ("low", "medium", "high", "none"),
        "text_legibility": ("clear", "partial"),
    }
    return {
        split: {
            dimension: {
                category: int((frame[dimension].fillna("").str.lower() == category).sum())
                for category in categories
            }
            for dimension, categories in dimensions.items()
        }
        for split, frame in splits.items()
    }


def _pie_labels(names: list[str], values: list[int]) -> list[str]:
    total = sum(values)
    return [f"{name}\n{value * 100.0 / total:.1f}%" for name, value in zip(names, values, strict=True)]


def main() -> None:
    splits = _read_splits()
    combined = pd.concat(splits.values(), ignore_index=True)
    geography = _derive_geography(combined)
    combined["derived_continent"] = geography["derived_continent"]

    continent_counts = combined["derived_continent"].value_counts()
    continent_values = [int(continent_counts.get(name, 0)) for name in CONTINENT_ORDER]

    country_codes = combined["country"].where(
        combined["country"].notna(),
        geography["derived_country_code"],
    )
    country_counts = country_codes.value_counts()
    top_country_codes = list(country_counts.head(5).index)
    top_country_values = [int(country_counts[code]) for code in top_country_codes]
    country_names = [COUNTRY_NAMES.get(code, str(code)) for code in top_country_codes]
    other_count = len(combined) - sum(top_country_values)
    country_names.append("Other")
    top_country_values.append(other_count)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), facecolor="white")

    pie_specs = (
        (
            axes[0],
            list(CONTINENT_ORDER),
            continent_values,
            CONTINENT_COLORS,
            "Continent distribution",
        ),
        (
            axes[1],
            country_names,
            top_country_values,
            COUNTRY_COLORS,
            "Top countries",
        ),
    )
    for ax, names, values, colors, title in pie_specs:
        ax.pie(
            values,
            labels=_pie_labels(names, values),
            colors=colors,
            startangle=90,
            counterclock=False,
            labeldistance=1.08,
            textprops={"fontsize": 9},
            wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.0},
        )
        ax.text(
            0,
            0,
            f"GeoExp7K\nn={len(combined):,}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="semibold",
        )
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=18)
        ax.set_aspect("equal")

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.88, wspace=0.32)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "geoexp7k_geographic_distribution_pies"
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print("continent_counts", dict(zip(CONTINENT_ORDER, continent_values, strict=True)))
    print("country_counts", dict(zip(country_names, top_country_values, strict=True)))
    print("split_counts", _split_counts(splits))
    print(output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
