"""Verify a candidate location using satellite or rendered map tiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geoagent.core.coordinates import (
    baidu_tile_position_from_wgs84,
    baidu_tile_from_wgs84,
    google_xyz_from_wgs84,
    normalize_to_wgs84,
)
from geoagent.core.registry import tool_registry
from geoagent.models.vlm_client import brain_credentials_configured
from geoagent.tools.base import BaseTool
from geoagent.tools.maps.imagery_common import (
    MapImageryError,
    cache_dir_for,
    compare_with_vlm,
    compose_tile_grid,
    download_image,
    stable_token,
)


@tool_registry.register("map_tile_verify")
class MapTileVerifyTool(BaseTool):
    name = "map_tile_verify"
    description = "Verify an existing coordinate hypothesis with satellite or rendered map tiles."

    SUPPORTED_PROVIDERS = {"google", "baidu"}
    SUPPORTED_MAP_TYPES = {"satellite", "roadmap"}

    def is_available(self, state: Any = None) -> bool:
        return brain_credentials_configured(self.app_config)

    def run(self, **kwargs: Any):
        map_provider = str(kwargs.get("map_provider", "")).strip().lower()
        map_type = str(kwargs.get("map_type", "satellite")).strip().lower()
        if map_provider not in self.SUPPORTED_PROVIDERS:
            return self.result(
                success=False,
                data={"supported_map_providers": sorted(self.SUPPORTED_PROVIDERS)},
                error="Missing or unsupported map_provider. Choose google or baidu.",
            )
        if map_type not in self.SUPPORTED_MAP_TYPES:
            return self.result(
                success=False,
                data={"supported_map_types": sorted(self.SUPPORTED_MAP_TYPES)},
                error="Missing or unsupported map_type. Choose satellite or roadmap.",
            )
        lat = kwargs.get("lat")
        lon = kwargs.get("lon", kwargs.get("lng"))
        if lat is None or lon is None:
            return self.result(success=False, error="map_tile_verify requires lat and lon for an existing candidate.")

        coordinate_system = str(kwargs.get("coordinate_system", "wgs84") or "wgs84")
        wgs_lon, wgs_lat = normalize_to_wgs84(float(lon), float(lat), coordinate_system)
        zoom = int(kwargs.get("zoom", 19))
        zoom = max(3, min(20, zoom))
        tile_radius = int(kwargs.get("tile_radius", 1))
        tile_radius = max(0, min(2, tile_radius))
        task_id = str(kwargs.get("task_id") or "manual")
        image_path = str(kwargs.get("image_path") or "")
        candidate_name = str(kwargs.get("candidate_name") or kwargs.get("name") or "candidate location")
        candidate_id = kwargs.get("candidate_id")
        reason = str(kwargs.get("reason") or "")

        cache_dir = cache_dir_for(task_id, self.name)
        token = stable_token(map_provider, map_type, wgs_lat, wgs_lon, zoom, tile_radius, candidate_name)
        tile_dir = cache_dir / token
        tile_dir.mkdir(parents=True, exist_ok=True)

        try:
            tiles = self._retrieve_tiles(
                provider=map_provider,
                map_type=map_type,
                lon=wgs_lon,
                lat=wgs_lat,
                zoom=zoom,
                radius=tile_radius,
                tile_dir=tile_dir,
            )
            candidate_marker = None
            if map_provider == "baidu":
                _, _, pixel_x, pixel_y = baidu_tile_position_from_wgs84(wgs_lon, wgs_lat, zoom)
                candidate_marker = {
                    "x": tile_radius * 256 + pixel_x,
                    "y": tile_radius * 256 + pixel_y,
                }
            mosaic_path = compose_tile_grid(
                tiles,
                tile_dir / f"{map_provider}_{map_type}_mosaic.png",
                candidate_marker=candidate_marker,
            )
        except Exception as exc:  # noqa: BLE001
            return self.result(
                success=False,
                data={
                    "map_provider": map_provider,
                    "map_type": map_type,
                    "lat": wgs_lat,
                    "lon": wgs_lon,
                    "coordinate_system": "WGS84",
                    "input_coordinate_system": coordinate_system,
                    "zoom": zoom,
                    "tile_radius": tile_radius,
                },
                error=f"Map tile retrieval failed: {exc}",
            )

        verification = compare_with_vlm(
            app_config=self.app_config,
            source_image_path=image_path,
            reference_image_path=str(mosaic_path),
            prompt=self._comparison_prompt(
                candidate_name=candidate_name,
                map_provider=map_provider,
                map_type=map_type,
                lat=wgs_lat,
                lon=wgs_lon,
                reason=reason,
            ),
        )
        return self.result(
            data={
                "source_type": "map_tile_verification",
                "map_provider": map_provider,
                "provider": f"{map_provider}_maps",
                "provider_display_name": "Google Maps" if map_provider == "google" else "Baidu Maps",
                "map_type": map_type,
                "candidate_name": candidate_name,
                "candidate_id": candidate_id,
                "lat": wgs_lat,
                "lon": wgs_lon,
                "coordinate_system": "WGS84",
                "input_coordinate_system": coordinate_system,
                "zoom": zoom,
                "tile_radius": tile_radius,
                "cache_dir": str(tile_dir),
                "mosaic_path": str(mosaic_path),
                "tiles": tiles,
                "verification": verification,
                "verdict": verification["verdict"],
                "confidence": verification["confidence"],
            }
        )

    def _retrieve_tiles(
        self,
        provider: str,
        map_type: str,
        lon: float,
        lat: float,
        zoom: int,
        radius: int,
        tile_dir: Path,
    ) -> list[dict[str, Any]]:
        center_x, center_y = (
            google_xyz_from_wgs84(lon, lat, zoom)
            if provider == "google"
            else baidu_tile_from_wgs84(lon, lat, zoom)
        )
        tiles: list[dict[str, Any]] = []
        y_values = (
            range(center_y - radius, center_y + radius + 1)
            if provider == "google"
            else range(center_y + radius, center_y - radius - 1, -1)
        )
        for grid_y, y in enumerate(y_values):
            for grid_x, x in enumerate(range(center_x - radius, center_x + radius + 1)):
                if provider == "google":
                    max_tile = (2**zoom) - 1
                    if y < 0 or y > max_tile:
                        continue
                    x = x % (2**zoom)
                url = self._tile_url(provider, map_type, x, y, zoom)
                path = tile_dir / f"{provider}_{map_type}_z{zoom}_x{x}_y{y}.png"
                label = f"{provider}/{map_type} z{zoom} x{x} y{y}"
                download_image(url, path)
                tiles.append(
                    {
                        "provider": provider,
                        "map_type": map_type,
                        "x": x,
                        "y": y,
                        "z": zoom,
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "url": url,
                        "path": str(path),
                        "label": label,
                    }
                )
        if not tiles:
            raise MapImageryError("No map tiles were retrieved.")
        return tiles

    def _tile_url(self, provider: str, map_type: str, x: int, y: int, zoom: int) -> str:
        if provider == "google":
            layer = "s" if map_type == "satellite" else "m"
            server = abs(x + y) % 4
            return f"https://mt{server}.google.com/vt/lyrs={layer}&x={x}&y={y}&z={zoom}"

        server = abs(x + y) % 4
        if map_type == "satellite":
            return f"http://shangetu{server}.map.bdimg.com/it/u=x={x};y={y};z={zoom};v=009;type=sate&fm=46"
        return (
            f"http://online{server}.map.bdimg.com/tile/"
            f"?qt=vtile&x={x}&y={y}&z={zoom}&styles=pl&scaler=1"
        )

    def _comparison_prompt(
        self,
        candidate_name: str,
        map_provider: str,
        map_type: str,
        lat: float,
        lon: float,
        reason: str,
    ) -> str:
        return f"""
You are verifying a geolocation hypothesis by comparing two images.

Image 1 is the original user-provided ground-level image.
Image 2 is a stitched {map_type} map tile mosaic around the candidate location from {map_provider}.
On Baidu mosaics, the red crosshair marks the exact candidate coordinate.

Candidate:
- name: {candidate_name}
- coordinates_wgs84: lat={lat}, lon={lon}
- reason_for_verification: {reason or "not specified"}

Use satellite imagery mainly for broad spatial context: terrain, water, mountains, coastlines, rivers, bridges, large building footprints, and road layout.
Use rendered roadmap imagery mainly for road network, street names, POI labels, and spatial relationships.
Do not require a ground-level object to be visible in satellite imagery. If the map cannot test the relevant visual clues, return inconclusive.

Return only JSON:
{{
  "verdict": "supports | contradicts | inconclusive",
  "confidence": 0.0,
  "matching_features": ["features that match"],
  "contradictions": ["features that conflict"],
  "limitations": ["coverage/viewpoint/time/scale limits"],
  "recommended_next_step": "short next step or empty string"
}}
""".strip()
