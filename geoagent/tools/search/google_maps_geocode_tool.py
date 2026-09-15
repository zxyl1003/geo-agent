"""Google Maps Geocoding tool."""

from __future__ import annotations

from typing import Any

import requests

from geoagent.core.coordinates import normalize_coordinate_fields
from geoagent.core.registry import tool_registry
from geoagent.tools.base import BaseTool


@tool_registry.register("google_maps_geocode")
class GoogleMapsGeocodeTool(BaseTool):
    """
    Google API 每月有200美元的免费额度，大概相当于40000次地理编码，由于Google POI检索和详情只包含基础字段，免费不消耗额度
    因此，只有地理编码会产生费用
    """
    name = "google_maps_geocode"
    description = "Geocode an address to WGS84 coordinates or reverse geocode WGS84 coordinates to a human-readable address using Google Maps Geocoding API."
    # Internal provider behind the geocode/reverse_geocode facades, used as
    # the fallback for international geocoding. Never exposed to the Brain.
    hidden = True

    def run(self, **kwargs: Any):
        address = str(kwargs.get("address", "")).strip()
        lat = kwargs.get("lat")
        lon = kwargs.get("lon", kwargs.get("lng"))
        latlng = str(kwargs.get("latlng", "")).strip()
        top_k = int(kwargs.get("top_k", 3))
        language = str(kwargs.get("language", kwargs.get("language_code", "en")) or "en")
        region = str(kwargs.get("region", "") or "").strip()

        mode = self._mode(address=address, lat=lat, lon=lon, latlng=latlng)
        if mode == "missing":
            return self.result(success=False, error="Missing geocode input. Provide address, lat/lon, or latlng.")

        api_key = self.app_config.env.google_maps_api_key
        if api_key is None:
            return self.result(success=False, error="GOOGLE_MAPS_API_KEY is not configured.")
        try:
            data = self._google_geocode(
                api_key=api_key.get_secret_value(),
                mode=mode,
                address=address,
                lat=lat,
                lon=lon,
                latlng=latlng,
                language=language,
                region=region,
                top_k=top_k,
                extra=kwargs,
            )
            return self.result(data=data)
        except Exception as exc:  # noqa: BLE001
            return self.result(
                success=False,
                data={"provider": "google_maps_geocoding", "mode": mode, "address": address, "lat": lat, "lon": lon},
                error=f"Google Maps geocoding failed: {exc}",
            )

    def _mode(self, address: str, lat: Any, lon: Any, latlng: str) -> str:
        if address:
            return "geocode"
        if latlng or (lat is not None and lon is not None):
            return "reverse_geocode"
        return "missing"

    def _google_geocode(
        self,
        api_key: str,
        mode: str,
        address: str,
        lat: Any,
        lon: Any,
        latlng: str,
        language: str,
        region: str,
        top_k: int,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"key": api_key, "language": language}
        if mode == "geocode":
            params["address"] = address
            if region:
                params["region"] = region
            components = extra.get("components")
            if components:
                params["components"] = components
        else:
            params["latlng"] = latlng or f"{float(lat)},{float(lon)}"
            if extra.get("result_type"):
                params["result_type"] = extra["result_type"]
            if extra.get("location_type"):
                params["location_type"] = extra["location_type"]

        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params=params,
            timeout=45,
        )
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        status = payload.get("status")
        if status not in {"OK", "ZERO_RESULTS"}:
            raise RuntimeError(f"Geocoding status {status}: {payload.get('error_message', '')}")

        results = [self._normalize_result(item) for item in payload.get("results", [])[: max(1, top_k)]]
        top_result = results[0] if results else None
        return {
            "provider": "google_maps_geocoding",
            "provider_display_name": "Google Maps",
            "map_provider": "google",
            "source_type": "geocoding_api",
            "coordinate_system": "WGS84",
            "using_configured_api_key": True,
            "mode": mode,
            "status": status,
            "address": address or None,
            "latlng": params.get("latlng"),
            "results": results,
            "top_result": top_result,
            "lat": top_result.get("lat") if top_result else None,
            "lon": top_result.get("lon") if top_result else None,
            "formatted_address": top_result.get("formatted_address") if top_result else None,
        }

    def _normalize_result(self, item: dict[str, Any]) -> dict[str, Any]:
        geometry = item.get("geometry") or {}
        location = geometry.get("location") or {}
        viewport = geometry.get("viewport") or {}
        coordinates = normalize_coordinate_fields(location.get("lat"), location.get("lng"), "wgs84")
        return {
            "formatted_address": item.get("formatted_address"),
            "place_id": item.get("place_id"),
            "types": item.get("types", []),
            **coordinates,
            "location_type": geometry.get("location_type"),
            "viewport": viewport,
            "partial_match": item.get("partial_match", False),
            "address_components": item.get("address_components", []),
            "plus_code": item.get("plus_code"),
        }

    def is_available(self, state: Any = None) -> bool:
        return bool(self.app_config.env.google_maps_api_key)
