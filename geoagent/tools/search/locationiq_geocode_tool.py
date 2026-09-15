"""LocationIQ forward and reverse geocoding for international addresses.

LocationIQ is the primary provider for non-mainland-China geocoding. It offers
a free tier (150k requests/month, 5k/day) with a single API key covering both
forward and reverse geocoding.
"""

from __future__ import annotations

from typing import Any

import requests

from geoagent.core.coordinates import normalize_coordinate_fields
from geoagent.core.registry import tool_registry
from geoagent.tools.base import BaseTool


# LocationIQ 'type' values (Nominatim-compatible) to system granularity level.
_TYPE_GRANULARITY_MAP: dict[str, str] = {
    "house": "coordinates",
    "building": "coordinates",
    "residential": "street",
    "street": "street",
    "road": "street",
    "neighbourhood": "street",
    "suburb": "city",
    "city": "city",
    "town": "city",
    "village": "city",
    "county": "region",
    "state": "region",
    "country": "country",
}


@tool_registry.register("locationiq_geocode")
class LocationIQGeocodeTool(BaseTool):
    """Convert an address/place name to WGS84 coordinates (forward) or
    reverse-geocode WGS84 coordinates to a human-readable address using
    LocationIQ."""

    name = "locationiq_geocode"
    description = (
        "Geocode an address, street name, or place name to WGS84 coordinates, or reverse-geocode "
        "WGS84 coordinates to a human-readable address, using LocationIQ. Intended for regions "
        "outside mainland China. Returns candidates with granularity and admin context."
    )
    # Internal provider behind the geocode/reverse_geocode facades. Never
    # exposed directly to the Brain.
    hidden = True

    _MAX_NETWORK_RETRIES = 3

    def run(self, **kwargs: Any) -> Any:
        address = str(kwargs.get("address", kwargs.get("q", kwargs.get("query", "")))).strip()
        lat = kwargs.get("lat")
        lon = kwargs.get("lon", kwargs.get("lng"))
        latlng = str(kwargs.get("latlng", "")).strip()
        top_k = max(1, min(int(kwargs.get("top_k", 3)), 10))
        language = str(kwargs.get("language", kwargs.get("accept_language", "en")) or "en").strip()
        country_code = str(kwargs.get("country_code", kwargs.get("countrycodes", ""))).strip().lower() or None

        mode = self._mode(address=address, lat=lat, lon=lon, latlng=latlng)
        if mode == "missing":
            return self.result(success=False, error="Missing geocode input. Provide address, lat/lon, or latlng.")

        api_key = self.app_config.env.locationiq_api_key
        if api_key is None:
            return self.result(success=False, error="LOCATIONIQ_KEY is not configured.")

        try:
            if mode == "geocode":
                data = self._locationiq_forward(
                    api_key=api_key.get_secret_value(),
                    address=address,
                    country_code=country_code,
                    language=language,
                    top_k=top_k,
                )
            else:
                data = self._locationiq_reverse(
                    api_key=api_key.get_secret_value(),
                    lat=lat,
                    lon=lon,
                    latlng=latlng,
                    language=language,
                )
            return self.result(data=data)
        except Exception as exc:  # noqa: BLE001
            return self.result(
                success=False,
                data={"provider": "locationiq", "mode": mode, "address": address or None, "lat": lat, "lon": lon},
                error=f"LocationIQ geocoding failed: {exc}",
            )

    def _mode(self, address: str, lat: Any, lon: Any, latlng: str) -> str:
        if address:
            return "geocode"
        if latlng or (lat is not None and lon is not None):
            return "reverse_geocode"
        return "missing"

    def _locationiq_forward(
        self,
        api_key: str,
        address: str,
        country_code: str | None,
        language: str,
        top_k: int,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "key": api_key,
            "q": address,
            "format": "json",
            "addressdetails": 1,
            "limit": top_k,
            "accept-language": language,
        }
        if country_code:
            params["countrycodes"] = country_code

        response = self._get(
            self.app_config.env.locationiq_forward_url,
            params=params,
        )
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"LocationIQ status {payload.get('code', 'error')}: {payload.get('error')}")

        results = [self._normalize_candidate(item, idx) for idx, item in enumerate(payload[:top_k]) if isinstance(item, dict)]
        top_result = results[0] if results else None
        return {
            "provider": "locationiq",
            "provider_display_name": "LocationIQ",
            "map_provider": "locationiq",
            "source_type": "geocoding_api",
            "coordinate_system": "WGS84",
            "using_configured_api_key": True,
            "mode": "geocode",
            "address": address,
            "country_code": country_code,
            "candidates": results,
            "top_result": top_result,
            "lat": top_result.get("lat") if top_result else None,
            "lon": top_result.get("lon") if top_result else None,
            "formatted_address": top_result.get("display_name") if top_result else None,
        }

    def _locationiq_reverse(
        self,
        api_key: str,
        lat: Any,
        lon: Any,
        latlng: str,
        language: str,
    ) -> dict[str, Any]:
        if latlng:
            lat_text, lon_text = latlng.split(",", 1)
            lat, lon = float(lat_text), float(lon_text)
        params: dict[str, Any] = {
            "key": api_key,
            "lat": float(lat),
            "lon": float(lon),
            "format": "json",
            "addressdetails": 1,
            "accept-language": language,
            "zoom": 18,
        }

        response = self._get(
            self.app_config.env.locationiq_reverse_url,
            params=params,
        )
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("LocationIQ reverse geocoding returned an unexpected response.")
        if payload.get("error"):
            raise RuntimeError(f"LocationIQ status {payload.get('code', 'error')}: {payload.get('error')}")

        coordinates = normalize_coordinate_fields(payload.get("lat"), payload.get("lon"), "wgs84")
        result = {
            "formatted_address": payload.get("display_name"),
            "place_id": payload.get("place_id"),
            "types": [payload.get("type")] if payload.get("type") else [],
            **coordinates,
            "location_type": "reverse_geocode",
            "address_components": payload.get("address") or {},
            "bounding_box": payload.get("boundingbox"),
        }
        return {
            "provider": "locationiq",
            "provider_display_name": "LocationIQ",
            "map_provider": "locationiq",
            "source_type": "reverse_geocoding_api",
            "coordinate_system": "WGS84",
            "using_configured_api_key": True,
            "mode": "reverse_geocode",
            "latlng": f"{float(lat)},{float(lon)}",
            "results": [result],
            "top_result": result,
            "lat": result.get("lat"),
            "lon": result.get("lon"),
            "formatted_address": result.get("formatted_address"),
        }

    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        for attempt in range(self._MAX_NETWORK_RETRIES + 1):
            try:
                return requests.get(url, params=params, timeout=45)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt == self._MAX_NETWORK_RETRIES:
                    error_type = type(exc).__name__
                    raise RuntimeError(
                        f"LocationIQ network request failed after {attempt + 1} attempts ({error_type})."
                    ) from None
        raise AssertionError("unreachable")

    def _normalize_candidate(self, item: dict[str, Any], index: int) -> dict[str, Any]:
        """Normalize a LocationIQ/Nominatim-style search result."""
        lat = float(item.get("lat", 0))
        lon = float(item.get("lon", 0))
        place_type = str(item.get("type", "") or "")
        place_class = str(item.get("class", "") or "")
        display_name = item.get("display_name", "")

        granularity = _TYPE_GRANULARITY_MAP.get(place_type, _TYPE_GRANULARITY_MAP.get(place_class, "street"))

        address = item.get("address", {}) or {}
        name = (
            address.get("building")
            or address.get("house")
            or address.get("road")
            or address.get("street")
            or address.get("suburb")
            or address.get("city")
            or address.get("town")
            or address.get("village")
            or item.get("name")
        )
        city = address.get("city") or address.get("town") or address.get("village")
        region = address.get("state") or address.get("county")
        country = address.get("country")

        bbox = item.get("boundingbox")
        bbox_formatted = None
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_formatted = {
                "south": float(bbox[0]),
                "north": float(bbox[1]),
                "west": float(bbox[2]),
                "east": float(bbox[3]),
            }

        return {
            "candidate_id": f"locationiq_{index}",
            "name": name or display_name,
            "display_name": display_name,
            "lat": lat,
            "lon": lon,
            "granularity": granularity,
            "city": city,
            "region": region,
            "country": country,
            "locationiq_type": place_type,
            "locationiq_class": place_class,
            "bounding_box": bbox_formatted,
            "address": address,
            "importance": item.get("importance"),
        }

    def is_available(self, state: Any = None) -> bool:
        return bool(self.app_config.env.locationiq_api_key)
