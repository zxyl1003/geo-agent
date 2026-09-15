"""Unified reverse-geocode tool with internal provider routing.

Routing (hidden from the Brain — the Brain only calls `reverse_geocode`):
- Coordinates known to be in mainland China (country hint) -> Baidu Maps
- Otherwise -> LocationIQ

The result always exposes the final `provider` / `provider_display_name` and an
`attempts` chain so API usage can be traced in the audit log.
"""

from __future__ import annotations

from typing import Any

from geoagent.core.registry import tool_registry
from geoagent.tools.base import BaseTool
from geoagent.tools.search.geocode_tool import _is_china_hint


@tool_registry.register("reverse_geocode")
class ReverseGeocodeTool(BaseTool):
    """Reverse-geocode WGS84 coordinates to a human-readable address using the
    best available provider."""

    name = "reverse_geocode"
    description = (
        "Convert WGS84 coordinates to a human-readable address. "
        "Routes automatically: mainland China coordinates use Baidu Maps; "
        "international coordinates use LocationIQ. "
        "Returns the formatted address with administrative context."
    )

    def run(self, **kwargs: Any) -> Any:
        lat = kwargs.get("lat")
        lon = kwargs.get("lon", kwargs.get("lng"))
        latlng = str(kwargs.get("latlng", "")).strip()
        country = str(kwargs.get("country", "") or "").strip()
        region = str(kwargs.get("region", "") or "").strip()
        language = str(kwargs.get("language", "") or "").strip() or "en"

        if not (latlng or (lat is not None and lon is not None)):
            return self.result(success=False, error="reverse_geocode requires lat/lon or latlng.")

        # Internal routing: the Brain does not choose the provider.
        if _is_china_hint(country, region):
            return self._run_baidu(lat=lat, lon=lon, latlng=latlng, language=language, kwargs=kwargs)
        return self._run_international(lat=lat, lon=lon, latlng=latlng, language=language, kwargs=kwargs)

    def _run_baidu(self, lat: Any, lon: Any, latlng: str, language: str, kwargs: dict[str, Any]) -> Any:
        baidu = self._tool("baidu_maps_geocode")
        if baidu is None:
            return self.result(
                success=False,
                data={"provider": "baidu_maps", "attempts": [{"provider": "baidu_maps", "status": "skipped", "reason": "not_registered"}]},
                error="Baidu Maps reverse geocoding is not available. Configure BAIDU_MAPS_API_KEY and enable the baidu_maps_geocode tool.",
            )
        merged = dict(kwargs)
        if latlng:
            merged["latlng"] = latlng
        else:
            merged["lat"] = lat
            merged["lon"] = lon
        merged.setdefault("coordtype", "WGS84")
        result = baidu.safe_run(**merged)
        return self._normalize_routed_result(result, final_provider="baidu_maps", chain_providers=["baidu_maps"])

    def _run_international(self, lat: Any, lon: Any, latlng: str, language: str, kwargs: dict[str, Any]) -> Any:
        locationiq = self._tool("locationiq_geocode")
        if locationiq is None:
            return self.result(
                success=False,
                data={"provider": "locationiq", "attempts": ["locationiq"]},
                error="International reverse geocoding requires LOCATIONIQ_KEY in .env.",
            )

        merged = dict(kwargs)
        if latlng:
            merged["latlng"] = latlng
        else:
            merged["lat"] = lat
            merged["lon"] = lon
        merged["language"] = language
        result = locationiq.safe_run(**merged)
        return self._normalize_routed_result(result, final_provider="locationiq", chain_providers=["locationiq"])

    def _normalize_routed_result(self, result: Any, final_provider: str, chain_providers: list[str]) -> Any:
        # The facade is the tool the Brain called; keep the result attributed to
        # it so tool history, hypothesis merge, and resource accounting see
        # `reverse_geocode`, not the internal provider implementation.
        result.tool_name = self.name
        if not result.success:
            result.data = dict(result.data or {})
            result.data["provider"] = final_provider
            result.data["provider_display_name"] = self._display_name(final_provider)
            result.data["attempts"] = chain_providers
            return result
        data = dict(result.data)
        data["provider"] = final_provider
        data["provider_display_name"] = self._display_name(final_provider)
        data["attempts"] = chain_providers
        data["routed"] = True
        result.data = data
        return result

    def _display_name(self, provider: str) -> str:
        return {
            "baidu_maps": "Baidu Maps",
            "locationiq": "LocationIQ",
        }.get(provider, provider)

    def _tool(self, name: str) -> Any | None:
        """Instantiate a provider tool (from the workflow-bound set when
        available, otherwise from the registry) and return it if enabled and
        has credentials configured."""
        from geoagent.core.registry import tool_registry as _registry

        candidate = None
        bound = getattr(self, "tools", None)
        if isinstance(bound, dict):
            candidate = bound.get(name)
        if candidate is None:
            tool_config = self.app_config.tools.get(name)
            if tool_config is None or not tool_config.enable:
                return None
            cls = _registry.maybe_get(name)
            if cls is None:
                cls = _registry.register_from_class_path(name, tool_config.class_path)
            candidate = cls(config=tool_config, app_config=self.app_config)
        try:
            if not candidate.is_available():
                return None
        except TypeError:
            if not candidate.is_available(None):
                return None
        return candidate

    def is_available(self, state: Any = None) -> bool:
        return bool(
            self.app_config.env.baidu_maps_api_key
            or self.app_config.env.locationiq_api_key
        )
