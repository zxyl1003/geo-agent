"""Unified geocode tool with internal provider routing.

Routing (hidden from the Brain — the Brain only calls `geocode`):
- Mainland China address (country/region hints indicate China) -> Baidu Maps
- International address -> LocationIQ

The result always exposes the final `provider` / `provider_display_name` and an
`attempts` chain so API usage can be traced in the audit log.
"""

from __future__ import annotations

from typing import Any

from geoagent.core.registry import tool_registry
from geoagent.tools.base import BaseTool


_MAINLAND_CHINA_ALIASES = {
    "cn",
    "prc",
    "china",
    "mainland china",
    "mainland",
    "people's republic of china",
    "中国",
    "中国大陆",
    "中华人民共和国",
}


def _normalize_country(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _is_china_hint(country: Any, region: Any) -> bool:
    country = _normalize_country(country)
    if country in _MAINLAND_CHINA_ALIASES:
        return True
    region = _normalize_country(region)
    # Chinese-script region/address text strongly implies mainland China.
    if any("一" <= char <= "鿿" for char in region):
        return True
    return False


@tool_registry.register("geocode")
class GeocodeTool(BaseTool):
    """Geocode an address to WGS84 coordinates using the best available provider."""

    name = "geocode"
    description = (
        "Convert an address, street name, or place name to WGS84 coordinates. "
        "Routes automatically: mainland China addresses use Baidu Maps; "
        "international addresses use LocationIQ. "
        "Returns candidate coordinates with granularity and administrative context."
    )

    def run(self, **kwargs: Any) -> Any:
        address = str(kwargs.get("address", "")).strip()
        if not address:
            return self.result(success=False, error="geocode requires a non-empty address.")
        country = str(kwargs.get("country", "") or "").strip()
        region = str(kwargs.get("region", "") or "").strip()
        top_k = max(1, min(int(kwargs.get("top_k", 3)), 10))
        language = str(kwargs.get("language", "") or "").strip() or "en"

        # Internal routing: the Brain does not choose the provider.
        if self._is_mainland_china_context(country, region):
            return self._run_baidu(address=address, region=region, top_k=top_k, language=language, kwargs=kwargs)
        return self._run_international(address=address, country=country, region=region, top_k=top_k, language=language, kwargs=kwargs)

    def _is_mainland_china_context(self, country: str, region: str) -> bool:
        return _is_china_hint(country, region)

    def _run_baidu(self, address: str, region: str, top_k: int, language: str, kwargs: dict[str, Any]) -> Any:
        baidu = self._baidu_tool()
        if baidu is None:
            return self.result(
                success=False,
                data={"provider": "baidu_maps", "attempts": [{"provider": "baidu_maps", "status": "skipped", "reason": "not_registered"}]},
                error="Baidu Maps geocoding is not available. Configure BAIDU_MAPS_API_KEY and enable the baidu_maps_geocode tool.",
            )
        merged = dict(kwargs)
        merged["address"] = address
        if region and not merged.get("city"):
            merged["city"] = region
        merged["top_k"] = top_k
        result = baidu.safe_run(**merged)
        return self._normalize_routed_result(result, final_provider="baidu_maps", chain_providers=["baidu_maps"])

    def _run_international(self, address: str, country: str, region: str, top_k: int, language: str, kwargs: dict[str, Any]) -> Any:
        locationiq = self._locationiq_tool()
        if locationiq is None:
            return self.result(
                success=False,
                data={"provider": "locationiq", "attempts": ["locationiq"]},
                error="International geocoding requires LOCATIONIQ_KEY in .env.",
            )

        merged = dict(kwargs)
        merged["address"] = address
        merged["top_k"] = top_k
        merged["language"] = language
        if country:
            merged["country_code"] = country
        result = locationiq.safe_run(**merged)
        return self._normalize_routed_result(result, final_provider="locationiq", chain_providers=["locationiq"])

    def _normalize_routed_result(self, result: Any, final_provider: str, chain_providers: list[str]) -> Any:
        # The facade is the tool the Brain called; keep the result attributed to
        # it so tool history, hypothesis merge, and resource accounting see
        # `geocode`, not the internal provider implementation.
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

    def _baidu_tool(self) -> Any | None:
        return self._tool("baidu_maps_geocode")

    def _locationiq_tool(self) -> Any | None:
        return self._tool("locationiq_geocode")

    def _tool(self, name: str) -> Any | None:
        """Instantiate a provider tool (from the workflow-bound set when
        available, otherwise from the registry) and return it if enabled and
        has credentials configured."""
        from geoagent.core.registry import tool_registry as _registry
        from geoagent.core.config import ToolConfig

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
