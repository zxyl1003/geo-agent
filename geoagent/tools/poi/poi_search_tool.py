"""POI search with internal provider routing.

Routing (hidden from the Brain — the Brain only calls `poi_search`):
- Mainland China (country/region hints indicate China, or query contains
  Chinese script) -> Baidu Maps POI search.
- International -> Serper Places (Google Places semantics).

The result always exposes the final `provider` / `provider_display_name` and an
`attempts` chain so API usage can be traced in the audit log.
"""

from __future__ import annotations

from typing import Any

import pycountry
import requests

from geoagent.core.coordinates import normalize_coordinate_fields
from geoagent.core.registry import tool_registry
from geoagent.core.retrieval_cache import RetrievalCache
from geoagent.tools.base import BaseTool
from geoagent.tools.search.geocode_tool import _is_china_hint, _normalize_country


_COUNTRY_DEFAULT_LANGUAGES: dict[str, str] = {
    "ae": "ar", "ar": "es", "au": "en", "br": "pt", "ca": "en", "ch": "de",
    "cn": "zh-cn", "de": "de", "eg": "ar", "es": "es", "fr": "fr", "gb": "en",
    "gh": "en", "hk": "zh-hk", "id": "id", "in": "hi", "it": "it", "jp": "ja",
    "ke": "sw", "kr": "ko", "mx": "es", "my": "ms", "ng": "en", "nl": "nl",
    "no": "no", "nz": "en", "ph": "en", "pl": "pl", "ru": "ru", "sa": "ar", "se": "sv",
    "sg": "en", "th": "th", "tr": "tr", "tw": "zh-tw", "us": "en", "vn": "vi",
    "za": "en",
}

_COUNTRY_NAME_ALIASES = {
    "russia": "RU",
}


@tool_registry.register("poi_search")
class POISearchTool(BaseTool):
    name = "poi_search"
    description = (
        "Search POIs and return candidate metadata and WGS84 coordinates. "
        "Routes automatically: mainland China POIs use Baidu Maps; "
        "international POIs require an ISO 3166-1 country name or code and use "
        "Serper Places (Google map semantics)."
    )

    def is_available(self, state: Any = None) -> bool:
        return bool(self.app_config.env.serper_api_key or self.app_config.env.baidu_maps_api_key)

    def run(self, **kwargs: Any):
        allowed_arguments = {"query", "region", "country", "observed_entity_id", "top_k"}
        unsupported_arguments = sorted(set(kwargs) - allowed_arguments)
        if unsupported_arguments:
            return self.result(
                success=False,
                error=f"Unsupported poi_search arguments: {', '.join(unsupported_arguments)}.",
            )

        query = " ".join(str(kwargs.get("query", "")).split())
        region = " ".join(str(kwargs.get("region", "")).split())
        country = str(kwargs.get("country", "") or "").strip()
        observed_entity_id = str(kwargs.get("observed_entity_id", "") or "").strip()
        top_k = max(1, min(int(kwargs.get("top_k", 5)), 10))
        if not query and not region:
            return self.result(success=False, error="Missing POI search query or region.")

        map_provider = "baidu" if self._is_china_hint(country, region, query) else "google"

        public_args: dict[str, Any] = {
            "query": query,
            "region": region,
            "country": country or None,
            "map_provider": map_provider,
            "observed_entity_id": observed_entity_id,
            "top_k": top_k,
        }
        serper_locale: tuple[str, str, str] | None = None
        if map_provider == "google":
            try:
                serper_locale = self._resolve_serper_locale(
                    region=region,
                    country=country,
                )
            except ValueError as exc:
                return self.result(
                    success=False,
                    data={"query": query, "region": region, "map_provider": "google"},
                    error=f"Invalid Serper Places localization: {exc}",
                )
            location, gl, hl = serper_locale
            public_args.update({"location": location, "gl": gl, "hl": hl})
        cache_provider = "serper_places" if map_provider == "google" else map_provider
        cache = RetrievalCache.from_config(self.app_config)
        cached = cache.get(self.name, cache_provider, public_args)
        if cached is not None:
            if isinstance(cached.get("candidates"), list):
                cached["candidates"] = cached["candidates"][:top_k]
            cached["cache_hit"] = True
            return self.result(data=cached)

        full_query = f"{query} {region}".strip()
        if map_provider == "google":
            location, gl, hl = serper_locale or ("", "", "")
            result = self._run_google(query or region, region, location, gl, hl, top_k, observed_entity_id)
        else:
            result = self._run_baidu(query or region, region, top_k, observed_entity_id)
        if result.success:
            cache.put(self.name, cache_provider, public_args, result.data)
            result.data["cache_hit"] = False
        return result

    def _is_china_hint(self, country: str, region: str, query: str) -> bool:
        if _is_china_hint(country, region):
            return True
        # Chinese-script query without an explicit international country hint is
        # treated as mainland China.
        if any("一" <= char <= "鿿" for char in query):
            return not _normalize_country(country)
        return False

    def _run_google(
        self,
        query: str,
        region: str,
        location: str,
        gl: str,
        hl: str,
        top_k: int,
        observed_entity_id: str,
    ):
        api_key = self.app_config.env.serper_api_key
        if not api_key:
            return self.result(
                success=False,
                data={
                    "query": query,
                    "region": region,
                    "location": location,
                    "gl": gl,
                    "hl": hl,
                    "map_provider": "google",
                },
                error="SERPER_API_KEY is not configured for Google POI search.",
            )
        try:
            candidates = self._attach_observed_entity_id(
                self._serper_places_search(api_key.get_secret_value(), query, location, gl, hl, top_k), observed_entity_id
            )
            data = self._response_data(query, region, observed_entity_id, "google", "serper_places", candidates)
            data.update({"location": location, "gl": gl, "hl": hl})
            return self.result(data=data)
        except Exception as exc:  # noqa: BLE001
            return self.result(
                success=False,
                data={
                    "query": query, "region": region, "location": location, "gl": gl, "hl": hl,
                    "map_provider": "google", "provider": "serper_places",
                },
                error=f"Serper Places POI search failed: {exc}",
            )

    def _run_baidu(self, query: str, region: str, top_k: int, observed_entity_id: str):
        api_key = self.app_config.env.baidu_maps_api_key
        if not api_key:
            return self.result(success=False, error="BAIDU_MAPS_API_KEY is not configured.")
        try:
            candidates = self._attach_observed_entity_id(
                self._baidu_region_search(api_key.get_secret_value(), query, region, top_k), observed_entity_id
            )
            return self.result(data=self._response_data(query, region, observed_entity_id, "baidu", "baidu_maps", candidates))
        except Exception as exc:  # noqa: BLE001
            return self.result(success=False, error=f"Baidu Maps POI search failed: {exc}")

    def _response_data(
        self,
        query: str,
        region: str,
        observed_entity_id: str,
        map_provider: str,
        provider: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "query": query,
            "region": region,
            "observed_entity_id": observed_entity_id or None,
            "map_provider": map_provider,
            "provider": provider,
            "coordinate_system": "WGS84",
            "using_configured_api_key": True,
            "candidates": candidates,
        }

    def _serper_places_search(
        self,
        api_key: str,
        query: str,
        location: str,
        gl: str,
        hl: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        response = requests.post(
            self.app_config.env.serper_places_base_url,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "location": location, "gl": gl, "hl": hl, "num": top_k},
            timeout=45,
        )
        data = self._json_or_raise(response)
        return [
            self._normalize_serper_place(item)
            for item in data.get("places") or []
            if isinstance(item, dict)
        ][:top_k]

    def _resolve_serper_locale(self, region: str, country: str) -> tuple[str, str, str]:
        if not country:
            raise ValueError(
                "country is required for international POI search and must be an "
                "ISO 3166-1 name, alpha-2 code, or alpha-3 code"
            )
        try:
            lookup_value = _COUNTRY_NAME_ALIASES.get(country.casefold(), country)
            iso_country = pycountry.countries.lookup(lookup_value)
        except LookupError as exc:
            raise ValueError(
                f"country must be a standard ISO 3166-1 name or code: {country}"
            ) from exc

        gl = iso_country.alpha_2.lower()
        hl = _COUNTRY_DEFAULT_LANGUAGES.get(gl, "en")
        location = " ".join(region.split()) or iso_country.name
        return location, gl, hl

    def _baidu_region_search(self, api_key: str, query: str, region: str, top_k: int) -> list[dict[str, Any]]:
        response = requests.get(
            "https://api.map.baidu.com/place/v3/region",
            params={
                "query": query,
                "region": region or "全国",
                "region_limit": "true" if region else "false",
                "scope": 2,
                "page_size": top_k,
                "page_num": 0,
                "output": "json",
                "ak": api_key,
            },
            timeout=45,
        )
        data = self._json_or_raise(response)
        if str(data.get("status")) != "0":
            raise RuntimeError(f"Baidu Maps status {data.get('status')}: {data.get('message', '')}")
        return [
            self._normalize_baidu_place(item)
            for item in data.get("results") or []
            if isinstance(item, dict)
        ][:top_k]

    def _json_or_raise(self, response: requests.Response) -> dict[str, Any]:
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Expected JSON object response.")
        return data

    def _normalize_serper_place(self, place: dict[str, Any]) -> dict[str, Any]:
        place_id = place.get("placeId") or place.get("cid")
        category = place.get("type") or place.get("category")
        return {
            "place_id": place_id,
            "resource_name": place_id,
            "name": place.get("title"),
            "address": place.get("address"),
            "category": category,
            "types": place.get("types") or self._split_types(category),
            **normalize_coordinate_fields(place.get("latitude"), place.get("longitude"), "wgs84"),
            "url": place.get("website"),
            "website": place.get("website"),
            "phone_number": place.get("phoneNumber"),
            "description": place.get("description"),
            "rating": place.get("rating"),
            "rating_count": place.get("ratingCount"),
            "position": place.get("position"),
            "cid": place.get("cid"),
            "score": place.get("rating"),
            "map_provider": "google",
            "provider": "serper_places",
            "provider_detail": "serper_google_places",
            "details_complete": True,
        }

    def _normalize_baidu_place(self, place: dict[str, Any]) -> dict[str, Any]:
        detail = place.get("detail_info") or {}
        category = detail.get("classified_poi_tag") or detail.get("type") or detail.get("tag")
        location = place.get("location") or {}
        return {
            "place_id": place.get("uid"),
            "resource_name": place.get("uid"),
            "name": place.get("name"),
            "address": place.get("address"),
            "category": category,
            "types": self._split_types(category),
            **normalize_coordinate_fields(location.get("lat"), location.get("lng"), "bd09ll"),
            "url": detail.get("detail_url"),
            "phone_number": place.get("telephone"),
            "rating": detail.get("overall_rating"),
            "score": detail.get("overall_rating"),
            "map_provider": "baidu",
            "provider": "baidu_maps",
            "province": place.get("province"),
            "city": place.get("city"),
            "area": place.get("area"),
            "adcode": place.get("adcode"),
            "details_complete": True,
        }

    def _split_types(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item]
        return [item.strip() for item in str(value).replace(",", ";").split(";") if item.strip()]

    def _attach_observed_entity_id(self, candidates: list[dict[str, Any]], observed_entity_id: str) -> list[dict[str, Any]]:
        if observed_entity_id:
            for candidate in candidates:
                candidate.setdefault("observed_entity_id", observed_entity_id)
        return candidates
