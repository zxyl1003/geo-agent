"""Baidu Maps geocoding and reverse geocoding tool."""

from __future__ import annotations

from typing import Any

import requests

from geoagent.core.coordinates import normalize_coordinate_fields
from geoagent.core.registry import tool_registry
from geoagent.tools.base import BaseTool


@tool_registry.register("baidu_maps_geocode")
class BaiduMapsGeocodeTool(BaseTool):
    name = "baidu_maps_geocode"
    description = "Geocode a mainland China address or reverse geocode coordinates using Baidu Maps Geocoding API v3; public lat/lon outputs are WGS84."
    # Internal provider behind the geocode/reverse_geocode facades. Never
    # exposed directly to the Brain.
    hidden = True

    def run(self, **kwargs: Any):
        address = str(kwargs.get("address", "")).strip()
        city = str(kwargs.get("city", kwargs.get("region", "")) or "").strip()
        lat = kwargs.get("lat")
        lon = kwargs.get("lon", kwargs.get("lng"))
        latlng = str(kwargs.get("latlng", "") or kwargs.get("location", "") or "").strip()
        top_k = int(kwargs.get("top_k", 3))
        coordtype = self._canonical_coordtype(kwargs.get("coordtype") or "WGS84")
        raw_ret_coordtype = str(kwargs.get("ret_coordtype", "") or "").strip()
        ret_coordtype = self._canonical_coordtype(raw_ret_coordtype) if raw_ret_coordtype else ""
        # Baidu's geocoding/reverse-geocoding endpoints document ret_coordtype as
        # gcj02ll / bd09mc only (default bd09ll); wgs84ll is NOT a valid return
        # coordinate system. Requesting WGS84 output is achieved by leaving
        # ret_coordtype unset (default BD-09) and converting internally.
        if ret_coordtype == "WGS84":
            ret_coordtype = ""

        mode = self._mode(address=address, lat=lat, lon=lon, latlng=latlng)
        if mode == "missing":
            return self.result(success=False, error="Missing geocode input. Provide address, lat/lon, or latlng.")

        api_key = self.app_config.env.baidu_maps_api_key
        if api_key is None:
            return self.result(success=False, error="BAIDU_MAPS_API_KEY is not configured.")
        try:
            if mode == "geocode":
                data = self._baidu_geocode(
                    api_key=api_key.get_secret_value(),
                    address=address,
                    city=city,
                    ret_coordtype=ret_coordtype,
                    top_k=top_k,
                    extra=kwargs,
                )
            else:
                data = self._baidu_reverse_geocode(
                    api_key=api_key.get_secret_value(),
                    lat=lat,
                    lon=lon,
                    latlng=latlng,
                    coordtype=coordtype,
                    ret_coordtype=ret_coordtype,
                    top_k=top_k,
                    extra=kwargs,
                )
            return self.result(data=data)
        except Exception as exc:  # noqa: BLE001
            return self.result(
                success=False,
                data={"provider": "baidu_maps_geocoding", "mode": mode, "address": address, "lat": lat, "lon": lon},
                error=f"Baidu Maps geocoding failed: {exc}",
            )

    def _mode(self, address: str, lat: Any, lon: Any, latlng: str) -> str:
        if address:
            return "geocode"
        if latlng or (lat is not None and lon is not None):
            return "reverse_geocode"
        return "missing"

    def _baidu_geocode(
        self,
        api_key: str,
        address: str,
        city: str,
        ret_coordtype: str,
        top_k: int,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"address": address, "output": "json", "ak": api_key}
        if city:
            params["city"] = city
        if ret_coordtype:
            params["ret_coordtype"] = self._baidu_api_coordtype(ret_coordtype)
        if "extension_analys_level" in extra:
            params["extension_analys_level"] = extra["extension_analys_level"]
        if "extension_poi_infos" in extra:
            params["extension_poi_infos"] = extra["extension_poi_infos"]

        response = requests.get("https://api.map.baidu.com/geocoding/v3/", params=params, timeout=45)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        status = payload.get("status")
        if str(status) != "0":
            raise RuntimeError(f"Baidu Maps status {status}: {payload.get('message', '') or payload.get('msg', '')}")

        result = self._normalize_geocode_result(
            payload.get("result") or {},
            address=address,
            city=city,
            coordinate_system=ret_coordtype or "BD-09",
            top_k=top_k,
        )
        return {
            "provider": "baidu_maps_geocoding",
            "provider_display_name": "Baidu Maps",
            "map_provider": "baidu",
            "source_type": "geocoding_api",
            "coordinate_system": "WGS84",
            "using_configured_api_key": True,
            "mode": "geocode",
            "status": status,
            "address": address,
            "city": city or None,
            "coordtype": None,
            "ret_coordtype": ret_coordtype or "BD-09",
            "results": [result] if result else [],
            "top_result": result,
            "lat": result.get("lat") if result else None,
            "lon": result.get("lon") if result else None,
            "raw_lat": result.get("raw_lat") if result else None,
            "raw_lon": result.get("raw_lon") if result else None,
            "raw_coordinate_system": result.get("raw_coordinate_system") if result else None,
            "formatted_address": result.get("formatted_address") if result else address,
        }

    def _baidu_reverse_geocode(
        self,
        api_key: str,
        lat: Any,
        lon: Any,
        latlng: str,
        coordtype: str,
        ret_coordtype: str,
        top_k: int,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        location = latlng or f"{float(lat)},{float(lon)}"
        api_coordtype = self._baidu_api_coordtype(coordtype)
        params: dict[str, Any] = {
            "ak": api_key,
            "location": location,
            "coordtype": api_coordtype,
            "output": "json",
            "extensions_poi": extra.get("extensions_poi", 1),
        }
        if ret_coordtype:
            params["ret_coordtype"] = self._baidu_api_coordtype(ret_coordtype)
        for key in (
            "radius",
            "poi_types",
            "extensions_road",
            "extensions_town",
            "region_data_source",
            "entire_poi",
            "sort_strategy",
        ):
            if key in extra and extra[key] is not None:
                params[key] = extra[key]

        response = requests.get("https://api.map.baidu.com/reverse_geocoding/v3/", params=params, timeout=45)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        status = payload.get("status")
        if str(status) != "0":
            raise RuntimeError(f"Baidu Maps status {status}: {payload.get('message', '') or payload.get('msg', '')}")

        result = self._normalize_reverse_result(
            payload.get("result") or {},
            fallback_lat=lat,
            fallback_lon=lon,
            fallback_latlng=latlng,
            result_coordinate_system=ret_coordtype or "BD-09",
            fallback_coordinate_system=coordtype,
            top_k=top_k,
        )
        return {
            "provider": "baidu_maps_geocoding",
            "provider_display_name": "Baidu Maps",
            "map_provider": "baidu",
            "source_type": "reverse_geocoding_api",
            "coordinate_system": "WGS84",
            "using_configured_api_key": True,
            "mode": "reverse_geocode",
            "status": status,
            "address": None,
            "latlng": location,
            "coordtype": coordtype,
            "ret_coordtype": ret_coordtype or "BD-09",
            "results": [result] if result else [],
            "top_result": result,
            "lat": result.get("lat") if result else None,
            "lon": result.get("lon") if result else None,
            "raw_lat": result.get("raw_lat") if result else None,
            "raw_lon": result.get("raw_lon") if result else None,
            "raw_coordinate_system": result.get("raw_coordinate_system") if result else None,
            "formatted_address": result.get("formatted_address") if result else None,
        }

    def _canonical_coordtype(self, coordtype: Any) -> str:
        coord = str(coordtype or "WGS84").strip().lower().replace("-", "").replace("_", "")
        if coord in {"wgs84", "wgs84ll", "gps"}:
            return "WGS84"
        if coord in {"gcj02", "gcj02ll"}:
            return "GCJ-02"
        if coord in {"bd09", "bd09ll", "bd09latlon"}:
            return "BD-09"
        if coord in {"bd09mc", "baidumc"}:
            return "BD-09MC"
        return str(coordtype or "WGS84").strip()

    def _baidu_api_coordtype(self, coordtype: str) -> str:
        coord = str(coordtype or "WGS84").strip().lower().replace("-", "").replace("_", "")
        if coord in {"wgs84", "wgs84ll", "gps"}:
            return "wgs84ll"
        if coord in {"gcj02", "gcj02ll"}:
            return "gcj02ll"
        if coord in {"bd09", "bd09ll", "bd09latlon"}:
            return "bd09ll"
        if coord in {"bd09mc", "baidumc"}:
            return "bd09mc"
        return str(coordtype or "WGS84").strip()

    def _normalize_geocode_result(
        self,
        result: dict[str, Any],
        address: str,
        city: str,
        coordinate_system: str,
        top_k: int,
    ) -> dict[str, Any]:
        location = result.get("location") or {}
        poi_infos = result.get("poi_infos") or []
        primary_poi = poi_infos[0] if poi_infos else {}
        formatted_address = primary_poi.get("formatted_address") or address
        coordinates = normalize_coordinate_fields(location.get("lat"), location.get("lng"), coordinate_system)
        return {
            "formatted_address": formatted_address,
            "place_id": primary_poi.get("uid"),
            "types": [item for item in [result.get("level"), result.get("analys_level")] if item],
            **coordinates,
            "location_type": "precise" if result.get("precise") == 1 else "approximate",
            "precise": result.get("precise"),
            "confidence": result.get("confidence"),
            "comprehension": result.get("comprehension"),
            "level": result.get("level"),
            "analys_level": result.get("analys_level"),
            "city": city or primary_poi.get("city"),
            "address_components": self._address_components_from_poi(primary_poi),
            "pois": [self._normalize_geocode_poi(item, coordinate_system) for item in poi_infos[: max(1, top_k)]],
        }

    def _normalize_reverse_result(
        self,
        result: dict[str, Any],
        fallback_lat: Any,
        fallback_lon: Any,
        fallback_latlng: str,
        result_coordinate_system: str,
        fallback_coordinate_system: str,
        top_k: int,
    ) -> dict[str, Any]:
        location = result.get("location") or {}
        lat, lon = self._coerce_lat_lon(fallback_lat, fallback_lon, fallback_latlng)
        if location:
            coordinates = normalize_coordinate_fields(location.get("lat"), location.get("lng"), result_coordinate_system)
        else:
            coordinates = normalize_coordinate_fields(lat, lon, fallback_coordinate_system)
        pois = result.get("pois") or []
        formatted_address = result.get("formatted_address_poi") or result.get("formatted_address")
        return {
            "formatted_address": formatted_address,
            "place_id": pois[0].get("uid") if pois else None,
            "types": ["reverse_geocode"],
            **coordinates,
            "location_type": "reverse_geocode",
            "address_components": result.get("addressComponent") or {},
            "business": result.get("business"),
            "business_info": result.get("business_info") or [],
            "semantic_description": result.get("sematic_description"),
            "formatted_address_poi": result.get("formatted_address_poi"),
            "roads": result.get("roads") or [],
            "pois": [self._normalize_reverse_poi(item, result_coordinate_system) for item in pois[: max(1, top_k)]],
        }

    def _normalize_geocode_poi(self, item: dict[str, Any], coordinate_system: str) -> dict[str, Any]:
        location = item.get("location") or {}
        coordinates = normalize_coordinate_fields(location.get("lat"), location.get("lng"), coordinate_system)
        return {
            "uid": item.get("uid"),
            "name": item.get("name"),
            "formatted_address": item.get("formatted_address"),
            **coordinates,
            "level": item.get("level"),
            "confidence": item.get("confidence"),
            "province": item.get("province"),
            "city": item.get("city"),
            "district": item.get("district"),
            "adcode": item.get("adcode"),
        }

    def _normalize_reverse_poi(self, item: dict[str, Any], coordinate_system: str) -> dict[str, Any]:
        point = item.get("point") or {}
        coordinates = normalize_coordinate_fields(point.get("y") or point.get("lat"), point.get("x") or point.get("lng"), coordinate_system)
        return {
            "uid": item.get("uid"),
            "name": item.get("name"),
            "address": item.get("addr"),
            "tag": item.get("tag"),
            "distance": item.get("distance"),
            "direction": item.get("direction"),
            **coordinates,
        }

    def _address_components_from_poi(self, item: dict[str, Any]) -> dict[str, Any]:
        keys = ("country", "province", "city", "district", "town", "street", "street_number", "adcode")
        return {key: item.get(key) for key in keys if item.get(key) is not None}

    def _coerce_lat_lon(self, lat: Any, lon: Any, latlng: str) -> tuple[float | None, float | None]:
        if lat is not None and lon is not None:
            return float(lat), float(lon)
        if "," in latlng:
            lat_text, lon_text = latlng.split(",", 1)
            return float(lat_text), float(lon_text)
        return None, None

    def is_available(self, state: Any = None) -> bool:
        return bool(self.app_config.env.baidu_maps_api_key)
