"""Verify a candidate location using street-view imagery."""

from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from geoagent.core.coordinates import normalize_to_wgs84
from geoagent.core.registry import tool_registry
from geoagent.models.vlm_client import brain_credentials_configured
from geoagent.tools.base import BaseTool
from geoagent.tools.maps.imagery_common import (
    cache_dir_for,
    compare_with_vlm,
    compose_contact_sheet,
    download_image,
    stable_token,
)


@tool_registry.register("streetview_verify")
class StreetViewVerifyTool(BaseTool):
    name = "streetview_verify"
    description = "Verify an existing coordinate hypothesis with Google or Baidu street-view imagery."

    SUPPORTED_PROVIDERS = {"google", "baidu"}

    def is_available(self, state: Any = None) -> bool:
        has_streetview_key = bool(
            self.app_config.env.google_maps_api_key or self.app_config.env.baidu_maps_api_key
        )
        return brain_credentials_configured(self.app_config) and has_streetview_key

    def run(self, **kwargs: Any):
        map_provider = str(kwargs.get("map_provider", "")).strip().lower()
        if map_provider not in self.SUPPORTED_PROVIDERS:
            return self.result(
                success=False,
                data={"supported_map_providers": sorted(self.SUPPORTED_PROVIDERS)},
                error="Missing or unsupported map_provider. Choose google or baidu.",
            )

        lat = kwargs.get("lat")
        lon = kwargs.get("lon", kwargs.get("lng"))
        if lat is None or lon is None:
            return self.result(success=False, error="streetview_verify requires lat and lon for an existing candidate.")

        coordinate_system = str(kwargs.get("coordinate_system", "wgs84") or "wgs84")
        wgs_lon, wgs_lat = normalize_to_wgs84(float(lon), float(lat), coordinate_system)
        radius_m = int(kwargs.get("radius_m", 60))
        radius_m = max(5, min(500, radius_m))
        headings = self._headings(kwargs.get("headings"))
        pitch = int(kwargs.get("pitch", 0))
        pitch = max(-90, min(90, pitch)) if map_provider == "google" else max(0, min(90, pitch))
        fov = int(kwargs.get("fov", kwargs.get("fovy", 90)))
        fov = max(10, min(120, fov))
        task_id = str(kwargs.get("task_id") or "manual")
        image_path = str(kwargs.get("image_path") or "")
        candidate_name = str(kwargs.get("candidate_name") or kwargs.get("name") or "candidate location")
        candidate_id = kwargs.get("candidate_id")
        reason = str(kwargs.get("reason") or "")
        metadata_only = bool(kwargs.get("metadata_only", False))

        cache_dir = cache_dir_for(task_id, self.name)
        token = stable_token(map_provider, wgs_lat, wgs_lon, radius_m, headings, pitch, fov, candidate_name)
        street_dir = cache_dir / token
        street_dir.mkdir(parents=True, exist_ok=True)

        try:
            metadata = self._fetch_metadata(
                provider=map_provider,
                lat=wgs_lat,
                lon=wgs_lon,
                radius_m=radius_m,
                pano_id=kwargs.get("pano_id"),
            )
        except Exception as exc:  # noqa: BLE001
            return self.result(
                success=False,
                data={
                    "map_provider": map_provider,
                    "lat": wgs_lat,
                    "lon": wgs_lon,
                    "coordinate_system": "WGS84",
                    "input_coordinate_system": coordinate_system,
                    "radius_m": radius_m,
                },
                error=f"Street-view metadata lookup failed: {exc}",
            )

        if not metadata.get("available"):
            return self.result(
                data={
                    "source_type": "streetview_verification",
                    "map_provider": map_provider,
                    "provider": f"{map_provider}_streetview",
                    "provider_display_name": "Google Street View" if map_provider == "google" else "Baidu Street View",
                    "candidate_name": candidate_name,
                    "candidate_id": candidate_id,
                    "lat": wgs_lat,
                    "lon": wgs_lon,
                    "coordinate_system": "WGS84",
                    "input_coordinate_system": coordinate_system,
                    "radius_m": radius_m,
                    "streetview_available": False,
                    "metadata": metadata,
                    "images": [],
                    "verification": {
                        "verdict": "inconclusive",
                        "confidence": 0.2,
                        "matching_features": [],
                        "contradictions": [],
                        "limitations": ["No street-view coverage was found near the candidate."],
                        "recommended_next_step": "",
                        "provider": "metadata",
                        "raw": {},
                    },
                    "verdict": "inconclusive",
                    "confidence": 0.2,
                }
            )

        if metadata_only:
            return self.result(
                data={
                    "source_type": "streetview_metadata",
                    "map_provider": map_provider,
                    "provider": f"{map_provider}_streetview",
                    "provider_display_name": "Google Street View" if map_provider == "google" else "Baidu Street View",
                    "candidate_name": candidate_name,
                    "candidate_id": candidate_id,
                    "lat": wgs_lat,
                    "lon": wgs_lon,
                    "coordinate_system": "WGS84",
                    "radius_m": radius_m,
                    "streetview_available": True,
                    "metadata": metadata,
                    "images": [],
                    "verification": {
                        "verdict": "inconclusive",
                        "confidence": 0.3,
                        "matching_features": [],
                        "contradictions": [],
                        "limitations": ["Street-view coverage exists, but imagery was not downloaded."],
                        "recommended_next_step": "",
                        "provider": "metadata",
                        "raw": {},
                    },
                    "verdict": "inconclusive",
                    "confidence": 0.3,
                }
            )

        try:
            images = self._download_views(
                provider=map_provider,
                metadata=metadata,
                headings=headings,
                pitch=pitch,
                fov=fov,
                street_dir=street_dir,
            )
            contact_sheet = compose_contact_sheet(
                images,
                street_dir / f"{map_provider}_streetview_contact_sheet.png",
                columns=2 if len(images) <= 4 else 3,
            )
        except Exception as exc:  # noqa: BLE001
            return self.result(
                success=False,
                data={
                    "map_provider": map_provider,
                    "lat": wgs_lat,
                    "lon": wgs_lon,
                    "coordinate_system": "WGS84",
                    "input_coordinate_system": coordinate_system,
                    "metadata": metadata,
                },
                error=f"Street-view image retrieval failed: {exc}",
            )

        verification = compare_with_vlm(
            app_config=self.app_config,
            source_image_path=image_path,
            reference_image_path=str(contact_sheet),
            prompt=self._comparison_prompt(
                candidate_name=candidate_name,
                map_provider=map_provider,
                lat=wgs_lat,
                lon=wgs_lon,
                headings=headings,
                reason=reason,
            ),
        )
        return self.result(
            data={
                "source_type": "streetview_verification",
                "map_provider": map_provider,
                "provider": f"{map_provider}_streetview",
                "provider_display_name": "Google Street View" if map_provider == "google" else "Baidu Street View",
                "candidate_name": candidate_name,
                "candidate_id": candidate_id,
                "lat": wgs_lat,
                "lon": wgs_lon,
                "coordinate_system": "WGS84",
                "input_coordinate_system": coordinate_system,
                "radius_m": radius_m,
                "headings": headings,
                "pitch": pitch,
                "fov": fov,
                "streetview_available": True,
                "metadata": metadata,
                "cache_dir": str(street_dir),
                "contact_sheet_path": str(contact_sheet),
                "images": images,
                "verification": verification,
                "verdict": verification["verdict"],
                "confidence": verification["confidence"],
            }
        )

    def _fetch_metadata(
        self,
        provider: str,
        lat: float,
        lon: float,
        radius_m: int,
        pano_id: Any | None = None,
    ) -> dict[str, Any]:
        if provider == "google":
            return self._fetch_google_metadata(lat=lat, lon=lon, radius_m=radius_m, pano_id=pano_id)
        return self._fetch_baidu_metadata(lat=lat, lon=lon, pano_id=pano_id)

    def _fetch_google_metadata(
        self,
        lat: float,
        lon: float,
        radius_m: int,
        pano_id: Any | None = None,
    ) -> dict[str, Any]:
        api_key = self.app_config.env.google_maps_api_key
        if not api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is required for Google Street View Static API.")
        radius = max(5, min(500, int(radius_m)))
        params: dict[str, Any] = {"key": api_key.get_secret_value()}
        if pano_id:
            params["pano"] = str(pano_id)
        else:
            params["location"] = f"{lat},{lon}"
            params["radius"] = radius
        url = self._signed_google_url(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params,
        )
        response = requests.get(url, timeout=45)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        payload = response.json()
        status = str(payload.get("status") or "UNKNOWN_ERROR").upper()
        if status in {"ZERO_RESULTS", "NOT_FOUND"}:
            return {
                "available": False,
                "provider": "google_streetview_static_api",
                "status": status,
                "lat": lat,
                "lon": lon,
                "coordinate_system": "WGS84",
                "raw": payload,
            }
        if status != "OK":
            message = str(payload.get("error_message") or "request failed")
            raise RuntimeError(f"Google Street View Static API returned {status}: {message}")
        location = payload.get("location") or {}
        return {
            "available": True,
            "provider": "google_streetview_static_api",
            "status": status,
            "pano_id": payload.get("pano_id"),
            "lat": float(location.get("lat", lat)),
            "lon": float(location.get("lng", lon)),
            "coordinate_system": "WGS84",
            "date": payload.get("date"),
            "copyright": payload.get("copyright"),
            "raw": payload,
        }

    def _fetch_baidu_metadata(
        self,
        lat: float,
        lon: float,
        pano_id: Any | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "pano_id": str(pano_id) if pano_id else None,
            "lat": lat,
            "lon": lon,
            "coordinate_system": "WGS84",
        }
        url = self._baidu_streetview_image_url(metadata, heading=0, pitch=0, fov=90, width=10, height=10)
        response = requests.get(url, timeout=45)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        content_type = str(response.headers.get("Content-Type") or "").lower()
        if content_type.startswith("image/"):
            return {
                "available": True,
                "provider": "baidu_panorama_static_api",
                "status": "OK",
                **metadata,
            }
        payload = response.json()
        status = str(payload.get("status") or payload.get("code") or "UNKNOWN_ERROR")
        if status in {"402", "405"}:
            return {
                "available": False,
                "provider": "baidu_panorama_static_api",
                "status": status,
                **metadata,
                "raw": payload,
            }
        message = str(payload.get("message") or payload.get("msg") or "request failed")
        raise RuntimeError(f"Baidu Panorama Static API returned {status}: {message}")

    def _download_views(
        self,
        provider: str,
        metadata: dict[str, Any],
        headings: list[int],
        pitch: int,
        fov: int,
        street_dir: Path,
    ) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for heading in headings:
            path = street_dir / f"{provider}_streetview_h{heading}_p{pitch}_f{fov}.jpg"
            label = f"{provider} streetview heading {heading}"
            url = self._streetview_image_url(provider, metadata, heading, pitch, fov)
            download_image(url, path)
            images.append(
                {
                    "provider": provider,
                    "heading": heading,
                    "pitch": pitch,
                    "fov": fov,
                    "path": str(path),
                    "url": self._safe_url(url),
                    "label": label,
                }
            )
        return images

    def _streetview_image_url(
        self,
        provider: str,
        metadata: dict[str, Any],
        heading: int,
        pitch: int,
        fov: int,
    ) -> str:
        pano_id = str(metadata.get("pano_id") or "")
        if provider == "google":
            api_key = self.app_config.env.google_maps_api_key
            if not api_key:
                raise RuntimeError("GOOGLE_MAPS_API_KEY is required for Google Street View Static API.")
            params: dict[str, Any] = {
                "size": "640x640",
                "heading": heading,
                "pitch": pitch,
                "fov": fov,
                "key": api_key.get_secret_value(),
                "return_error_code": "true",
            }
            if pano_id:
                params["pano"] = pano_id
            else:
                params["location"] = f"{metadata['lat']},{metadata['lon']}"
            return self._signed_google_url(
                "https://maps.googleapis.com/maps/api/streetview",
                params,
            )
        return self._baidu_streetview_image_url(metadata, heading, pitch, fov)

    def _baidu_streetview_image_url(
        self,
        metadata: dict[str, Any],
        heading: int,
        pitch: int,
        fov: int,
        width: int = 1024,
        height: int = 512,
    ) -> str:
        api_key = self.app_config.env.baidu_maps_api_key
        if not api_key:
            raise RuntimeError("BAIDU_MAPS_API_KEY is required for Baidu Panorama Static API.")
        params: dict[str, Any] = {
            "ak": api_key.get_secret_value(),
            "width": width,
            "height": height,
            "coordtype": "wgs84ll",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
        }
        pano_id = str(metadata.get("pano_id") or "")
        if pano_id:
            params["panoid"] = pano_id
        else:
            params["location"] = f"{metadata['lon']},{metadata['lat']}"
        return f"https://api.map.baidu.com/panorama/v2?{urlencode(params)}"

    def _signed_google_url(self, base_url: str, params: dict[str, Any]) -> str:
        query = urlencode(params)
        unsigned = f"{base_url}?{query}"
        secret = self.app_config.env.google_maps_url_signing_secret
        if not secret:
            return unsigned
        parsed = urlparse(unsigned)
        path_and_query = f"{parsed.path}?{parsed.query}"
        raw_secret = secret.get_secret_value()
        padding = "=" * (-len(raw_secret) % 4)
        try:
            decoded_secret = base64.urlsafe_b64decode(raw_secret + padding)
        except Exception:
            return unsigned
        digest = hmac.new(decoded_secret, path_and_query.encode("utf-8"), hashlib.sha1).digest()
        signature = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        pairs.append(("signature", signature))
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode(pairs), ""))

    def _safe_url(self, url: str) -> str:
        parsed = urlparse(url)
        pairs = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if key.lower() in {"ak", "key", "signature"}:
                pairs.append((key, "***"))
            else:
                pairs.append((key, value))
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode(pairs), ""))

    def _headings(self, raw: Any) -> list[int]:
        if raw is None or raw == "":
            return [0, 90, 180, 270]
        if isinstance(raw, str):
            values = [item.strip() for item in raw.split(",")]
        elif isinstance(raw, list):
            values = raw
        else:
            values = [raw]
        headings: list[int] = []
        for value in values:
            try:
                headings.append(int(float(value)) % 360)
            except (TypeError, ValueError):
                continue
        deduped = []
        for heading in headings:
            if heading not in deduped:
                deduped.append(heading)
        return deduped[:8] or [0, 90, 180, 270]

    def _comparison_prompt(
        self,
        candidate_name: str,
        map_provider: str,
        lat: float,
        lon: float,
        headings: list[int],
        reason: str,
    ) -> str:
        return f"""
You are verifying a geolocation hypothesis by comparing two images.

Image 1 is the original user-provided ground-level image.
Image 2 is a contact sheet of street-view images near the candidate location from {map_provider}.

Candidate:
- name: {candidate_name}
- coordinates_wgs84: lat={lat}, lon={lon}
- sampled_headings: {headings}
- reason_for_verification: {reason or "not specified"}

Use street-view imagery for fine-grained comparison: storefronts, facade shapes, road markings, lane layout, signs, text, poles, road width, vegetation, and nearby buildings.
Do not over-penalize viewpoint, date, seasonal, or occlusion differences. If the target object may simply be outside the sampled headings, return inconclusive rather than contradicts.

Return only JSON:
{{
  "verdict": "supports | contradicts | inconclusive",
  "confidence": 0.0,
  "matching_features": ["features that match"],
  "contradictions": ["features that conflict"],
  "limitations": ["coverage/viewpoint/time/occlusion limits"],
  "recommended_next_step": "short next step or empty string"
}}
""".strip()
