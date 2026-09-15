"""Verify a candidate location using street-view imagery."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from geoagent.core.coordinates import normalize_to_wgs84, wgs84_to_bd09mc
from geoagent.core.registry import tool_registry
from geoagent.models.vlm_client import brain_credentials_configured
from geoagent.tools.base import BaseTool
from geoagent.tools.maps.imagery_common import (
    browser_headers,
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
        return brain_credentials_configured(self.app_config)

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
        if provider == "google" and pano_id:
            return {
                "available": True,
                "provider": "google_streetview_unofficial",
                "pano_id": str(pano_id),
                "lat": lat,
                "lon": lon,
                "coordinate_system": "WGS84",
                "status": "provided_pano_id",
            }
        if provider == "google":
            return self._fetch_google_metadata(lat=lat, lon=lon, radius_m=radius_m)
        return self._fetch_baidu_metadata(lat=lat, lon=lon)

    def _fetch_google_metadata(self, lat: float, lon: float, radius_m: int) -> dict[str, Any]:
        radius = max(5, min(500, int(radius_m)))
        url = (
            "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
            f"?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}!2d{radius}"
            "!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2"
            "!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2"
            "&callback=_xdc_._geoagent"
        )
        response = requests.get(url, timeout=45, headers=browser_headers("https://www.google.com/maps/"))
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        matches = re.findall(
            r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+(?:\.[0-9]+)?),(-?[0-9]+(?:\.[0-9]+)?)',
            response.text,
        )
        seen: set[str] = set()
        panoramas = []
        for pano_id, pano_lat, pano_lon in matches:
            if pano_id in seen:
                continue
            seen.add(pano_id)
            panoramas.append(
                {
                    "pano_id": pano_id,
                    "lat": float(pano_lat),
                    "lon": float(pano_lon),
                    "coordinate_system": "WGS84",
                }
            )
        if not panoramas:
            return {
                "available": False,
                "provider": "google_streetview_unofficial",
                "status": "ZERO_RESULTS",
                "lat": lat,
                "lon": lon,
                "coordinate_system": "WGS84",
                "raw_preview": response.text[:500],
            }
        top = panoramas[0]
        return {
            "available": True,
            "provider": "google_streetview_unofficial",
            "status": "OK",
            "pano_id": top["pano_id"],
            "lat": top["lat"],
            "lon": top["lon"],
            "coordinate_system": "WGS84",
            "panorama_count": len(panoramas),
            "raw": {"panoramas": panoramas[:8]},
        }

    def _fetch_baidu_metadata(self, lat: float, lon: float) -> dict[str, Any]:
        mc_x, mc_y = wgs84_to_bd09mc(lon, lat)
        qs_url = f"https://mapsv0.bdimg.com/?qt=qsdata&x={mc_x}&y={mc_y}"
        response = requests.get(qs_url, timeout=45)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        qs_payload = response.json()
        content = qs_payload.get("content") or {}
        pano_id = content.get("id")
        if not pano_id:
            return {
                "available": False,
                "provider": "baidu_streetview",
                "status": "ZERO_RESULTS",
                "bd09mc": {"x": mc_x, "y": mc_y},
                "raw": qs_payload,
            }
        s_content: dict[str, Any] = {}
        if pano_id:
            sdata_url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={pano_id}&pc=1"
            sdata_response = requests.get(sdata_url, timeout=45)
            if sdata_response.ok:
                sdata_payload = sdata_response.json()
                s_content = sdata_payload.get("content")[0] or {}
        return {
            "available": bool(pano_id),
            "provider": "baidu_streetview",
            "status": "OK" if pano_id else "NO_PANO_ID",
            "pano_id": pano_id,
            "lat": lat,
            "lon": lon,
            "coordinate_system": "WGS84",
            "bd09mc": {"x": mc_x, "y": mc_y},
            "street_name": content.get("RoadName"),
            "raw": {"qsdata": qs_payload, "sdata": s_content},
        }

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
            if provider == "google":
                url = self._download_google_unofficial_view(
                    metadata=metadata,
                    heading=heading,
                    pitch=pitch,
                    fov=fov,
                    path=path,
                )
            else:
                url = self._streetview_image_url(provider, metadata, heading, pitch, fov)
                headers = browser_headers("https://map.baidu.com/") if provider == "baidu" else None
                download_image(url, path, headers=headers)
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
            if pano_id:
                return self._google_unofficial_image_urls(pano_id, heading, pitch, fov)[0]
            raise RuntimeError("Google unofficial street-view download requires a pano_id.")

        if not pano_id:
            raise RuntimeError("Baidu street-view pano_id is missing.")
        return (
            "https://mapsv0.bdimg.com/"
            f"?qt=pr3d&fovy={fov}&quality=90&panoid={pano_id}"
            f"&heading={heading}&pitch={pitch}&width=1024&height=1024"
        )

    def _download_google_unofficial_view(
        self,
        metadata: dict[str, Any],
        heading: int,
        pitch: int,
        fov: int,
        path: Path,
    ) -> str:
        pano_id = str(metadata.get("pano_id") or "")
        if not pano_id:
            raise RuntimeError("Google unofficial street-view download requires a pano_id.")

        errors: list[str] = []
        headers = browser_headers("https://www.google.com/maps/")
        for url in self._google_unofficial_image_urls(pano_id, heading, pitch, fov):
            try:
                download_image(url, path, headers=headers)
                return url
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{self._safe_url(url)} -> {exc}")
        raise RuntimeError("Google unofficial street-view image download failed: " + " ; ".join(errors))

    def _google_unofficial_image_urls(self, pano_id: str, heading: int, pitch: int, fov: int) -> list[str]:
        common = f"panoid={pano_id}&w=1024&h=768&yaw={heading}&pitch={pitch}"
        return [
            (
                "https://streetviewpixels-pa.googleapis.com/v1/thumbnail?"
                f"{common}&cb_client=maps_sv.tactile.gps&thumbfov={fov}"
            ),
            (
                "https://streetviewpixels-pa.googleapis.com/v1/thumbnail?"
                f"{common}&cb_client=maps_sv.tactile&thumbfov={fov}"
            ),
            (
                "https://geo2.ggpht.com/cbk?"
                f"panoid={pano_id}&output=thumbnail&cb_client=maps_sv.tactile.gps&thumb=2"
                f"&w=1024&h=768&yaw={heading}&pitch={pitch}&thumbfov={fov}"
            ),
        ]

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
            if key.lower() in {"key", "signature"}:
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
