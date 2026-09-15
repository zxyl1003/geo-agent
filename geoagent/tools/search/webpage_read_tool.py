"""Controlled static webpage reader for evidence pages returned by search."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from geoagent.core.registry import tool_registry
from geoagent.core.retrieval_cache import RetrievalCache
from geoagent.tools.base import BaseTool


@tool_registry.register("webpage_read")
class WebpageReadTool(BaseTool):
    name = "webpage_read"
    description = "Read and extract bounded text from a public static HTML page; no paid API or JavaScript execution."

    def run(self, **kwargs: Any):
        url = str(kwargs.get("url") or "").strip()
        extra = self.config.extra if self.config else {}
        max_chars = max(500, min(int(kwargs.get("max_chars") or extra.get("max_chars") or 12000), 50000))
        max_bytes = max(65536, min(int(extra.get("max_bytes") or 2_000_000), 10_000_000))
        timeout = max(3, min(int(extra.get("timeout_seconds") or 15), 60))
        max_redirects = max(0, min(int(extra.get("max_redirects") or 4), 8))
        if not url:
            return self.result(success=False, error="Missing webpage URL.")

        public_args = {"url": url, "max_chars": max_chars}
        cache = RetrievalCache.from_config(self.app_config)
        cached = cache.get(self.name, "static_html", public_args)
        if cached is not None:
            cached["cache_hit"] = True
            return self.result(data=cached)
        try:
            data = self._fetch(url, max_chars=max_chars, max_bytes=max_bytes, timeout=timeout, max_redirects=max_redirects)
            cache.put(self.name, "static_html", public_args, data)
            data["cache_hit"] = False
            return self.result(data=data)
        except Exception as exc:  # noqa: BLE001
            return self.result(success=False, data={"url": url}, error=f"Webpage read failed: {exc}")

    def _fetch(self, url: str, *, max_chars: int, max_bytes: int, timeout: int, max_redirects: int) -> dict[str, Any]:
        current_url = url
        response: requests.Response | None = None
        try:
            for redirect_count in range(max_redirects + 1):
                self._validate_public_url(current_url)
                response = requests.get(
                    current_url,
                    headers={"User-Agent": "GeoAgentResearch/1.0 (+static evidence reader)"},
                    timeout=timeout,
                    allow_redirects=False,
                    stream=True,
                )
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    response.close()
                    response = None
                    if not location:
                        raise RuntimeError("Redirect response is missing Location header.")
                    if redirect_count >= max_redirects:
                        raise RuntimeError("Too many redirects.")
                    current_url = urljoin(current_url, location)
                    continue
                break
            if response is None or not response.ok:
                status = response.status_code if response is not None else "unknown"
                raise RuntimeError(f"HTTP {status}.")

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                raise RuntimeError(f"Unsupported content type: {content_type or 'missing'}.")
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise RuntimeError(f"Page exceeds {max_bytes} byte limit.")
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                raw.extend(chunk)
                if len(raw) > max_bytes:
                    raise RuntimeError(f"Page exceeds {max_bytes} byte limit.")
            encoding = response.encoding or "utf-8"
            html = bytes(raw).decode(encoding, errors="replace")
        finally:
            if response is not None:
                response.close()

        if content_type == "text/plain":
            text = self._normalize_text(html)
            title = ""
            description = ""
            canonical_url = current_url
            language = ""
        else:
            soup = BeautifulSoup(html, "html.parser")
            for element in soup(["script", "style", "noscript", "svg", "canvas", "template"]):
                element.decompose()
            title = self._normalize_text(soup.title.get_text(" ") if soup.title else "")
            description_tag = soup.find("meta", attrs={"name": lambda value: value and value.lower() == "description"})
            description = self._normalize_text(str(description_tag.get("content") or "") if description_tag else "")
            canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
            canonical_url = urljoin(current_url, str(canonical_tag.get("href"))) if canonical_tag and canonical_tag.get("href") else current_url
            language = str(soup.html.get("lang") or "") if soup.html else ""
            main = soup.find("article") or soup.find("main") or soup.body or soup
            text = self._normalize_text(main.get_text("\n"))

        truncated = len(text) > max_chars
        content = text[:max_chars]
        return {
            "url": url,
            "final_url": current_url,
            "canonical_url": canonical_url,
            "title": title,
            "description": description,
            "language": language,
            "content_type": content_type,
            "content": content,
            "char_count": len(content),
            "original_char_count": len(text),
            "truncated": truncated,
            "source_type": "webpage_read",
        }

    def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise RuntimeError("Only http and https URLs are allowed.")
        if not parsed.hostname or parsed.username or parsed.password:
            raise RuntimeError("URL must contain a public hostname and no credentials.")
        try:
            addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror as exc:
            raise RuntimeError(f"Hostname resolution failed: {exc}") from exc
        if not addresses:
            raise RuntimeError("Hostname did not resolve.")
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise RuntimeError("Private, loopback, link-local, or reserved network addresses are blocked.")

    def _normalize_text(self, value: str) -> str:
        lines = [" ".join(line.split()) for line in value.splitlines()]
        return "\n".join(line for line in lines if line)
