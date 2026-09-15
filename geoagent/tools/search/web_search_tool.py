"""Web search through Serper with persistent result caching."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

from geoagent.core.registry import tool_registry
from geoagent.core.retrieval_cache import RetrievalCache
from geoagent.tools.base import BaseTool


@tool_registry.register("web_search")
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the public web with Serper; successful results are cached locally."

    def is_available(self, state: Any = None) -> bool:
        return bool(self.app_config.env.serper_api_key)

    def run(self, **kwargs: Any):
        query = " ".join(str(kwargs.get("query", "")).split())
        top_k = max(1, min(int(kwargs.get("top_k", 5)), 10))
        provider = "serper"
        if not query:
            return self.result(success=False, error="Missing search query.")

        public_args = self._public_args(query, top_k, kwargs)
        cache = RetrievalCache.from_config(self.app_config)
        cached = cache.get(self.name, provider, public_args)
        if cached is not None:
            if isinstance(cached.get("results"), list):
                cached["results"] = cached["results"][:top_k]
            cached["cache_hit"] = True
            return self.result(data=cached)

        credentials = self._credentials()
        if credentials is None:
            return self.result(
                success=False,
                data={"query": query, "top_k": top_k, "provider": provider},
                error="SERPER_API_KEY is not configured.",
            )
        api_key, base_url = credentials

        try:
            normalized = self._search_serper(base_url, api_key, public_args)
            data = {
                "query": query,
                "top_k": top_k,
                "provider": provider,
                "using_configured_api_key": True,
                **normalized,
            }
            cache.put(self.name, provider, public_args, data)
            data["cache_hit"] = False
            return self.result(data=data)
        except Exception as exc:  # noqa: BLE001
            return self.result(
                success=False,
                data={"query": query, "top_k": top_k, "provider": provider, "using_configured_api_key": True},
                error=f"Serper web search failed: {exc}",
            )

    def _credentials(self) -> tuple[str, str] | None:
        env = self.app_config.env
        key = env.serper_api_key
        url = env.serper_search_base_url
        if not key:
            return None
        return key.get_secret_value(), url.rstrip("/")

    def _public_args(self, query: str, top_k: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        for key in ("location", "gl", "hl"):
            value = str(kwargs.get(key) or "").strip()
            if value:
                data[key] = value
        return data

    def _search_serper(self, url: str, api_key: str, args: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"q": args["query"], "num": args["top_k"]}
        for key in ("location", "gl", "hl"):
            if args.get(key):
                payload[key] = args[key]
        response = requests.post(
            url,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
        data = self._json_or_raise(response)
        results = [
            self._result(
                item.get("title"),
                item.get("snippet"),
                item.get("link"),
                position=item.get("position"),
            )
            for item in data.get("organic") or []
            if isinstance(item, dict)
        ]
        return {
            "results": results[: args["top_k"]],
            "provider_metadata": {
                "search_parameters": data.get("searchParameters") or {},
                "credits": data.get("credits"),
            },
            "knowledge_graph": data.get("knowledgeGraph"),
            "answer_box": data.get("answerBox"),
            "people_also_ask": data.get("peopleAlsoAsk") or [],
            "related_searches": data.get("relatedSearches") or [],
        }

    def _json_or_raise(self, response: requests.Response) -> dict[str, Any]:
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Expected JSON object response.")
        return data

    def _result(
        self,
        title: Any,
        snippet: Any,
        url: Any,
        *,
        position: Any = None,
        engine: Any = None,
        score: Any = None,
    ) -> dict[str, Any]:
        normalized_url = str(url or "").strip()
        source_name = self._source_name(normalized_url)
        return {
            "title": str(title or "").strip(),
            "snippet": str(snippet or "").strip(),
            "url": normalized_url,
            "source_type": "web_search",
            "source_name": source_name,
            "source_label": f"Web Search/{source_name}",
            "position": position,
            "engine": engine,
            "score": score,
        }

    def _source_name(self, url: str) -> str:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        known_sources = (
            ("dianping.com", "Dianping"),
            ("trip.com", "Trip.com"),
            ("ctrip.com", "Trip.com"),
            ("baidu.com", "Baidu"),
            ("google.", "Google"),
            ("meituan.com", "Meituan"),
        )
        for marker, label in known_sources:
            if marker in host:
                return label
        return host or "Web"
