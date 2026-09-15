"""Tool registry utilities."""

from __future__ import annotations

from geoagent.core.config import AppConfig
from geoagent.core.registry import tool_registry
from geoagent.tools.base import BaseTool


def build_tools(app_config: AppConfig) -> dict[str, BaseTool]:
    """Instantiate enabled tools from config class paths."""

    tools: dict[str, BaseTool] = {}
    for name, tool_config in app_config.tools.items():
        if not tool_config.enable:
            continue
        cls = tool_registry.register_from_class_path(name, tool_config.class_path)
        tools[name] = cls(config=tool_config, app_config=app_config)
    return tools


__all__ = ["build_tools", "tool_registry"]
