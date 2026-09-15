"""Base interface for all tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar

from geoagent.core.config import AppConfig, ToolConfig
from geoagent.core.logging import log_tool_result
from geoagent.core.schemas import ToolResult

if TYPE_CHECKING:
    from geoagent.state.task_state import GeoLocalizationState


class BaseTool(ABC):
    name: ClassVar[str] = "base_tool"
    description: ClassVar[str] = ""
    cost_level: ClassVar[str] = "low"
    max_calls_per_task: ClassVar[int] = 10
    internal: ClassVar[bool] = False
    # Hidden tools are enabled and executable but never exposed to the Brain.
    # Used for internal provider implementations behind a routing facade
    # (e.g. baidu/google/locationiq geocoding behind `geocode`).
    hidden: ClassVar[bool] = False

    def __init__(self, config: ToolConfig | None = None, app_config: AppConfig | None = None) -> None:
        self.config = config
        self.app_config = app_config or AppConfig()
        if config is not None:
            self.name = config.name
            self.cost_level = config.cost_level
            self.max_calls_per_task = config.max_calls_per_task

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute a tool and return a normalized result."""

    def is_available(self, state: "GeoLocalizationState | None" = None) -> bool:
        """Return whether the tool should be exposed to the Brain right now."""

        return True

    def safe_run(self, **kwargs: Any) -> ToolResult:
        start = perf_counter()
        try:
            result = self.run(**kwargs)
            if result.latency_ms is None:
                result.latency_ms = (perf_counter() - start) * 1000
            log_tool_result(self.name, result.success, result.model_dump(), result.error)
            return result
        except Exception as exc:  # noqa: BLE001 - tools must never crash workflows
            result = ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
                latency_ms=(perf_counter() - start) * 1000,
            )
            log_tool_result(self.name, False, result.model_dump(), result.error)
            return result

    def result(self, success: bool = True, data: dict[str, Any] | None = None, error: str | None = None) -> ToolResult:
        return ToolResult(tool_name=self.name, success=success, data=data or {}, error=error)
