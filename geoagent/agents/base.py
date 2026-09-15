"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from geoagent.core.config import AppConfig
from geoagent.core.schemas import AgentOutput
from geoagent.state.task_state import GeoLocalizationState
from geoagent.tools.base import BaseTool


class BaseAgent(ABC):
    name: ClassVar[str] = "base_agent"
    description: ClassVar[str] = ""

    def __init__(self, app_config: AppConfig | None = None, tools: dict[str, BaseTool] | None = None) -> None:
        self.app_config = app_config or AppConfig()
        self.tools = tools or {}

    @abstractmethod
    def run(self, state: GeoLocalizationState) -> AgentOutput:
        """Run agent logic against the shared task state."""

    def record_model_usage(self, state: GeoLocalizationState, role: str, response: dict[str, Any]) -> None:
        state.add_model_usage(
            role=role,
            model=str(response.get("model") or ""),
            provider=str(response.get("provider") or ""),
            usage=response.get("usage") if isinstance(response.get("usage"), dict) else {},
        )
