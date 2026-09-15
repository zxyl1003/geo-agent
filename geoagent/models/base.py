"""Base model client interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from geoagent.core.config import AppConfig, ModelConfig


class BaseModelClient(ABC):
    def __init__(self, config: ModelConfig | None = None, app_config: AppConfig | None = None) -> None:
        self.config = config
        self.app_config = app_config or AppConfig()

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a model response."""
