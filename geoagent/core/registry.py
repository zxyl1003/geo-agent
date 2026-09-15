"""Component registry with decorator registration and dynamic imports."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from geoagent.core.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple named registry for framework components."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._items: dict[str, type[T]] = {}

    def register(self, name: str, item: type[T] | None = None) -> Callable[[type[T]], type[T]] | type[T]:
        """Register a class by name.

        Can be used as `registry.register("x", X)` or as a decorator:
        `@registry.register("x")`.
        """

        def decorator(cls: type[T]) -> type[T]:
            self._items[name] = cls
            return cls

        if item is not None:
            return decorator(item)
        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise RegistryError(f"{self.namespace!r} registry has no item named {name!r}")
        return self._items[name]

    def maybe_get(self, name: str) -> type[T] | None:
        return self._items.get(name)

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        cls = self.get(name)
        return cls(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._items.keys())

    def import_from_path(self, class_path: str) -> type[T]:
        try:
            module_name, class_name = class_path.rsplit(".", 1)
        except ValueError as exc:
            raise RegistryError(f"Invalid class path: {class_path}") from exc

        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise RegistryError(f"Could not import {class_path}: {exc}") from exc
        return cls

    def register_from_class_path(self, name: str, class_path: str) -> type[T]:
        cls = self.import_from_path(class_path)
        self.register(name, cls)
        return cls


agent_registry: Registry[Any] = Registry("agent")
tool_registry: Registry[Any] = Registry("tool")
workflow_registry: Registry[Any] = Registry("workflow")
