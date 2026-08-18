from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any


class PluginRegistry:
    """Built-ins plus lazy Python entry-point discovery."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Callable[..., Any]]] = {
            "providers": {},
            "policies": {},
        }

    def register(self, group: str, name: str, factory: Callable[..., Any]) -> None:
        if group not in self._items:
            raise KeyError(f"unknown plugin group: {group}")
        if name in self._items[group]:
            raise ValueError(f"duplicate {group} plugin: {name}")
        self._items[group][name] = factory

    def discover(self) -> None:
        for group, ep_group in {
            "providers": "fusion_runtime.providers",
            "policies": "fusion_runtime.policies",
        }.items():
            for item in entry_points(group=ep_group):
                self._items[group].setdefault(item.name, item.load())

    def create(self, group: str, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            factory = self._items[group][name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items.get(group, {}))) or "none"
            raise KeyError(f"unknown {group} plugin {name!r}; available: {available}") from exc
        return factory(*args, **kwargs)
