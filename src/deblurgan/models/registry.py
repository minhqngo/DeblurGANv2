"""A tiny name -> factory registry.

Replaces the string if-chains in ``models/networks.py`` and the ``eval()``-based
dispatch in ``adversarial_trainer.py``. Unknown names raise with the available names
listed, so typos are caught immediately.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._entries:
                raise KeyError(f"{self._kind} {name!r} is already registered")
            self._entries[name] = factory
            return factory

        return deco

    def create(self, name: str, *args, **kwargs) -> T:
        if name not in self._entries:
            raise KeyError(
                f"Unknown {self._kind} {name!r}. Available: {sorted(self._entries)}"
            )
        return self._entries[name](*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)
