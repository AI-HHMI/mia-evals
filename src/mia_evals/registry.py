from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic name -> subclass-of-T registry, selected via config."""

    def __init__(self, base_cls: type[Any]) -> None:
        # `base_cls` is typed as `type[Any]` rather than `type[T]` on purpose: it is
        # never instantiated here (only used for `issubclass` checks in `register`),
        # so it may legitimately be an abstract class. Typing it as `type[T]` makes
        # mypy's type-abstract check fire whenever T is bound to an ABC, which is the
        # expected usage for this class (e.g. `Registry(BaseAlgorithm)`).
        self._base_cls: type[T] = base_cls
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def _decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(f"'{name}' is already registered")
            if not issubclass(cls, self._base_cls):
                raise TypeError(f"{cls.__name__} must subclass {self._base_cls.__name__}")
            self._registry[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> type[T]:
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(f"unknown '{name}'; available: {self.available()}") from None

    def build(self, name: str, *args: Any, **kwargs: Any) -> T:
        return self.get(name)(*args, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._registry)
