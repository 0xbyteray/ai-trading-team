"""Base indicator class."""

from abc import ABC, abstractmethod
from typing import Any


class Indicator(ABC):
    """Abstract base class for technical indicators.

    Wraps talipp indicators with a unified interface.
    """

    def __init__(self, name: str, **params: Any) -> None:
        self._name = name
        self._params = params
        self._value: Any = None

    @property
    def name(self) -> str:
        """Indicator name."""
        return self._name

    @property
    def params(self) -> dict[str, Any]:
        """Indicator parameters."""
        return self._params

    @property
    def value(self) -> Any:
        """Current indicator value."""
        return self._value

    @abstractmethod
    def update(self, data: dict[str, Any]) -> Any:
        """Update indicator with new data.

        Args:
            data: OHLCV data dict with keys: open, high, low, close, volume

        Returns:
            Updated indicator value
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset indicator state."""
        ...
