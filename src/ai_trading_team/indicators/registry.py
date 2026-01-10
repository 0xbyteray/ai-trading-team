"""Indicator registry for managing active indicators."""

from typing import Any

from ai_trading_team.indicators.base import Indicator


class IndicatorRegistry:
    """Registry for managing technical indicators.

    Provides a central place to register, update, and query indicators.
    """

    def __init__(self) -> None:
        self._indicators: dict[str, Indicator] = {}

    def register(self, indicator: Indicator) -> None:
        """Register an indicator.

        Args:
            indicator: Indicator instance to register
        """
        self._indicators[indicator.name] = indicator

    def unregister(self, name: str) -> None:
        """Unregister an indicator by name."""
        if name in self._indicators:
            del self._indicators[name]

    def get(self, name: str) -> Indicator | None:
        """Get an indicator by name."""
        return self._indicators.get(name)

    def update_all(self, data: dict[str, Any]) -> dict[str, Any]:
        """Update all registered indicators.

        Args:
            data: OHLCV data dict

        Returns:
            Dict of indicator name -> value
        """
        results = {}
        for name, indicator in self._indicators.items():
            results[name] = indicator.update(data)
        return results

    def get_all_values(self) -> dict[str, Any]:
        """Get current values of all indicators."""
        return {name: ind.value for name, ind in self._indicators.items()}

    def reset_all(self) -> None:
        """Reset all indicators."""
        for indicator in self._indicators.values():
            indicator.reset()

    @property
    def names(self) -> list[str]:
        """List of registered indicator names."""
        return list(self._indicators.keys())
