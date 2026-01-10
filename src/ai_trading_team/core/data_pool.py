"""Real-time data pool with thread-safe storage."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Any

from ai_trading_team.core.types import EventType


@dataclass
class DataSnapshot:
    """Point-in-time snapshot of all data."""

    timestamp: datetime
    ticker: dict[str, Any] | None = None
    klines: dict[str, list[dict[str, Any]]] | None = None  # interval -> klines
    orderbook: dict[str, Any] | None = None
    indicators: dict[str, Any] | None = None
    position: dict[str, Any] | None = None
    orders: list[dict[str, Any]] | None = None
    account: dict[str, Any] | None = None


class DataPool:
    """Thread-safe real-time data storage.

    Central repository for all market and account data.
    Supports subscriptions for reactive updates.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._ticker: dict[str, Any] | None = None
        self._klines: dict[str, list[dict[str, Any]]] = {}  # interval -> klines
        self._orderbook: dict[str, Any] | None = None
        self._indicators: dict[str, Any] = {}
        self._position: dict[str, Any] | None = None
        self._orders: list[dict[str, Any]] = []
        self._account: dict[str, Any] | None = None
        self._subscribers: list[Callable[[EventType, Any], None]] = []

    def subscribe(self, callback: Callable[[EventType, Any], None]) -> None:
        """Subscribe to data updates.

        Args:
            callback: Function called with (event_type, data) on updates
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[EventType, Any], None]) -> None:
        """Unsubscribe from data updates."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify(self, event_type: EventType, data: Any) -> None:
        """Notify all subscribers of an update."""
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            callback(event_type, data)

    def update_ticker(self, ticker: dict[str, Any]) -> None:
        """Update ticker data."""
        with self._lock:
            self._ticker = ticker
        self._notify(EventType.TICKER_UPDATE, ticker)

    def update_klines(self, interval: str, klines: list[dict[str, Any]]) -> None:
        """Update kline data for an interval."""
        with self._lock:
            self._klines[interval] = klines
        self._notify(EventType.KLINE_UPDATE, {"interval": interval, "klines": klines})

    def update_orderbook(self, orderbook: dict[str, Any]) -> None:
        """Update orderbook data."""
        with self._lock:
            self._orderbook = orderbook
        self._notify(EventType.ORDERBOOK_UPDATE, orderbook)

    def update_indicator(self, name: str, value: Any) -> None:
        """Update indicator value."""
        with self._lock:
            self._indicators[name] = value
        self._notify(EventType.INDICATOR_UPDATE, {"name": name, "value": value})

    def update_position(self, position: dict[str, Any] | None) -> None:
        """Update position data."""
        with self._lock:
            self._position = position
        self._notify(EventType.POSITION_UPDATED, position)

    def update_orders(self, orders: list[dict[str, Any]]) -> None:
        """Update open orders."""
        with self._lock:
            self._orders = orders

    def update_account(self, account: dict[str, Any]) -> None:
        """Update account data."""
        with self._lock:
            self._account = account

    def get_snapshot(self) -> DataSnapshot:
        """Get a point-in-time snapshot of all data."""
        with self._lock:
            return DataSnapshot(
                timestamp=datetime.now(),
                ticker=self._ticker.copy() if self._ticker else None,
                klines={k: list(v) for k, v in self._klines.items()},
                orderbook=self._orderbook.copy() if self._orderbook else None,
                indicators=self._indicators.copy(),
                position=self._position.copy() if self._position else None,
                orders=list(self._orders),
                account=self._account.copy() if self._account else None,
            )

    @property
    def ticker(self) -> dict[str, Any] | None:
        """Get current ticker."""
        with self._lock:
            return self._ticker.copy() if self._ticker else None

    @property
    def indicators(self) -> dict[str, Any]:
        """Get current indicators."""
        with self._lock:
            return self._indicators.copy()

    def get_klines(self, interval: str) -> list[dict[str, Any]]:
        """Get klines for an interval."""
        with self._lock:
            return list(self._klines.get(interval, []))
