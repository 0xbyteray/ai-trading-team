"""WEEX WebSocket stream for account updates."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class WEEXStream(ABC):
    """Abstract WEEX WebSocket stream.

    Handles real-time account, position, and order updates.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        ...

    @abstractmethod
    async def subscribe_account(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to account updates."""
        ...

    @abstractmethod
    async def subscribe_positions(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to position updates."""
        ...

    @abstractmethod
    async def subscribe_orders(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to order updates."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        ...
