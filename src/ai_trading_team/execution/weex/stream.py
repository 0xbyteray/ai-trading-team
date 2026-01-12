"""WEEX WebSocket stream for account updates."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from weex_sdk import AsyncWeexWebSocket

logger = logging.getLogger(__name__)


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


class WEEXPrivateStream(WEEXStream):
    """Private WEEX WebSocket stream for account, position, and order updates."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
    ) -> None:
        self._ws = AsyncWeexWebSocket(
            api_key=api_key,
            secret_key=api_secret,
            passphrase=passphrase,
            is_private=True,
        )

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        await self._ws.connect()
        logger.info("WEEX private WebSocket connected")

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        await self._ws.close()
        logger.info("WEEX private WebSocket disconnected")

    async def subscribe_account(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to account updates."""
        await self._ws.subscribe_account(callback)

    async def subscribe_positions(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to position updates."""
        await self._ws.subscribe_position(callback)

    async def subscribe_orders(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to order updates."""
        await self._ws.subscribe_order(callback)

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return bool(self._ws.connected)
