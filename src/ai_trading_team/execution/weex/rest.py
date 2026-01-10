"""WEEX REST API client."""

from abc import ABC, abstractmethod
from decimal import Decimal

from ai_trading_team.core.types import OrderType, Side, TimeInForce


class WEEXRestClient(ABC):
    """Abstract WEEX REST API client.

    Handles order placement, account queries, and position management.
    """

    @abstractmethod
    async def get_account_info(self) -> dict:
        """Get account information."""
        ...

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[dict]:
        """Get positions."""
        ...

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        size: Decimal,
        price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: str | None = None,
    ) -> dict:
        """Place an order."""
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> list[dict]:
        """Get open orders."""
        ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage."""
        ...
