"""Agent command definitions."""

from dataclasses import dataclass
from enum import Enum

from ai_trading_team.core.types import OrderType, Side


class AgentAction(str, Enum):
    """Actions that agent can take."""

    OPEN = "open"  # Open a new position
    CLOSE = "close"  # Close existing position
    ADD = "add"  # Add to existing position
    REDUCE = "reduce"  # Reduce existing position
    CANCEL = "cancel"  # Cancel pending order
    OBSERVE = "observe"  # Do nothing, just observe


@dataclass
class AgentCommand:
    """Structured command from agent decision."""

    action: AgentAction
    symbol: str
    reason: str  # Agent's explanation (required for auditing)
    side: Side | None = None
    size: float | None = None
    price: float | None = None
    order_type: OrderType | None = None

    def is_actionable(self) -> bool:
        """Check if command requires execution."""
        return self.action != AgentAction.OBSERVE

    def validate(self) -> list[str]:
        """Validate command fields.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.action in (AgentAction.OPEN, AgentAction.ADD):
            if self.side is None:
                errors.append("side is required for open/add actions")
            if self.size is None or self.size <= 0:
                errors.append("positive size is required for open/add actions")

        if self.action == AgentAction.CLOSE and self.side is None:
            errors.append("side is required for close action")

        if self.order_type == OrderType.LIMIT and self.price is None:
            errors.append("price is required for limit orders")

        if not self.reason:
            errors.append("reason is required for auditing")

        return errors
