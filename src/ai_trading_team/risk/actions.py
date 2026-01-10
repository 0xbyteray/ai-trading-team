"""Risk control actions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class RiskAction:
    """Risk control action to be executed."""

    action_type: Literal["close", "close_all", "reduce", "cancel_orders"]
    symbol: str
    reason: str
    priority: int = 0
    reduce_percent: float | None = None  # For "reduce" action
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "symbol": self.symbol,
            "reason": self.reason,
            "priority": self.priority,
            "reduce_percent": self.reduce_percent,
            "timestamp": self.timestamp.isoformat(),
        }
