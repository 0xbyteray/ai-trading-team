"""Risk control rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.execution.models import Account, Position
from ai_trading_team.risk.actions import RiskAction


class RuleType(str, Enum):
    """Types of risk rules."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_POSITION_SIZE = "max_position_size"
    MAX_LEVERAGE = "max_leverage"
    DAILY_LOSS_LIMIT = "daily_loss_limit"


@dataclass
class RiskRule(ABC):
    """Abstract risk rule base class."""

    name: str
    enabled: bool = True
    priority: int = 0  # Higher = checked first

    @abstractmethod
    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Evaluate the risk rule.

        Args:
            snapshot: Current market data snapshot
            position: Current position (if any)
            account: Current account state

        Returns:
            Risk action if triggered, None otherwise
        """
        ...

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get rule configuration."""
        ...


@dataclass
class StopLossRule(RiskRule):
    """Fixed stop loss rule."""

    stop_loss_percent: Decimal = Decimal("2.0")

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if position has exceeded stop loss threshold."""
        if not position:
            return None

        # Calculate P&L percentage
        pnl_percent = (position.unrealized_pnl / position.margin) * 100

        if pnl_percent <= -self.stop_loss_percent:
            return RiskAction(
                action_type="close",
                symbol=position.symbol,
                reason=f"Stop loss triggered: PnL {pnl_percent:.2f}% <= -{self.stop_loss_percent}%",
                priority=self.priority,
            )

        return None

    def get_config(self) -> dict[str, Any]:
        return {"stop_loss_percent": float(self.stop_loss_percent)}


@dataclass
class TakeProfitRule(RiskRule):
    """Fixed take profit rule."""

    take_profit_percent: Decimal = Decimal("4.0")

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if position has reached take profit threshold."""
        if not position:
            return None

        # Calculate P&L percentage
        pnl_percent = (position.unrealized_pnl / position.margin) * 100

        if pnl_percent >= self.take_profit_percent:
            return RiskAction(
                action_type="close",
                symbol=position.symbol,
                reason=f"Take profit triggered: PnL {pnl_percent:.2f}% >= {self.take_profit_percent}%",
                priority=self.priority,
            )

        return None

    def get_config(self) -> dict[str, Any]:
        return {"take_profit_percent": float(self.take_profit_percent)}


@dataclass
class MaxDrawdownRule(RiskRule):
    """Maximum account drawdown rule."""

    max_drawdown_percent: Decimal = Decimal("10.0")
    peak_equity: Decimal = Decimal("0")

    def evaluate(
        self,
        snapshot: DataSnapshot,
        position: Position | None,
        account: Account,
    ) -> RiskAction | None:
        """Check if account drawdown exceeds threshold."""
        current_equity = account.total_equity

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            return None

        if self.peak_equity <= 0:
            return None

        drawdown_percent = ((self.peak_equity - current_equity) / self.peak_equity) * 100

        if drawdown_percent >= self.max_drawdown_percent:
            return RiskAction(
                action_type="close_all",
                symbol="*",
                reason=f"Max drawdown triggered: {drawdown_percent:.2f}% >= {self.max_drawdown_percent}%",
                priority=100,  # Highest priority
            )

        return None

    def get_config(self) -> dict[str, Any]:
        return {"max_drawdown_percent": float(self.max_drawdown_percent)}
