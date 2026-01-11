"""Signal Aggregator - Manages signal sources and collects signals.

This replaces the old orchestrator's periodic evaluation with event-driven signals.
The aggregator:
1. Manages multiple signal sources
2. Updates sources with new data
3. Collects emitted signals
4. Optionally detects confluence (multiple aligned signals)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.strategy.signals.base import SignalSource
from ai_trading_team.strategy.signals.bollinger_breakout import BollingerBreakoutSignal
from ai_trading_team.strategy.signals.funding_rate import FundingRateSignal
from ai_trading_team.strategy.signals.liquidation import LiquidationSignal
from ai_trading_team.strategy.signals.ls_ratio import LongShortRatioSignal
from ai_trading_team.strategy.signals.ma_crossover import MACrossoverSignal
from ai_trading_team.strategy.signals.macd_crossover import MACDCrossoverSignal
from ai_trading_team.strategy.signals.open_interest import OpenInterestSignal
from ai_trading_team.strategy.signals.order_status import OrderStatusSignal
from ai_trading_team.strategy.signals.pnl_change import PnLChangeSignal
from ai_trading_team.strategy.signals.risk_signal import RiskSignal
from ai_trading_team.strategy.signals.rsi_extreme import RSIExtremeSignal
from ai_trading_team.strategy.signals.types import (
    ALL_TIMEFRAMES,
    Signal,
    SignalDirection,
    SignalStrength,
    SignalType,
    Timeframe,
)

logger = logging.getLogger(__name__)


@dataclass
class SignalWindow:
    """Configuration for signal confluence detection."""

    # Time window to look for aligned signals
    window_seconds: int = 300  # 5 minutes

    # Minimum signals needed for confluence
    min_signals_for_confluence: int = 2

    # Weights for different signal types (for confluence scoring)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "ma_crossover": 1.5,  # MA signals weighted higher
            "macd_crossover": 1.5,
            "rsi_extreme": 1.2,
            "bollinger_breakout": 1.2,
            "funding_rate": 1.0,
            "long_short_ratio": 1.0,
            "open_interest": 0.8,
            "liquidation": 1.3,
        }
    )


class SignalAggregator:
    """Aggregates signals from multiple sources.

    Key differences from old orchestrator:
    - No periodic scoring - only collects actual signals
    - Signals are event-driven (state changes only)
    - Optional confluence detection for aligned signals
    - Multi-timeframe support (5m, 15m, 1h, 4h)
    """

    def __init__(
        self,
        data_pool: DataPool,
        symbol: str,
        confluence_config: SignalWindow | None = None,
    ) -> None:
        """Initialize signal aggregator.

        Args:
            data_pool: Shared data pool
            symbol: Trading symbol
            confluence_config: Optional confluence detection config
        """
        self._data_pool = data_pool
        self._symbol = symbol
        self._confluence_config = confluence_config or SignalWindow()

        # Signal sources
        self._sources: list[SignalSource] = []

        # Recent signals (for confluence detection)
        self._recent_signals: deque[Signal] = deque(maxlen=100)

        # Callbacks for new signals
        self._signal_callbacks: list[Any] = []

        # Initialize default signal sources
        self._init_default_sources()

    def _init_default_sources(self) -> None:
        """Initialize default signal sources per USER_INSTRUCTIONS.md."""
        # ===========================================
        # Technical Indicator Signals (5m, 15m, 1h, 4h)
        # ===========================================

        # SMA(60) crossover on all timeframes
        self.add_source(
            MACrossoverSignal(
                ma_period=60,
                timeframes=ALL_TIMEFRAMES,
            )
        )

        # RSI(14) extremes on all timeframes
        self.add_source(
            RSIExtremeSignal(
                oversold_threshold=30.0,
                overbought_threshold=70.0,
                rsi_period=14,
                timeframes=ALL_TIMEFRAMES,
            )
        )

        # MACD(12, 26, 9) crossover on all timeframes
        self.add_source(
            MACDCrossoverSignal(
                fast_period=12,
                slow_period=26,
                signal_period=9,
                timeframes=ALL_TIMEFRAMES,
            )
        )

        # Bollinger Bands(20, 2) breakout on all timeframes
        self.add_source(
            BollingerBreakoutSignal(
                period=20,
                std_dev=2.0,
                timeframes=ALL_TIMEFRAMES,
            )
        )

        # ===========================================
        # Market Data Signals
        # ===========================================

        # Funding rate shifts
        self.add_source(FundingRateSignal())

        # Long/Short ratio change >5% in 5 minutes
        self.add_source(
            LongShortRatioSignal(
                change_threshold_percent=5.0,
                window_minutes=5,
            )
        )

        # Open Interest change >5% in 5 minutes
        self.add_source(
            OpenInterestSignal(
                change_threshold_percent=5.0,
                window_minutes=5,
            )
        )

        # Liquidation >1M USD
        self.add_source(
            LiquidationSignal(
                min_value_usd=1_000_000.0,
            )
        )

        # ===========================================
        # Account/Risk Signals
        # ===========================================

        # P&L change ±5%
        self.add_source(
            PnLChangeSignal(
                threshold_percent=5.0,
            )
        )

        # Risk module triggers
        self.add_source(RiskSignal())

        # Order status changes
        self.add_source(OrderStatusSignal())

        logger.info(f"Initialized {len(self._sources)} signal sources")
        for source in self._sources:
            logger.debug(
                f"  - {source.name}: timeframes={[tf.value for tf in source.timeframes]}"
            )

    def add_source(self, source: SignalSource) -> None:
        """Add a signal source.

        Args:
            source: SignalSource to add
        """
        self._sources.append(source)
        logger.debug(f"Added signal source: {source.name}")

    def remove_source(self, name: str) -> bool:
        """Remove a signal source by name.

        Args:
            name: Source name to remove

        Returns:
            True if removed
        """
        for i, source in enumerate(self._sources):
            if source.name == name:
                self._sources.pop(i)
                return True
        return False

    def on_signal(self, callback: Any) -> None:
        """Register a callback for new signals.

        Args:
            callback: Function to call with new Signal
        """
        self._signal_callbacks.append(callback)

    def update(self, timeframe: Timeframe | None = None) -> list[Signal]:
        """Update all sources and collect new signals.

        This should be called when new data arrives for a timeframe.

        Args:
            timeframe: Specific timeframe to update, or None for all

        Returns:
            List of new signals emitted
        """
        snapshot = self._data_pool.get_snapshot()
        signals: list[Signal] = []

        timeframes = [timeframe] if timeframe else list(Timeframe)

        for tf in timeframes:
            for source in self._sources:
                if not source.enabled:
                    continue

                signal = source.update(snapshot, tf)
                if signal:
                    signals.append(signal)
                    self._recent_signals.append(signal)

                    # Notify callbacks
                    for callback in self._signal_callbacks:
                        try:
                            callback(signal)
                        except Exception as e:
                            logger.error(f"Signal callback error: {e}")

        # Check for confluence
        if len(signals) > 0:
            confluence = self._check_confluence()
            if confluence:
                signals.append(confluence)
                self._recent_signals.append(confluence)

        return signals

    def _check_confluence(self) -> Signal | None:
        """Check for signal confluence (multiple aligned signals).

        Returns:
            Confluence signal if detected
        """
        config = self._confluence_config
        cutoff = datetime.now() - timedelta(seconds=config.window_seconds)

        # Filter recent signals within window
        recent = [s for s in self._recent_signals if s.timestamp > cutoff]

        if len(recent) < config.min_signals_for_confluence:
            return None

        # Count bullish vs bearish signals
        bullish_score = 0.0
        bearish_score = 0.0

        for signal in recent:
            weight = config.weights.get(signal.source, 1.0)
            if signal.strength == SignalStrength.STRONG:
                weight *= 1.5
            elif signal.strength == SignalStrength.WEAK:
                weight *= 0.5

            if signal.direction == SignalDirection.BULLISH:
                bullish_score += weight
            elif signal.direction == SignalDirection.BEARISH:
                bearish_score += weight

        # Check for confluence
        min_score = config.min_signals_for_confluence

        if bullish_score >= min_score and bullish_score > bearish_score * 1.5:
            return Signal(
                signal_type=SignalType.BULLISH_CONFLUENCE,
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.STRONG if bullish_score >= min_score * 2 else SignalStrength.MODERATE,
                timeframe=Timeframe.H1,  # Confluence is multi-timeframe
                source="aggregator",
                data={
                    "bullish_score": bullish_score,
                    "bearish_score": bearish_score,
                    "signal_count": len(recent),
                    "signals": [s.signal_type.value for s in recent if s.is_bullish],
                },
                description=(
                    f"BULLISH CONFLUENCE: {bullish_score:.1f} bullish vs "
                    f"{bearish_score:.1f} bearish from {len(recent)} signals"
                ),
            )

        elif bearish_score >= min_score and bearish_score > bullish_score * 1.5:
            return Signal(
                signal_type=SignalType.BEARISH_CONFLUENCE,
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.STRONG if bearish_score >= min_score * 2 else SignalStrength.MODERATE,
                timeframe=Timeframe.H1,
                source="aggregator",
                data={
                    "bullish_score": bullish_score,
                    "bearish_score": bearish_score,
                    "signal_count": len(recent),
                    "signals": [s.signal_type.value for s in recent if s.is_bearish],
                },
                description=(
                    f"BEARISH CONFLUENCE: {bearish_score:.1f} bearish vs "
                    f"{bullish_score:.1f} bullish from {len(recent)} signals"
                ),
            )

        return None

    def get_recent_signals(
        self,
        seconds: int = 300,
        direction: SignalDirection | None = None,
    ) -> list[Signal]:
        """Get recent signals within a time window.

        Args:
            seconds: Time window in seconds
            direction: Optional filter by direction

        Returns:
            List of matching signals
        """
        cutoff = datetime.now() - timedelta(seconds=seconds)
        signals = [s for s in self._recent_signals if s.timestamp > cutoff]

        if direction:
            signals = [s for s in signals if s.direction == direction]

        return signals

    def get_source_states(self) -> dict[str, Any]:
        """Get current state of all sources for debugging.

        Returns:
            Dictionary of source states
        """
        states = {}
        for source in self._sources:
            states[source.name] = {
                "enabled": source.enabled,
                "timeframes": [tf.value for tf in source.timeframes],
                "states": {
                    tf.value: str(source.get_state(tf)) for tf in source.timeframes
                },
            }
        return states

    def reset(self) -> None:
        """Reset all source states."""
        for source in self._sources:
            source.reset()
        self._recent_signals.clear()
        logger.info("Signal aggregator reset")
