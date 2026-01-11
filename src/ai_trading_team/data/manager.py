"""Data manager - coordinates REST and WebSocket data sources."""

import logging
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.data.binance.rest import BinanceRestClient
from ai_trading_team.data.binance.stream import BinanceStreamClient
from ai_trading_team.data.models import Kline, Ticker

logger = logging.getLogger(__name__)


class BinanceDataManager:
    """Binance data manager.

    Coordinates REST API and WebSocket streams,
    writes updates to the DataPool.
    """

    # Required intervals for signal system
    REQUIRED_INTERVALS = ["5m", "15m", "1h", "4h"]
    MIN_KLINES_PER_INTERVAL = 100  # Need enough for MA60 + buffer

    def __init__(
        self,
        data_pool: DataPool,
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        """Initialize Binance data manager.

        Args:
            data_pool: Shared data pool for storing real-time data
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
        """
        self._data_pool = data_pool
        self._rest_client = BinanceRestClient(api_key, api_secret)
        self._stream_client = BinanceStreamClient()
        self._running = False
        self._symbol = ""
        self._kline_interval = "1m"
        self._data_ready = False

    async def start(self, symbol: str, kline_interval: str = "1m") -> None:
        """Start data collection for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            kline_interval: Kline interval for streaming (e.g., "1m", "5m")
        """
        self._symbol = symbol
        self._kline_interval = kline_interval
        self._running = True

        # Initialize historical data first
        await self.initialize(symbol)

        # Set up callbacks
        self._stream_client.on_ticker(self._on_ticker)
        self._stream_client.on_kline(self._on_kline)

        # Connect to WebSocket streams
        await self._stream_client.connect(symbol, kline_interval)
        logger.info(f"Started data collection for {symbol}")

    async def stop(self) -> None:
        """Stop data collection."""
        self._running = False
        await self._stream_client.close()
        logger.info("Stopped data collection")

    async def initialize(self, symbol: str) -> None:
        """Initialize historical data via REST API.

        Fetches klines for all required timeframes (5m, 15m, 1h, 4h)
        to ensure signal system has sufficient data.

        Args:
            symbol: Trading pair
        """
        logger.info(f"Initializing historical data for {symbol}")
        self._data_ready = False

        try:
            # Fetch initial ticker
            ticker = await self._rest_client.get_ticker(symbol)
            self._data_pool.update_ticker(self._ticker_to_dict(ticker))

            # Fetch historical klines for all required intervals
            for interval in self.REQUIRED_INTERVALS:
                klines = await self._rest_client.get_klines(
                    symbol, interval, limit=self.MIN_KLINES_PER_INTERVAL
                )
                kline_dicts = [self._kline_to_dict(k) for k in klines]
                self._data_pool.update_klines(interval, kline_dicts)
                logger.info(f"Initialized {len(klines)} klines for {symbol} {interval}")

            # Also fetch the streaming interval if different
            if self._kline_interval not in self.REQUIRED_INTERVALS:
                klines = await self._rest_client.get_klines(
                    symbol, self._kline_interval, limit=self.MIN_KLINES_PER_INTERVAL
                )
                kline_dicts = [self._kline_to_dict(k) for k in klines]
                self._data_pool.update_klines(self._kline_interval, kline_dicts)
                logger.info(
                    f"Initialized {len(klines)} klines for {symbol} {self._kline_interval}"
                )

            self._data_ready = True
            logger.info(
                f"Data initialization complete: {len(self.REQUIRED_INTERVALS)} timeframes loaded"
            )

        except Exception as e:
            logger.error(f"Failed to initialize data: {e}")
            raise

    def _on_ticker(self, ticker: Ticker) -> None:
        """Handle ticker update from WebSocket."""
        self._data_pool.update_ticker(self._ticker_to_dict(ticker))

    def _on_kline(self, interval: str, kline: Kline) -> None:
        """Handle kline update from WebSocket."""
        # Get existing klines and update/append
        existing = self._data_pool.get_klines(interval)
        kline_dict = self._kline_to_dict(kline)

        # Check if this is an update to the last kline or a new one
        if existing and existing[-1].get("open_time") == kline_dict.get("open_time"):
            # Update the last kline
            existing[-1] = kline_dict
        else:
            # Append new kline
            existing.append(kline_dict)
            # Keep only the last 500 klines
            if len(existing) > 500:
                existing = existing[-500:]

        self._data_pool.update_klines(interval, existing)

    @staticmethod
    def _ticker_to_dict(ticker: Ticker) -> dict[str, Any]:
        """Convert Ticker to dict with serializable values."""
        result = asdict(ticker)
        for key, value in result.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
        return result

    @staticmethod
    def _kline_to_dict(kline: Kline) -> dict[str, Any]:
        """Convert Kline to dict with serializable values."""
        result = asdict(kline)
        for key, value in result.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
        return result

    @property
    def is_running(self) -> bool:
        """Check if data collection is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._stream_client.is_connected

    @property
    def is_data_ready(self) -> bool:
        """Check if all required historical data is loaded."""
        return self._data_ready
