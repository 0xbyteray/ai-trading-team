"""Binance REST API client implementation."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

from ai_trading_team.data.models import (
    FundingRate,
    Kline,
    LongShortRatio,
    OpenInterest,
    OrderBook,
    OrderBookLevel,
    Ticker,
)

logger = logging.getLogger(__name__)


class BinanceRestClient:
    """Binance Futures REST API client.

    Fetches market data from Binance USDS-M Futures.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        """Initialize Binance REST client.

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._client: DerivativesTradingUsdsFutures | None = None

    def _get_client(self) -> DerivativesTradingUsdsFutures:
        """Get or create Binance client."""
        if self._client is None:
            config = ConfigurationRestAPI(
                api_key=self._api_key if self._api_key else None,
                api_secret=self._api_secret if self._api_secret else None,
                base_path=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
            )
            self._client = DerivativesTradingUsdsFutures(config_rest_api=config)
        return self._client

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Ticker data
        """
        client = self._get_client()

        def _fetch() -> dict[str, Any]:
            response = client.rest_api.ticker24hr_price_change_statistics(symbol=symbol)
            return response.data().to_dict()  # type: ignore[no-any-return]

        data = await asyncio.to_thread(_fetch)

        return Ticker(
            symbol=data.get("symbol", symbol),
            last_price=Decimal(str(data.get("lastPrice", 0))),
            bid_price=Decimal(str(data.get("bidPrice", 0))),
            ask_price=Decimal(str(data.get("askPrice", 0))),
            high_24h=Decimal(str(data.get("highPrice", 0))),
            low_24h=Decimal(str(data.get("lowPrice", 0))),
            volume_24h=Decimal(str(data.get("volume", 0))),
            price_change_percent=Decimal(str(data.get("priceChangePercent", 0))),
            timestamp=datetime.now(),
        )

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> list[Kline]:
        """Get historical klines.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1m", "5m", "1h")
            limit: Number of klines to fetch

        Returns:
            List of Kline objects
        """
        client = self._get_client()

        def _fetch() -> list[list[Any]]:
            response = client.rest_api.kline_candlestick_data(
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            return response.data()  # type: ignore[no-any-return]

        data = await asyncio.to_thread(_fetch)

        klines = []
        for item in data:
            # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
            klines.append(
                Kline(
                    open_time=datetime.fromtimestamp(item[0] / 1000),
                    open=Decimal(str(item[1])),
                    high=Decimal(str(item[2])),
                    low=Decimal(str(item[3])),
                    close=Decimal(str(item[4])),
                    volume=Decimal(str(item[5])),
                    close_time=datetime.fromtimestamp(item[6] / 1000),
                    quote_volume=Decimal(str(item[7])),
                    trades=int(item[8]) if len(item) > 8 else None,
                )
            )
        return klines

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book snapshot.

        Args:
            symbol: Trading pair
            limit: Depth limit

        Returns:
            OrderBook snapshot
        """
        client = self._get_client()

        def _fetch() -> dict[str, Any]:
            response = client.rest_api.order_book(symbol=symbol, limit=limit)
            return response.data()  # type: ignore[no-any-return]

        data = await asyncio.to_thread(_fetch)

        bids = [
            OrderBookLevel(price=Decimal(str(b[0])), quantity=Decimal(str(b[1])))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=Decimal(str(a[0])), quantity=Decimal(str(a[1])))
            for a in data.get("asks", [])
        ]

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            last_update_id=data.get("lastUpdateId"),
        )

    async def get_funding_rate(self, symbol: str) -> FundingRate:
        """Get current funding rate.

        Args:
            symbol: Trading pair

        Returns:
            FundingRate data
        """
        client = self._get_client()

        def _fetch() -> dict[str, Any]:
            response = client.rest_api.get_funding_rate_info()
            data = response.data()
            # Find the specific symbol
            for item in data:
                if item.get("symbol") == symbol:
                    return item  # type: ignore[no-any-return]
            return {}

        data = await asyncio.to_thread(_fetch)

        return FundingRate(
            symbol=symbol,
            funding_rate=Decimal(str(data.get("lastFundingRate", 0))),
            funding_time=datetime.now(),
            mark_price=None,
        )

    async def get_mark_price(self, symbol: str) -> dict[str, Any]:
        """Get mark price.

        Args:
            symbol: Trading pair

        Returns:
            Mark price data
        """
        client = self._get_client()

        def _fetch() -> dict[str, Any]:
            response = client.rest_api.mark_price(symbol=symbol)
            return response.data()  # type: ignore[no-any-return]

        return await asyncio.to_thread(_fetch)

    async def get_open_interest(self, symbol: str) -> OpenInterest:
        """Get open interest.

        Args:
            symbol: Trading pair

        Returns:
            OpenInterest data
        """
        client = self._get_client()

        def _fetch() -> dict[str, Any]:
            response = client.rest_api.open_interest(symbol=symbol)
            return response.data()  # type: ignore[no-any-return]

        data = await asyncio.to_thread(_fetch)

        return OpenInterest(
            symbol=symbol,
            open_interest=Decimal(str(data.get("openInterest", 0))),
            timestamp=datetime.now(),
        )

    async def get_long_short_ratio(
        self, symbol: str, period: str = "5m"
    ) -> LongShortRatio:
        """Get long/short ratio.

        Args:
            symbol: Trading pair
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)

        Returns:
            LongShortRatio data
        """
        client = self._get_client()

        def _fetch() -> list[dict[str, Any]]:
            response = client.rest_api.top_trader_long_short_ratio_positions(
                symbol=symbol,
                period=period,
                limit=1,
            )
            return response.data()  # type: ignore[no-any-return]

        data = await asyncio.to_thread(_fetch)

        if data:
            item = data[0]
            return LongShortRatio(
                symbol=symbol,
                long_ratio=Decimal(str(item.get("longAccount", 0))),
                short_ratio=Decimal(str(item.get("shortAccount", 0))),
                long_short_ratio=Decimal(str(item.get("longShortRatio", 0))),
                timestamp=datetime.fromtimestamp(item.get("timestamp", 0) / 1000),
            )

        return LongShortRatio(
            symbol=symbol,
            long_ratio=Decimal("0.5"),
            short_ratio=Decimal("0.5"),
            long_short_ratio=Decimal("1.0"),
            timestamp=datetime.now(),
        )
