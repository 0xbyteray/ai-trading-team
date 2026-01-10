"""Configuration management for AI Trading Team."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class APIConfig:
    """API configuration for exchanges and LLM."""

    anthropic_base_url: str = ""
    anthropic_api_key: str = ""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    weex_api_key: str = ""
    weex_api_secret: str = ""
    weex_passphrase: str = ""


@dataclass
class TradingConfig:
    """Trading parameters configuration."""

    symbol: str = "BTCUSDT"
    leverage: int = 10
    max_position_size: float = 1000.0  # USDT
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 4.0


@dataclass
class Config:
    """Main configuration container."""

    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Config":
        """Load configuration from environment variables.

        Args:
            env_path: Path to .env file (optional)

        Returns:
            Config instance populated from environment
        """
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        api = APIConfig(
            anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
            weex_api_key=os.getenv("WEEX_API_KEY", ""),
            weex_api_secret=os.getenv("WEEX_API_SECRET", ""),
            weex_passphrase=os.getenv("WEEX_PASSPHRASE", ""),
        )

        trading = TradingConfig(
            symbol=os.getenv("TRADING_SYMBOL", "BTCUSDT"),
            leverage=int(os.getenv("TRADING_LEVERAGE", "10")),
            max_position_size=float(os.getenv("TRADING_MAX_POSITION", "1000.0")),
            stop_loss_percent=float(os.getenv("TRADING_STOP_LOSS", "2.0")),
            take_profit_percent=float(os.getenv("TRADING_TAKE_PROFIT", "4.0")),
        )

        return cls(api=api, trading=trading)
