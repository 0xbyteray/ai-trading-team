"""AI Trading Team - Entry point."""

from ai_trading_team.config import Config
from ai_trading_team.logging import setup_logging


def main() -> None:
    """Application entry point."""
    logger = setup_logging()
    config = Config.from_env()

    logger.info("AI Trading Team starting...")
    logger.info(f"Trading symbol: {config.trading.symbol}")
    logger.info(f"Leverage: {config.trading.leverage}x")


if __name__ == "__main__":
    main()
