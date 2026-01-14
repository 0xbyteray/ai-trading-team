#!/usr/bin/env python3
"""Script to manually adjust WEEX leverage.

Usage:
    uv run python scripts/set_leverage.py
    uv run python scripts/set_leverage.py 10
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_trading_team.config import Config
from ai_trading_team.execution.weex.executor import WEEXExecutor


async def main() -> None:
    """Main entry point."""
    config = Config.from_env()

    if not config.api.weex_api_key:
        print("Error: WEEX API credentials not configured in .env")
        sys.exit(1)

    symbol = f"cmt_{config.trading.symbol.lower()}"
    print(f"Symbol: {symbol}")
    print("-" * 50)

    executor = WEEXExecutor(
        api_key=config.api.weex_api_key,
        api_secret=config.api.weex_api_secret,
        passphrase=config.api.weex_passphrase,
    )

    try:
        await executor.connect()
        print("Connected to WEEX\n")

        # Get current position to show current leverage
        position = await executor.get_position(symbol)
        if position:
            print(f"Current position:")
            print(f"  Side: {position.side.value}")
            print(f"  Size: {position.size}")
            print(f"  Leverage: {position.leverage}x")
            print(f"  Entry Price: {position.entry_price}")
            print()
        else:
            print("No open position.\n")

        # Check if leverage was passed as argument
        if len(sys.argv) > 1:
            try:
                new_leverage = int(sys.argv[1])
            except ValueError:
                print(f"Invalid leverage value: {sys.argv[1]}")
                return
        else:
            print("Enter new leverage (1-20):")
            print("  Enter 'q' to quit")
            print()

            choice = input("Leverage: ").strip().lower()

            if choice == "q":
                print("Cancelled.")
                return

            try:
                new_leverage = int(choice)
            except ValueError:
                print("Invalid input. Enter a number between 1 and 20.")
                return

        # Validate leverage
        if new_leverage < 1 or new_leverage > 20:
            print(f"Leverage must be between 1 and 20. Got: {new_leverage}")
            return

        print(f"\nSetting leverage to {new_leverage}x...")

        success = await executor.set_leverage(symbol, new_leverage)

        if success:
            print(f"Leverage set to {new_leverage}x successfully.")
        else:
            print("Failed to set leverage.")

    finally:
        await executor.disconnect()
        print("\nDisconnected from WEEX")


if __name__ == "__main__":
    asyncio.run(main())
