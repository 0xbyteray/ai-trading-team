#!/usr/bin/env python3
"""Script to manually cancel WEEX stop-loss orders.

Usage:
    uv run python scripts/cancel_stop_loss.py
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

        # Get all plan orders
        plans = await executor.get_current_plan_orders(symbol)

        if not plans:
            print("No plan orders found.")
            return

        # Display all plan orders
        print(f"Found {len(plans)} plan order(s):\n")
        for i, plan in enumerate(plans):
            order_id = plan.get("orderId") or plan.get("order_id")
            plan_type = plan.get("planType") or plan.get("plan_type") or plan.get("type") or "unknown"
            trigger_price = plan.get("triggerPrice") or plan.get("trigger_price")
            size = plan.get("size")
            side = plan.get("positionSide") or plan.get("position_side") or plan.get("side")
            created_at = plan.get("cTime") or plan.get("created_at") or ""
            print(f"  [{i + 1}] ID: {order_id}")
            print(f"      Type: {plan_type}")
            print(f"      Side: {side}")
            print(f"      Trigger Price: {trigger_price}")
            print(f"      Size: {size}")
            if created_at:
                print(f"      Created: {created_at}")
            print()

        print("-" * 50)
        print("Options:")
        print("  Enter number(s) to cancel specific orders (e.g., '1' or '1,2,3')")
        print("  Enter 'all' to cancel all plan orders")
        print("  Enter 'q' to quit")
        print()

        choice = input("Your choice: ").strip().lower()

        if choice == "q":
            print("Cancelled.")
            return

        if choice == "all":
            confirm = input(f"Cancel ALL {len(plans)} plan orders? (y/n): ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return

            cancelled = 0
            for plan in plans:
                order_id = plan.get("orderId") or plan.get("order_id")
                if order_id:
                    success = await executor.cancel_plan_order(str(order_id))
                    if success:
                        print(f"  Cancelled order {order_id}")
                        cancelled += 1
                    else:
                        print(f"  Failed to cancel order {order_id}")
            print(f"\nCancelled {cancelled}/{len(plans)} orders.")
        else:
            # Parse individual selections
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
            except ValueError:
                print("Invalid input. Enter numbers separated by commas.")
                return

            for idx in indices:
                if idx < 0 or idx >= len(plans):
                    print(f"Invalid selection: {idx + 1}")
                    continue

                plan = plans[idx]
                order_id = plan.get("orderId") or plan.get("order_id")
                if order_id:
                    success = await executor.cancel_plan_order(str(order_id))
                    if success:
                        print(f"Cancelled order {order_id}")
                    else:
                        print(f"Failed to cancel order {order_id}")

    finally:
        await executor.disconnect()
        print("\nDisconnected from WEEX")


if __name__ == "__main__":
    asyncio.run(main())
