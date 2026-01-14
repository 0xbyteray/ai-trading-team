#!/usr/bin/env python3
"""Script to manually place stop-loss or take-profit orders on WEEX.

Usage:
    uv run python scripts/place_tp_sl.py
"""

import asyncio
import sys
import uuid
from decimal import ROUND_DOWN, ROUND_UP
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_trading_team.config import Config
from ai_trading_team.core.types import Side
from ai_trading_team.execution.weex.executor import WEEXExecutor


async def place_tp_sl_order(
    executor: WEEXExecutor,
    symbol: str,
    plan_type: str,  # "loss_plan" or "profit_plan"
    side: Side,
    size: float,
    trigger_price: float,
) -> str | None:
    """Place a take-profit or stop-loss plan order."""
    client = executor._ensure_connected()
    contract = await executor._get_contract(symbol)
    size_step = executor._size_step_from_contract(contract)
    price_step = executor._price_step_from_contract(contract)

    size_decimal = executor._decimal_or_zero(size)
    adjusted_size = executor._quantize_step(size_decimal, size_step, ROUND_DOWN)
    if adjusted_size <= 0:
        raise ValueError(f"Size {size_decimal} invalid after step adjustment (step={size_step})")

    # For stop loss: round down for long, round up for short
    # For take profit: opposite
    if plan_type == "loss_plan":
        rounding = ROUND_DOWN if side == Side.LONG else ROUND_UP
    else:
        rounding = ROUND_UP if side == Side.LONG else ROUND_DOWN

    adjusted_trigger = executor._quantize_step(
        executor._decimal_or_zero(trigger_price), price_step, rounding
    )
    if adjusted_trigger <= 0:
        raise ValueError("Trigger price must be > 0")

    client_order_id = f"manual_{uuid.uuid4().hex[:12]}"
    position_side = "long" if side == Side.LONG else "short"

    data = {
        "symbol": symbol,
        "clientOrderId": client_order_id,
        "planType": plan_type,
        "triggerPrice": executor._format_decimal(adjusted_trigger),
        "executePrice": "0",  # Market execution
        "size": executor._format_decimal(adjusted_size),
        "positionSide": position_side,
        "marginMode": 3,
    }

    response = await client.post("/capi/v2/order/placeTpSlOrder", data=data)
    if isinstance(response, list) and response:
        item = response[0]
        if item.get("success"):
            return str(item.get("orderId") or "")
    return None


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

        # Get current position
        position = await executor.get_position(symbol)
        if not position:
            print("No open position found.")
            print("You need an open position to place TP/SL orders.")
            return

        print(f"Current position:")
        print(f"  Side: {position.side.value}")
        print(f"  Size: {position.size}")
        print(f"  Entry Price: {position.entry_price}")
        print(f"  Leverage: {position.leverage}x")
        print(f"  Unrealized PnL: {position.unrealized_pnl}")
        print()

        # Show existing plan orders
        plans = await executor.get_current_plan_orders(symbol)
        if plans:
            print(f"Existing plan orders ({len(plans)}):")
            for plan in plans:
                order_id = plan.get("orderId") or plan.get("order_id")
                plan_type = plan.get("planType") or plan.get("plan_type") or plan.get("type")
                trigger = plan.get("triggerPrice") or plan.get("trigger_price")
                size = plan.get("size")
                print(f"  - ID={order_id} Type={plan_type} Trigger={trigger} Size={size}")
            print()

        print("-" * 50)
        print("Select order type:")
        print("  [1] Stop Loss (loss_plan)")
        print("  [2] Take Profit (profit_plan)")
        print("  [q] Quit")
        print()

        type_choice = input("Order type (1/2): ").strip().lower()

        if type_choice == "q":
            print("Cancelled.")
            return

        if type_choice == "1":
            plan_type = "loss_plan"
            plan_name = "Stop Loss"
        elif type_choice == "2":
            plan_type = "profit_plan"
            plan_name = "Take Profit"
        else:
            print("Invalid choice.")
            return

        print(f"\nPlacing {plan_name} order for {position.side.value} position")
        print()

        # Get trigger price
        if plan_type == "loss_plan":
            if position.side == Side.LONG:
                hint = f"(below entry {position.entry_price})"
            else:
                hint = f"(above entry {position.entry_price})"
        else:
            if position.side == Side.LONG:
                hint = f"(above entry {position.entry_price})"
            else:
                hint = f"(below entry {position.entry_price})"

        trigger_input = input(f"Trigger price {hint}: ").strip()
        try:
            trigger_price = float(trigger_input)
        except ValueError:
            print("Invalid price.")
            return

        # Get size (default to full position)
        size_input = input(f"Size [default: {position.size}]: ").strip()
        if size_input:
            try:
                size = float(size_input)
            except ValueError:
                print("Invalid size.")
                return
        else:
            size = float(position.size)

        # Validate
        if plan_type == "loss_plan":
            if position.side == Side.LONG and trigger_price >= float(position.entry_price):
                print(f"Warning: Stop loss for LONG should be below entry price {position.entry_price}")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    return
            elif position.side == Side.SHORT and trigger_price <= float(position.entry_price):
                print(f"Warning: Stop loss for SHORT should be above entry price {position.entry_price}")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    return

        print()
        print(f"Placing {plan_name}:")
        print(f"  Side: {position.side.value}")
        print(f"  Trigger: {trigger_price}")
        print(f"  Size: {size}")
        print()

        confirm = input("Confirm? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

        order_id = await place_tp_sl_order(
            executor=executor,
            symbol=symbol,
            plan_type=plan_type,
            side=position.side,
            size=size,
            trigger_price=trigger_price,
        )

        if order_id:
            print(f"\n{plan_name} order placed successfully!")
            print(f"Order ID: {order_id}")
        else:
            print(f"\nFailed to place {plan_name} order.")

    finally:
        await executor.disconnect()
        print("\nDisconnected from WEEX")


if __name__ == "__main__":
    asyncio.run(main())
