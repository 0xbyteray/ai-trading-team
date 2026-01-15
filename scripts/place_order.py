#!/usr/bin/env python3
"""Script to manually place orders on WEEX.

Supports:
- Open long/short positions with market or limit orders
- Close positions fully or partially
- View current position and account info

Usage:
    uv run python scripts/place_order.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_trading_team.config import Config
from ai_trading_team.core.types import OrderType, Side
from ai_trading_team.execution.weex.executor import WEEXExecutor


def print_separator() -> None:
    print("-" * 60)


def print_menu() -> None:
    print_separator()
    print("Select action:")
    print("  [1] Open Long (市价做多)")
    print("  [2] Open Short (市价做空)")
    print("  [3] Open Long Limit (限价做多)")
    print("  [4] Open Short Limit (限价做空)")
    print("  [5] Close Position (平仓)")
    print("  [6] Reduce Position (部分平仓)")
    print("  [7] View Account Info (查看账户)")
    print("  [8] View Position (查看仓位)")
    print("  [9] View Open Orders (查看挂单)")
    print("  [q] Quit (退出)")
    print_separator()


async def show_account_info(executor: WEEXExecutor) -> None:
    """Display account information."""
    account = await executor.get_account()
    print("\nAccount Info:")
    print(f"  Total Equity: {account.total_equity:.2f} USDT")
    print(f"  Available Balance: {account.available_balance:.2f} USDT")
    print(f"  Used Margin: {account.used_margin:.2f} USDT")
    print(f"  Unrealized PnL: {account.unrealized_pnl:.2f} USDT")
    print()


async def show_position(executor: WEEXExecutor, symbol: str) -> None:
    """Display current position."""
    position = await executor.get_position(symbol)
    if not position:
        print("\nNo open position.\n")
        return

    # Calculate PnL percentage
    margin = float(position.margin)
    pnl = float(position.unrealized_pnl)
    pnl_pct = (pnl / margin * 100) if margin > 0 else 0

    print("\nCurrent Position:")
    print(f"  Side: {position.side.value.upper()}")
    print(f"  Size: {position.size}")
    print(f"  Entry Price: {position.entry_price}")
    print(f"  Leverage: {position.leverage}x")
    print(f"  Margin: {position.margin:.2f} USDT")
    print(f"  Unrealized PnL: {pnl:+.2f} USDT ({pnl_pct:+.2f}%)")
    print()


async def show_orders(executor: WEEXExecutor, symbol: str) -> None:
    """Display open orders."""
    orders = await executor.get_open_orders(symbol)
    if not orders:
        print("\nNo open orders.\n")
        return

    print(f"\nOpen Orders ({len(orders)}):")
    for order in orders:
        print(f"  - ID={order.order_id} Side={order.side.value} "
              f"Size={order.size} Price={order.price} Status={order.status.value}")
    print()

    # Also show plan orders
    plans = await executor.get_current_plan_orders(symbol)
    if plans:
        print(f"Plan Orders ({len(plans)}):")
        for plan in plans:
            order_id = plan.get("orderId") or plan.get("order_id")
            plan_type = plan.get("planType") or plan.get("plan_type")
            trigger = plan.get("triggerPrice") or plan.get("trigger_price")
            size = plan.get("size")
            print(f"  - ID={order_id} Type={plan_type} Trigger={trigger} Size={size}")
        print()


async def open_position(
    executor: WEEXExecutor,
    symbol: str,
    side: Side,
    order_type: OrderType,
) -> None:
    """Open a new position."""
    # Get account for balance info
    account = await executor.get_account()
    available = float(account.available_balance)
    max_margin = available / 20  # 1/20 rule

    print(f"\nOpening {side.value.upper()} position")
    print(f"Available balance: {available:.2f} USDT")
    print(f"Max margin (1/20 rule): {max_margin:.2f} USDT")
    print()

    # Get size
    size_input = input("Size (e.g., 0.001 BTC): ").strip()
    try:
        size = float(size_input)
        if size <= 0:
            print("Size must be positive.")
            return
    except ValueError:
        print("Invalid size.")
        return

    # Get price for limit orders
    price = None
    if order_type == OrderType.LIMIT:
        price_input = input("Limit price: ").strip()
        try:
            price = float(price_input)
            if price <= 0:
                print("Price must be positive.")
                return
        except ValueError:
            print("Invalid price.")
            return

    # Get stop loss price (optional)
    sl_input = input("Stop loss price (optional, press Enter to skip): ").strip()
    stop_loss = None
    if sl_input:
        try:
            stop_loss = float(sl_input)
        except ValueError:
            print("Invalid stop loss price, skipping.")

    # Get take profit price (optional)
    tp_input = input("Take profit price (optional, press Enter to skip): ").strip()
    take_profit = None
    if tp_input:
        try:
            take_profit = float(tp_input)
        except ValueError:
            print("Invalid take profit price, skipping.")

    # Confirmation
    print()
    print("Order Summary:")
    print(f"  Action: OPEN {side.value.upper()}")
    print(f"  Type: {order_type.value}")
    print(f"  Size: {size}")
    if price:
        print(f"  Price: {price}")
    if stop_loss:
        print(f"  Stop Loss: {stop_loss}")
    if take_profit:
        print(f"  Take Profit: {take_profit}")
    print()

    confirm = input("Confirm order? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        order = await executor.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            action="open",
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )
        print(f"\nOrder placed successfully!")
        print(f"Order ID: {order.order_id}")
        print(f"Status: {order.status.value}")
    except Exception as e:
        print(f"\nFailed to place order: {e}")


async def close_position(executor: WEEXExecutor, symbol: str, partial: bool = False) -> None:
    """Close an existing position."""
    position = await executor.get_position(symbol)
    if not position:
        print("\nNo open position to close.")
        return

    await show_position(executor, symbol)

    if partial:
        size_input = input(f"Size to close (max {position.size}): ").strip()
        try:
            size = float(size_input)
            if size <= 0:
                print("Size must be positive.")
                return
            if size > float(position.size):
                print(f"Size exceeds position size {position.size}.")
                return
        except ValueError:
            print("Invalid size.")
            return
    else:
        size = None  # Close full position

    print()
    print("Close Order Summary:")
    print(f"  Side: {position.side.value.upper()}")
    print(f"  Size: {size if size else position.size} (full)" if not size else f"  Size: {size}")
    print()

    confirm = input("Confirm close? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        order = await executor.close_position(
            symbol=symbol,
            side=position.side,
            size=size,
        )
        if order:
            print(f"\nPosition closed successfully!")
            print(f"Order ID: {order.order_id}")
        else:
            print("\nFailed to close position.")
    except Exception as e:
        print(f"\nFailed to close position: {e}")


async def main() -> None:
    """Main entry point."""
    config = Config.from_env()

    if not config.api.weex_api_key:
        print("Error: WEEX API credentials not configured in .env")
        sys.exit(1)

    symbol = f"cmt_{config.trading.symbol.lower()}"
    print(f"\nWEEX Manual Trading Script")
    print(f"Symbol: {symbol}")

    executor = WEEXExecutor(
        api_key=config.api.weex_api_key,
        api_secret=config.api.weex_api_secret,
        passphrase=config.api.weex_passphrase,
    )

    try:
        await executor.connect()
        print("Connected to WEEX")

        # Show initial position and account info
        await show_account_info(executor)
        await show_position(executor, symbol)

        while True:
            print_menu()
            choice = input("Select action: ").strip().lower()

            if choice == "q":
                print("Exiting...")
                break

            elif choice == "1":
                await open_position(executor, symbol, Side.LONG, OrderType.MARKET)

            elif choice == "2":
                await open_position(executor, symbol, Side.SHORT, OrderType.MARKET)

            elif choice == "3":
                await open_position(executor, symbol, Side.LONG, OrderType.LIMIT)

            elif choice == "4":
                await open_position(executor, symbol, Side.SHORT, OrderType.LIMIT)

            elif choice == "5":
                await close_position(executor, symbol, partial=False)

            elif choice == "6":
                await close_position(executor, symbol, partial=True)

            elif choice == "7":
                await show_account_info(executor)

            elif choice == "8":
                await show_position(executor, symbol)

            elif choice == "9":
                await show_orders(executor, symbol)

            else:
                print("Invalid choice. Please select 1-9 or q.")

    finally:
        await executor.disconnect()
        print("\nDisconnected from WEEX")


if __name__ == "__main__":
    asyncio.run(main())
