"""AI Trading Team - Entry point for MVP.

This is a minimal viable product that:
1. Fetches market data from Binance (BTCUSDT)
2. Runs MA crossover strategy to generate signals
3. Uses LangChain agent to make trading decisions
4. Executes trades on WEEX (cmt_btcusdt)
"""

import asyncio
import contextlib
import logging
import signal

from ai_trading_team.agent.commands import AgentAction
from ai_trading_team.agent.trader import LangChainTradingAgent
from ai_trading_team.config import Config
from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.core.signal_queue import SignalQueue
from ai_trading_team.core.types import OrderType
from ai_trading_team.data.manager import BinanceDataManager
from ai_trading_team.execution.weex.executor import WEEXExecutor
from ai_trading_team.logging import setup_logging
from ai_trading_team.strategy.factors.ma_crossover import MACrossoverStrategy

# Trading pair mappings
BINANCE_SYMBOL = "BTCUSDT"  # Binance uses uppercase
WEEX_SYMBOL = "cmt_btcusdt"  # WEEX competition symbol


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: Config) -> None:
        """Initialize trading bot.

        Args:
            config: Application configuration
        """
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._running = False

        # Core components
        self._data_pool = DataPool()
        self._signal_queue = SignalQueue()

        # Modules
        self._data_manager = BinanceDataManager(
            self._data_pool,
            config.api.binance_api_key,
            config.api.binance_api_secret,
        )
        self._strategy = MACrossoverStrategy(
            self._data_pool,
            self._signal_queue,
            short_period=7,
            long_period=25,
            ma_type="ema",
            kline_interval="1m",
        )
        self._agent = LangChainTradingAgent(config, WEEX_SYMBOL)
        self._executor = WEEXExecutor(
            config.api.weex_api_key,
            config.api.weex_api_secret,
            config.api.weex_passphrase,
        )

    async def start(self) -> None:
        """Start the trading bot."""
        self._logger.info("Starting trading bot...")
        self._running = True

        # Connect to WEEX
        try:
            await self._executor.connect()
            self._logger.info("Connected to WEEX")

            # Set leverage
            await self._executor.set_leverage(WEEX_SYMBOL, self._config.trading.leverage)

        except Exception as e:
            self._logger.error(f"Failed to connect to WEEX: {e}")
            self._logger.warning("Continuing without WEEX connection (dry run mode)")

        # Start data collection from Binance
        await self._data_manager.start(BINANCE_SYMBOL, kline_interval="1m")
        self._logger.info(f"Started data collection for {BINANCE_SYMBOL}")

        # Main loop
        await self._run_loop()

    async def _run_loop(self) -> None:
        """Main trading loop."""
        strategy_check_interval = 5  # Check strategy every 5 seconds
        last_check = 0

        while self._running:
            try:
                current_time = asyncio.get_event_loop().time()

                # Periodically check strategy conditions
                if current_time - last_check >= strategy_check_interval:
                    last_check = current_time

                    # Evaluate strategy
                    signal = self._strategy.evaluate()
                    if signal:
                        self._logger.info(
                            f"Signal generated: {signal.signal_type.value}"
                        )

                # Process any signals in the queue
                signal = self._signal_queue.pop()
                if signal:
                    await self._process_signal(signal)

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    async def _process_signal(self, signal) -> None:  # type: ignore[no-untyped-def]
        """Process a strategy signal through the agent.

        Args:
            signal: Strategy signal to process
        """
        self._logger.info(f"Processing signal: {signal.signal_type.value}")

        # Get market snapshot
        snapshot = self._data_pool.get_snapshot()

        # Get current position from WEEX
        try:
            position = await self._executor.get_position(WEEX_SYMBOL)
            if position:
                snapshot.position = {
                    "symbol": position.symbol,
                    "side": position.side.value,
                    "size": float(position.size),
                    "entry_price": float(position.entry_price),
                    "unrealized_pnl": float(position.unrealized_pnl),
                }

            # Get account info
            account = await self._executor.get_account()
            snapshot.account = {
                "balance": float(account.total_equity),
                "available": float(account.available_balance),
                "margin": float(account.used_margin),
            }

        except Exception as e:
            self._logger.warning(f"Failed to get WEEX data: {e}")

        # Get agent decision
        decision = await self._agent.process_signal(signal, snapshot)
        self._logger.info(
            f"Agent decision: {decision.command.action.value} - {decision.command.reason[:100]}"
        )

        # Execute the decision
        if decision.command.is_actionable():
            await self._execute_command(decision)

        # Upload AI log to WEEX
        try:
            await self._executor.upload_ai_log(
                stage="Signal Processing",
                model=decision.model,
                input_data={
                    "signal_type": signal.signal_type.value,
                    "signal_data": signal.data,
                    "market_snapshot": decision.market_snapshot,
                },
                output={
                    "action": decision.command.action.value,
                    "side": decision.command.side.value if decision.command.side else None,
                    "size": decision.command.size,
                },
                explanation=decision.command.reason,
            )
        except Exception as e:
            self._logger.warning(f"Failed to upload AI log: {e}")

    async def _execute_command(self, decision) -> None:  # type: ignore[no-untyped-def]
        """Execute agent command.

        Args:
            decision: Agent decision with command to execute
        """
        command = decision.command
        self._logger.info(f"Executing command: {command.action.value}")

        try:
            if command.action == AgentAction.OPEN:
                if command.side and command.size:
                    order = await self._executor.place_order(
                        symbol=WEEX_SYMBOL,
                        side=command.side,
                        order_type=command.order_type or OrderType.MARKET,
                        size=command.size,
                        price=command.price,
                        action="open",
                    )
                    self._logger.info(f"Opened position: {order.order_id}")

            elif command.action == AgentAction.CLOSE:
                if command.side:
                    order = await self._executor.close_position(
                        symbol=WEEX_SYMBOL,
                        side=command.side,
                        size=command.size,
                    )
                    if order:
                        self._logger.info(f"Closed position: {order.order_id}")

            elif command.action == AgentAction.ADD:
                if command.side and command.size:
                    order = await self._executor.place_order(
                        symbol=WEEX_SYMBOL,
                        side=command.side,
                        order_type=command.order_type or OrderType.MARKET,
                        size=command.size,
                        price=command.price,
                        action="open",
                    )
                    self._logger.info(f"Added to position: {order.order_id}")

            elif command.action == AgentAction.REDUCE:
                if command.side and command.size:
                    order = await self._executor.place_order(
                        symbol=WEEX_SYMBOL,
                        side=command.side,
                        order_type=command.order_type or OrderType.MARKET,
                        size=command.size,
                        price=command.price,
                        action="close",
                    )
                    self._logger.info(f"Reduced position: {order.order_id}")

            elif command.action == AgentAction.CANCEL:
                count = await self._executor.cancel_all_orders(WEEX_SYMBOL)
                self._logger.info(f"Cancelled {count} orders")

        except Exception as e:
            self._logger.error(f"Failed to execute command: {e}")

    async def stop(self) -> None:
        """Stop the trading bot."""
        self._logger.info("Stopping trading bot...")
        self._running = False

        # Stop data collection
        await self._data_manager.stop()

        # Disconnect from WEEX
        await self._executor.disconnect()

        self._logger.info("Trading bot stopped")


async def main_async() -> None:
    """Async main entry point."""
    logger = setup_logging()
    config = Config.from_env()

    logger.info("=" * 50)
    logger.info("AI Trading Team MVP")
    logger.info("=" * 50)
    logger.info(f"Binance Symbol: {BINANCE_SYMBOL}")
    logger.info(f"WEEX Symbol: {WEEX_SYMBOL}")
    logger.info(f"Leverage: {config.trading.leverage}x")
    logger.info("Strategy: MA Crossover (EMA 7/25)")
    logger.info("=" * 50)

    bot = TradingBot(config)

    # Handle shutdown gracefully
    loop = asyncio.get_event_loop()

    def shutdown_handler() -> None:
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await bot.stop()


def main() -> None:
    """Application entry point."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main_async())


if __name__ == "__main__":
    main()
