"""Trading agent main logic."""

import contextlib
import json
import logging
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ai_trading_team.agent.commands import AgentAction, AgentCommand
from ai_trading_team.agent.prompts import DECISION_PROMPT, SYSTEM_PROMPT
from ai_trading_team.agent.schemas import AgentDecision
from ai_trading_team.config import Config
from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.core.signal_queue import StrategySignal
from ai_trading_team.core.types import OrderType, Side

logger = logging.getLogger(__name__)


class LangChainTradingAgent:
    """LangChain-based trading agent.

    Uses Claude via langchain-anthropic to process signals and generate commands.
    """

    def __init__(self, config: Config, symbol: str) -> None:
        """Initialize trading agent.

        Args:
            config: Application configuration
            symbol: Trading symbol (e.g., "cmt_btcusdt")
        """
        self._config = config
        self._symbol = symbol
        self._llm = self._create_llm()

    def _create_llm(self) -> Any:
        """Create LangChain LLM client."""
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=self._config.api.anthropic_api_key,
            anthropic_api_url=self._config.api.anthropic_base_url,
            max_tokens=2048,
        )

    async def process_signal(
        self,
        signal: StrategySignal,
        snapshot: DataSnapshot,
    ) -> AgentDecision:
        """Process a strategy signal and generate a decision.

        Args:
            signal: Strategy signal that triggered the agent
            snapshot: Current market data snapshot

        Returns:
            Agent decision with command and metadata
        """
        start_time = time.time()

        # Format context for the prompt
        context = self.format_context(signal, snapshot)

        # Create messages
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=DECISION_PROMPT.format(**context)),
        ]

        try:
            # Invoke the LLM
            response = await self._llm.ainvoke(messages)
            raw_response = response.content

            # Parse the response
            command = self.parse_response(str(raw_response))

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Get token usage if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = response.usage_metadata.get("input_tokens", 0)
                completion_tokens = response.usage_metadata.get("output_tokens", 0)

            decision = AgentDecision(
                signal_type=signal.signal_type.value,
                signal_data=signal.data,
                market_snapshot=self._snapshot_to_dict(snapshot),
                command=command,
                timestamp=datetime.now(),
                model="claude-sonnet-4-20250514",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

            logger.info(
                f"Agent decision: action={command.action.value}, "
                f"side={command.side}, reason={command.reason[:50]}..."
            )

            return decision

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            # Return observe command on error
            latency_ms = (time.time() - start_time) * 1000
            return AgentDecision(
                signal_type=signal.signal_type.value,
                signal_data=signal.data,
                market_snapshot=self._snapshot_to_dict(snapshot),
                command=AgentCommand(
                    action=AgentAction.OBSERVE,
                    symbol=self._symbol,
                    reason=f"Agent error: {e}. Observing for safety.",
                ),
                timestamp=datetime.now(),
                model="claude-sonnet-4-20250514",
                latency_ms=latency_ms,
            )

    def parse_response(self, response: str) -> AgentCommand:
        """Parse LLM response into structured command.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed agent command

        Raises:
            ValueError: If response cannot be parsed
        """
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, response: {response}")
            raise ValueError(f"Failed to parse JSON: {e}") from e

        # Parse action
        action_str = data.get("action", "observe").lower()
        try:
            action = AgentAction(action_str)
        except ValueError:
            action = AgentAction.OBSERVE

        # Parse side
        side_str = data.get("side")
        side: Side | None = None
        if side_str:
            with contextlib.suppress(ValueError):
                side = Side(side_str.lower())

        # Parse order type
        order_type_str = data.get("order_type")
        order_type: OrderType | None = None
        if order_type_str:
            try:
                order_type = OrderType(order_type_str.lower())
            except ValueError:
                order_type = OrderType.MARKET

        return AgentCommand(
            action=action,
            symbol=data.get("symbol", self._symbol),
            side=side,
            size=float(data["size"]) if data.get("size") else None,
            price=float(data["price"]) if data.get("price") else None,
            order_type=order_type,
            reason=data.get("reason", "No reason provided"),
        )

    def format_context(
        self,
        signal: StrategySignal,
        snapshot: DataSnapshot,
    ) -> dict[str, Any]:
        """Format signal and snapshot into prompt context.

        Args:
            signal: Strategy signal
            snapshot: Market data snapshot

        Returns:
            Context dict for prompt formatting
        """
        # Format ticker
        ticker_str = "N/A"
        if snapshot.ticker:
            ticker = snapshot.ticker
            ticker_str = (
                f"Last: {ticker.get('last_price', 'N/A')}, "
                f"Bid: {ticker.get('bid_price', 'N/A')}, "
                f"Ask: {ticker.get('ask_price', 'N/A')}, "
                f"24h Change: {ticker.get('price_change_percent', 'N/A')}%"
            )

        # Format klines (last 5)
        klines_str = "N/A"
        if snapshot.klines:
            for interval, klines in snapshot.klines.items():
                if klines:
                    last_5 = klines[-5:]
                    klines_str = f"Interval: {interval}\n"
                    for k in last_5:
                        klines_str += (
                            f"  O:{k.get('open'):.2f} H:{k.get('high'):.2f} "
                            f"L:{k.get('low'):.2f} C:{k.get('close'):.2f}\n"
                        )
                    break

        # Format orderbook
        orderbook_str = "N/A"
        if snapshot.orderbook:
            ob = snapshot.orderbook
            bids = ob.get("bids", [])[:3]
            asks = ob.get("asks", [])[:3]
            orderbook_str = f"Bids: {bids}, Asks: {asks}"

        # Format indicators
        indicators_str = "N/A"
        if snapshot.indicators:
            indicators_str = "\n".join(
                f"{k}: {v}" for k, v in snapshot.indicators.items()
            )

        # Format position
        position_str = "无持仓"
        if snapshot.position:
            pos = snapshot.position
            position_str = (
                f"Symbol: {pos.get('symbol')}, "
                f"Side: {pos.get('side')}, "
                f"Size: {pos.get('size')}, "
                f"Entry: {pos.get('entry_price')}, "
                f"PnL: {pos.get('unrealized_pnl')}"
            )

        # Format orders
        orders_str = "无挂单"
        if snapshot.orders:
            orders_str = "\n".join(
                f"ID:{o.get('order_id')}, Side:{o.get('side')}, "
                f"Price:{o.get('price')}, Size:{o.get('size')}"
                for o in snapshot.orders
            )

        # Format account
        account_str = "N/A"
        if snapshot.account:
            acc = snapshot.account
            account_str = (
                f"Balance: {acc.get('balance')} USDT, "
                f"Available: {acc.get('available')} USDT, "
                f"Margin: {acc.get('margin')} USDT"
            )

        return {
            "signal_type": signal.signal_type.value,
            "signal_data": json.dumps(signal.data, default=str),
            "ticker": ticker_str,
            "klines": klines_str,
            "orderbook": orderbook_str,
            "indicators": indicators_str,
            "position": position_str,
            "orders": orders_str,
            "account": account_str,
        }

    def _snapshot_to_dict(self, snapshot: DataSnapshot) -> dict[str, Any]:
        """Convert snapshot to serializable dict."""
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "ticker": snapshot.ticker,
            "klines_count": sum(len(v) for v in (snapshot.klines or {}).values()),
            "indicators": snapshot.indicators,
            "position": snapshot.position,
            "orders_count": len(snapshot.orders or []),
        }
