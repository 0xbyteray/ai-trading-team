# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent cryptocurrency trading bot using LangChain agents to make trading decisions based on market data and technical indicators. Currently targets WEEX exchange for order execution while using Binance as the primary data source (due to its market dominance and pricing authority).

## Commands

```bash
# Install dependencies (including dev tools)
uv sync --all-extras

# Run the application
uv run python main.py

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_file.py::test_function_name

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Textual dev console (run in separate terminal for debugging)
uv run textual console

# Run app with dev console connection
uv run textual run --dev main.py

# Serve app in browser
uv run textual serve "python main.py"
```

## Environment Setup

Copy `env.example` to `.env` and configure API keys and trading parameters. See `env.example` for all available options.

## Project Structure

```
src/ai_trading_team/
├── config.py              # Configuration management (loads from .env)
├── logging.py             # Logging setup (outputs to logs/)
├── core/                  # Core infrastructure
│   ├── types.py           # Global enums (Side, OrderType, etc.)
│   ├── events.py          # Event system for inter-module communication
│   ├── data_pool.py       # Thread-safe real-time data storage
│   └── signal_queue.py    # Timestamped signal queue
├── data/                  # Market data (Binance source)
│   ├── models.py          # Data models (Kline, Ticker, OrderBook)
│   ├── binance/           # Binance REST + WebSocket
│   └── manager.py         # Data manager
├── indicators/            # Technical indicators (talipp wrapper)
│   ├── base.py            # Indicator base class
│   ├── registry.py        # Indicator registry
│   └── technical.py       # RSI, MACD, BB, ATR implementations
├── strategy/              # Mechanical strategies
│   ├── base.py            # Strategy base class
│   ├── signals.py         # Signal definitions
│   ├── conditions.py      # Condition definitions
│   └── factors/           # Single-factor strategies
├── agent/                 # LangChain trading agent
│   ├── llm.py             # LLM client (langchain-anthropic)
│   ├── prompts.py         # Prompt templates
│   ├── schemas.py         # Agent decision schemas
│   ├── commands.py        # Command definitions (AgentAction, AgentCommand)
│   └── trader.py          # Trading agent logic
├── execution/             # Order execution
│   ├── models.py          # Position, Order, Account models
│   ├── base.py            # Exchange abstract interface
│   ├── weex/              # WEEX implementation
│   └── manager.py         # Execution manager
├── risk/                  # Independent risk control
│   ├── rules.py           # Risk rules (StopLoss, TakeProfit, MaxDrawdown)
│   ├── actions.py         # Risk actions
│   └── monitor.py         # Risk monitor
├── audit/                 # AI decision logging
│   ├── models.py          # Log models (AgentLog, OrderLog)
│   ├── writer.py          # Local file writer
│   ├── uploaders/         # Pluggable uploaders
│   │   ├── base.py        # Uploader interface
│   │   └── weex.py        # WEEX API uploader
│   └── manager.py         # Audit manager
└── ui/                    # Textual TUI
    ├── app.py             # Main application
    ├── screens/           # Dashboard, Logs screens
    └── widgets/           # Ticker, Positions, Signals widgets

tests/                     # Test files
logs/                      # Runtime logs (gitignored)
docs/                      # Reference materials (do not import)
```

## Architecture

The system is designed with the following modular layers:

### UI Layer
- **Textual TUI**: Terminal-based user interface built with Textual framework
  - Async-native, can integrate with WebSocket data streams
  - Supports running in terminal or browser via `textual serve`
  - Debug with `textual console` in a separate terminal
  - Command palette available via `Ctrl+P`

### Data Layer
- **Data Module**: Fetches real-time market data from Binance (REST + WebSocket)
  - Klines, ticker, orderbook, trades
  - Long/short ratio, funding rate, mark price
  - Open interest, liquidation events
- **Shared Data Pool**: Central real-time data repository that all modules read from

### Analysis Layer
- **Indicator Module**: Computes technical indicators using `talipp` library
- **Mechanical Strategy Module**: Triggers single-factor signals when predefined conditions are met (e.g., RSI overbought, MACD crossover, price levels)

### Decision Layer
- **Agent Module**: LangChain-based agents receive strategy signals with full market context (data snapshot, indicators, positions, orders) and return structured JSON commands
- **Risk Control Module**: Independent mechanical risk management operating outside agent logic

### Execution Layer
- **Account & Execution Module**: Manages positions, orders, and executes trades on WEEX
  - Must maintain real-time sync via REST + WebSocket
  - Handle reconnection automatically
  - Ensure reliable execution of critical operations (close, reduce, cancel)

### Logging Layer
- Complete AI decision logs (input, output, reasoning) for WEEX AI competition compliance
- Local file persistence for analysis and auditing

## Key Design Principles

1. **Signal Queue**: Strategy signals must be timestamped and queued to prevent signal overlap
2. **Data Snapshot**: When signals trigger, capture a point-in-time snapshot from the data pool for agent context
3. **Dual Data Sources**: Binance for market data, target exchange (WEEX) for account/order data
4. **Structured Agent Output**: Agent responses must be JSON-formatted commands executable by the strategy engine

## Important Notes

- Files in `docs/` are reference materials only - do not import or use code from this directory
- Use `talipp` library for all technical indicator calculations
