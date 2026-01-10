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
src/ai_trading_team/     # Main package
  config.py              # Configuration management (loads from .env)
  logging.py             # Logging setup (outputs to logs/)
tests/                   # Test files
logs/                    # Runtime logs (gitignored)
docs/                    # Reference materials (do not import)
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
