"""Main dashboard screen."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Static

from ai_trading_team.ui.widgets.positions import PositionsWidget
from ai_trading_team.ui.widgets.signals import SignalsWidget
from ai_trading_team.ui.widgets.ticker import TickerWidget


class DashboardScreen(Screen):
    """Main trading dashboard screen."""

    CSS = """
    DashboardScreen {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 1;
    }

    .panel {
        border: solid $primary;
        padding: 1;
    }

    .panel-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    #ticker-panel {
        column-span: 2;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        with Container(id="ticker-panel", classes="panel"):
            yield Static("Market Data", classes="panel-title")
            yield TickerWidget()

        with Container(id="positions-panel", classes="panel"):
            yield Static("Positions", classes="panel-title")
            yield PositionsWidget()

        with Container(id="signals-panel", classes="panel"):
            yield Static("Strategy Signals", classes="panel-title")
            yield SignalsWidget()
