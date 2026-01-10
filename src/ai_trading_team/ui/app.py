"""Main Textual application."""

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from ai_trading_team.ui.screens.dashboard import DashboardScreen


class TradingApp(App):
    """AI Trading Team TUI Application."""

    TITLE = "AI Trading Team"
    CSS = """
    Screen {
        background: $surface;
    }
    """

    BINDINGS = [
        ("d", "switch_screen('dashboard')", "Dashboard"),
        ("l", "switch_screen('logs')", "Logs"),
        ("q", "quit", "Quit"),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
    }

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.push_screen("dashboard")

    def action_switch_screen(self, screen_name: str) -> None:
        """Switch to a different screen."""
        if screen_name in self.SCREENS:
            self.switch_screen(screen_name)
