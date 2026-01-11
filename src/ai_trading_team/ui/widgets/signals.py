"""Strategy signals display widget."""

from textual.app import ComposeResult
from textual.widgets import DataTable, Static


class SignalsWidget(Static):
    """Strategy signals display widget."""

    DEFAULT_CSS = """
    SignalsWidget {
        height: auto;
    }

    DataTable {
        height: auto;
        max-height: 10;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the signals widget."""
        yield DataTable(id="signals-table")

    def on_mount(self) -> None:
        """Initialize the signals table."""
        table = self.query_one("#signals-table", DataTable)
        table.add_columns("Time", "Type", "Details")
        # Will be updated via signal queue subscription

    def add_signal(self, timestamp: str, signal_type: str, details: str) -> None:
        """Add a new signal to the display."""
        table = self.query_one("#signals-table", DataTable)
        table.add_row(timestamp, signal_type, details)

        # Keep only last 20 signals
        while table.row_count > 20:
            # Get the first row key from the rows dict
            first_row_key = next(iter(table.rows.keys()))
            table.remove_row(first_row_key)

    def clear_signals(self) -> None:
        """Clear all signals."""
        table = self.query_one("#signals-table", DataTable)
        table.clear()
