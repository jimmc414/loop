"""Game session logger — captures all terminal output and player input to a file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.console import Console

from .config import DATA_DIR


LOG_DIR = DATA_DIR / "logs"


class LoggingConsoleProxy:
    """Wraps a Rich Console, duplicating all output to a plain-text log file.

    Drop-in replacement for Console — all display.py call sites work unchanged.
    """

    def __init__(self, log_path: str | Path | None = None):
        self._console = Console()
        self._log_file = None
        self._log_console = None

        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(path, "w", encoding="utf-8")
            self._log_console = Console(
                file=self._log_file,
                width=120,
                no_color=True,
                highlight=False,
                force_terminal=False,
            )
            # Write header
            self._log_console.print(f"LOOP — Game Session Log")
            self._log_console.print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log_console.print("=" * 80)
            self._log_console.print()
            self._log_file.flush()

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)
        if self._log_console:
            self._log_console.print(*args, **kwargs)
            self._log_file.flush()

    def clear(self):
        self._console.clear()
        if self._log_console:
            self._log_console.print()
            self._log_console.print("=" * 80)
            self._log_console.print()
            self._log_file.flush()

    def close(self):
        if self._log_file:
            self._log_console.print()
            self._log_console.print("=" * 80)
            self._log_console.print(
                f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None
            self._log_console = None

    def log_only(self, text: str):
        """Write a line to the log file only (not to the terminal)."""
        if self._log_console and self._log_file:
            self._log_console.print(text)
            self._log_file.flush()

    @property
    def logging_active(self) -> bool:
        return self._log_file is not None

    @property
    def log_path(self) -> str | None:
        if self._log_file:
            return self._log_file.name
        return None


def generate_log_path() -> Path:
    """Generate a timestamped log file path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"session_{timestamp}.log"
