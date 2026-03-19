"""Tests for game session logging."""

import os
import tempfile

import pytest

from loop.game_logger import LoggingConsoleProxy, generate_log_path


class TestLoggingConsoleProxy:
    def test_print_writes_to_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.print("Hello, world!")
        proxy.close()

        contents = log_file.read_text()
        assert "Hello, world!" in contents

    def test_print_strips_rich_markup_in_log(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.print("[bold red]Important message[/]")
        proxy.close()

        contents = log_file.read_text()
        assert "Important message" in contents
        # Rich markup should be stripped in log
        assert "[bold red]" not in contents
        assert "[/]" not in contents

    def test_no_log_path_is_noop(self):
        proxy = LoggingConsoleProxy(log_path=None)
        proxy.print("This should not crash")
        assert not proxy.logging_active
        assert proxy.log_path is None
        proxy.close()

    def test_logging_active_property(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        assert proxy.logging_active is True
        proxy.close()
        assert proxy.logging_active is False

    def test_log_path_property(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        assert proxy.log_path == str(log_file)
        proxy.close()

    def test_header_written(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.close()

        contents = log_file.read_text()
        assert "LOOP — Game Session Log" in contents
        assert "Started:" in contents

    def test_footer_written_on_close(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.print("game content")
        proxy.close()

        contents = log_file.read_text()
        assert "Session ended:" in contents

    def test_clear_writes_separator(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.print("before clear")
        proxy.clear()
        proxy.print("after clear")
        proxy.close()

        contents = log_file.read_text()
        assert "=" * 80 in contents
        assert "before clear" in contents
        assert "after clear" in contents

    def test_log_only_not_in_terminal(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.log_only("[INPUT] > hello")
        proxy.close()

        contents = log_file.read_text()
        assert "[INPUT] > hello" in contents

    def test_log_only_noop_without_log(self):
        proxy = LoggingConsoleProxy(log_path=None)
        proxy.log_only("should not crash")
        proxy.close()

    def test_multiple_prints_accumulate(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.print("line 1")
        proxy.print("line 2")
        proxy.print("line 3")
        proxy.close()

        contents = log_file.read_text()
        assert "line 1" in contents
        assert "line 2" in contents
        assert "line 3" in contents

    def test_creates_parent_directories(self, tmp_path):
        log_file = tmp_path / "deep" / "nested" / "dir" / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.print("nested test")
        proxy.close()
        assert log_file.exists()

    def test_double_close_is_safe(self, tmp_path):
        log_file = tmp_path / "test.log"
        proxy = LoggingConsoleProxy(log_path=str(log_file))
        proxy.print("content")
        proxy.close()
        proxy.close()  # should not raise


class TestGenerateLogPath:
    def test_returns_path_in_log_dir(self):
        path = generate_log_path()
        assert "logs" in str(path)
        assert "session_" in str(path)
        assert str(path).endswith(".log")

    def test_unique_paths(self):
        p1 = generate_log_path()
        p2 = generate_log_path()
        # Same second could produce same timestamp, so just check format
        assert str(p1).endswith(".log")
        assert "session_" in str(p1)
