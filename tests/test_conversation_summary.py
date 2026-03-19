"""Tests for post-conversation summary display (loop/display.py)."""

from io import StringIO

import pytest
from rich.console import Console

from loop.display import GameDisplay


@pytest.fixture
def display():
    d = GameDisplay()
    d.console = Console(file=StringIO(), force_terminal=True, width=80)
    return d


def _get_output(display: GameDisplay) -> str:
    display.console.file.seek(0)
    return display.console.file.read()


class TestConversationSummary:
    def test_summary_shows_character_name(self, display):
        display.show_conversation_summary(
            char_name="Alice", exchanges=5, trust_change=3,
            trust_now=23, evidence_learned=[], rumors_planted=0,
        )
        output = _get_output(display)
        assert "Alice" in output
        assert "5 exchanges" in output

    def test_summary_shows_positive_trust(self, display):
        display.show_conversation_summary(
            char_name="Bob", exchanges=3, trust_change=5,
            trust_now=25, evidence_learned=[], rumors_planted=0,
        )
        output = _get_output(display)
        assert "+5" in output
        assert "25" in output

    def test_summary_shows_negative_trust(self, display):
        display.show_conversation_summary(
            char_name="Bob", exchanges=2, trust_change=-3,
            trust_now=7, evidence_learned=[], rumors_planted=0,
        )
        output = _get_output(display)
        assert "-3" in output

    def test_summary_shows_evidence(self, display):
        display.show_conversation_summary(
            char_name="Alice", exchanges=4, trust_change=1,
            trust_now=11, evidence_learned=["evidence_03", "evidence_07"],
            rumors_planted=0,
        )
        output = _get_output(display)
        assert "evidence_03" in output
        assert "evidence_07" in output

    def test_summary_shows_rumors_planted(self, display):
        display.show_conversation_summary(
            char_name="Alice", exchanges=3, trust_change=0,
            trust_now=10, evidence_learned=[], rumors_planted=2,
        )
        output = _get_output(display)
        assert "Rumors planted: 2" in output

    def test_summary_omits_zero_rumors(self, display):
        display.show_conversation_summary(
            char_name="Alice", exchanges=1, trust_change=0,
            trust_now=0, evidence_learned=[], rumors_planted=0,
        )
        output = _get_output(display)
        assert "Rumors planted" not in output

    def test_summary_omits_empty_evidence(self, display):
        display.show_conversation_summary(
            char_name="Alice", exchanges=1, trust_change=0,
            trust_now=0, evidence_learned=[], rumors_planted=0,
        )
        output = _get_output(display)
        assert "Learned" not in output
