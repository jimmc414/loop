"""Tests for conversation engine truncation (loop/conversation_engine.py)."""

import re

import pytest


def truncate_text(result_text: str) -> str:
    """Replicate the sentence-based truncation logic from _get_npc_response."""
    result_text = result_text.strip()
    if len(result_text) > 500:
        sentences = re.split(r'(?<=[.!?])\s+', result_text)
        truncated = ""
        for s in sentences:
            if len(truncated) + len(s) + 1 > 500:
                break
            truncated = f"{truncated} {s}" if truncated else s
        result_text = truncated or sentences[0]
    return result_text or "*says nothing*"


class TestSentenceTruncation:
    def test_short_text_unchanged(self):
        text = "Hello there. How are you doing today?"
        result = truncate_text(text)
        assert result == text

    def test_long_text_truncated_at_sentence_boundary(self):
        # Build text with known sentences
        sentences = [
            "The camp was quiet this morning.",
            "Birds sang in the pine trees.",
            "A cold breeze swept across the lake.",
            "Something felt wrong about the whole place.",
            "Alice had been acting strange lately.",
        ]
        # Repeat sentences to exceed 500 chars
        long_text = " ".join(sentences * 5)
        assert len(long_text) > 500

        result = truncate_text(long_text)
        assert len(result) <= 500
        # Must end with a complete sentence (period)
        assert result.rstrip().endswith(".")

    def test_truncation_preserves_complete_sentences(self):
        s1 = "First sentence here." # 20 chars
        s2 = "Second sentence here." # 21 chars
        s3 = "A" * 500  # one huge sentence to force truncation
        long_text = f"{s1} {s2} {s3}"
        result = truncate_text(long_text)
        # Should keep s1 and s2 but not s3
        assert s1 in result
        assert s2 in result
        assert len(result) <= 500

    def test_single_long_sentence_fallback(self):
        """If the first sentence alone exceeds 500 chars, keep it as fallback."""
        long_sentence = "A" * 600 + "."
        result = truncate_text(long_sentence)
        # Fallback: sentences[0] is the full text (no sentence breaks inside)
        assert result == long_sentence

    def test_exclamation_and_question_boundaries(self):
        text = "Watch out! " + "Are you okay? " + "A" * 500
        result = truncate_text(text)
        assert "Watch out!" in result
        assert len(result) <= 500

    def test_empty_text_returns_says_nothing(self):
        assert truncate_text("") == "*says nothing*"
        assert truncate_text("   ") == "*says nothing*"

    def test_exactly_500_chars_unchanged(self):
        text = "A" * 500
        result = truncate_text(text)
        assert result == text
        assert len(result) == 500
