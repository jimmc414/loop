"""Tests for ConversationEngine trust heuristic pattern matching."""

import re
from unittest.mock import MagicMock

import pytest

from loop.config import (
    TRUST_DELTA_ACCUSATION_HARSH,
    TRUST_DELTA_ACCUSATION_MILD,
    TRUST_DELTA_EMPATHY,
    TRUST_DELTA_IMPOSSIBLE_EXPLAINED,
    TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED,
    TRUST_DELTA_PRESSURE,
    TRUST_DELTA_REMEMBERED,
)
from loop.conversation_engine import ConversationEngine
from loop.models import (
    Character,
    CharacterTier,
    KnowledgeEntry,
    PersistentKnowledge,
    ScheduleEntry,
    TimeSlot,
    WorldState,
)
from tests.conftest import make_character, make_knowledge, make_world, TOTAL_SLOTS


def _make_conv_engine(characters=None, evidence_registry=None):
    """Create a ConversationEngine with a dummy display."""
    chars = characters or []
    world = make_world(characters=chars, evidence_registry=evidence_registry or [])
    display = MagicMock()
    return ConversationEngine(world, display)


def _calc_trust(engine, msg, character, history=None, knowledge=None):
    """Call _calculate_trust_delta with sensible defaults."""
    result = engine._calculate_trust_delta(
        msg,
        character,
        history or [],
        knowledge or make_knowledge(),
    )
    # Support both old int return and new (int, str) tuple return
    if isinstance(result, tuple):
        return result[0]
    return result


# ── Empathy patterns ─────────────────────────────────────────────────


class TestEmpathyPatterns:
    def test_sorry(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "I'm sorry about that", char)
        assert delta == TRUST_DELTA_EMPATHY

    def test_understand(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "I understand how you feel", char)
        assert delta == TRUST_DELTA_EMPATHY

    def test_how_are_you(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "how are you doing today?", char)
        assert delta == TRUST_DELTA_EMPATHY

    def test_must_be_hard(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "that must be hard for you", char)
        assert delta == TRUST_DELTA_EMPATHY

    def test_no_empathy_no_bonus(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "tell me about the lodge", char)
        assert delta == 0


# ── Harsh accusation ─────────────────────────────────────────────────


class TestHarshAccusation:
    def test_liar(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "you're a liar", char)
        assert delta == TRUST_DELTA_ACCUSATION_HARSH

    def test_guilty(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "you are guilty of this", char)
        assert delta == TRUST_DELTA_ACCUSATION_HARSH

    def test_murderer(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "you're a murderer", char)
        assert delta == TRUST_DELTA_ACCUSATION_HARSH

    def test_harsh_is_early_return(self):
        """Harsh accusation returns immediately, overriding everything else."""
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        # Message has empathy AND harsh accusation — harsh wins
        delta = _calc_trust(engine, "I'm sorry but you're a liar", char)
        assert delta == TRUST_DELTA_ACCUSATION_HARSH


# ── Mild accusation ──────────────────────────────────────────────────


class TestMildAccusation:
    def test_admit_it(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "just admit it already", char)
        assert delta == TRUST_DELTA_ACCUSATION_MILD

    def test_hiding_something(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "you're hiding something from me", char)
        assert delta == TRUST_DELTA_ACCUSATION_MILD


# ── Pressure ──────────────────────────────────────────────────────────


class TestPressure:
    def test_tell_me_now(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "tell me now what happened", char)
        assert delta == TRUST_DELTA_PRESSURE

    def test_you_must(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "you must tell me the truth", char)
        assert delta == TRUST_DELTA_PRESSURE

    def test_spit_it_out(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "spit it out already", char)
        assert delta == TRUST_DELTA_PRESSURE


# ── Remembered words ─────────────────────────────────────────────────


class TestRememberedWords:
    def test_echoing_npc_words_bonus(self):
        """If player echoes >= 2 non-common NPC words (>= 3 chars), +2."""
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        history = [{"player": "hi", "npc": "The generator is completely broken down"}]
        # Player echoes "generator" and "broken" (both >= 3 chars, non-common)
        delta = _calc_trust(engine, "I saw the generator was broken too", char, history=history)
        assert delta >= TRUST_DELTA_REMEMBERED

    def test_common_words_excluded(self):
        """Common words like 'that', 'this', 'with' don't count as echoes."""
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        history = [{"player": "hi", "npc": "I think that they would just have been here"}]
        # All NPC words >=3 chars are in the common_words set
        delta = _calc_trust(engine, "I think that they would have been there", char, history=history)
        # Should NOT get the remembered bonus (only common words overlap)
        assert delta == 0

    def test_short_words_excluded(self):
        """Words shorter than 3 chars don't count."""
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        history = [{"player": "hi", "npc": "Go to it by me"}]
        delta = _calc_trust(engine, "Go to it by me", char, history=history)
        # All words are <= 2 chars, so no remembered bonus
        assert delta == 0

    def test_no_history_no_bonus(self):
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "generator broken", char, history=[])
        assert delta == 0


# ── Impossible knowledge ─────────────────────────────────────────────


class TestImpossibleKnowledge:
    def test_explained_with_i_heard(self):
        """Secret word overlap >= 2 with 'I heard' explanation -> +3."""
        char = make_character(
            name="NPC",
            secrets=["secretly meeting someone at the boathouse nightly"],
        )
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(
            engine,
            "I heard you were meeting someone at the boathouse secretly",
            char,
        )
        assert delta == TRUST_DELTA_IMPOSSIBLE_EXPLAINED

    def test_explained_with_i_noticed(self):
        char = make_character(
            name="NPC",
            secrets=["secretly meeting someone at the boathouse nightly"],
        )
        engine = _make_conv_engine(characters=[char])
        # Words >= 5 chars in secret: secretly, meeting, someone, boathouse, nightly
        # Must overlap >= 2 of those exact words in the message
        delta = _calc_trust(
            engine,
            "I noticed you were meeting someone at the boathouse nightly",
            char,
        )
        assert delta == TRUST_DELTA_IMPOSSIBLE_EXPLAINED

    def test_unexplained_impossible_knowledge(self):
        """Secret word overlap >= 2 without explanation -> -2."""
        char = make_character(
            name="NPC",
            secrets=["secretly meeting someone at the boathouse nightly"],
        )
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(
            engine,
            "You were meeting someone at the boathouse secretly",
            char,
        )
        assert delta == TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED

    def test_no_overlap_no_effect(self):
        char = make_character(
            name="NPC",
            secrets=["secretly meeting someone at the boathouse nightly"],
        )
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "nice weather today", char)
        assert delta == 0

    def test_single_word_overlap_no_effect(self):
        """Need >= 2 overlapping words, 1 is not enough."""
        char = make_character(
            name="NPC",
            secrets=["secretly meeting someone at the boathouse nightly"],
        )
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "the boathouse is nice", char)
        assert delta == 0


# ── Compound interactions ────────────────────────────────────────────


class TestCompoundInteractions:
    def test_empathy_plus_remembered(self):
        """Empathy (+1) and remembered words (+2) stack."""
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        history = [{"player": "hi", "npc": "The generator is completely broken down"}]
        delta = _calc_trust(
            engine,
            "I'm sorry about the generator being broken",
            char,
            history=history,
        )
        assert delta == TRUST_DELTA_EMPATHY + TRUST_DELTA_REMEMBERED

    def test_harsh_overrides_empathy_and_remembered(self):
        """Harsh accusation early return means other bonuses don't apply."""
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        history = [{"player": "hi", "npc": "The generator is completely broken down"}]
        delta = _calc_trust(
            engine,
            "I'm sorry but you're a guilty liar who broke the generator",
            char,
            history=history,
        )
        assert delta == TRUST_DELTA_ACCUSATION_HARSH

    def test_mild_overrides_empathy(self):
        """Mild accusation is early return — empathy doesn't add."""
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "I'm sorry but you're hiding something", char)
        assert delta == TRUST_DELTA_ACCUSATION_MILD

    def test_pressure_overrides_empathy(self):
        """Pressure is early return — empathy doesn't add."""
        char = make_character(name="NPC")
        engine = _make_conv_engine(characters=[char])
        delta = _calc_trust(engine, "I understand but tell me now", char)
        assert delta == TRUST_DELTA_PRESSURE
