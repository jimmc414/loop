"""Tests for NPC Déjà Vu: Cross-Loop Memory Echoes."""

from unittest.mock import MagicMock

import pytest

from loop.config import TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED
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
from loop.prompts.conversation import (
    _build_deja_vu_section,
    _build_time_loop_rule,
    build_system_prompt,
)
from tests.conftest import (
    make_character,
    make_engine,
    make_knowledge,
    make_loop_state,
    make_world,
    TOTAL_SLOTS,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_conv_engine(characters=None, evidence_registry=None):
    chars = characters or []
    world = make_world(characters=chars, evidence_registry=evidence_registry or [])
    display = MagicMock()
    return ConversationEngine(world, display)


def _calc_trust(engine, msg, character, history=None, knowledge=None):
    result = engine._calculate_trust_delta(
        msg, character, history or [], knowledge or make_knowledge(),
    )
    if isinstance(result, tuple):
        return result
    return result, ""


def _make_knowledge_entry():
    return KnowledgeEntry(available_topics=["camp life"], mood="neutral")


# ── Data model tests ─────────────────────────────────────────────────


class TestDataModel:
    def test_new_fields_default_empty(self):
        k = PersistentKnowledge()
        assert k.npc_interaction_counts == {}
        assert k.npc_previous_topics == {}

    def test_json_roundtrip(self):
        k = PersistentKnowledge(
            npc_interaction_counts={"Alice": 10, "Bob": 3},
            npc_previous_topics={"Alice": ["fire", "boathouse"]},
        )
        data = k.model_dump_json()
        k2 = PersistentKnowledge.model_validate_json(data)
        assert k2.npc_interaction_counts == {"Alice": 10, "Bob": 3}
        assert k2.npc_previous_topics == {"Alice": ["fire", "boathouse"]}

    def test_backwards_compat_missing_fields(self):
        """Old save data without new fields loads cleanly."""
        old_data = {
            "evidence_discovered": ["ev1"],
            "characters_met": ["Alice"],
        }
        k = PersistentKnowledge.model_validate(old_data)
        assert k.npc_interaction_counts == {}
        assert k.npc_previous_topics == {}


# ── Tracking tests (state_machine.py) ────────────────────────────────


class TestLoopResetTracking:
    def test_reset_accumulates_interaction_counts(self):
        knowledge = make_knowledge(characters_met=["Alice", "Bob"])
        loop = make_loop_state(
            conversations_this_loop={"Alice": 5, "Bob": 3},
            character_trust={"Alice": 20, "Bob": 20},
        )
        engine = make_engine(loop=loop, knowledge=knowledge)
        engine.reset_loop()
        assert engine.knowledge.npc_interaction_counts["Alice"] == 5
        assert engine.knowledge.npc_interaction_counts["Bob"] == 3

    def test_reset_accumulates_across_multiple_resets(self):
        knowledge = make_knowledge(characters_met=["Alice"])
        loop = make_loop_state(
            conversations_this_loop={"Alice": 5},
            character_trust={"Alice": 20},
        )
        engine = make_engine(loop=loop, knowledge=knowledge)
        engine.reset_loop()
        assert engine.knowledge.npc_interaction_counts["Alice"] == 5

        # Simulate second loop
        engine.loop.conversations_this_loop["Alice"] = 7
        engine.reset_loop()
        assert engine.knowledge.npc_interaction_counts["Alice"] == 12

    def test_reset_extracts_topics_from_journal(self):
        knowledge = make_knowledge(
            characters_met=["Alice"],
            conversation_journal=[
                "Loop 1, Day 2 MORNING: Spoke with Alice (3 exchanges). Trust: 20. Topics: fire, boathouse.",
            ],
        )
        loop = make_loop_state(
            loop_number=1,
            conversations_this_loop={"Alice": 3},
            character_trust={"Alice": 20},
        )
        engine = make_engine(loop=loop, knowledge=knowledge)
        engine.reset_loop()
        assert "fire" in engine.knowledge.npc_previous_topics.get("Alice", [])
        assert "boathouse" in engine.knowledge.npc_previous_topics.get("Alice", [])

    def test_reset_caps_topics_at_ten(self):
        # Create journal entries that would produce more than 10 topics
        topics = [f"topic{i}" for i in range(15)]
        entries = []
        for i in range(0, 15, 3):
            batch = ", ".join(topics[i:i+3])
            entries.append(
                f"Loop 1, Day 1 MORNING: Spoke with Alice (2 exchanges). Trust: 20. Topics: {batch}."
            )
        knowledge = make_knowledge(
            characters_met=["Alice"],
            conversation_journal=entries,
        )
        loop = make_loop_state(
            loop_number=1,
            conversations_this_loop={"Alice": 10},
            character_trust={"Alice": 20},
        )
        engine = make_engine(loop=loop, knowledge=knowledge)
        engine.reset_loop()
        assert len(engine.knowledge.npc_previous_topics.get("Alice", [])) <= 10

    def test_reset_deja_vu_trust_bonus(self):
        """NPC with 15+ interactions gets +5 extra trust on reset."""
        knowledge = make_knowledge(
            characters_met=["Alice", "Bob"],
            npc_interaction_counts={"Alice": 20, "Bob": 5},
        )
        loop = make_loop_state(
            conversations_this_loop={"Alice": 1, "Bob": 1},
            character_trust={"Alice": 20, "Bob": 20},
        )
        engine = make_engine(loop=loop, knowledge=knowledge)
        engine.reset_loop()
        # Alice: 20+ interactions → base trust + 5
        # Bob: 5 interactions → just base trust
        alice_trust = engine.loop.character_trust["Alice"]
        bob_trust = engine.loop.character_trust["Bob"]
        assert alice_trust == bob_trust + 5


# ── Prompt generation tests ──────────────────────────────────────────


class TestDejaVuPromptGeneration:
    def _make_char(self, tier=CharacterTier.TIER3, name="Alice"):
        return make_character(name=name, tier=tier)

    def _make_pk(self, counts=None, topics=None):
        return make_knowledge(
            npc_interaction_counts=counts or {},
            npc_previous_topics=topics or {},
        )

    def test_no_deja_vu_loop_1(self):
        char = self._make_char()
        pk = self._make_pk(counts={"Alice": 10})
        result = _build_deja_vu_section(char, pk, loop_number=1)
        assert result == ""

    def test_no_deja_vu_loop_2(self):
        char = self._make_char()
        pk = self._make_pk(counts={"Alice": 10})
        result = _build_deja_vu_section(char, pk, loop_number=2)
        assert result == ""

    def test_no_deja_vu_no_prior_interaction(self):
        char = self._make_char()
        pk = self._make_pk()  # empty counts
        result = _build_deja_vu_section(char, pk, loop_number=5)
        assert result == ""

    def test_subtle_loop_3(self):
        char = self._make_char()
        pk = self._make_pk(counts={"Alice": 5})
        result = _build_deja_vu_section(char, pk, loop_number=3)
        assert "SUBTLE" in result
        assert "Have we met" in result

    def test_growing_loop_4(self):
        char = self._make_char()
        pk = self._make_pk(counts={"Alice": 8})
        result = _build_deja_vu_section(char, pk, loop_number=4)
        assert "GROWING" in result
        assert "dreams" in result.lower()

    def test_growing_loop_5(self):
        char = self._make_char()
        pk = self._make_pk(counts={"Alice": 8})
        result = _build_deja_vu_section(char, pk, loop_number=5)
        assert "GROWING" in result

    def test_fracture_loop_6(self):
        char = self._make_char()
        pk = self._make_pk(counts={"Alice": 20})
        result = _build_deja_vu_section(char, pk, loop_number=6)
        assert "FRACTURE" in result

    def test_fracture_loop_7(self):
        char = self._make_char()
        pk = self._make_pk(counts={"Alice": 20})
        result = _build_deja_vu_section(char, pk, loop_number=7)
        assert "FRACTURE" in result

    def test_tier3_more_open(self):
        char = self._make_char(tier=CharacterTier.TIER3)
        pk = self._make_pk(counts={"Alice": 8})
        result = _build_deja_vu_section(char, pk, loop_number=4)
        assert "OPEN" in result

    def test_tier3_ally_high_interaction(self):
        char = self._make_char(tier=CharacterTier.TIER3)
        pk = self._make_pk(counts={"Alice": 20})
        result = _build_deja_vu_section(char, pk, loop_number=6)
        assert "ALLY" in result

    def test_tier2_conflicted(self):
        char = self._make_char(tier=CharacterTier.TIER2, name="Bob")
        pk = self._make_pk(counts={"Bob": 8})
        result = _build_deja_vu_section(char, pk, loop_number=4)
        assert "CONFLICTED" in result

    def test_tier2_terrified_fracture(self):
        char = self._make_char(tier=CharacterTier.TIER2, name="Bob")
        pk = self._make_pk(counts={"Bob": 10})
        result = _build_deja_vu_section(char, pk, loop_number=6)
        assert "TERRIFIED" in result

    def test_tier1_suppresses(self):
        char = self._make_char(tier=CharacterTier.TIER1, name="Villain")
        pk = self._make_pk(counts={"Villain": 8})
        result = _build_deja_vu_section(char, pk, loop_number=4)
        assert "SUPPRESS" in result or "guarded" in result.lower()

    def test_tier1_paranoid_fracture(self):
        char = self._make_char(tier=CharacterTier.TIER1, name="Villain")
        pk = self._make_pk(counts={"Villain": 20})
        result = _build_deja_vu_section(char, pk, loop_number=6)
        assert "PARANOID" in result

    def test_topics_injected(self):
        char = self._make_char()
        pk = self._make_pk(
            counts={"Alice": 8},
            topics={"Alice": ["fire", "boathouse", "missing key"]},
        )
        result = _build_deja_vu_section(char, pk, loop_number=4)
        assert "fire" in result
        assert "boathouse" in result

    def test_topics_injected_subtle(self):
        char = self._make_char()
        pk = self._make_pk(
            counts={"Alice": 5},
            topics={"Alice": ["fire", "boathouse"]},
        )
        result = _build_deja_vu_section(char, pk, loop_number=3)
        assert "fire" in result

    def test_time_loop_rule_unchanged_loop5(self):
        char = self._make_char(tier=CharacterTier.TIER3)
        pk = self._make_pk(counts={"Alice": 20})
        rule = _build_time_loop_rule(char, pk, loop_number=5)
        assert "do NOT know this is a time loop" in rule

    def test_time_loop_rule_unchanged_tier1_loop7(self):
        char = self._make_char(tier=CharacterTier.TIER1, name="Villain")
        pk = self._make_pk(counts={"Villain": 20})
        rule = _build_time_loop_rule(char, pk, loop_number=7)
        assert "do NOT know this is a time loop" in rule

    def test_time_loop_rule_changed_loop6_tier3_high(self):
        char = self._make_char(tier=CharacterTier.TIER3)
        pk = self._make_pk(counts={"Alice": 20})
        rule = _build_time_loop_rule(char, pk, loop_number=6)
        assert "deeply wrong with time" in rule

    def test_time_loop_rule_changed_loop7_tier3_high(self):
        char = self._make_char(tier=CharacterTier.TIER3)
        pk = self._make_pk(counts={"Alice": 20})
        rule = _build_time_loop_rule(char, pk, loop_number=7)
        assert "deeply wrong with time" in rule

    def test_time_loop_rule_unchanged_tier3_low_interaction(self):
        char = self._make_char(tier=CharacterTier.TIER3)
        pk = self._make_pk(counts={"Alice": 5})
        rule = _build_time_loop_rule(char, pk, loop_number=7)
        assert "do NOT know this is a time loop" in rule


# ── Full system prompt integration ───────────────────────────────────


class TestDejaVuInSystemPrompt:
    def test_deja_vu_appears_in_system_prompt_loop4(self):
        char = make_character(name="Alice", tier=CharacterTier.TIER3)
        pk = make_knowledge(npc_interaction_counts={"Alice": 8})
        kentry = KnowledgeEntry(available_topics=["camp life"])
        prompt = build_system_prompt(
            character=char, knowledge=kentry, trust_level=20,
            player_knowledge=pk, loop_number=4, day=1, slot="MORNING",
        )
        assert "DEJA VU" in prompt
        assert "GROWING" in prompt

    def test_no_deja_vu_in_system_prompt_loop1(self):
        char = make_character(name="Alice", tier=CharacterTier.TIER3)
        pk = make_knowledge(npc_interaction_counts={"Alice": 8})
        kentry = KnowledgeEntry(available_topics=["camp life"])
        prompt = build_system_prompt(
            character=char, knowledge=kentry, trust_level=20,
            player_knowledge=pk, loop_number=1, day=1, slot="MORNING",
        )
        assert "DEJA VU" not in prompt


# ── Trust mechanic tests ─────────────────────────────────────────────


class TestDejaVuTrustMechanic:
    def test_impossible_knowledge_no_penalty_with_deja_vu(self):
        """NPC with prior interactions: 0 penalty for impossible knowledge."""
        char = make_character(
            name="NPC",
            secrets=["The chemical storage facility is leaking toxic waste"],
        )
        engine = _make_conv_engine(characters=[char])
        pk = make_knowledge(npc_interaction_counts={"NPC": 5})
        delta, reason = _calc_trust(
            engine, "The chemical storage facility is leaking toxic waste", char,
            knowledge=pk,
        )
        assert delta == 0
        assert "déjà vu" in reason.lower()

    def test_impossible_knowledge_penalty_without_deja_vu(self):
        """NPC with no prior interactions: still gets -2 penalty."""
        char = make_character(
            name="NPC",
            secrets=["The chemical storage facility is leaking toxic waste"],
        )
        engine = _make_conv_engine(characters=[char])
        pk = make_knowledge()  # no interaction counts
        delta, reason = _calc_trust(
            engine, "The chemical storage facility is leaking toxic waste", char,
            knowledge=pk,
        )
        assert delta == TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED
        assert "unexplained" in reason.lower()

    def test_impossible_knowledge_explained_still_works(self):
        """Explained impossible knowledge still gives bonus regardless of déjà vu."""
        char = make_character(
            name="NPC",
            secrets=["The chemical storage facility is leaking toxic waste"],
        )
        engine = _make_conv_engine(characters=[char])
        pk = make_knowledge(npc_interaction_counts={"NPC": 5})
        delta, reason = _calc_trust(
            engine,
            "I heard the chemical storage facility is leaking toxic waste",
            char, knowledge=pk,
        )
        assert delta > 0
        assert "explained" in reason.lower()


# ── Impossible knowledge prompt modification tests ───────────────────


class TestImposibleKnowledgePromptModification:
    def test_tier3_less_suspicious_with_deja_vu(self):
        char = make_character(
            name="Alice", tier=CharacterTier.TIER3,
            secrets=["knows about the hidden tunnel"],
        )
        pk = make_knowledge(
            npc_interaction_counts={"Alice": 5},
            evidence_discovered=["alice_tunnel"],
        )
        kentry = KnowledgeEntry(available_topics=["camp life"])
        prompt = build_system_prompt(
            character=char, knowledge=kentry, trust_level=20,
            player_knowledge=pk, loop_number=4, day=1, slot="MORNING",
        )
        assert "LESS suspicious" in prompt

    def test_tier2_torn_with_deja_vu(self):
        char = make_character(
            name="Bob", tier=CharacterTier.TIER2,
            secrets=["knows about the hidden tunnel"],
        )
        pk = make_knowledge(
            npc_interaction_counts={"Bob": 5},
            evidence_discovered=["bob_tunnel"],
        )
        kentry = KnowledgeEntry(available_topics=["camp life"])
        prompt = build_system_prompt(
            character=char, knowledge=kentry, trust_level=20,
            player_knowledge=pk, loop_number=4, day=1, slot="MORNING",
        )
        assert "torn" in prompt.lower()

    def test_tier1_no_deja_vu_modification(self):
        char = make_character(
            name="Villain", tier=CharacterTier.TIER1,
            secrets=["planned the whole thing"],
        )
        pk = make_knowledge(
            npc_interaction_counts={"Villain": 20},
            evidence_discovered=["villain_planned"],
        )
        kentry = KnowledgeEntry(available_topics=["camp life"])
        prompt = build_system_prompt(
            character=char, knowledge=kentry, trust_level=20,
            player_knowledge=pk, loop_number=4, day=1, slot="MORNING",
        )
        assert "suspicion or confusion" in prompt or "LESS suspicious" not in prompt
