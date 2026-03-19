"""Tests for the 'Present Evidence' confrontation mechanic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loop.config import CONFRONT_TRUST_THRESHOLD
from loop.conversation_engine import ConversationEngine
from loop.models import (
    CharacterTier,
    ConversationResult,
    Evidence,
    EvidenceType,
)
from loop.prompts.conversation import build_evidence_confrontation_prompt
from tests.conftest import make_character, make_knowledge, make_loop_state, make_world


# ── Helpers ──────────────────────────────────────────────────────────


def _make_evidence(**overrides):
    defaults = {
        "id": "ev_test",
        "type": EvidenceType.TESTIMONY,
        "description": "A suspicious letter mentioning secret plans",
        "source_character": "",
        "source_location": "main_lodge",
        "connects_to": [],
    }
    defaults.update(overrides)
    return Evidence(**defaults)


def _make_engine(characters=None, evidence_registry=None):
    chars = characters or []
    world = make_world(characters=chars, evidence_registry=evidence_registry or [])
    display = MagicMock()
    return ConversationEngine(world, display)


# ══════════════════════════════════════════════════════════════════════
# TestClassifyEvidenceRelevance
# ══════════════════════════════════════════════════════════════════════


class TestClassifyEvidenceRelevance:
    def test_source_match(self):
        """Evidence whose source_character matches NPC name -> 'source'."""
        char = make_character(name="Alice", tier=CharacterTier.TIER1)
        ev = _make_evidence(source_character="Alice")
        engine = _make_engine(characters=[char], evidence_registry=[ev])
        assert engine._classify_evidence_relevance(ev, char) == "source"

    def test_source_case_insensitive(self):
        char = make_character(name="alice", tier=CharacterTier.TIER1)
        ev = _make_evidence(source_character="Alice")
        engine = _make_engine(characters=[char], evidence_registry=[ev])
        assert engine._classify_evidence_relevance(ev, char) == "source"

    def test_subject_npc_name_in_description(self):
        """NPC name appears in evidence description -> 'subject'."""
        char = make_character(name="Bob", tier=CharacterTier.TIER2)
        ev = _make_evidence(description="Bob was seen near the boathouse at midnight")
        engine = _make_engine(characters=[char], evidence_registry=[ev])
        assert engine._classify_evidence_relevance(ev, char) == "subject"

    def test_subject_case_insensitive(self):
        char = make_character(name="bob", tier=CharacterTier.TIER2)
        ev = _make_evidence(description="BOB was seen near the lake")
        engine = _make_engine(characters=[char], evidence_registry=[ev])
        assert engine._classify_evidence_relevance(ev, char) == "subject"

    def test_connected_via_connects_to(self):
        """connects_to refs evidence whose source is this NPC -> 'connected'."""
        char = make_character(name="Carol", tier=CharacterTier.TIER2)
        ev_connected = _make_evidence(id="ev_carol", source_character="Carol")
        ev_main = _make_evidence(id="ev_main", connects_to=["ev_carol"])
        engine = _make_engine(
            characters=[char],
            evidence_registry=[ev_main, ev_connected],
        )
        assert engine._classify_evidence_relevance(ev_main, char) == "connected"

    def test_connected_via_keyword_overlap(self):
        """Keyword overlap between evidence description and NPC secrets -> 'connected'."""
        char = make_character(
            name="Dave",
            tier=CharacterTier.TIER2,
            secrets=["secretly stashing chemicals in the boathouse storage"],
        )
        ev = _make_evidence(description="Chemicals found in the boathouse storage area")
        engine = _make_engine(characters=[char], evidence_registry=[ev])
        assert engine._classify_evidence_relevance(ev, char) == "connected"

    def test_unrelated(self):
        """No match at all -> 'unrelated'."""
        char = make_character(name="Eve", tier=CharacterTier.TIER3)
        ev = _make_evidence(
            description="A torn map of the forest trails",
            source_character="Frank",
        )
        engine = _make_engine(characters=[char], evidence_registry=[ev])
        assert engine._classify_evidence_relevance(ev, char) == "unrelated"

    def test_empty_source_character(self):
        """Empty source_character should not match anyone as source."""
        char = make_character(name="Grace", tier=CharacterTier.TIER3)
        ev = _make_evidence(source_character="", description="Random note")
        engine = _make_engine(characters=[char], evidence_registry=[ev])
        result = engine._classify_evidence_relevance(ev, char)
        assert result != "source"


# ══════════════════════════════════════════════════════════════════════
# TestComputeEvidenceTrustDelta
# ══════════════════════════════════════════════════════════════════════


class TestComputeEvidenceTrustDelta:
    """One test per cell in the 4x3 trust delta matrix + threshold edges."""

    def _char(self, tier):
        return make_character(name="NPC", tier=tier)

    # ── source row ────────────────────────────────────────────────────

    def test_source_tier1(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("source", self._char(CharacterTier.TIER1), 50) == -5

    def test_source_tier2_low_trust(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("source", self._char(CharacterTier.TIER2), 20) == -2

    def test_source_tier2_high_trust(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("source", self._char(CharacterTier.TIER2), 40) == 3

    def test_source_tier3(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("source", self._char(CharacterTier.TIER3), 50) == 5

    # ── subject row ───────────────────────────────────────────────────

    def test_subject_tier1(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("subject", self._char(CharacterTier.TIER1), 50) == -3

    def test_subject_tier2_low(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("subject", self._char(CharacterTier.TIER2), 39) == -1

    def test_subject_tier2_high(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("subject", self._char(CharacterTier.TIER2), 40) == 2

    def test_subject_tier3(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("subject", self._char(CharacterTier.TIER3), 50) == 4

    # ── connected row ─────────────────────────────────────────────────

    def test_connected_tier1(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("connected", self._char(CharacterTier.TIER1), 50) == -2

    def test_connected_tier2_low(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("connected", self._char(CharacterTier.TIER2), 10) == 0

    def test_connected_tier2_high(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta("connected", self._char(CharacterTier.TIER2), 50) == 1

    # ── unrelated row ─────────────────────────────────────────────────

    def test_unrelated_all_zero(self):
        engine = _make_engine()
        for tier in CharacterTier:
            assert engine._compute_evidence_trust_delta("unrelated", self._char(tier), 50) == 0

    # ── threshold edge cases ──────────────────────────────────────────

    def test_tier2_at_exact_threshold(self):
        """Trust exactly at CONFRONT_TRUST_THRESHOLD uses high-trust branch."""
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta(
            "source", self._char(CharacterTier.TIER2), CONFRONT_TRUST_THRESHOLD,
        ) == 3

    def test_tier2_one_below_threshold(self):
        engine = _make_engine()
        assert engine._compute_evidence_trust_delta(
            "source", self._char(CharacterTier.TIER2), CONFRONT_TRUST_THRESHOLD - 1,
        ) == -2


# ══════════════════════════════════════════════════════════════════════
# TestGetTierGuidance
# ══════════════════════════════════════════════════════════════════════


class TestGetTierGuidance:
    """One test per cell in the 3x3 tier/trust_bracket matrix."""

    def test_tier1_low(self):
        char = make_character(tier=CharacterTier.TIER1)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 10)
        assert "defensive" in g.lower() or "deny" in g.lower()

    def test_tier1_mid(self):
        char = make_character(tier=CharacterTier.TIER1)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 45)
        assert "uneasy" in g.lower() or "composure" in g.lower()

    def test_tier1_high(self):
        char = make_character(tier=CharacterTier.TIER1)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 70)
        assert "shaken" in g.lower() or "cracks" in g.lower()

    def test_tier2_low(self):
        char = make_character(tier=CharacterTier.TIER2)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 10)
        assert "nervous" in g.lower() or "reluctant" in g.lower()

    def test_tier2_mid(self):
        char = make_character(tier=CharacterTier.TIER2)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 45)
        assert "conflicted" in g.lower()

    def test_tier2_high(self):
        char = make_character(tier=CharacterTier.TIER2)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 70)
        assert "crack" in g.lower() or "reveal" in g.lower()

    def test_tier3_low(self):
        char = make_character(tier=CharacterTier.TIER3)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 10)
        assert "curiosity" in g.lower() or "confusion" in g.lower()

    def test_tier3_mid(self):
        char = make_character(tier=CharacterTier.TIER3)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 45)
        assert "intrigued" in g.lower()

    def test_tier3_high(self):
        char = make_character(tier=CharacterTier.TIER3)
        engine = _make_engine(characters=[char])
        g = engine._get_tier_guidance(char, 70)
        assert "helpful" in g.lower() or "engage" in g.lower()


# ══════════════════════════════════════════════════════════════════════
# TestHandleEvidencePresentation
# ══════════════════════════════════════════════════════════════════════


class TestHandleEvidencePresentation:
    """Integration tests for _handle_evidence_presentation."""

    def _setup(self, char_tier=CharacterTier.TIER2, evidence_kwargs=None, extra_evidence=None):
        char = make_character(name="TestNPC", tier=char_tier)
        ev_kwargs = {"id": "ev_test", "source_character": "TestNPC"}
        ev_kwargs.update(evidence_kwargs or {})
        ev = _make_evidence(**ev_kwargs)
        registry = [ev] + (extra_evidence or [])
        world = make_world(characters=[char], evidence_registry=registry)
        display = MagicMock()
        engine = ConversationEngine(world, display)
        loop_state = make_loop_state(character_trust={"TestNPC": 50})
        knowledge = make_knowledge(evidence_discovered=["ev_test"])
        return engine, char, ev, loop_state, knowledge

    @pytest.mark.asyncio
    async def test_invalid_evidence_id(self):
        engine, char, ev, ls, k = self._setup()
        with patch.object(engine, '_get_npc_response', new_callable=AsyncMock):
            resp, delta, disc = await engine._handle_evidence_presentation(
                "nonexistent", char, 50, ls, k, [], "sys",
            )
        assert delta == 0
        assert disc == []
        assert "don't have" in resp.lower()

    @pytest.mark.asyncio
    async def test_undiscovered_evidence(self):
        engine, char, ev, ls, k = self._setup()
        k.evidence_discovered.clear()  # not discovered
        with patch.object(engine, '_get_npc_response', new_callable=AsyncMock):
            resp, delta, disc = await engine._handle_evidence_presentation(
                "ev_test", char, 50, ls, k, [], "sys",
            )
        assert delta == 0
        assert "haven't discovered" in resp.lower()

    @pytest.mark.asyncio
    async def test_valid_presentation_returns_response_and_delta(self):
        engine, char, ev, ls, k = self._setup()
        with patch.object(engine, '_get_npc_response', new_callable=AsyncMock, return_value="I... how did you get that?"):
            resp, delta, disc = await engine._handle_evidence_presentation(
                "ev_test", char, 50, ls, k, [], "sys",
            )
        assert resp == "I... how did you get that?"
        assert delta != 0  # source match at high trust should give non-zero

    @pytest.mark.asyncio
    async def test_evidence_discovery_on_source_high_trust(self):
        """Source relevance + high trust -> reveal first undiscovered connected evidence."""
        ev_connected = _make_evidence(id="ev_hidden", description="Hidden clue")
        engine, char, ev, ls, k = self._setup(
            evidence_kwargs={"connects_to": ["ev_hidden"]},
            extra_evidence=[ev_connected],
        )
        with patch.object(engine, '_get_npc_response', new_callable=AsyncMock, return_value="Fine..."):
            resp, delta, disc = await engine._handle_evidence_presentation(
                "ev_test", char, 50, ls, k, [], "sys",
            )
        assert disc == ["ev_hidden"]

    @pytest.mark.asyncio
    async def test_no_discovery_at_low_trust(self):
        """Source relevance + low trust -> no evidence discovered."""
        ev_connected = _make_evidence(id="ev_hidden", description="Hidden clue")
        engine, char, ev, ls, k = self._setup(
            evidence_kwargs={"connects_to": ["ev_hidden"]},
            extra_evidence=[ev_connected],
        )
        with patch.object(engine, '_get_npc_response', new_callable=AsyncMock, return_value="No."):
            resp, delta, disc = await engine._handle_evidence_presentation(
                "ev_test", char, 20, ls, k, [], "sys",  # low trust
            )
        assert disc == []

    @pytest.mark.asyncio
    async def test_schedule_disruption_tier1_source(self):
        """TIER1 + source relevance -> schedule modification."""
        engine, char, ev, ls, k = self._setup(char_tier=CharacterTier.TIER1)
        with patch.object(engine, '_get_npc_response', new_callable=AsyncMock, return_value="I deny everything."):
            await engine._handle_evidence_presentation(
                "ev_test", char, 50, ls, k, [], "sys",
            )
        # Should have schedule modification for next slot
        assert "TestNPC" in ls.schedule_modifications
        mods = ls.schedule_modifications["TestNPC"]
        # At least one modification exists
        assert len(mods) >= 1
        # The modification should point to a private location
        for idx_str, entry in mods.items():
            assert entry.location in ("old_boathouse", "storage_cellar", "maintenance_shed")

    @pytest.mark.asyncio
    async def test_no_schedule_disruption_tier2(self):
        """TIER2 + source relevance -> NO schedule modification."""
        engine, char, ev, ls, k = self._setup(char_tier=CharacterTier.TIER2)
        with patch.object(engine, '_get_npc_response', new_callable=AsyncMock, return_value="..."):
            await engine._handle_evidence_presentation(
                "ev_test", char, 50, ls, k, [], "sys",
            )
        assert "TestNPC" not in ls.schedule_modifications

    @pytest.mark.asyncio
    async def test_already_discovered_connected_skipped(self):
        """If connected evidence is already discovered, skip it."""
        ev_connected = _make_evidence(id="ev_known", description="Already known")
        engine, char, ev, ls, k = self._setup(
            evidence_kwargs={"connects_to": ["ev_known"]},
            extra_evidence=[ev_connected],
        )
        k.evidence_discovered.append("ev_known")  # already discovered
        with patch.object(engine, '_get_npc_response', new_callable=AsyncMock, return_value="..."):
            resp, delta, disc = await engine._handle_evidence_presentation(
                "ev_test", char, 50, ls, k, [], "sys",
            )
        assert disc == []

    @pytest.mark.asyncio
    async def test_history_passed_to_llm(self):
        """Conversation history is included in the LLM call."""
        engine, char, ev, ls, k = self._setup()
        mock_llm = AsyncMock(return_value="Response")
        history = [{"player": "hello", "npc": "hi there"}]
        with patch.object(engine, '_get_npc_response', mock_llm):
            await engine._handle_evidence_presentation(
                "ev_test", char, 50, ls, k, history, "sys",
            )
        # The user_prompt passed to LLM should contain history
        call_args = mock_llm.call_args
        user_prompt = call_args[0][1]  # second positional arg
        assert "hello" in user_prompt
        assert "hi there" in user_prompt


# ══════════════════════════════════════════════════════════════════════
# TestBuildEvidenceConfrontationPrompt
# ══════════════════════════════════════════════════════════════════════


class TestBuildEvidenceConfrontationPrompt:
    def _evidence(self):
        return _make_evidence(description="A torn letter mentioning the dock")

    def test_includes_evidence_description(self):
        prompt = build_evidence_confrontation_prompt(
            make_character(), self._evidence(), 50, "source",
            "Be defensive.",
        )
        assert "torn letter mentioning the dock" in prompt

    def test_includes_relevance(self):
        prompt = build_evidence_confrontation_prompt(
            make_character(), self._evidence(), 50, "connected",
            "Be curious.",
        )
        assert "connected" in prompt

    def test_includes_tier_guidance(self):
        guidance = "You are shaken and cracks show."
        prompt = build_evidence_confrontation_prompt(
            make_character(), self._evidence(), 50, "source", guidance,
        )
        assert guidance in prompt

    def test_includes_trust_context(self):
        prompt_high = build_evidence_confrontation_prompt(
            make_character(), self._evidence(), 60, "source", "guidance",
        )
        assert "high" in prompt_high

        prompt_low = build_evidence_confrontation_prompt(
            make_character(), self._evidence(), 20, "source", "guidance",
        )
        assert "low" in prompt_low


# ══════════════════════════════════════════════════════════════════════
# TestConversationResultExtension
# ══════════════════════════════════════════════════════════════════════


class TestConversationResultExtension:
    def test_default_evidence_presented(self):
        """New field has empty list default."""
        result = ConversationResult()
        assert result.evidence_presented == []

    def test_serialization_roundtrip(self):
        """Field survives JSON serialization."""
        result = ConversationResult(evidence_presented=["ev_a", "ev_b"])
        data = result.model_dump()
        restored = ConversationResult(**data)
        assert restored.evidence_presented == ["ev_a", "ev_b"]
