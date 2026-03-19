"""Tests for KnowledgeBase (loop/knowledge_base.py)."""

import pytest

from loop.knowledge_base import KnowledgeBase
from loop.models import (
    Evidence,
    EvidenceConnection,
    EvidenceType,
    PersistentKnowledge,
    TimeSlot,
)
from tests.conftest import make_knowledge, make_world


def _make_kb(evidence=None, knowledge=None):
    ev_list = evidence or []
    world = make_world(evidence_registry=ev_list)
    k = knowledge or make_knowledge()
    return KnowledgeBase(k, world)


# ── Evidence CRUD ─────────────────────────────────────────────────────


class TestEvidenceCRUD:
    def test_discover_evidence(self):
        ev = Evidence(id="ev1", type=EvidenceType.PHYSICAL, description="A clue")
        kb = _make_kb(evidence=[ev])
        result = kb.discover_evidence("ev1")
        assert result is not None
        assert result.id == "ev1"
        assert "ev1" in kb.knowledge.evidence_discovered

    def test_discover_duplicate_returns_none(self):
        ev = Evidence(id="ev1", type=EvidenceType.PHYSICAL, description="A clue")
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        kb = _make_kb(evidence=[ev], knowledge=knowledge)
        result = kb.discover_evidence("ev1")
        assert result is None

    def test_discover_nonexistent_returns_none(self):
        kb = _make_kb()
        result = kb.discover_evidence("nonexistent")
        assert result is None

    def test_has_evidence(self):
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        kb = _make_kb(knowledge=knowledge)
        assert kb.has_evidence("ev1") is True
        assert kb.has_evidence("ev2") is False

    def test_get_discovered_evidence(self):
        ev1 = Evidence(id="ev1", type=EvidenceType.PHYSICAL, description="Clue 1")
        ev2 = Evidence(id="ev2", type=EvidenceType.TESTIMONY, description="Clue 2")
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        kb = _make_kb(evidence=[ev1, ev2], knowledge=knowledge)
        discovered = kb.get_discovered_evidence()
        assert len(discovered) == 1
        assert discovered[0].id == "ev1"


# ── Schedule observations ────────────────────────────────────────────


class TestScheduleObservation:
    def test_record_schedule_observation_str_keys(self):
        """record_schedule_observation uses str keys."""
        kb = _make_kb()
        kb.record_schedule_observation("Alice", 5, "dining_hall")
        known = kb.knowledge.character_schedules_known
        assert "Alice" in known
        assert "5" in known["Alice"]
        assert known["Alice"]["5"] == "dining_hall"

    def test_multiple_observations(self):
        kb = _make_kb()
        kb.record_schedule_observation("Alice", 0, "main_lodge")
        kb.record_schedule_observation("Alice", 5, "dining_hall")
        known = kb.knowledge.character_schedules_known["Alice"]
        assert len(known) == 2
        assert known["0"] == "main_lodge"
        assert known["5"] == "dining_hall"


# ── Evidence connections ─────────────────────────────────────────────


class TestConnections:
    def test_add_connection(self):
        knowledge = make_knowledge(evidence_discovered=["ev1", "ev2"])
        kb = _make_kb(knowledge=knowledge)
        result = kb.add_connection("ev1", "ev2", "Both found at same spot")
        assert result is True
        assert len(kb.knowledge.evidence_connections) == 1

    def test_add_connection_undiscovered_fails(self):
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        kb = _make_kb(knowledge=knowledge)
        result = kb.add_connection("ev1", "ev2")
        assert result is False

    def test_add_duplicate_connection_fails(self):
        knowledge = make_knowledge(
            evidence_discovered=["ev1", "ev2"],
            evidence_connections=[
                EvidenceConnection(evidence_a="ev1", evidence_b="ev2")
            ],
        )
        kb = _make_kb(knowledge=knowledge)
        result = kb.add_connection("ev1", "ev2")
        assert result is False

    def test_add_duplicate_reversed_fails(self):
        """Connection {a, b} == {b, a} — duplicates detected either way."""
        knowledge = make_knowledge(
            evidence_discovered=["ev1", "ev2"],
            evidence_connections=[
                EvidenceConnection(evidence_a="ev1", evidence_b="ev2")
            ],
        )
        kb = _make_kb(knowledge=knowledge)
        result = kb.add_connection("ev2", "ev1")
        assert result is False

    def test_remove_connection(self):
        knowledge = make_knowledge(
            evidence_discovered=["ev1", "ev2"],
            evidence_connections=[
                EvidenceConnection(evidence_a="ev1", evidence_b="ev2")
            ],
        )
        kb = _make_kb(knowledge=knowledge)
        result = kb.remove_connection("ev1", "ev2")
        assert result is True
        assert len(kb.knowledge.evidence_connections) == 0

    def test_remove_nonexistent_connection(self):
        kb = _make_kb()
        result = kb.remove_connection("ev1", "ev2")
        assert result is False

    def test_confirm_connection(self):
        ev1 = Evidence(
            id="ev1", type=EvidenceType.PHYSICAL,
            description="A", connects_to=["ev2"],
        )
        ev2 = Evidence(id="ev2", type=EvidenceType.PHYSICAL, description="B")
        knowledge = make_knowledge(
            evidence_discovered=["ev1", "ev2"],
            evidence_connections=[
                EvidenceConnection(evidence_a="ev1", evidence_b="ev2")
            ],
        )
        kb = _make_kb(evidence=[ev1, ev2], knowledge=knowledge)
        result = kb.confirm_connection("ev1", "ev2")
        assert result is True
        assert kb.knowledge.evidence_connections[0].confirmed is True


# ── Theories ──────────────────────────────────────────────────────────


class TestTheories:
    def test_add_theory(self):
        kb = _make_kb()
        kb.add_theory("The butler did it")
        assert "The butler did it" in kb.knowledge.theories

    def test_add_duplicate_theory_ignored(self):
        kb = _make_kb()
        kb.add_theory("The butler did it")
        kb.add_theory("The butler did it")
        assert len(kb.knowledge.theories) == 1

    def test_remove_theory(self):
        knowledge = make_knowledge(theories=["Theory A", "Theory B"])
        kb = _make_kb(knowledge=knowledge)
        result = kb.remove_theory(0)
        assert result is True
        assert "Theory A" not in kb.knowledge.theories
        assert "Theory B" in kb.knowledge.theories

    def test_remove_theory_invalid_index(self):
        kb = _make_kb()
        result = kb.remove_theory(0)
        assert result is False


# ── Knowledge percentage ──────────────────────────────────────────────


class TestKnowledgePercentage:
    def test_zero_evidence(self):
        kb = _make_kb()
        assert kb.get_knowledge_percentage() == 0.0

    def test_all_discovered(self):
        ev1 = Evidence(id="ev1", type=EvidenceType.PHYSICAL, description="A")
        ev2 = Evidence(id="ev2", type=EvidenceType.PHYSICAL, description="B")
        knowledge = make_knowledge(evidence_discovered=["ev1", "ev2"])
        kb = _make_kb(evidence=[ev1, ev2], knowledge=knowledge)
        assert kb.get_knowledge_percentage() == pytest.approx(1.0)

    def test_half_discovered(self):
        ev1 = Evidence(id="ev1", type=EvidenceType.PHYSICAL, description="A")
        ev2 = Evidence(id="ev2", type=EvidenceType.PHYSICAL, description="B")
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        kb = _make_kb(evidence=[ev1, ev2], knowledge=knowledge)
        assert kb.get_knowledge_percentage() == pytest.approx(0.5)


# ── Player knowledge flags ───────────────────────────────────────────


class TestPlayerKnowledgeFlags:
    def test_future_schedule_knowledge_flagged(self):
        knowledge = make_knowledge(
            character_schedules_known={"Alice": {"0": "main_lodge", "10": "dining_hall"}}
        )
        kb = _make_kb(knowledge=knowledge)
        flags = kb.get_player_knowledge_flags("Alice", 5)
        # Slot 10 > 5, so it should be flagged
        assert any("later" in f for f in flags)

    def test_past_schedule_not_flagged(self):
        knowledge = make_knowledge(
            character_schedules_known={"Alice": {"0": "main_lodge"}}
        )
        kb = _make_kb(knowledge=knowledge)
        flags = kb.get_player_knowledge_flags("Alice", 5)
        assert len(flags) == 0
