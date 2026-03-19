"""Tests for world validation (loop/world_generator.py)."""

import pytest

from loop.config import TOTAL_SLOTS
from loop.models import (
    CausalChainEvent,
    Character,
    CharacterTier,
    EndingCondition,
    EndingType,
    Evidence,
    EvidenceType,
    KnowledgeEntry,
    Location,
    ScheduleEntry,
    TimeSlot,
    WorldState,
)
from loop.world_generator import _validate_world
from tests.conftest import make_character, make_location, make_world


class TestScheduleValidation:
    def test_bad_schedule_length_detected(self):
        char = make_character(
            name="BadSched",
            schedule=[ScheduleEntry(location="main_lodge", activity="idle")] * 10,
        )
        world = make_world(characters=[char])
        issues = _validate_world(world)
        assert any("schedule entries" in issue and "BadSched" in issue for issue in issues)

    def test_correct_schedule_length_ok(self):
        char = make_character(name="GoodSched")  # default has TOTAL_SLOTS entries
        world = make_world(characters=[char])
        issues = _validate_world(world)
        assert not any("schedule entries" in issue for issue in issues)

    def test_bad_knowledge_timeline_length(self):
        char = make_character(
            name="BadKnow",
            knowledge_timeline=[KnowledgeEntry()] * 5,
        )
        world = make_world(characters=[char])
        issues = _validate_world(world)
        assert any("knowledge entries" in issue and "BadKnow" in issue for issue in issues)


class TestLocationReferences:
    def test_unknown_location_in_schedule(self):
        char = make_character(
            name="LostChar",
            schedule=[ScheduleEntry(location="nonexistent_place", activity="wandering")]
            * TOTAL_SLOTS,
        )
        world = make_world(characters=[char])
        issues = _validate_world(world)
        assert any("unknown location" in issue.lower() and "nonexistent_place" in issue
                    for issue in issues)


class TestCausalChainValidation:
    def test_chronological_order_violation(self):
        e1 = CausalChainEvent(
            id="e1", day=3, time_slot=TimeSlot.EVENING,
            character="V", action="a", location="main_lodge",
        )
        e2 = CausalChainEvent(
            id="e2", day=1, time_slot=TimeSlot.MORNING,
            character="V", action="b", location="main_lodge",
        )
        char = make_character(name="V")
        world = make_world(characters=[char], causal_chain=[e1, e2])
        issues = _validate_world(world)
        assert any("out of order" in issue.lower() for issue in issues)

    def test_chronological_order_correct(self):
        e1 = CausalChainEvent(
            id="e1", day=1, time_slot=TimeSlot.MORNING,
            character="V", action="a", location="main_lodge",
        )
        e2 = CausalChainEvent(
            id="e2", day=3, time_slot=TimeSlot.EVENING,
            character="V", action="b", location="main_lodge",
        )
        char = make_character(name="V")
        world = make_world(characters=[char], causal_chain=[e1, e2])
        issues = _validate_world(world)
        assert not any("out of order" in issue.lower() for issue in issues)

    def test_unknown_character_in_event(self):
        event = CausalChainEvent(
            id="e1", day=1, time_slot=TimeSlot.MORNING,
            character="Ghost", action="haunt", location="main_lodge",
        )
        world = make_world(causal_chain=[event])
        issues = _validate_world(world)
        assert any("unknown character" in issue.lower() for issue in issues)


class TestEndingConditions:
    def test_missing_ending_conditions(self):
        world = make_world(ending_conditions=[])
        issues = _validate_world(world)
        assert any("ending" in issue.lower() for issue in issues)

    def test_with_ending_conditions_ok(self):
        endings = [EndingCondition(type=EndingType.FAILURE, description="fail")]
        world = make_world(ending_conditions=endings)
        issues = _validate_world(world)
        assert not any("no ending" in issue.lower() for issue in issues)


class TestEvidencePrerequisiteCycles:
    def test_diamond_dag_not_false_cycle(self):
        """Diamond DAG (A->B, A->C, B->D, C->D) is NOT a cycle."""
        ev_a = Evidence(id="A", type=EvidenceType.PHYSICAL, description="A")
        ev_b = Evidence(
            id="B", type=EvidenceType.PHYSICAL, description="B",
            prerequisites=["A"],
        )
        ev_c = Evidence(
            id="C", type=EvidenceType.PHYSICAL, description="C",
            prerequisites=["A"],
        )
        ev_d = Evidence(
            id="D", type=EvidenceType.PHYSICAL, description="D",
            prerequisites=["B", "C"],
        )
        world = make_world(
            evidence_registry=[ev_a, ev_b, ev_c, ev_d],
            ending_conditions=[EndingCondition(type=EndingType.FAILURE, description="f")],
        )
        issues = _validate_world(world)
        assert not any("cycle" in issue.lower() for issue in issues)

    def test_actual_cycle_detected(self):
        """A -> B -> C -> A is a real cycle and must be detected."""
        ev_a = Evidence(
            id="A", type=EvidenceType.PHYSICAL, description="A",
            prerequisites=["C"],
        )
        ev_b = Evidence(
            id="B", type=EvidenceType.PHYSICAL, description="B",
            prerequisites=["A"],
        )
        ev_c = Evidence(
            id="C", type=EvidenceType.PHYSICAL, description="C",
            prerequisites=["B"],
        )
        world = make_world(
            evidence_registry=[ev_a, ev_b, ev_c],
            ending_conditions=[EndingCondition(type=EndingType.FAILURE, description="f")],
        )
        issues = _validate_world(world)
        assert any("cycle" in issue.lower() for issue in issues)

    def test_self_referencing_cycle(self):
        """An evidence that requires itself is a cycle."""
        ev = Evidence(
            id="A", type=EvidenceType.PHYSICAL, description="A",
            prerequisites=["A"],
        )
        world = make_world(
            evidence_registry=[ev],
            ending_conditions=[EndingCondition(type=EndingType.FAILURE, description="f")],
        )
        issues = _validate_world(world)
        assert any("cycle" in issue.lower() for issue in issues)

    def test_unknown_prerequisite(self):
        ev = Evidence(
            id="A", type=EvidenceType.PHYSICAL, description="A",
            prerequisites=["nonexistent"],
        )
        world = make_world(
            evidence_registry=[ev],
            ending_conditions=[EndingCondition(type=EndingType.FAILURE, description="f")],
        )
        issues = _validate_world(world)
        assert any("unknown prerequisite" in issue.lower() for issue in issues)
