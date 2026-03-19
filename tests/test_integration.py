"""Integration tests exercising multiple modules together."""

import pytest

from loop.config import DAYS_PER_LOOP, MAX_LOOPS, TOTAL_SLOTS, slot_index
from loop.intervention import InterventionManager
from loop.models import (
    CausalChainEvent,
    Character,
    CharacterTier,
    EndingCondition,
    EndingType,
    Evidence,
    EvidenceType,
    InterventionNode,
    KnowledgeEntry,
    LoopState,
    PersistentKnowledge,
    ScheduleEntry,
    TimeSlot,
    WorldState,
)
from loop.state_machine import ClockworkEngine
from tests.conftest import (
    make_character,
    make_engine,
    make_knowledge,
    make_location,
    make_loop_state,
    make_world,
)


class TestFullLoopRun:
    def test_twenty_slot_loop_completes(self):
        """A full 20-slot loop runs from start to end without error."""
        engine = make_engine()
        results = []
        for i in range(TOTAL_SLOTS):
            result = engine.advance_time()
            results.append(result)
        # Last result should be loop_end
        assert results[-1]["type"] == "loop_end"
        # All others should be slot_advance or day_advance
        for r in results[:-1]:
            assert r["type"] in ("slot_advance", "day_advance")

    def test_day_transitions_correct(self):
        """Over 20 slots, we go through 5 days."""
        engine = make_engine()
        days_seen = {1}
        for _ in range(TOTAL_SLOTS):
            engine.advance_time()
            days_seen.add(engine.loop.current_day)
        assert days_seen == {1, 2, 3, 4, 5}


class TestInterventionPrevents:
    def _build_intervention_scenario(self):
        char = make_character(name="Villain", tier=CharacterTier.TIER1)
        e1 = CausalChainEvent(
            id="ce1", day=1, time_slot=TimeSlot.MORNING,
            character="Villain", action="prepare", location="main_lodge",
            downstream_effects=["ce2"],
        )
        e2 = CausalChainEvent(
            id="ce2", day=3, time_slot=TimeSlot.AFTERNOON,
            character="Villain", action="escalate", location="main_lodge",
            downstream_effects=["ce3"],
        )
        e3 = CausalChainEvent(
            id="ce3", day=5, time_slot=TimeSlot.NIGHT,
            character="Villain", action="catastrophe", location="main_lodge",
        )
        iv = InterventionNode(
            id="iv1", causal_event_id="ce1",
            required_day=1, required_slot=TimeSlot.MORNING,
            required_location="main_lodge",
            required_evidence=["ev1"],
            cascade_interrupts=["ce2"],
        )
        ev = Evidence(
            id="ev1", type=EvidenceType.PHYSICAL,
            description="Trap evidence", source_location="main_lodge",
        )
        ending_full = EndingCondition(
            type=EndingType.FULL_PREVENTION,
            required_interrupted_events=["ce1", "ce2", "ce3"],
            description="Fully prevented",
        )
        ending_fail = EndingCondition(
            type=EndingType.FAILURE,
            description="Failed",
        )
        world = make_world(
            characters=[char],
            causal_chain=[e1, e2, e3],
            intervention_tree=[iv],
            evidence_registry=[ev],
            ending_conditions=[ending_full, ending_fail],
        )
        return world

    def test_intervention_prevents_catastrophe(self):
        """apply_intervention -> check_catastrophe returns False."""
        world = self._build_intervention_scenario()
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        engine = make_engine(world=world, knowledge=knowledge)

        # Apply the intervention
        result = engine.apply_intervention("iv1")
        assert result["success"] is True
        assert "ce1" in engine.loop.events_interrupted
        assert "ce2" in engine.loop.events_interrupted  # cascade

        # Manually interrupt ce3 as well (it would be caught by recalculate)
        engine.loop.events_interrupted.append("ce3")

        assert engine.check_catastrophe() is False

    def test_no_intervention_catastrophe_occurs(self):
        world = self._build_intervention_scenario()
        engine = make_engine(world=world)
        assert engine.check_catastrophe() is True


class TestLoopResetIntegration:
    def test_loop_reset_preserves_knowledge_grants_trust(self):
        """Loop reset: knowledge preserved, trust reset with +10 bonus for met chars."""
        char = make_character(name="Alice")
        world = make_world(characters=[char])
        knowledge = make_knowledge(
            evidence_discovered=["ev1"],
            characters_met=["Alice"],
        )
        loop = make_loop_state(
            character_trust={"Alice": 50},
            current_day=5,
            current_slot=TimeSlot.NIGHT,
        )
        engine = make_engine(world=world, loop=loop, knowledge=knowledge)

        result = engine.reset_loop()
        assert result is True

        # Knowledge preserved
        assert "ev1" in engine.knowledge.evidence_discovered
        # Trust reset with +10 bonus
        assert engine.loop.character_trust.get("Alice") == 10
        # Loop number incremented
        assert engine.loop.loop_number == 2
        # History recorded
        assert len(engine.knowledge.loop_history) == 1

    def test_second_loop_works(self):
        """After reset, a second loop can run."""
        engine = make_engine()
        engine.reset_loop()

        # Run the second loop
        for _ in range(TOTAL_SLOTS):
            result = engine.advance_time()
        assert result["type"] == "loop_end"
        assert engine.loop.loop_number == 2


class TestRecalculateCausalChain:
    def test_any_upstream_interrupts_downstream(self):
        """recalculate_causal_chain uses ANY upstream interrupted semantics."""
        char = make_character(name="V")
        # Two independent paths lead to e3
        e1 = CausalChainEvent(
            id="e1", day=1, time_slot=TimeSlot.MORNING,
            character="V", action="a", location="main_lodge",
            downstream_effects=["e3"],
        )
        e2 = CausalChainEvent(
            id="e2", day=2, time_slot=TimeSlot.MORNING,
            character="V", action="b", location="main_lodge",
            downstream_effects=["e3"],
        )
        e3 = CausalChainEvent(
            id="e3", day=5, time_slot=TimeSlot.NIGHT,
            character="V", action="catastrophe", location="main_lodge",
        )
        world = make_world(characters=[char], causal_chain=[e1, e2, e3])
        loop = make_loop_state(events_interrupted=["e1"])
        engine = make_engine(world=world, loop=loop)

        # Create a minimal InterventionManager to test recalculate
        from unittest.mock import MagicMock
        display = MagicMock()
        conv = MagicMock()
        mgr = InterventionManager(world, engine, display, conv)

        mgr.recalculate_causal_chain(loop)
        # ANY upstream interrupted -> e3 should be interrupted
        assert "e3" in loop.events_interrupted

    def test_no_upstream_interrupted_keeps_event(self):
        char = make_character(name="V")
        e1 = CausalChainEvent(
            id="e1", day=1, time_slot=TimeSlot.MORNING,
            character="V", action="a", location="main_lodge",
            downstream_effects=["e2"],
        )
        e2 = CausalChainEvent(
            id="e2", day=5, time_slot=TimeSlot.NIGHT,
            character="V", action="catastrophe", location="main_lodge",
        )
        world = make_world(characters=[char], causal_chain=[e1, e2])
        loop = make_loop_state()
        engine = make_engine(world=world, loop=loop)

        from unittest.mock import MagicMock
        display = MagicMock()
        conv = MagicMock()
        mgr = InterventionManager(world, engine, display, conv)

        mgr.recalculate_causal_chain(loop)
        assert "e2" not in loop.events_interrupted

    def test_cascade_propagation(self):
        """Interruption cascades through multiple levels."""
        char = make_character(name="V")
        e1 = CausalChainEvent(
            id="e1", day=1, time_slot=TimeSlot.MORNING,
            character="V", action="a", location="main_lodge",
            downstream_effects=["e2"],
        )
        e2 = CausalChainEvent(
            id="e2", day=2, time_slot=TimeSlot.MORNING,
            character="V", action="b", location="main_lodge",
            downstream_effects=["e3"],
        )
        e3 = CausalChainEvent(
            id="e3", day=5, time_slot=TimeSlot.NIGHT,
            character="V", action="c", location="main_lodge",
        )
        world = make_world(characters=[char], causal_chain=[e1, e2, e3])
        loop = make_loop_state(events_interrupted=["e1"])
        engine = make_engine(world=world, loop=loop)

        from unittest.mock import MagicMock
        display = MagicMock()
        conv = MagicMock()
        mgr = InterventionManager(world, engine, display, conv)

        mgr.recalculate_causal_chain(loop)
        assert "e2" in loop.events_interrupted
        assert "e3" in loop.events_interrupted


class TestScheduleModificationEndToEnd:
    def test_str_keys_through_advance_time(self):
        """Schedule modification str keys work end-to-end through advance_time."""
        char = make_character(name="Alice")
        # Modify slot 1 (AFTERNOON of day 1) to go to dining_hall
        loop = make_loop_state(
            schedule_modifications={
                "Alice": {
                    "1": ScheduleEntry(location="dining_hall", activity="eating lunch")
                }
            }
        )
        world = make_world(characters=[char])
        engine = make_engine(world=world, loop=loop)

        # Advance from MORNING to AFTERNOON
        engine.advance_time()
        assert engine.loop.current_slot == TimeSlot.AFTERNOON

        # Check Alice's location uses the modification
        entry = engine.get_character_location("Alice")
        assert entry.location == "dining_hall"
        assert entry.activity == "eating lunch"

    def test_intervention_schedule_changes_persist(self):
        """Schedule modifications from interventions use str keys and persist."""
        char = make_character(name="Villain", tier=CharacterTier.TIER1)
        event = CausalChainEvent(
            id="ce1", day=1, time_slot=TimeSlot.MORNING,
            character="Villain", action="x", location="main_lodge",
        )
        iv = InterventionNode(
            id="iv1", causal_event_id="ce1",
            required_day=1, required_slot=TimeSlot.MORNING,
            required_location="main_lodge",
            success_schedule_changes={
                "Villain": [
                    ScheduleEntry(location="cabin_row", activity="confined"),
                    ScheduleEntry(location="cabin_row", activity="resting"),
                ]
            },
        )
        world = make_world(
            characters=[char],
            causal_chain=[event],
            intervention_tree=[iv],
        )
        engine = make_engine(world=world)

        engine.apply_intervention("iv1")

        mods = engine.loop.schedule_modifications.get("Villain", {})
        assert len(mods) == 2
        # All keys should be strings
        for key in mods:
            assert isinstance(key, str)
        # The start index is 0 (current slot), so keys "0" and "1"
        assert mods["0"].location == "cabin_row"
        assert mods["1"].location == "cabin_row"

        # Verify the modification takes effect in schedule queries
        entry = engine.get_character_location("Villain", day=1, slot=TimeSlot.MORNING)
        assert entry.location == "cabin_row"
