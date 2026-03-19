"""Tests for ClockworkEngine (loop/state_machine.py)."""

import random
from unittest.mock import patch

import pytest

from loop.config import (
    DAYS_PER_LOOP,
    FOLLOW_BASE_DETECTION,
    FOLLOW_HIGH_TRUST_REDUCTION,
    FOLLOW_TIER1_BONUS,
    MAX_LOOPS,
    SLOTS_PER_DAY,
    TOTAL_SLOTS,
    slot_index,
)
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
    Location,
    LoopState,
    PersistentKnowledge,
    ScheduleEntry,
    TimeSlot,
    WildCardEvent,
    WorldState,
)
from loop.state_machine import ClockworkEngine, SLOT_ORDER

# Re-use factory helpers from conftest (imported automatically by pytest)
from tests.conftest import (
    make_character,
    make_engine,
    make_knowledge,
    make_location,
    make_loop_state,
    make_world,
)


# ── Time advancement ─────────────────────────────────────────────────


class TestTimeAdvancement:
    def test_slot_advance_morning_to_afternoon(self):
        engine = make_engine()
        result = engine.advance_time()
        assert result["type"] == "slot_advance"
        assert engine.loop.current_slot == TimeSlot.AFTERNOON

    def test_slot_advance_afternoon_to_evening(self):
        engine = make_engine(loop=make_loop_state(current_slot=TimeSlot.AFTERNOON))
        engine.advance_time()
        assert engine.loop.current_slot == TimeSlot.EVENING

    def test_slot_advance_evening_to_night(self):
        engine = make_engine(loop=make_loop_state(current_slot=TimeSlot.EVENING))
        engine.advance_time()
        assert engine.loop.current_slot == TimeSlot.NIGHT

    def test_day_advance_night_to_next_morning(self):
        engine = make_engine(
            loop=make_loop_state(current_day=1, current_slot=TimeSlot.NIGHT)
        )
        result = engine.advance_time()
        assert result["type"] == "day_advance"
        assert engine.loop.current_day == 2
        assert engine.loop.current_slot == TimeSlot.MORNING

    def test_full_day_cycle(self):
        """MORNING -> AFTERNOON -> EVENING -> NIGHT -> next day MORNING."""
        engine = make_engine()
        assert engine.loop.current_slot == TimeSlot.MORNING
        engine.advance_time()
        assert engine.loop.current_slot == TimeSlot.AFTERNOON
        engine.advance_time()
        assert engine.loop.current_slot == TimeSlot.EVENING
        engine.advance_time()
        assert engine.loop.current_slot == TimeSlot.NIGHT
        engine.advance_time()
        assert engine.loop.current_day == 2
        assert engine.loop.current_slot == TimeSlot.MORNING

    def test_loop_end_at_day5_night(self):
        engine = make_engine(
            loop=make_loop_state(
                current_day=DAYS_PER_LOOP, current_slot=TimeSlot.NIGHT
            )
        )
        result = engine.advance_time()
        assert result["type"] == "loop_end"
        assert "catastrophe" in result

    def test_twenty_slots_total(self):
        """Advancing 19 times from start should reach end of loop on 20th."""
        engine = make_engine()
        for i in range(TOTAL_SLOTS - 1):
            result = engine.advance_time()
            assert result["type"] in ("slot_advance", "day_advance"), f"Failed at step {i}"
        # The 20th advance should end the loop
        result = engine.advance_time()
        assert result["type"] == "loop_end"


# ── Schedule queries ──────────────────────────────────────────────────


class TestScheduleQueries:
    def test_default_schedule(self):
        char = make_character(name="Alice")
        engine = make_engine(world=make_world(characters=[char]))
        entry = engine.get_character_location("Alice")
        assert entry is not None
        assert entry.location == "main_lodge"

    def test_wild_card_overrides_default(self):
        char = make_character(name="Alice")
        wc = WildCardEvent(
            id="wc1",
            day=1,
            time_slot=TimeSlot.MORNING,
            description="Storm!",
            schedule_overrides={
                "Alice": ScheduleEntry(location="dining_hall", activity="sheltering")
            },
        )
        engine = make_engine(world=make_world(characters=[char], wild_cards=[wc]))
        entry = engine.get_character_location("Alice", day=1, slot=TimeSlot.MORNING)
        assert entry.location == "dining_hall"

    def test_schedule_modification_overrides_wild_card(self):
        """schedule_modifications take priority over wild cards and defaults."""
        char = make_character(name="Alice")
        wc = WildCardEvent(
            id="wc1",
            day=1,
            time_slot=TimeSlot.MORNING,
            description="Storm!",
            schedule_overrides={
                "Alice": ScheduleEntry(location="dining_hall", activity="sheltering")
            },
        )
        loop = make_loop_state(
            schedule_modifications={
                "Alice": {
                    "0": ScheduleEntry(location="kitchen", activity="cooking")
                }
            }
        )
        engine = make_engine(
            world=make_world(characters=[char], wild_cards=[wc]), loop=loop
        )
        entry = engine.get_character_location("Alice", day=1, slot=TimeSlot.MORNING)
        assert entry.location == "kitchen"

    def test_schedule_modifications_use_str_keys(self):
        """schedule_modifications dict uses str keys, not int keys."""
        char = make_character(name="Bob")
        loop = make_loop_state(
            schedule_modifications={
                "Bob": {
                    "5": ScheduleEntry(location="lake_shore", activity="fishing")
                }
            }
        )
        engine = make_engine(world=make_world(characters=[char]), loop=loop)
        # slot_index(2, "AFTERNOON") = 5
        entry = engine.get_character_location("Bob", day=2, slot=TimeSlot.AFTERNOON)
        assert entry.location == "lake_shore"

    def test_get_characters_at_location(self):
        alice = make_character(name="Alice")
        bob = make_character(name="Bob")
        engine = make_engine(world=make_world(characters=[alice, bob]))
        chars = engine.get_characters_at_location("main_lodge")
        assert "Alice" in chars
        assert "Bob" in chars


# ── Follow mechanic ──────────────────────────────────────────────────


class TestFollow:
    def test_follow_tier1_higher_detection(self):
        """TIER1 chars get +0.15 detection bonus."""
        char = make_character(name="Villain", tier=CharacterTier.TIER1)
        engine = make_engine(world=make_world(characters=[char]))
        # Detection = BASE(0.15) + TIER1(0.15) = 0.30
        with patch("random.random", return_value=0.29):
            result = engine.resolve_follow("Villain")
            assert result.get("detected") is True

    def test_follow_high_trust_reduces_detection(self):
        """Trust >= 40 reduces detection by 0.15."""
        char = make_character(name="Friend", tier=CharacterTier.TIER3)
        loop = make_loop_state(character_trust={"Friend": 50})
        engine = make_engine(world=make_world(characters=[char]), loop=loop)
        # Detection = BASE(0.15) - HIGH_TRUST(0.15) = 0.00
        with patch("random.random", return_value=0.01):
            result = engine.resolve_follow("Friend")
            assert result["success"] is True

    def test_follow_tier3_base_detection(self):
        char = make_character(name="NPC")
        engine = make_engine(world=make_world(characters=[char]))
        # Detection = BASE(0.15)
        with patch("random.random", return_value=0.16):
            result = engine.resolve_follow("NPC")
            assert result["success"] is True

    def test_follow_detected_trust_penalty(self):
        char = make_character(name="NPC")
        loop = make_loop_state(character_trust={"NPC": 20})
        engine = make_engine(world=make_world(characters=[char]), loop=loop)
        with patch("random.random", return_value=0.01):
            result = engine.resolve_follow("NPC")
            assert result["detected"] is True
            assert engine.loop.character_trust["NPC"] == 15  # -5

    def test_follow_end_of_loop_fails(self):
        char = make_character(name="NPC")
        loop = make_loop_state(
            current_day=DAYS_PER_LOOP, current_slot=TimeSlot.NIGHT
        )
        engine = make_engine(world=make_world(characters=[char]), loop=loop)
        result = engine.resolve_follow("NPC")
        assert result["success"] is False
        assert "End of loop" in result["reason"]

    def test_follow_records_schedule_observation_str_key(self):
        """Follow records schedule observations using str keys."""
        schedule = [ScheduleEntry(location="main_lodge", activity="idle")] * TOTAL_SLOTS
        schedule[1] = ScheduleEntry(location="dining_hall", activity="eating")
        char = make_character(name="NPC", schedule=schedule)
        engine = make_engine(world=make_world(characters=[char]))
        with patch("random.random", return_value=0.99):  # not detected
            engine.resolve_follow("NPC")
        known = engine.knowledge.character_schedules_known.get("NPC", {})
        # next_idx = 0 + 1 = 1, key is str "1"
        assert "1" in known
        assert known["1"] == "dining_hall"

    def test_follow_nonexistent_character(self):
        engine = make_engine()
        result = engine.resolve_follow("Nobody")
        assert result["success"] is False


# ── Search mechanic ──────────────────────────────────────────────────


class TestSearch:
    def test_evidence_found_when_slot_matches(self):
        ev = Evidence(
            id="ev1",
            type=EvidenceType.PHYSICAL,
            description="A clue",
            source_location="main_lodge",
            available_day=1,
            available_slot=TimeSlot.MORNING,
        )
        engine = make_engine(world=make_world(evidence_registry=[ev]))
        result = engine.resolve_search("main_lodge")
        assert len(result["found"]) == 1
        assert result["found"][0].id == "ev1"

    def test_evidence_not_found_before_available_slot(self):
        ev = Evidence(
            id="ev1",
            type=EvidenceType.PHYSICAL,
            description="A clue",
            source_location="main_lodge",
            available_day=2,
            available_slot=TimeSlot.MORNING,
        )
        engine = make_engine(world=make_world(evidence_registry=[ev]))
        result = engine.resolve_search("main_lodge")
        assert len(result["found"]) == 0

    def test_evidence_prerequisites_must_be_discovered(self):
        ev = Evidence(
            id="ev2",
            type=EvidenceType.PHYSICAL,
            description="Hidden clue",
            source_location="main_lodge",
            available_day=1,
            available_slot=TimeSlot.MORNING,
            prerequisites=["ev1"],
        )
        engine = make_engine(world=make_world(evidence_registry=[ev]))
        result = engine.resolve_search("main_lodge")
        assert len(result["found"]) == 0

    def test_evidence_found_after_prereqs_discovered(self):
        ev = Evidence(
            id="ev2",
            type=EvidenceType.PHYSICAL,
            description="Hidden clue",
            source_location="main_lodge",
            available_day=1,
            available_slot=TimeSlot.MORNING,
            prerequisites=["ev1"],
        )
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        engine = make_engine(
            world=make_world(evidence_registry=[ev]), knowledge=knowledge
        )
        result = engine.resolve_search("main_lodge")
        assert len(result["found"]) == 1

    def test_already_discovered_evidence_skipped(self):
        ev = Evidence(
            id="ev1",
            type=EvidenceType.PHYSICAL,
            description="A clue",
            source_location="main_lodge",
            available_day=1,
            available_slot=TimeSlot.MORNING,
        )
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        engine = make_engine(
            world=make_world(evidence_registry=[ev]), knowledge=knowledge
        )
        result = engine.resolve_search("main_lodge")
        assert len(result["found"]) == 0

    def test_observers_are_tier1_and_tier2(self):
        t1 = make_character(name="Villain", tier=CharacterTier.TIER1)
        t2 = make_character(name="Witness", tier=CharacterTier.TIER2)
        t3 = make_character(name="Bystander", tier=CharacterTier.TIER3)
        engine = make_engine(world=make_world(characters=[t1, t2, t3]))
        result = engine.resolve_search("main_lodge")
        assert "Villain" in result["observers"]
        assert "Witness" in result["observers"]
        assert "Bystander" not in result["observers"]
        assert result["is_risky"] is True


# ── Travel ────────────────────────────────────────────────────────────


class TestTravel:
    def test_travel_to_adjacent(self):
        engine = make_engine()
        result = engine.resolve_travel("dining_hall")
        assert result["success"] is True
        assert engine.loop.player_location == "dining_hall"

    def test_travel_non_adjacent_fails(self):
        loc_a = make_location(id="main_lodge", name="Main Lodge", adjacent_locations=["dining_hall"])
        loc_b = make_location(id="dining_hall", name="Dining Hall", adjacent_locations=["main_lodge"])
        loc_c = make_location(id="kitchen", name="Kitchen", adjacent_locations=["dining_hall"])
        engine = make_engine(world=make_world(locations=[loc_a, loc_b, loc_c]))
        result = engine.resolve_travel("kitchen")
        assert result["success"] is False

    def test_travel_to_locked_location_fails(self):
        loc_a = make_location(id="main_lodge", name="Main Lodge", adjacent_locations=["vault"])
        loc_b = make_location(id="vault", name="Vault", locked=True, lock_conditions="Need a key", adjacent_locations=["main_lodge"])
        engine = make_engine(world=make_world(locations=[loc_a, loc_b]))
        result = engine.resolve_travel("vault")
        assert result["success"] is False
        assert "Locked" in result["reason"]

    def test_travel_to_distant_costs_slot(self):
        loc_a = make_location(id="main_lodge", name="Main Lodge", adjacent_locations=["lake_shore"])
        loc_b = make_location(id="lake_shore", name="Lake Shore", is_distant=True, adjacent_locations=["main_lodge"])
        engine = make_engine(world=make_world(locations=[loc_a, loc_b]))
        result = engine.resolve_travel("lake_shore")
        assert result["success"] is True
        assert result["costs_slot"] is True


# ── Interventions ─────────────────────────────────────────────────────


class TestInterventions:
    def _make_intervention_world(self):
        char = make_character(name="Villain", tier=CharacterTier.TIER1)
        event = CausalChainEvent(
            id="ce1",
            day=1,
            time_slot=TimeSlot.MORNING,
            character="Villain",
            action="set trap",
            location="main_lodge",
            downstream_effects=["ce2"],
        )
        event2 = CausalChainEvent(
            id="ce2",
            day=2,
            time_slot=TimeSlot.MORNING,
            character="Villain",
            action="trigger trap",
            location="main_lodge",
        )
        intervention = InterventionNode(
            id="iv1",
            causal_event_id="ce1",
            required_evidence=["ev1"],
            required_location="main_lodge",
            required_day=1,
            required_slot=TimeSlot.MORNING,
            action_description="Disarm the trap",
            cascade_interrupts=["ce2"],
        )
        ev = Evidence(
            id="ev1",
            type=EvidenceType.PHYSICAL,
            description="Trap evidence",
            source_location="main_lodge",
        )
        return make_world(
            characters=[char],
            causal_chain=[event, event2],
            intervention_tree=[intervention],
            evidence_registry=[ev],
        )

    def test_intervention_requires_evidence(self):
        world = self._make_intervention_world()
        engine = make_engine(world=world)
        available = engine._get_available_interventions()
        assert len(available) == 0

    def test_intervention_available_with_evidence(self):
        world = self._make_intervention_world()
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        engine = make_engine(world=world, knowledge=knowledge)
        available = engine._get_available_interventions()
        assert len(available) == 1
        assert available[0].id == "iv1"

    def test_intervention_requires_location(self):
        world = self._make_intervention_world()
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        loop = make_loop_state(player_location="dining_hall")
        engine = make_engine(world=world, loop=loop, knowledge=knowledge)
        available = engine._get_available_interventions()
        assert len(available) == 0

    def test_intervention_requires_correct_day_slot(self):
        world = self._make_intervention_world()
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        loop = make_loop_state(current_day=2)
        engine = make_engine(world=world, loop=loop, knowledge=knowledge)
        available = engine._get_available_interventions()
        assert len(available) == 0

    def test_apply_intervention_interrupts_event(self):
        world = self._make_intervention_world()
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        engine = make_engine(world=world, knowledge=knowledge)
        result = engine.apply_intervention("iv1")
        assert result["success"] is True
        assert "ce1" in engine.loop.events_interrupted

    def test_apply_intervention_cascade_interrupts(self):
        world = self._make_intervention_world()
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        engine = make_engine(world=world, knowledge=knowledge)
        engine.apply_intervention("iv1")
        assert "ce2" in engine.loop.events_interrupted

    def test_apply_intervention_schedule_modifications_str_keys(self):
        """Intervention schedule changes use str keys."""
        char = make_character(name="Villain", tier=CharacterTier.TIER1)
        event = CausalChainEvent(
            id="ce1", day=1, time_slot=TimeSlot.MORNING,
            character="Villain", action="set trap", location="main_lodge",
        )
        intervention = InterventionNode(
            id="iv1", causal_event_id="ce1",
            required_day=1, required_slot=TimeSlot.MORNING,
            required_location="main_lodge",
            success_schedule_changes={
                "Villain": [ScheduleEntry(location="cabin_row", activity="resting")]
            },
        )
        world = make_world(
            characters=[char], causal_chain=[event],
            intervention_tree=[intervention],
        )
        engine = make_engine(world=world)
        engine.apply_intervention("iv1")
        mods = engine.loop.schedule_modifications
        assert "Villain" in mods
        # Keys should be str
        for key in mods["Villain"]:
            assert isinstance(key, str)

    def test_intervention_requires_trust(self):
        char = make_character(name="Villain", tier=CharacterTier.TIER1)
        event = CausalChainEvent(
            id="ce1", day=1, time_slot=TimeSlot.MORNING,
            character="Villain", action="set trap", location="main_lodge",
        )
        intervention = InterventionNode(
            id="iv1", causal_event_id="ce1",
            required_day=1, required_slot=TimeSlot.MORNING,
            required_location="main_lodge",
            trust_required=50,
        )
        world = make_world(
            characters=[char], causal_chain=[event],
            intervention_tree=[intervention],
        )
        loop = make_loop_state(character_trust={"Villain": 30})
        engine = make_engine(world=world, loop=loop)
        available = engine._get_available_interventions()
        assert len(available) == 0

    def test_intervention_trust_met(self):
        char = make_character(name="Villain", tier=CharacterTier.TIER1)
        event = CausalChainEvent(
            id="ce1", day=1, time_slot=TimeSlot.MORNING,
            character="Villain", action="set trap", location="main_lodge",
        )
        intervention = InterventionNode(
            id="iv1", causal_event_id="ce1",
            required_day=1, required_slot=TimeSlot.MORNING,
            required_location="main_lodge",
            trust_required=50,
        )
        world = make_world(
            characters=[char], causal_chain=[event],
            intervention_tree=[intervention],
        )
        loop = make_loop_state(character_trust={"Villain": 60})
        engine = make_engine(world=world, loop=loop)
        available = engine._get_available_interventions()
        assert len(available) == 1


# ── Catastrophe check ────────────────────────────────────────────────


class TestCatastrophe:
    def test_chain_intact_no_interruptions(self):
        event = CausalChainEvent(
            id="final", day=5, time_slot=TimeSlot.NIGHT,
            character="Villain", action="catastrophe", location="main_lodge",
        )
        world = make_world(causal_chain=[event])
        engine = make_engine(world=world)
        assert engine.check_catastrophe() is True

    def test_chain_broken_by_interruption(self):
        event = CausalChainEvent(
            id="final", day=5, time_slot=TimeSlot.NIGHT,
            character="Villain", action="catastrophe", location="main_lodge",
        )
        world = make_world(causal_chain=[event])
        loop = make_loop_state(events_interrupted=["final"])
        engine = make_engine(world=world, loop=loop)
        assert engine.check_catastrophe() is False

    def test_chain_intact_uses_memo(self):
        """_chain_intact takes a memo dict for memoization."""
        e1 = CausalChainEvent(
            id="e1", day=1, time_slot=TimeSlot.MORNING,
            character="V", action="a", location="main_lodge",
            downstream_effects=["e2"],
        )
        e2 = CausalChainEvent(
            id="e2", day=5, time_slot=TimeSlot.NIGHT,
            character="V", action="b", location="main_lodge",
        )
        world = make_world(causal_chain=[e1, e2])
        engine = make_engine(world=world)
        memo: dict[str, bool] = {}
        result = engine._chain_intact("e2", memo)
        assert result is True
        # memo should be populated
        assert "e2" in memo
        assert "e1" in memo

    def test_chain_intact_memo_caches_false(self):
        e1 = CausalChainEvent(
            id="e1", day=1, time_slot=TimeSlot.MORNING,
            character="V", action="a", location="main_lodge",
            downstream_effects=["e2"],
        )
        e2 = CausalChainEvent(
            id="e2", day=5, time_slot=TimeSlot.NIGHT,
            character="V", action="b", location="main_lodge",
        )
        world = make_world(causal_chain=[e1, e2])
        loop = make_loop_state(events_interrupted=["e1"])
        engine = make_engine(world=world, loop=loop)
        memo: dict[str, bool] = {}
        result = engine._chain_intact("e2", memo)
        assert result is False
        assert memo["e1"] is False
        assert memo["e2"] is False

    def test_chain_intact_upstream_check(self):
        """If an upstream event is interrupted, the chain is broken."""
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
        world = make_world(causal_chain=[e1, e2, e3])
        loop = make_loop_state(events_interrupted=["e1"])
        engine = make_engine(world=world, loop=loop)
        assert engine.check_catastrophe() is False


# ── Ending evaluation ────────────────────────────────────────────────


class TestEndingEvaluation:
    def test_deeper_truth_highest_priority(self):
        endings = [
            EndingCondition(type=EndingType.FULL_PREVENTION, description="Full prevention"),
            EndingCondition(type=EndingType.DEEPER_TRUTH, description="Deeper truth"),
        ]
        engine = make_engine(world=make_world(ending_conditions=endings))
        result = engine.evaluate_ending()
        assert result.type == EndingType.DEEPER_TRUTH

    def test_full_prevention_over_partial(self):
        endings = [
            EndingCondition(type=EndingType.PARTIAL_PREVENTION, description="Partial"),
            EndingCondition(type=EndingType.FULL_PREVENTION, description="Full"),
        ]
        engine = make_engine(world=make_world(ending_conditions=endings))
        result = engine.evaluate_ending()
        assert result.type == EndingType.FULL_PREVENTION

    def test_ending_requires_interrupted_events(self):
        endings = [
            EndingCondition(
                type=EndingType.FULL_PREVENTION,
                required_interrupted_events=["ce1"],
                description="Full prevention",
            ),
        ]
        engine = make_engine(world=make_world(ending_conditions=endings))
        result = engine.evaluate_ending()
        # Should default to failure since ce1 is not interrupted
        assert result.type == EndingType.FAILURE

    def test_ending_requires_evidence(self):
        endings = [
            EndingCondition(
                type=EndingType.DEEPER_TRUTH,
                required_evidence=["ev1", "ev2"],
                description="Full truth",
            ),
        ]
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        engine = make_engine(
            world=make_world(ending_conditions=endings), knowledge=knowledge
        )
        result = engine.evaluate_ending()
        assert result.type == EndingType.FAILURE

    def test_ending_met_with_evidence_and_events(self):
        endings = [
            EndingCondition(
                type=EndingType.FULL_PREVENTION,
                required_interrupted_events=["ce1"],
                required_evidence=["ev1"],
                description="Full prevention",
            ),
        ]
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        loop = make_loop_state(events_interrupted=["ce1"])
        engine = make_engine(
            world=make_world(ending_conditions=endings),
            loop=loop,
            knowledge=knowledge,
        )
        result = engine.evaluate_ending()
        assert result.type == EndingType.FULL_PREVENTION

    def test_priority_order_all_five(self):
        """Priority: deeper_truth > full_prevention > partial > exposure > failure."""
        endings = [
            EndingCondition(type=EndingType.FAILURE, description="fail"),
            EndingCondition(type=EndingType.EXPOSURE, description="exposure"),
            EndingCondition(type=EndingType.PARTIAL_PREVENTION, description="partial"),
            EndingCondition(type=EndingType.FULL_PREVENTION, description="full"),
            EndingCondition(type=EndingType.DEEPER_TRUTH, description="truth"),
        ]
        engine = make_engine(world=make_world(ending_conditions=endings))
        result = engine.evaluate_ending()
        assert result.type == EndingType.DEEPER_TRUTH

    def test_default_failure_when_no_endings_match(self):
        engine = make_engine(world=make_world(ending_conditions=[]))
        result = engine.evaluate_ending()
        assert result.type == EndingType.FAILURE


# ── Loop reset ────────────────────────────────────────────────────────


class TestLoopReset:
    def test_reset_increments_loop_number(self):
        engine = make_engine()
        engine.reset_loop()
        assert engine.loop.loop_number == 2

    def test_reset_clears_mutable_state(self):
        loop = make_loop_state(
            current_day=3,
            current_slot=TimeSlot.EVENING,
            player_location="dining_hall",
            interventions_made=["iv1"],
            schedule_modifications={"A": {"0": ScheduleEntry(location="x", activity="y")}},
            events_triggered=["wc1"],
            events_interrupted=["ce1"],
            conversations_this_loop={"Bob": 3},
            active_claims=[],
            npc_heard_claims={"Bob": ["c1"]},
        )
        engine = make_engine(loop=loop)
        engine.reset_loop()
        assert engine.loop.current_day == 1
        assert engine.loop.current_slot == TimeSlot.MORNING
        assert engine.loop.player_location == "main_lodge"
        assert len(engine.loop.interventions_made) == 0
        assert len(engine.loop.schedule_modifications) == 0
        assert len(engine.loop.events_triggered) == 0
        assert len(engine.loop.events_interrupted) == 0
        assert len(engine.loop.conversations_this_loop) == 0
        assert len(engine.loop.active_claims) == 0
        assert len(engine.loop.npc_heard_claims) == 0

    def test_reset_preserves_knowledge(self):
        knowledge = make_knowledge(
            evidence_discovered=["ev1", "ev2"],
            characters_met=["Alice", "Bob"],
        )
        engine = make_engine(knowledge=knowledge)
        engine.reset_loop()
        assert "ev1" in engine.knowledge.evidence_discovered
        assert "Alice" in engine.knowledge.characters_met

    def test_reset_appends_loop_history(self):
        engine = make_engine()
        engine.reset_loop()
        assert len(engine.knowledge.loop_history) == 1
        assert engine.knowledge.loop_history[0].loop_number == 1

    def test_reset_max_loops_enforced(self):
        loop = make_loop_state(loop_number=MAX_LOOPS)
        engine = make_engine(loop=loop)
        result = engine.reset_loop()
        assert result is False

    def test_reset_just_under_max_loops(self):
        loop = make_loop_state(loop_number=MAX_LOOPS - 1)
        engine = make_engine(loop=loop)
        result = engine.reset_loop()
        assert result is True

    def test_reset_trust_bonus_for_met_characters(self):
        """reset_loop grants +10 trust for previously met characters."""
        knowledge = make_knowledge(characters_met=["Alice", "Bob"])
        loop = make_loop_state(character_trust={"Alice": 30, "Bob": 50, "Charlie": 10})
        engine = make_engine(loop=loop, knowledge=knowledge)
        engine.reset_loop()
        assert engine.loop.character_trust.get("Alice") == 10
        assert engine.loop.character_trust.get("Bob") == 10
        # Charlie not in characters_met, so no trust bonus
        assert engine.loop.character_trust.get("Charlie") is None


# ── Available actions ─────────────────────────────────────────────────


class TestAvailableActions:
    def test_search_always_available(self):
        """Search action is always available (dead if/else removed)."""
        engine = make_engine()
        actions = engine.get_available_actions()
        action_types = [a["type"] for a in actions]
        assert "search" in action_types

    def test_observe_always_available(self):
        engine = make_engine()
        actions = engine.get_available_actions()
        action_types = [a["type"] for a in actions]
        assert "observe" in action_types
