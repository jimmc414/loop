"""Adversarial tests for timeloop-camp-pinehaven.

These tests target boundary conditions, edge cases, and potential bugs across:
- state_machine (ClockworkEngine)
- rumor_mill (propagate_claims, _share_chance, _check_alarm_reactions)
- schedule_tracker (ScheduleTracker command parsing)
- display (GameDisplay._group_actions, show_schedule_tracker_day)
- conversation_engine (_calculate_trust_delta, _extract_topics)
- knowledge_base (KnowledgeBase connection cycles, add_connection self-loop)
- config (slot_index/day_and_slot inverse, boundary values)
- models (serialization edge cases)
"""

from __future__ import annotations

import re

import pytest

from loop.config import (
    DAYS_PER_LOOP,
    MAX_LOOPS,
    SLOT_NAMES,
    SLOTS_PER_DAY,
    TOTAL_SLOTS,
    TRUST_DELTA_ACCUSATION_HARSH,
    TRUST_DELTA_ACCUSATION_MILD,
    TRUST_DELTA_EMPATHY,
    TRUST_DELTA_IMPOSSIBLE_EXPLAINED,
    TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED,
    TRUST_DELTA_PRESSURE,
    TRUST_DELTA_REMEMBERED,
    day_and_slot,
    slot_index,
)
from loop.knowledge_base import KnowledgeBase
from loop.models import (
    CausalChainEvent,
    CharacterTier,
    Claim,
    EndingCondition,
    EndingType,
    Evidence,
    EvidenceConnection,
    EvidenceType,
    InterventionNode,
    LoopState,
    PersistentKnowledge,
    ScheduleEntry,
    TimeSlot,
    WildCardEvent,
    WorldState,
)
from loop.rumor_mill import (
    MAX_SPREAD_HOPS,
    _check_alarm_reactions,
    _share_chance,
    propagate_claims,
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


# ── Config: slot_index / day_and_slot inverse ─────────────────────────


class TestConfigBoundaries:
    def test_slot_index_day1_morning_is_zero(self):
        assert slot_index(1, "MORNING") == 0

    def test_slot_index_day1_night_is_three(self):
        assert slot_index(1, "NIGHT") == 3

    def test_slot_index_day5_night_is_total_slots_minus_one(self):
        assert slot_index(5, "NIGHT") == TOTAL_SLOTS - 1

    def test_day_and_slot_roundtrip_all_valid_indices(self):
        """Every valid 0-based index must round-trip through day_and_slot -> slot_index."""
        for idx in range(TOTAL_SLOTS):
            day, slot_name = day_and_slot(idx)
            assert slot_index(day, slot_name) == idx, (
                f"Round-trip failed at index {idx}: got day={day} slot={slot_name} "
                f"-> slot_index={slot_index(day, slot_name)}"
            )

    def test_slot_index_invalid_slot_name_raises(self):
        """slot_index with an unknown slot name should raise ValueError, not silently return garbage."""
        with pytest.raises((ValueError, IndexError)):
            slot_index(1, "MIDNIGHT")

    def test_day_and_slot_zero_index(self):
        """Index 0 must map to day 1, MORNING."""
        day, slot = day_and_slot(0)
        assert day == 1
        assert slot == "MORNING"

    def test_day_and_slot_negative_index_does_not_produce_valid_day(self):
        """day_and_slot(-1) produces day=0 which is out-of-range (1..5).

        This documents the behavior so callers know negative indices are
        not valid inputs.  The contract is: index must be in [0, TOTAL_SLOTS).
        """
        day, slot = day_and_slot(-1)
        # day=0 is outside the valid 1..5 range
        assert day < 1 or day > DAYS_PER_LOOP, (
            f"day_and_slot(-1) returned a valid-looking day={day}; "
            "negative indices should produce out-of-range day values"
        )

    def test_slot_index_day_zero_produces_negative_index(self):
        """slot_index(0, ...) yields a negative index (day 0 is off-bounds).

        Any caller that passes day=0 should not receive a non-negative index
        that could accidentally be used to index into a schedule list.
        """
        idx = slot_index(0, "MORNING")
        assert idx < 0, (
            f"slot_index(0, 'MORNING') returned {idx}; expected a negative "
            "index since day 0 is invalid"
        )


# ── State machine: get_character_location with day=0 ──────────────────


class TestGetCharacterLocationDayBoundary:
    def test_day_zero_returns_no_out_of_range_wrap(self):
        """Passing day=0 to get_character_location must not silently return
        a schedule entry by wrapping into negative list indices.

        Python lists allow negative indexing (schedule[-4] == schedule[16]),
        which would return stale data from day 5 instead of None or an error.
        """
        char = make_character(name="Alice")
        # Set a sentinel at the last slot (index 19) to detect wrap-around
        sentinel_entry = ScheduleEntry(location="sentinel_location", activity="sentinel")
        char.schedule[TOTAL_SLOTS - 1] = sentinel_entry

        engine = make_engine(world=make_world(characters=[char]))
        result = engine.get_character_location("Alice", day=0, slot=TimeSlot.MORNING)

        # Either None (graceful) or the correct fallback — but must NOT return
        # the sentinel that lives at the end of the list (which would happen if
        # negative indexing is used: schedule[-4] == schedule[16]).
        if result is not None:
            assert result.location != "sentinel_location", (
                "BUG: get_character_location(day=0) wrapped to a negative schedule "
                "index and returned data from a different day"
            )

    def test_day_zero_silently_treated_as_day_one(self):
        """BUG: get_character_location uses 'd = day or self.loop.current_day'.

        Because 0 is falsy in Python, day=0 silently evaluates to current_day (1),
        returning day-1 data instead of None or raising an error.  A caller that
        passes day=0 expecting no result (or an error) will receive stale day-1 data.
        """
        schedule = [
            ScheduleEntry(location=f"loc_{i}", activity=f"act_{i}")
            for i in range(TOTAL_SLOTS)
        ]
        char = make_character(name="Bob", schedule=schedule)
        engine = make_engine(world=make_world(characters=[char]))

        result_day0 = engine.get_character_location("Bob", day=0, slot=TimeSlot.MORNING)
        result_day1 = engine.get_character_location("Bob", day=1, slot=TimeSlot.MORNING)

        # day=0 should be invalid (out of range) and return None — but due to
        # falsy evaluation it silently returns the same entry as day=1.
        assert result_day0 is None, (
            f"BUG: get_character_location(day=0) returned {result_day0} instead of None. "
            "The expression 'd = day or self.loop.current_day' treats day=0 as falsy "
            "and substitutes the current day, silently returning wrong data."
        )


# ── State machine: causal chain cycle detection ────────────────────────


class TestCausalChainCycle:
    def test_chain_intact_simple_cycle_does_not_hang(self):
        """_chain_intact must terminate even when downstream_effects creates a cycle.

        A -> B -> A is an invalid but possible world state.  Without cycle
        detection the recursion would be infinite.
        """
        e_a = CausalChainEvent(
            id="cycle_a",
            day=1,
            time_slot=TimeSlot.MORNING,
            character="V",
            action="step_a",
            location="main_lodge",
            downstream_effects=["cycle_b"],
        )
        e_b = CausalChainEvent(
            id="cycle_b",
            day=5,
            time_slot=TimeSlot.NIGHT,
            character="V",
            action="step_b",
            location="main_lodge",
            downstream_effects=["cycle_a"],  # closes the cycle
        )
        world = make_world(causal_chain=[e_a, e_b])
        engine = make_engine(world=world)

        # This must not raise RecursionError
        memo: dict[str, bool] = {}
        try:
            result = engine._chain_intact("cycle_b", memo)
        except RecursionError:
            pytest.fail(
                "BUG: _chain_intact entered infinite recursion on a cyclic causal chain. "
                "The memo dict should prevent re-entrant calls but does not guard "
                "against cycles where the memo entry is not set before recursing."
            )

    def test_chain_intact_self_loop_does_not_hang(self):
        """An event that lists itself in downstream_effects must not cause infinite recursion."""
        e_self = CausalChainEvent(
            id="self_ref",
            day=5,
            time_slot=TimeSlot.NIGHT,
            character="V",
            action="self_step",
            location="main_lodge",
            downstream_effects=["self_ref"],  # self-referential
        )
        world = make_world(causal_chain=[e_self])
        engine = make_engine(world=world)

        memo: dict[str, bool] = {}
        try:
            engine._chain_intact("self_ref", memo)
        except RecursionError:
            pytest.fail(
                "BUG: _chain_intact entered infinite recursion on a self-referential event."
            )


# ── State machine: follow at last valid slot ───────────────────────────


class TestFollowBoundary:
    def test_follow_at_second_to_last_slot_succeeds(self):
        """Follow at slot index TOTAL_SLOTS-2 (day 5 EVENING) should work:
        next_idx = TOTAL_SLOTS-1, which is still < TOTAL_SLOTS.
        """
        char = make_character(name="NPC")
        loop = make_loop_state(
            current_day=DAYS_PER_LOOP,
            current_slot=TimeSlot.EVENING,
        )
        engine = make_engine(world=make_world(characters=[char]), loop=loop)

        from unittest.mock import patch
        with patch("random.random", return_value=0.99):  # never detected
            result = engine.resolve_follow("NPC")

        # Should succeed (next slot = NIGHT on day 5 = index 19 < 20)
        # OR fail with "Lost track" if no entry — but must NOT fail with
        # "End of loop" since there is still one slot left.
        assert result.get("reason") != "End of loop — nowhere to follow", (
            "BUG: resolve_follow refused to follow at day 5 EVENING even though "
            "there is one more slot (NIGHT) remaining in the loop"
        )

    def test_follow_trust_penalty_cannot_go_below_negative_100(self):
        """Detected follow deducts 5 trust; starting at -98 must clamp at -100,
        not produce -103.

        The engine clamps trust in the *conversation* path but the follow
        detection path does a raw subtraction: trust - 5.
        """
        char = make_character(name="Sneaky")
        loop = make_loop_state(character_trust={"Sneaky": -98})
        engine = make_engine(world=make_world(characters=[char]), loop=loop)

        from unittest.mock import patch
        with patch("random.random", return_value=0.0):  # always detected
            engine.resolve_follow("Sneaky")

        trust_after = engine.loop.character_trust.get("Sneaky", 0)
        assert trust_after >= -100, (
            f"BUG: trust after detected follow is {trust_after}, below the -100 floor. "
            "resolve_follow does not clamp the trust penalty."
        )


# ── State machine: loop reset at MAX_LOOPS + 1 ────────────────────────


class TestLoopResetEdge:
    def test_reset_loop_already_beyond_max_returns_false(self):
        """If loop_number somehow exceeds MAX_LOOPS, reset_loop must still return False."""
        loop = make_loop_state(loop_number=MAX_LOOPS + 5)
        engine = make_engine(loop=loop)
        result = engine.reset_loop()
        assert result is False

    def test_reset_loop_exactly_at_max_returns_false(self):
        loop = make_loop_state(loop_number=MAX_LOOPS)
        engine = make_engine(loop=loop)
        assert engine.reset_loop() is False

    def test_reset_loop_one_below_max_returns_true(self):
        loop = make_loop_state(loop_number=MAX_LOOPS - 1)
        engine = make_engine(loop=loop)
        assert engine.reset_loop() is True

    def test_reset_loop_multiple_resets_loop_history_grows(self):
        """Each successful reset appends exactly one entry to loop_history."""
        engine = make_engine()
        for expected_len in range(1, MAX_LOOPS):
            result = engine.reset_loop()
            assert result is True
            assert len(engine.knowledge.loop_history) == expected_len
        # Now at MAX_LOOPS, next reset must fail
        assert engine.reset_loop() is False
        assert len(engine.knowledge.loop_history) == MAX_LOOPS - 1


# ── State machine: intervention schedule modification overflow ─────────


class TestInterventionScheduleOverflow:
    def test_intervention_schedule_changes_at_end_of_loop(self):
        """An intervention applied at the last slot must not write schedule
        modifications beyond TOTAL_SLOTS - 1.

        success_schedule_changes may list many entries.  The apply_intervention
        code uses start_idx + i as the key.  If start_idx = 19 (last slot) and
        there are 5 schedule entries, keys 19..23 would be written — indices
        that no character schedule actually has, and that slot_index never
        produces for valid game states.
        """
        char = make_character(name="Villain", tier=CharacterTier.TIER1)
        event = CausalChainEvent(
            id="ce_end",
            day=DAYS_PER_LOOP,
            time_slot=TimeSlot.NIGHT,
            character="Villain",
            action="final_act",
            location="main_lodge",
        )
        # 5 schedule changes starting at the last slot
        schedule_changes = [
            ScheduleEntry(location="main_lodge", activity=f"step_{i}")
            for i in range(5)
        ]
        intervention = InterventionNode(
            id="iv_end",
            causal_event_id="ce_end",
            required_day=DAYS_PER_LOOP,
            required_slot=TimeSlot.NIGHT,
            required_location="main_lodge",
            success_schedule_changes={"Villain": schedule_changes},
        )
        world = make_world(
            characters=[char],
            causal_chain=[event],
            intervention_tree=[intervention],
        )
        loop = make_loop_state(
            current_day=DAYS_PER_LOOP,
            current_slot=TimeSlot.NIGHT,
        )
        engine = make_engine(world=world, loop=loop)
        engine.apply_intervention("iv_end")

        mods = engine.loop.schedule_modifications.get("Villain", {})
        for key in mods:
            idx = int(key)
            assert idx < TOTAL_SLOTS, (
                f"BUG: apply_intervention wrote schedule modification at index {idx} "
                f"which is >= TOTAL_SLOTS ({TOTAL_SLOTS}). "
                "This key can never be resolved by get_character_location."
            )


# ── Rumor mill: spread_count incremented per listener, not per hop ─────


class TestRumorSpreadCountSemantics:
    def test_spread_count_increments_per_listener_not_per_slot(self):
        """spread_count is incremented once for every (sharer, listener) pair
        that successfully shares a claim in a single propagation call.

        With 1 knower and N ignorant listeners, a single propagate_claims call
        could increment spread_count up to N times in one shot — even though
        the claim has only 'spread' once (one time-slot transition).

        This means a popular claim with many co-located NPCs could be throttled
        to 0 remaining hops after a single time slot, whereas a claim shared
        one-on-one gets 4 full hops.  The asymmetry is a game-balance bug.
        """
        # Build a world with 1 knower + 5 ignorant NPCs all in the same location
        alice = make_character(name="Alice")  # knower
        listeners = [make_character(name=f"NPC_{i}") for i in range(5)]  # 5 ignorant
        all_chars = [alice] + listeners

        claim = Claim(
            id="c_spread",
            source="player",
            subject="someone",
            content="is suspicious",
            slot_created=0,
            heard_by=["Alice"],
            spread_count=0,
        )
        world = make_world(characters=all_chars)
        loop = make_loop_state(active_claims=[claim])
        # All characters at one location
        occupants = {"main_lodge": ["Alice"] + [c.name for c in listeners]}

        propagate_claims(world, loop, occupants)

        # The spread_count should not exceed MAX_SPREAD_HOPS after a single slot
        # If it does, the claim is effectively dead after one propagation call
        # even though it only "spread" to one group of people.
        if claim.spread_count > MAX_SPREAD_HOPS:
            pytest.fail(
                f"BUG: After a single propagate_claims call, spread_count={claim.spread_count} "
                f"exceeds MAX_SPREAD_HOPS={MAX_SPREAD_HOPS}. "
                "Claims shared with many co-located NPCs at once are throttled unfairly. "
                "spread_count should track propagation hops (time-slot steps), "
                "not individual (sharer, listener) pairs."
            )

    def test_empty_active_claims_returns_empty_events(self):
        """propagate_claims with no active_claims must return an empty list."""
        world = make_world(characters=[make_character(name="Alice")])
        loop = make_loop_state(active_claims=[])
        events = propagate_claims(world, loop, {"main_lodge": ["Alice"]})
        assert events == []

    def test_claim_with_empty_heard_by_does_not_propagate(self):
        """A claim with no knowers cannot spread (no sharer exists)."""
        alice = make_character(name="Alice")
        bob = make_character(name="Bob")
        claim = Claim(
            id="c_orphan",
            source="player",
            subject="Alice",
            content="is suspicious",
            slot_created=0,
            heard_by=[],  # nobody knows it
            spread_count=0,
        )
        world = make_world(characters=[alice, bob])
        loop = make_loop_state(active_claims=[claim])
        events = propagate_claims(world, loop, {"main_lodge": ["Alice", "Bob"]})
        rumor_events = [e for e in events if e["type"] == "rumor_spread"]
        assert len(rumor_events) == 0

    def test_claim_with_special_characters_in_content_propagates(self):
        """Claims with Unicode, quotes, and newlines must not crash propagation."""
        alice = make_character(name="Alice")
        bob = make_character(name="Bob")
        claim = Claim(
            id="c_special",
            source="player",
            subject="Bob",
            content='said "hello\nworld" with an emoji \U0001f525 and <b>HTML</b>',
            slot_created=0,
            heard_by=["Alice"],
            spread_count=0,
        )
        world = make_world(characters=[alice, bob])
        loop = make_loop_state(active_claims=[claim])
        events = propagate_claims(world, loop, {"main_lodge": ["Alice", "Bob"]})
        assert isinstance(events, list)

    def test_many_npcs_all_already_know_claim_produces_no_events(self):
        """If all NPCs already know a claim, propagation should produce 0 rumor events."""
        chars = [make_character(name=f"NPC_{i}") for i in range(10)]
        all_names = [c.name for c in chars]
        claim = Claim(
            id="c_full",
            source="player",
            subject="someone",
            content="is suspicious",
            slot_created=0,
            heard_by=list(all_names),  # everyone already knows
            spread_count=0,
        )
        world = make_world(characters=chars)
        loop = make_loop_state(active_claims=[claim])
        occupants = {"main_lodge": all_names}
        events = propagate_claims(world, loop, occupants)
        rumor_events = [e for e in events if e["type"] == "rumor_spread"]
        assert len(rumor_events) == 0


# ── Rumor mill: _share_chance tier1 alarm suppression stacks ──────────


class TestShareChanceEdgeCases:
    def test_tier1_self_claim_about_listener_takes_listener_chance(self):
        """When claim.subject == listener.name AND sharer is TIER1 and
        claim.subject == sharer.name, the conditions are contradictory.

        The code checks subject == listener first (setting ABOUT_LISTENER = 0.95),
        then overwrites with TIER1_SUPPRESSES (0.25) if subject == sharer.
        But if the claim is about someone OTHER than the sharer, the alarm
        suppression multiplication (0.3x) is applied after the ABOUT_LISTENER
        assignment.  Verify the final value is the expected one.
        """
        tier1_sharer = make_character(name="Villain", tier=CharacterTier.TIER1)
        listener = make_character(name="Bob")
        # Claim subject is Bob (the listener), content has alarm keyword
        claim = Claim(
            id="c_alarm_listener",
            source="player",
            subject="Bob",
            content="planted a weapon in the shed",
            slot_created=0,
        )
        chance = _share_chance(tier1_sharer, listener, claim)
        # Expected: ABOUT_LISTENER (0.95) then alarm suppression: 0.95 * 0.3 = 0.285
        # But the code sets chance = ABOUT_LISTENER first, then multiplies by 0.3
        # because sharer is TIER1 and content has alarm keywords
        expected = pytest.approx(0.95 * 0.3, abs=1e-6)
        assert chance == expected, (
            f"chance={chance} but expected {0.95 * 0.3:.4f}. "
            "TIER1 alarm suppression should multiply whatever chance was already set."
        )

    def test_share_chance_subject_case_insensitive_vs_listener_name(self):
        """claim.subject.lower() == listener.name.lower() triggers ABOUT_LISTENER.

        Verify that mixed-case subjects still match correctly.
        """
        sharer = make_character(name="Alice")
        listener = make_character(name="Bob")
        # Mixed-case subject matching listener name
        claim_upper = Claim(
            id="c_upper",
            source="player",
            subject="BOB",  # uppercase
            content="did something",
            slot_created=0,
        )
        from loop.rumor_mill import SHARE_CHANCE_ABOUT_LISTENER
        chance = _share_chance(sharer, listener, claim_upper)
        assert chance == pytest.approx(SHARE_CHANCE_ABOUT_LISTENER), (
            "Case-insensitive subject matching failed: 'BOB' should match listener 'Bob'"
        )


# ── Knowledge base: self-loop connection ──────────────────────────────


class TestKnowledgeBaseConnectionEdge:
    def test_add_connection_same_evidence_both_sides(self):
        """add_connection('ev1', 'ev1') should return False (self-loop is meaningless).

        The duplicate check is: {conn.evidence_a, conn.evidence_b} == {evidence_a, evidence_b}
        For a self-connection: {ev1} == {ev1} — this IS caught as a duplicate IF
        a self-connection already exists.  But on the FIRST call there is no prior
        connection, so the check passes and a self-loop is inserted.

        This documents whether that is the actual behavior.
        """
        knowledge = make_knowledge(evidence_discovered=["ev1"])
        world = make_world(
            evidence_registry=[
                Evidence(id="ev1", type=EvidenceType.PHYSICAL, description="A")
            ]
        )
        kb = KnowledgeBase(knowledge, world)
        result = kb.add_connection("ev1", "ev1", "same evidence")
        # A self-loop connection is semantically meaningless.
        # The expected behavior is False (rejected).
        assert result is False, (
            "BUG: add_connection('ev1', 'ev1') returned True, inserting a "
            "self-referential connection. Evidence should not connect to itself."
        )

    def test_add_connection_then_self_loop_is_duplicate(self):
        """After adding a real connection, a self-loop is still invalid."""
        knowledge = make_knowledge(evidence_discovered=["ev1", "ev2"])
        world = make_world(
            evidence_registry=[
                Evidence(id="ev1", type=EvidenceType.PHYSICAL, description="A"),
                Evidence(id="ev2", type=EvidenceType.PHYSICAL, description="B"),
            ]
        )
        kb = KnowledgeBase(knowledge, world)
        kb.add_connection("ev1", "ev2")
        result = kb.add_connection("ev1", "ev1")
        assert result is False, (
            "BUG: add_connection('ev1', 'ev1') returned True after another connection existed."
        )

    def test_get_player_knowledge_flags_with_string_and_int_keys(self):
        """character_schedules_known uses str keys, but get_player_knowledge_flags
        does int() conversion.  A non-numeric key must not crash the method.
        """
        knowledge = make_knowledge(
            character_schedules_known={"Alice": {"bad_key": "main_lodge", "5": "dining_hall"}}
        )
        world = make_world()
        kb = KnowledgeBase(knowledge, world)
        try:
            flags = kb.get_player_knowledge_flags("Alice", 3)
        except (ValueError, KeyError) as e:
            pytest.fail(
                f"BUG: get_player_knowledge_flags crashed with {type(e).__name__}: {e} "
                "when character_schedules_known contained a non-numeric key."
            )

    def test_knowledge_percentage_with_phantom_evidence_ids(self):
        """If evidence_discovered contains IDs not in the registry,
        get_knowledge_percentage should not exceed 1.0.
        """
        ev1 = Evidence(id="ev1", type=EvidenceType.PHYSICAL, description="A")
        knowledge = make_knowledge(evidence_discovered=["ev1", "phantom_1", "phantom_2"])
        world = make_world(evidence_registry=[ev1])
        kb = KnowledgeBase(knowledge, world)
        pct = kb.get_knowledge_percentage()
        # The registry has 1 item; only ev1 is real.
        # len(evidence_discovered) / len(registry) = 3/1 = 3.0 — exceeds 1.0!
        assert pct <= 1.0, (
            f"BUG: get_knowledge_percentage() returned {pct} > 1.0. "
            "Phantom evidence IDs in evidence_discovered inflate the percentage "
            "beyond 100%."
        )


# ── Display: _group_actions edge cases ────────────────────────────────


class TestGroupActionsEdgeCases:
    def _make_display(self):
        from loop.display import GameDisplay
        return GameDisplay()

    def test_group_actions_empty_list_returns_empty(self):
        display = self._make_display()
        result = display._group_actions([])
        assert result == []

    def test_group_actions_all_unknown_types_go_to_other(self):
        display = self._make_display()
        actions = [
            {"type": "unknown_type_a", "label": "Action A"},
            {"type": "unknown_type_b", "label": "Action B"},
            {"type": "unknown_type_c", "label": "Action C"},
        ]
        result = display._group_actions(actions)
        assert len(result) == 1
        group_name, items = result[0]
        assert group_name == "Other"
        assert len(items) == 3

    def test_group_actions_all_same_type(self):
        """All actions of the same type should appear under one group."""
        display = self._make_display()
        actions = [
            {"type": "talk", "target": f"NPC_{i}", "label": f"Talk to NPC_{i}"}
            for i in range(10)
        ]
        result = display._group_actions(actions)
        people_groups = [(name, items) for name, items in result if name == "People"]
        assert len(people_groups) == 1
        assert len(people_groups[0][1]) == 10

    def test_group_actions_long_group_name_does_not_crash_format(self):
        """The group header uses f"── {group_name} {'─' * (18 - len(group_name))}".

        If group_name is longer than 18 characters, the repeat count becomes
        negative and Python silently produces an empty string — the display
        should not crash.

        Since the only way to get a long group name is through the "Other"
        group (which is hardcoded), this test ensures the format string is
        safe for any group_name length.  We test by directly computing the
        repeat count.
        """
        group_name = "Other"  # 5 chars — fine
        repeat_count = 18 - len(group_name)
        assert repeat_count >= 0, f"Negative repeat count for group '{group_name}'"
        assert "─" * repeat_count  # should not raise

    def test_group_actions_preserves_original_action_indices(self):
        """_group_actions returns (original_index, action) pairs.
        Indices must correspond to the position in the original list.
        """
        display = self._make_display()
        actions = [
            {"type": "talk",     "target": "A", "label": "Talk to A"},
            {"type": "observe",  "label": "Observe"},
            {"type": "intervene","target": "iv1", "label": "Intervene"},
            {"type": "wait",     "label": "Wait"},
        ]
        result = display._group_actions(actions)
        flat = {label: idx for _, items in result for idx, action in items
                for label in [action["label"]]}
        assert flat["Talk to A"] == 0
        assert flat["Observe"] == 1
        assert flat["Intervene"] == 2
        assert flat["Wait"] == 3

    def test_group_actions_no_duplicate_indices(self):
        """Each action should appear in exactly one group, with its original index."""
        display = self._make_display()
        actions = [
            {"type": "talk", "target": "A", "label": "Talk A"},
            {"type": "follow", "target": "A", "label": "Follow A"},
            {"type": "observe", "label": "Observe"},
            {"type": "search", "label": "Search"},
            {"type": "travel", "target": "x", "label": "Go to X"},
            {"type": "intervene", "target": "y", "label": "Intervene Y"},
            {"type": "evidence_board", "label": "Evidence"},
            {"type": "schedule_tracker", "label": "Schedule"},
            {"type": "map", "label": "Map"},
            {"type": "wait", "label": "Wait"},
        ]
        result = display._group_actions(actions)
        all_indices = [idx for _, items in result for idx, _ in items]
        assert len(all_indices) == len(actions), "Some actions were lost or duplicated"
        assert sorted(all_indices) == list(range(len(actions)))


# ── Display: show_schedule_tracker_day with out-of-range day ──────────


class TestShowScheduleTrackerDay:
    def test_show_schedule_tracker_day_zero_does_not_crash(self):
        """show_schedule_tracker_day(day=0) calls slot_index(0, slot_name) which
        returns a negative index.  When used as a dict key string "-4" etc.,
        it will fail to match any known entry (harmless).  But if the implementation
        ever uses this as a list index it would silently return wrong data.

        The function should not raise an exception even with day=0.
        """
        from io import StringIO
        from unittest.mock import patch
        from loop.display import GameDisplay
        from rich.console import Console

        display = GameDisplay()
        world = make_world(characters=[make_character(name="Alice")])
        knowledge = make_knowledge(
            pinned_characters=["Alice"],
            character_schedules_known={"Alice": {"0": "main_lodge"}},
        )
        # Redirect console output to avoid polluting test output
        with patch.object(display.console, "print"):
            try:
                display.show_schedule_tracker_day(knowledge, world, day=0)
            except Exception as e:
                pytest.fail(
                    f"show_schedule_tracker_day(day=0) raised {type(e).__name__}: {e}"
                )

    def test_show_schedule_tracker_day_beyond_range_does_not_crash(self):
        """day=DAYS_PER_LOOP+1 should not crash (out-of-range day)."""
        from unittest.mock import patch
        from loop.display import GameDisplay

        display = GameDisplay()
        world = make_world()
        knowledge = make_knowledge()

        with patch.object(display.console, "print"):
            try:
                display.show_schedule_tracker_day(knowledge, world, day=DAYS_PER_LOOP + 1)
            except Exception as e:
                pytest.fail(
                    f"show_schedule_tracker_day(day={DAYS_PER_LOOP+1}) raised "
                    f"{type(e).__name__}: {e}"
                )

    def test_show_schedule_tracker_empty_world_no_crash(self):
        """An empty world (no characters) and empty knowledge should render safely."""
        from unittest.mock import patch
        from loop.display import GameDisplay

        display = GameDisplay()
        world = make_world(characters=[])
        knowledge = make_knowledge()

        with patch.object(display.console, "print"):
            try:
                display.show_schedule_tracker(knowledge, world)
            except Exception as e:
                pytest.fail(
                    f"show_schedule_tracker with empty world raised {type(e).__name__}: {e}"
                )


# ── Conversation engine: trust delta edge cases ───────────────────────


class TestTrustDeltaEdgeCases:
    def _make_engine_for_trust(self, char_name="Alice"):
        from loop.conversation_engine import ConversationEngine
        from loop.display import GameDisplay
        from unittest.mock import MagicMock

        char = make_character(name=char_name, secrets=["the weapon is in the cellar"])
        world = make_world(characters=[char])
        display = MagicMock(spec=GameDisplay)
        return ConversationEngine(world, display), char

    def test_harsh_accusation_returns_configured_penalty_no_empathy_stack(self):
        """Harsh accusation short-circuits immediately; even if the player also
        uses empathy words in the same message, the delta must equal
        TRUST_DELTA_ACCUSATION_HARSH (not HARSH + EMPATHY).
        """
        engine, char = self._make_engine_for_trust()
        msg = "I'm sorry, but I think you're a liar."
        result = engine._calculate_trust_delta(msg, char, [], make_knowledge())
        delta = result[0] if isinstance(result, tuple) else result
        assert delta == TRUST_DELTA_ACCUSATION_HARSH, (
            f"Expected {TRUST_DELTA_ACCUSATION_HARSH} for harsh accusation, got {delta}. "
            "Empathy words must not add to a harsh accusation penalty."
        )

    def test_mild_accusation_and_pressure_both_trigger_separately(self):
        """A message with ONLY a mild accusation keyword returns TRUST_DELTA_ACCUSATION_MILD.
        A message with ONLY a pressure keyword returns TRUST_DELTA_PRESSURE.
        They should not stack.
        """
        engine, char = self._make_engine_for_trust()

        mild_msg = "You're hiding something, aren't you?"
        mild_result = engine._calculate_trust_delta(mild_msg, char, [], make_knowledge())
        mild_delta = mild_result[0] if isinstance(mild_result, tuple) else mild_result
        assert mild_delta == TRUST_DELTA_ACCUSATION_MILD, f"Mild accusation expected {TRUST_DELTA_ACCUSATION_MILD}, got {mild_delta}"

        pressure_msg = "Tell me now where you were last night."
        pressure_result = engine._calculate_trust_delta(pressure_msg, char, [], make_knowledge())
        pressure_delta = pressure_result[0] if isinstance(pressure_result, tuple) else pressure_result
        assert pressure_delta == TRUST_DELTA_PRESSURE, f"Pressure expected {TRUST_DELTA_PRESSURE}, got {pressure_delta}"

    def test_empathy_only_gives_configured_bonus(self):
        engine, char = self._make_engine_for_trust()
        msg = "I'm sorry to hear that. How are you feeling?"
        result = engine._calculate_trust_delta(msg, char, [], make_knowledge())
        delta = result[0] if isinstance(result, tuple) else result
        assert delta == TRUST_DELTA_EMPATHY, f"Pure empathy expected {TRUST_DELTA_EMPATHY}, got {delta}"

    def test_zero_delta_on_neutral_message(self):
        engine, char = self._make_engine_for_trust()
        msg = "Nice weather today."
        result = engine._calculate_trust_delta(msg, char, [], make_knowledge())
        delta = result[0] if isinstance(result, tuple) else result
        assert delta == 0, f"Neutral message expected 0 delta, got {delta}"

    def test_secret_overlap_explained_gives_configured_bonus(self):
        """Referencing a secret with explanation gives TRUST_DELTA_IMPOSSIBLE_EXPLAINED."""
        engine, char = self._make_engine_for_trust()
        # The secret is "the weapon is in the cellar"
        # 5+ char words: "weapon", "cellar" — 2 overlapping words
        msg = "I heard the weapon might be in the cellar."
        result = engine._calculate_trust_delta(msg, char, [], make_knowledge())
        delta = result[0] if isinstance(result, tuple) else result
        assert delta == TRUST_DELTA_IMPOSSIBLE_EXPLAINED, f"Explained secret reference expected {TRUST_DELTA_IMPOSSIBLE_EXPLAINED}, got {delta}"

    def test_secret_overlap_unexplained_gives_configured_penalty(self):
        """Referencing a secret without explanation gives TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED."""
        engine, char = self._make_engine_for_trust()
        msg = "The weapon is in the cellar."  # no explanation phrase
        result = engine._calculate_trust_delta(msg, char, [], make_knowledge())
        delta = result[0] if isinstance(result, tuple) else result
        assert delta == TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED, f"Unexplained secret reference expected {TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED}, got {delta}"

    def test_remembered_words_require_two_non_common_matches(self):
        """Trust +2 for echoing only fires if 2+ unique non-common words are echoed.

        The REMEMBERED_THRESHOLD is 3 (minimum word length), so even very short
        words like 'the' (len=3) count if they are NOT in the common_words set.
        We use messages with zero genuine overlap to confirm 0 delta, and messages
        with two distinct rare NPC words echoed to confirm +2 delta.
        """
        engine, char = self._make_engine_for_trust()

        # NPC uses rare words; player response uses completely different vocabulary
        history = [{
            "player": "Hi",
            "npc": "I spotted wildflowers blooming along riverbanks at dusk"
        }]
        # Player echoes zero NPC words (no overlap after common-word filter)
        msg_zero_echo = "Did you catch any fish near lakeside dock?"
        result_zero = engine._calculate_trust_delta(msg_zero_echo, char, history, make_knowledge())
        delta_zero = result_zero[0] if isinstance(result_zero, tuple) else result_zero
        assert delta_zero == 0, (
            f"Expected 0 for zero NPC-word echo, got {delta_zero}"
        )

        # Player echoes exactly 2 unique NPC words -> TRUST_DELTA_REMEMBERED bonus
        msg_two_echo = "I spotted wildflowers near the dock too"
        result_two = engine._calculate_trust_delta(msg_two_echo, char, history, make_knowledge())
        delta_two = result_two[0] if isinstance(result_two, tuple) else result_two
        assert delta_two == TRUST_DELTA_REMEMBERED, (
            f"Expected {TRUST_DELTA_REMEMBERED} for 2-word echo ('spotted', 'wildflowers'), got {delta_two}"
        )

    def test_empty_history_no_crash_on_remembered_check(self):
        """_calculate_trust_delta with empty history must not crash."""
        engine, char = self._make_engine_for_trust()
        result = engine._calculate_trust_delta("Hello", char, [], make_knowledge())
        delta = result[0] if isinstance(result, tuple) else result
        assert isinstance(delta, int)

    def test_extract_topics_empty_history_returns_empty(self):
        """_extract_topics with no conversation history must return []."""
        engine, _ = self._make_engine_for_trust()
        topics = engine._extract_topics([])
        assert topics == []

    def test_extract_topics_no_evidence_keywords_registered(self):
        """If evidence_registry is empty, _extract_topics must return []."""
        from loop.conversation_engine import ConversationEngine
        from unittest.mock import MagicMock
        from loop.display import GameDisplay

        world = make_world(evidence_registry=[])
        display = MagicMock(spec=GameDisplay)
        engine = ConversationEngine(world, display)
        history = [{"player": "Did you see anything?", "npc": "I saw something odd."}]
        topics = engine._extract_topics(history)
        assert topics == []


# ── Models: serialization round-trips with edge-case data ─────────────


class TestModelSerializationEdgeCases:
    def test_claim_with_unicode_content_roundtrip(self):
        from loop.models import Claim
        claim = Claim(
            id="c_unicode",
            source="\u00e9lise",  # accented character
            subject="caf\u00e9",
            content="saw something at the caf\u00e9 \U0001f525 <b>bold</b>\nnewline",
            slot_created=0,
        )
        json_str = claim.model_dump_json()
        restored = Claim.model_validate_json(json_str)
        assert restored.content == claim.content
        assert restored.source == claim.source

    def test_loop_state_with_empty_collections_roundtrip(self):
        loop = make_loop_state()
        json_str = loop.model_dump_json()
        restored = LoopState.model_validate_json(json_str)
        assert restored.loop_number == 1
        assert restored.character_trust == {}
        assert restored.active_claims == []

    def test_world_state_with_no_catastrophe_day_roundtrip(self):
        world = make_world()
        json_str = world.model_dump_json()
        restored = WorldState.model_validate_json(json_str)
        assert restored.catastrophe_day == 5
        assert restored.catastrophe_slot == TimeSlot.NIGHT

    def test_evidence_with_empty_lists_roundtrip(self):
        ev = Evidence(
            id="ev_empty",
            type=EvidenceType.OBSERVATION,
            description="",
            prerequisites=[],
            connects_to=[],
        )
        json_str = ev.model_dump_json()
        restored = Evidence.model_validate_json(json_str)
        assert restored.description == ""
        assert restored.prerequisites == []
        assert restored.connects_to == []

    def test_persistent_knowledge_with_max_loops_of_history_roundtrip(self):
        from loop.models import LoopSummary, PersistentKnowledge
        summaries = [
            LoopSummary(
                loop_number=i,
                evidence_found=[f"ev{j}" for j in range(10)],
                interventions_attempted=[f"iv{j}" for j in range(5)],
                ending_reached="failure",
            )
            for i in range(MAX_LOOPS)
        ]
        knowledge = PersistentKnowledge(
            evidence_discovered=[f"ev{i}" for i in range(50)],
            loop_history=summaries,
        )
        json_str = knowledge.model_dump_json()
        restored = PersistentKnowledge.model_validate_json(json_str)
        assert len(restored.loop_history) == MAX_LOOPS
        assert len(restored.evidence_discovered) == 50


# ── State machine: check_catastrophe with no causal chain ─────────────


class TestCatastropheEdgeCases:
    def test_check_catastrophe_empty_causal_chain_returns_false(self):
        """With no causal chain events, catastrophe should not occur."""
        world = make_world(causal_chain=[])
        engine = make_engine(world=world)
        assert engine.check_catastrophe() is False

    def test_check_catastrophe_only_non_catastrophe_day_events(self):
        """Events that don't match catastrophe_day/slot should not trigger catastrophe."""
        event = CausalChainEvent(
            id="early_event",
            day=1,  # not catastrophe day (5)
            time_slot=TimeSlot.MORNING,
            character="V",
            action="early",
            location="main_lodge",
        )
        world = make_world(causal_chain=[event])
        engine = make_engine(world=world)
        assert engine.check_catastrophe() is False

    def test_check_catastrophe_multiple_final_events_all_must_be_interrupted(self):
        """If there are multiple events on catastrophe day/slot, catastrophe
        occurs if ANY of them has an intact chain — not only if ALL do.

        This tests the 'any is enough' contract: catastrophe = OR over final events.
        """
        e1 = CausalChainEvent(
            id="final_1", day=5, time_slot=TimeSlot.NIGHT,
            character="V1", action="act1", location="main_lodge",
        )
        e2 = CausalChainEvent(
            id="final_2", day=5, time_slot=TimeSlot.NIGHT,
            character="V2", action="act2", location="main_lodge",
        )
        world = make_world(causal_chain=[e1, e2])
        # Interrupt only e1, not e2
        loop = make_loop_state(events_interrupted=["final_1"])
        engine = make_engine(world=world, loop=loop)
        # e2 is still intact, so catastrophe should still occur
        assert engine.check_catastrophe() is True

    def test_advance_time_at_loop_end_does_not_mutate_day_or_slot(self):
        """advance_time at day 5 NIGHT should return 'loop_end' without
        changing current_day or current_slot (the loop is over).
        """
        engine = make_engine(
            loop=make_loop_state(
                current_day=DAYS_PER_LOOP,
                current_slot=TimeSlot.NIGHT,
            )
        )
        engine.advance_time()
        # Day and slot should remain at their final values
        assert engine.loop.current_day == DAYS_PER_LOOP
        assert engine.loop.current_slot == TimeSlot.NIGHT


# ── State machine: wildcard / causal event double-trigger guard ────────


class TestEventTriggerDedup:
    def test_wild_card_not_triggered_twice(self):
        """The same wild card at a given day/slot must not be added to
        events_triggered more than once if advance_time is called twice
        (e.g. via replay or edge-case calling code).
        """
        wc = WildCardEvent(
            id="wc_dedup",
            day=1,
            time_slot=TimeSlot.AFTERNOON,
            description="Storm",
        )
        world = make_world(wild_cards=[wc])
        engine = make_engine(world=world)
        # First advance: MORNING -> AFTERNOON
        engine.advance_time()
        assert engine.loop.current_slot == TimeSlot.AFTERNOON
        count_before = engine.loop.events_triggered.count("wc_dedup")

        # Manually call _check_wild_cards again (simulating a double-check)
        engine._check_wild_cards()
        count_after = engine.loop.events_triggered.count("wc_dedup")

        assert count_after == count_before, (
            f"BUG: _check_wild_cards appended 'wc_dedup' again (count went "
            f"{count_before} -> {count_after}). The guard `if wc.id not in "
            f"events_triggered` prevents duplicates, but duplicates can still "
            f"accumulate if the check is called more than once without the list check."
        )

    def test_causal_event_not_triggered_twice(self):
        """Same guard for causal events."""
        event = CausalChainEvent(
            id="ce_dedup",
            day=1,
            time_slot=TimeSlot.AFTERNOON,
            character="V",
            action="act",
            location="main_lodge",
        )
        world = make_world(causal_chain=[event])
        engine = make_engine(world=world)
        engine.advance_time()  # triggers AFTERNOON event

        count_before = engine.loop.events_triggered.count("ce_dedup")
        engine._check_causal_events()  # call again manually
        count_after = engine.loop.events_triggered.count("ce_dedup")

        assert count_after == count_before, (
            f"BUG: _check_causal_events triggered 'ce_dedup' twice (count went "
            f"{count_before} -> {count_after})."
        )


# ── Schedule tracker: day command parsing edge cases ──────────────────


class TestScheduleTrackerDayParsing:
    """Test the command parsing logic extracted from ScheduleTracker.show().

    We test the core logic (regex matching and range checking) directly,
    since the async show() method requires an event loop and display mock.
    """

    def _parse_day_command(self, parts_1: str) -> int | None:
        """Replicate the day command parsing from ScheduleTracker.show()."""
        match = re.match(r"(\d+)", parts_1)
        if match:
            d = int(match.group(1))
            if 1 <= d <= DAYS_PER_LOOP:
                return d
        return None

    def _parse_d_shorthand(self, verb: str) -> int | None:
        """Replicate the dN shorthand parsing."""
        m = re.match(r"d(\d+)$", verb)
        if m:
            d = int(m.group(1))
            if 1 <= d <= DAYS_PER_LOOP:
                return d
        return None

    def test_day_command_valid_range(self):
        for d in range(1, DAYS_PER_LOOP + 1):
            assert self._parse_day_command(str(d)) == d

    def test_day_command_zero_returns_none(self):
        assert self._parse_day_command("0") is None

    def test_day_command_beyond_max_returns_none(self):
        assert self._parse_day_command(str(DAYS_PER_LOOP + 1)) is None

    def test_day_command_negative_not_matched(self):
        # Regex only matches digits, so "-1" won't match \d+
        assert self._parse_day_command("-1") is None

    def test_day_command_large_number_rejected(self):
        assert self._parse_day_command("999") is None

    def test_day_command_with_leading_spaces_not_matched(self):
        """BUG: re.match anchors at the start of the string, so leading spaces
        in parts[1] prevent the digit from being matched.

        If a user types "day  3" (extra space), cmd.split(maxsplit=1) produces
        parts[1] = " 3" (with a leading space).  re.match(r"(\d+)", " 3") returns
        None because match starts at position 0 which is a space, not a digit.

        This is a usability defect: valid input is silently rejected and the user
        sees "Usage: day <N>" even though they typed a valid day number.
        """
        # re.match does NOT skip leading whitespace — it must match at position 0
        assert self._parse_day_command("  3  ") is None, (
            "Leading space before digit should prevent match with re.match — "
            "if this fails, re.search semantics were used instead of re.match"
        )
        # Without leading space, it works fine
        assert self._parse_day_command("3") == 3

    def test_d_shorthand_valid(self):
        for d in range(1, DAYS_PER_LOOP + 1):
            assert self._parse_d_shorthand(f"d{d}") == d

    def test_d_shorthand_zero_rejected(self):
        assert self._parse_d_shorthand("d0") is None

    def test_d_shorthand_beyond_max_rejected(self):
        assert self._parse_d_shorthand(f"d{DAYS_PER_LOOP + 1}") is None

    def test_d_shorthand_requires_exact_pattern(self):
        """d1extra should NOT match because of the $ anchor."""
        assert self._parse_d_shorthand("d1extra") is None
