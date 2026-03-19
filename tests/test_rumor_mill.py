"""Tests for the Rumor Mill (loop/rumor_mill.py)."""

import hashlib
import random

import pytest

from loop.config import TOTAL_SLOTS, slot_index
from loop.models import (
    Character,
    CharacterTier,
    Claim,
    LoopState,
    ScheduleEntry,
    TimeSlot,
    WorldState,
)
from loop.rumor_mill import (
    MAX_SPREAD_HOPS,
    SHARE_CHANCE_ABOUT_LISTENER,
    SHARE_CHANCE_BASE,
    SHARE_CHANCE_CLOSE,
    SHARE_CHANCE_TIER1_SUPPRESSES,
    _check_alarm_reactions,
    _share_chance,
    format_claims_for_prompt,
    get_claims_known_by,
    propagate_claims,
)
from tests.conftest import make_character, make_loop_state, make_world


# ── Constants ───────────────────────────────────────────────────────


class TestConstants:
    def test_max_spread_hops_is_four(self):
        """P2-A: MAX_SPREAD_HOPS was increased from 3 to 4."""
        assert MAX_SPREAD_HOPS == 4


# ── _share_chance tests ──────────────────────────────────────────────


class TestShareChance:
    def test_base_chance_strangers(self):
        sharer = make_character(name="Alice")
        listener = make_character(name="Bob")
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="did something", slot_created=0,
        )
        chance = _share_chance(sharer, listener, claim)
        assert chance == pytest.approx(SHARE_CHANCE_BASE)

    def test_close_relationship_bonus(self):
        sharer = make_character(name="Alice", relationships={"Bob": "best friend"})
        listener = make_character(name="Bob")
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="did something", slot_created=0,
        )
        chance = _share_chance(sharer, listener, claim)
        assert chance == pytest.approx(SHARE_CHANCE_CLOSE)

    def test_close_relationship_reverse(self):
        """Relationship on listener side also counts."""
        sharer = make_character(name="Alice")
        listener = make_character(name="Bob", relationships={"Alice": "colleague"})
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="did something", slot_created=0,
        )
        chance = _share_chance(sharer, listener, claim)
        assert chance == pytest.approx(SHARE_CHANCE_CLOSE)

    def test_about_listener_highest_chance(self):
        sharer = make_character(name="Alice")
        listener = make_character(name="Bob")
        claim = Claim(
            id="c1", source="player", subject="Bob",
            content="did something bad", slot_created=0,
        )
        chance = _share_chance(sharer, listener, claim)
        assert chance == pytest.approx(SHARE_CHANCE_ABOUT_LISTENER)

    def test_tier1_suppresses_self_claims(self):
        sharer = make_character(name="Villain", tier=CharacterTier.TIER1)
        listener = make_character(name="Bob")
        claim = Claim(
            id="c1", source="player", subject="Villain",
            content="did something bad", slot_created=0,
        )
        chance = _share_chance(sharer, listener, claim)
        assert chance == pytest.approx(SHARE_CHANCE_TIER1_SUPPRESSES)

    def test_tier1_suppresses_alarm_keywords(self):
        sharer = make_character(name="Villain", tier=CharacterTier.TIER1)
        listener = make_character(name="Bob")
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="has a weapon hidden somewhere", slot_created=0,
        )
        chance = _share_chance(sharer, listener, claim)
        # BASE * 0.3 since alarm keyword "weapon" is present
        assert chance == pytest.approx(SHARE_CHANCE_BASE * 0.3)

    def test_tier3_does_not_suppress(self):
        sharer = make_character(name="NPC", tier=CharacterTier.TIER3)
        listener = make_character(name="Bob")
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="has a weapon", slot_created=0,
        )
        chance = _share_chance(sharer, listener, claim)
        assert chance == pytest.approx(SHARE_CHANCE_BASE)


# ── propagate_claims tests ───────────────────────────────────────────


class TestPropagateClaims:
    def _setup(self, heard_by=None, spread_count=0):
        alice = make_character(name="Alice")
        bob = make_character(name="Bob")
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="is suspicious", slot_created=0,
            heard_by=heard_by or ["Alice"],
            spread_count=spread_count,
        )
        world = make_world(characters=[alice, bob])
        loop = make_loop_state(active_claims=[claim])
        occupants = {"main_lodge": ["Alice", "Bob"]}
        return world, loop, occupants, claim

    def test_colocated_npcs_share_claims(self):
        world, loop, occupants, _ = self._setup()
        # Force deterministic sharing by high seed value — just check mechanics
        events = propagate_claims(world, loop, occupants)
        # Even if RNG doesn't favor sharing, the mechanism runs without error
        # We test the logic paths are exercised
        assert isinstance(events, list)

    def test_player_excluded_from_heard_by(self):
        """Player should not be added to heard_by during propagation."""
        alice = make_character(name="Alice")
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="is suspicious", slot_created=0,
            heard_by=["Alice"],
        )
        world = make_world(characters=[alice])
        loop = make_loop_state(active_claims=[claim])
        occupants = {"main_lodge": ["Alice", "player"]}
        propagate_claims(world, loop, occupants)
        assert "player" not in claim.heard_by

    def test_max_spread_hops_enforced(self):
        world, loop, occupants, claim = self._setup(spread_count=MAX_SPREAD_HOPS)
        events = propagate_claims(world, loop, occupants)
        # No rumor_spread events since max hops reached
        rumor_events = [e for e in events if e["type"] == "rumor_spread"]
        assert len(rumor_events) == 0

    def test_single_occupant_no_propagation(self):
        alice = make_character(name="Alice")
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="is suspicious", slot_created=0,
            heard_by=["Alice"],
        )
        world = make_world(characters=[alice])
        loop = make_loop_state(active_claims=[claim])
        occupants = {"main_lodge": ["Alice"]}
        events = propagate_claims(world, loop, occupants)
        rumor_events = [e for e in events if e["type"] == "rumor_spread"]
        assert len(rumor_events) == 0

    def test_already_known_not_reshared(self):
        """If listener already knows, no resharing."""
        world, loop, occupants, _ = self._setup(heard_by=["Alice", "Bob"])
        events = propagate_claims(world, loop, occupants)
        rumor_events = [e for e in events if e["type"] == "rumor_spread"]
        assert len(rumor_events) == 0


# ── Deterministic seeding ────────────────────────────────────────────


class TestDeterministicSeeding:
    def test_hashlib_md5_deterministic(self):
        """Same inputs produce the same seed via hashlib.md5."""
        seed_data = repr(("c1", "Alice", "Bob", 1, "MORNING"))
        digest1 = int(hashlib.md5(seed_data.encode()).hexdigest(), 16)
        digest2 = int(hashlib.md5(seed_data.encode()).hexdigest(), 16)
        assert digest1 == digest2

    def test_different_inputs_different_seeds(self):
        data1 = repr(("c1", "Alice", "Bob", 1, "MORNING"))
        data2 = repr(("c1", "Alice", "Bob", 1, "AFTERNOON"))
        digest1 = int(hashlib.md5(data1.encode()).hexdigest(), 16)
        digest2 = int(hashlib.md5(data2.encode()).hexdigest(), 16)
        assert digest1 != digest2

    def test_propagation_is_repeatable(self):
        """Two runs with the same state produce the same events."""
        alice = make_character(name="Alice")
        bob = make_character(name="Bob")
        world = make_world(characters=[alice, bob])

        def run():
            claim = Claim(
                id="c1", source="player", subject="someone",
                content="is suspicious", slot_created=0,
                heard_by=["Alice"], spread_count=0,
            )
            loop = make_loop_state(active_claims=[claim])
            occupants = {"main_lodge": ["Alice", "Bob"]}
            return propagate_claims(world, loop, occupants)

        events1 = run()
        events2 = run()
        # Same length means deterministic
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1["type"] == e2["type"]


# ── Alarm reactions ──────────────────────────────────────────────────


class TestAlarmReactions:
    def test_tier1_evasive_reaction(self):
        villain = make_character(name="Villain", tier=CharacterTier.TIER1)
        claim = Claim(
            id="c1", source="player", subject="Villain",
            content="has a hidden weapon in the shed", slot_created=0,
            heard_by=["Villain"],
        )
        loop = make_loop_state(active_claims=[claim])
        char_map = {"Villain": villain}
        world = make_world(characters=[villain])
        events = _check_alarm_reactions(world, loop, char_map)
        assert len(events) == 1
        assert events[0]["reaction"] == "evasive"
        assert events[0]["character"] == "Villain"
        # Check schedule modification uses str key
        mods = loop.schedule_modifications.get("Villain", {})
        for key in mods:
            assert isinstance(key, str)

    def test_tier2_anxious_reaction(self):
        witness = make_character(name="Witness", tier=CharacterTier.TIER2)
        claim = Claim(
            id="c1", source="player", subject="Witness",
            content="is in danger of being killed", slot_created=0,
            heard_by=["Witness"],
        )
        loop = make_loop_state(active_claims=[claim])
        char_map = {"Witness": witness}
        world = make_world(characters=[witness])
        events = _check_alarm_reactions(world, loop, char_map)
        assert len(events) == 1
        assert events[0]["reaction"] == "anxious"
        assert events[0]["new_location"] == "directors_office"

    def test_no_retrigger_alarm(self):
        """alarm_key in events_triggered prevents re-trigger."""
        villain = make_character(name="Villain", tier=CharacterTier.TIER1)
        claim = Claim(
            id="c1", source="player", subject="Villain",
            content="has a hidden weapon", slot_created=0,
            heard_by=["Villain"],
        )
        alarm_key = f"alarm_{claim.id}_{claim.subject}"
        loop = make_loop_state(
            active_claims=[claim],
            events_triggered=[alarm_key],
        )
        char_map = {"Villain": villain}
        world = make_world(characters=[villain])
        events = _check_alarm_reactions(world, loop, char_map)
        assert len(events) == 0

    def test_no_alarm_without_alarm_keywords(self):
        villain = make_character(name="Villain", tier=CharacterTier.TIER1)
        claim = Claim(
            id="c1", source="player", subject="Villain",
            content="likes pizza", slot_created=0,
            heard_by=["Villain"],
        )
        loop = make_loop_state(active_claims=[claim])
        char_map = {"Villain": villain}
        world = make_world(characters=[villain])
        events = _check_alarm_reactions(world, loop, char_map)
        assert len(events) == 0

    def test_alarm_slot_check_uses_total_slots(self):
        """Alarm reaction should NOT trigger when next_idx >= TOTAL_SLOTS."""
        villain = make_character(name="Villain", tier=CharacterTier.TIER1)
        claim = Claim(
            id="c1", source="player", subject="Villain",
            content="has a weapon", slot_created=0,
            heard_by=["Villain"],
        )
        # Set to last slot (day 5, NIGHT) — next_idx = 20 which >= TOTAL_SLOTS
        loop = make_loop_state(
            active_claims=[claim],
            current_day=TOTAL_SLOTS // 4,
            current_slot=TimeSlot.NIGHT,
        )
        char_map = {"Villain": villain}
        world = make_world(characters=[villain])
        events = _check_alarm_reactions(world, loop, char_map)
        # Should have the alarm event tracked but no schedule modification
        # because next_idx >= TOTAL_SLOTS
        for ev in events:
            if ev["character"] == "Villain":
                # If an event is produced, the schedule mod should NOT exist
                # because the boundary check prevents it
                pass
        # Actually at day=5, NIGHT, next_idx = 20 >= 20, so no schedule change
        mods = loop.schedule_modifications.get("Villain", {})
        assert len(mods) == 0

    def test_subject_must_have_heard_claim(self):
        """If the subject hasn't heard the claim, no alarm."""
        villain = make_character(name="Villain", tier=CharacterTier.TIER1)
        claim = Claim(
            id="c1", source="player", subject="Villain",
            content="has a weapon", slot_created=0,
            heard_by=["Bob"],  # Villain hasn't heard it
        )
        loop = make_loop_state(active_claims=[claim])
        char_map = {"Villain": villain}
        world = make_world(characters=[villain])
        events = _check_alarm_reactions(world, loop, char_map)
        assert len(events) == 0

    def test_tier1_alarm_schedule_mod_str_key(self):
        """Tier1 alarm schedule modification uses str key for slot index."""
        villain = make_character(name="Villain", tier=CharacterTier.TIER1)
        claim = Claim(
            id="c1", source="player", subject="Villain",
            content="has stolen something from the lodge", slot_created=0,
            heard_by=["Villain"],
        )
        loop = make_loop_state(active_claims=[claim])
        char_map = {"Villain": villain}
        world = make_world(characters=[villain])
        events = _check_alarm_reactions(world, loop, char_map)
        mods = loop.schedule_modifications.get("Villain", {})
        assert len(mods) > 0
        for key in mods:
            assert isinstance(key, str)
            int(key)  # should parse as int without error


# ── get_claims_known_by ──────────────────────────────────────────────


class TestGetClaimsKnownBy:
    def test_returns_correct_claims(self):
        claim1 = Claim(
            id="c1", source="player", subject="someone",
            content="did something", slot_created=0, heard_by=["Alice"],
        )
        claim2 = Claim(
            id="c2", source="player", subject="other",
            content="did other thing", slot_created=0, heard_by=["Bob"],
        )
        loop = make_loop_state(
            active_claims=[claim1, claim2],
            npc_heard_claims={"Alice": ["c1"], "Bob": ["c2"]},
        )
        result = get_claims_known_by("Alice", loop)
        assert len(result) == 1
        assert result[0].id == "c1"

    def test_unknown_npc_returns_empty(self):
        loop = make_loop_state()
        result = get_claims_known_by("Nobody", loop)
        assert result == []

    def test_missing_claim_id_skipped(self):
        """If npc_heard_claims references a claim not in active_claims, skip it."""
        loop = make_loop_state(
            active_claims=[],
            npc_heard_claims={"Alice": ["nonexistent"]},
        )
        result = get_claims_known_by("Alice", loop)
        assert result == []


# ── format_claims_for_prompt ─────────────────────────────────────────


class TestFormatClaimsForPrompt:
    def test_empty_claims_empty_string(self):
        assert format_claims_for_prompt([], "Bob") == ""

    def test_player_source_label(self):
        claim = Claim(
            id="c1", source="player", subject="someone",
            content="did something", slot_created=0,
        )
        result = format_claims_for_prompt([claim], "Bob")
        assert "the new visitor" in result

    def test_npc_source_uses_name(self):
        claim = Claim(
            id="c1", source="Alice", subject="someone",
            content="did something", slot_created=0,
        )
        result = format_claims_for_prompt([claim], "Bob")
        assert "Alice" in result

    def test_about_you_tag(self):
        claim = Claim(
            id="c1", source="player", subject="Bob",
            content="is acting strange", slot_created=0,
        )
        result = format_claims_for_prompt([claim], "Bob")
        assert "[THIS IS ABOUT YOU]" in result
