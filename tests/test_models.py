"""Tests for Pydantic v2 model serialization (loop/models.py)."""

from datetime import datetime

import pytest

from loop.config import TOTAL_SLOTS
from loop.models import (
    CausalChainEvent,
    Character,
    CharacterTier,
    Claim,
    ConversationExchange,
    ConversationResult,
    EndingCondition,
    EndingType,
    Evidence,
    EvidenceConnection,
    EvidenceType,
    InterventionNode,
    KnowledgeEntry,
    Location,
    LoopState,
    LoopSummary,
    PersistentKnowledge,
    SaveGame,
    ScheduleEntry,
    TimeSlot,
    WildCardEvent,
    WorldState,
)
from tests.conftest import make_character, make_knowledge, make_loop_state, make_world


class TestModelRoundTrips:
    def test_schedule_entry_roundtrip(self):
        obj = ScheduleEntry(location="main_lodge", activity="idle")
        json_str = obj.model_dump_json()
        restored = ScheduleEntry.model_validate_json(json_str)
        assert restored.location == "main_lodge"
        assert restored.activity == "idle"

    def test_character_roundtrip(self):
        char = make_character(name="TestChar", age=30, tier=CharacterTier.TIER1)
        json_str = char.model_dump_json()
        restored = Character.model_validate_json(json_str)
        assert restored.name == "TestChar"
        assert restored.tier == CharacterTier.TIER1
        assert len(restored.schedule) == TOTAL_SLOTS

    def test_location_roundtrip(self):
        loc = Location(
            name="Main Lodge", id="main_lodge", area="central",
            description="A lodge", adjacent_locations=["dining_hall"],
            locked=True, lock_conditions="Need a key",
        )
        json_str = loc.model_dump_json()
        restored = Location.model_validate_json(json_str)
        assert restored.locked is True
        assert "dining_hall" in restored.adjacent_locations

    def test_claim_roundtrip(self):
        claim = Claim(
            id="c1", source="player", subject="Bob",
            content="stole something", slot_created=5,
            is_true=False, heard_by=["Alice", "Charlie"],
            spread_count=2,
        )
        json_str = claim.model_dump_json()
        restored = Claim.model_validate_json(json_str)
        assert restored.heard_by == ["Alice", "Charlie"]
        assert restored.spread_count == 2

    def test_evidence_roundtrip(self):
        ev = Evidence(
            id="ev1", type=EvidenceType.PHYSICAL,
            description="A knife", source_location="kitchen",
            available_day=2, available_slot=TimeSlot.EVENING,
            prerequisites=["ev0"], connects_to=["ev2"],
        )
        json_str = ev.model_dump_json()
        restored = Evidence.model_validate_json(json_str)
        assert restored.type == EvidenceType.PHYSICAL
        assert restored.available_slot == TimeSlot.EVENING
        assert restored.prerequisites == ["ev0"]

    def test_ending_condition_roundtrip(self):
        ec = EndingCondition(
            type=EndingType.DEEPER_TRUTH,
            required_interrupted_events=["ce1", "ce2"],
            required_evidence=["ev1"],
            description="You discovered the deeper truth",
        )
        json_str = ec.model_dump_json()
        restored = EndingCondition.model_validate_json(json_str)
        assert restored.type == EndingType.DEEPER_TRUTH
        assert len(restored.required_interrupted_events) == 2

    def test_loop_state_schedule_modifications_str_keys(self):
        """CRITICAL: schedule_modifications with str keys survive JSON round-trip."""
        loop = LoopState(
            schedule_modifications={
                "Alice": {
                    "0": ScheduleEntry(location="dining_hall", activity="eating"),
                    "5": ScheduleEntry(location="kitchen", activity="cooking"),
                }
            }
        )
        json_str = loop.model_dump_json()
        restored = LoopState.model_validate_json(json_str)
        alice_mods = restored.schedule_modifications["Alice"]
        assert "0" in alice_mods
        assert "5" in alice_mods
        assert alice_mods["0"].location == "dining_hall"

    def test_persistent_knowledge_schedules_known_str_keys(self):
        """CRITICAL: character_schedules_known with str keys survive round-trip."""
        pk = PersistentKnowledge(
            character_schedules_known={"Alice": {"0": "main_lodge", "5": "dining_hall"}}
        )
        json_str = pk.model_dump_json()
        restored = PersistentKnowledge.model_validate_json(json_str)
        alice_known = restored.character_schedules_known["Alice"]
        assert len(alice_known) == 2
        assert alice_known["0"] == "main_lodge"
        assert alice_known["5"] == "dining_hall"

    def test_world_state_roundtrip(self):
        world = make_world(
            catastrophe_description="Fire at the lodge",
            catastrophe_day=5,
            catastrophe_slot=TimeSlot.NIGHT,
        )
        json_str = world.model_dump_json()
        restored = WorldState.model_validate_json(json_str)
        assert restored.catastrophe_description == "Fire at the lodge"
        assert restored.catastrophe_day == 5

    def test_save_game_roundtrip(self):
        save = SaveGame(
            world_state=make_world(),
            loop_state=make_loop_state(),
            knowledge=make_knowledge(evidence_discovered=["ev1"]),
        )
        json_str = save.model_dump_json()
        restored = SaveGame.model_validate_json(json_str)
        assert "ev1" in restored.knowledge.evidence_discovered

    def test_enum_values_roundtrip(self):
        """All enum types survive serialization."""
        for ts in TimeSlot:
            data = {"current_slot": ts}
            ls = LoopState(current_slot=ts)
            json_str = ls.model_dump_json()
            restored = LoopState.model_validate_json(json_str)
            assert restored.current_slot == ts

    def test_nested_dict_roundtrip(self):
        loop = LoopState(
            character_trust={"Alice": 50, "Bob": -10},
            conversations_this_loop={"Alice": 3},
            npc_heard_claims={"Alice": ["c1", "c2"]},
        )
        json_str = loop.model_dump_json()
        restored = LoopState.model_validate_json(json_str)
        assert restored.character_trust["Alice"] == 50
        assert restored.npc_heard_claims["Alice"] == ["c1", "c2"]

    def test_evidence_connection_roundtrip(self):
        conn = EvidenceConnection(
            evidence_a="ev1", evidence_b="ev2",
            description="Both at the same location", confirmed=True,
        )
        json_str = conn.model_dump_json()
        restored = EvidenceConnection.model_validate_json(json_str)
        assert restored.confirmed is True

    def test_conversation_result_roundtrip(self):
        result = ConversationResult(
            trust_change=5,
            topics_discussed=["ev1"],
            exchanges_count=3,
            summary="Had a chat",
            exchanges=[ConversationExchange(player="hi", npc="hello")],
        )
        json_str = result.model_dump_json()
        restored = ConversationResult.model_validate_json(json_str)
        assert restored.trust_change == 5
        assert len(restored.exchanges) == 1

    def test_wild_card_event_roundtrip(self):
        wc = WildCardEvent(
            id="wc1", day=2, time_slot=TimeSlot.AFTERNOON,
            description="Storm rolls in",
            schedule_overrides={
                "Alice": ScheduleEntry(location="main_lodge", activity="sheltering"),
            },
        )
        json_str = wc.model_dump_json()
        restored = WildCardEvent.model_validate_json(json_str)
        assert "Alice" in restored.schedule_overrides
        assert restored.schedule_overrides["Alice"].location == "main_lodge"
