import pytest
from loop.models import (
    Character,
    CharacterTier,
    CausalChainEvent,
    Claim,
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
from loop.config import TOTAL_SLOTS
from loop.state_machine import ClockworkEngine
from loop.knowledge_base import KnowledgeBase


def make_character(**overrides):
    """Create a Character with sensible defaults."""
    defaults = {
        "name": "TestChar",
        "age": 25,
        "role": "counselor",
        "tier": CharacterTier.TIER3,
        "personality": "Friendly and outgoing",
        "speech_pattern": "Normal speech",
        "schedule": [
            ScheduleEntry(location="main_lodge", activity="idle")
            for _ in range(TOTAL_SLOTS)
        ],
        "knowledge_timeline": [KnowledgeEntry() for _ in range(TOTAL_SLOTS)],
    }
    defaults.update(overrides)
    return Character(**defaults)


def make_location(**overrides):
    defaults = {
        "name": "Test Location",
        "id": "test_loc",
        "area": "central",
        "description": "A test location",
        "adjacent_locations": [],
    }
    defaults.update(overrides)
    return Location(**defaults)


def make_world(**overrides):
    defaults = {
        "characters": [],
        "locations": [
            make_location(
                id="main_lodge",
                name="Main Lodge",
                adjacent_locations=["dining_hall"],
            ),
            make_location(
                id="dining_hall",
                name="Dining Hall",
                adjacent_locations=["main_lodge"],
            ),
        ],
        "causal_chain": [],
        "evidence_registry": [],
        "intervention_tree": [],
        "wild_cards": [],
        "ending_conditions": [],
    }
    defaults.update(overrides)
    return WorldState(**defaults)


def make_loop_state(**overrides):
    defaults = {}
    defaults.update(overrides)
    return LoopState(**defaults)


def make_knowledge(**overrides):
    defaults = {}
    defaults.update(overrides)
    return PersistentKnowledge(**defaults)


def make_engine(world=None, loop=None, knowledge=None):
    w = world or make_world()
    l = loop or make_loop_state()
    k = knowledge or make_knowledge()
    return ClockworkEngine(w, l, k)
