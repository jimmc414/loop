"""All Pydantic v2 data models for LOOP."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class TimeSlot(str, Enum):
    MORNING = "MORNING"
    AFTERNOON = "AFTERNOON"
    EVENING = "EVENING"
    NIGHT = "NIGHT"


class EvidenceType(str, Enum):
    TESTIMONY = "testimony"
    OBSERVATION = "observation"
    PHYSICAL = "physical"
    BEHAVIORAL = "behavioral"


class EndingType(str, Enum):
    FULL_PREVENTION = "full_prevention"
    PARTIAL_PREVENTION = "partial_prevention"
    EXPOSURE = "exposure"
    FAILURE = "failure"
    DEEPER_TRUTH = "deeper_truth"


class CharacterTier(str, Enum):
    TIER1 = "tier1"   # central to catastrophe
    TIER2 = "tier2"   # supporting / witnesses
    TIER3 = "tier3"   # color / red herrings


# ── Schedule & Knowledge ───────────────────────────────────────────────

class ScheduleEntry(BaseModel):
    location: str
    activity: str


class KnowledgeEntry(BaseModel):
    """What a character knows/reveals at a given time slot."""
    available_topics: list[str] = Field(default_factory=list)
    mood: str = "neutral"
    willingness: str = "normal"  # reluctant, normal, eager


# ── Characters ─────────────────────────────────────────────────────────

class Character(BaseModel):
    name: str
    age: int
    role: str                         # e.g. "camp counselor", "cook"
    tier: CharacterTier
    personality: str                  # paragraph describing personality
    speech_pattern: str               # how they talk
    backstory: str = ""
    schedule: list[ScheduleEntry]     # exactly 20 entries (5 days x 4 slots)
    secrets: list[str] = Field(default_factory=list)
    knowledge_timeline: list[KnowledgeEntry] = Field(default_factory=list)  # 20 entries
    trust_threshold: int = 40
    relationships: dict[str, str] = Field(default_factory=dict)  # name -> relationship description


# ── Locations ──────────────────────────────────────────────────────────

class Location(BaseModel):
    name: str
    id: str
    area: str
    description: str
    locked: bool = False
    lock_conditions: str = ""         # what unlocks it
    search_items: list[str] = Field(default_factory=list)
    adjacent_locations: list[str] = Field(default_factory=list)
    is_distant: bool = False


# ── Causal Chain ───────────────────────────────────────────────────────

class CausalChainEvent(BaseModel):
    id: str
    day: int
    time_slot: TimeSlot
    character: str                    # character name
    action: str
    location: str                     # location id
    is_interruptible: bool = True
    interrupt_method: str = ""
    downstream_effects: list[str] = Field(default_factory=list)  # ids of events this triggers


# ── Evidence ───────────────────────────────────────────────────────────

class Evidence(BaseModel):
    id: str
    type: EvidenceType
    description: str
    source_location: str = ""         # location id
    source_character: str = ""        # character name
    available_day: int = 1
    available_slot: TimeSlot = TimeSlot.MORNING
    prerequisites: list[str] = Field(default_factory=list)  # evidence ids
    significance: str = ""            # why this matters
    connects_to: list[str] = Field(default_factory=list)    # evidence ids


# ── Interventions ──────────────────────────────────────────────────────

class InterventionNode(BaseModel):
    id: str
    causal_event_id: str              # which causal chain event this interrupts
    required_evidence: list[str] = Field(default_factory=list)
    required_location: str = ""       # location id
    required_day: int = 1
    required_slot: TimeSlot = TimeSlot.MORNING
    action_description: str = ""
    requires_conversation: bool = False
    trust_required: int = 0           # trust level needed with relevant character
    success_schedule_changes: dict[str, list[ScheduleEntry]] = Field(default_factory=dict)
    cascade_interrupts: list[str] = Field(default_factory=list)  # other causal event ids


# ── Wild Cards ─────────────────────────────────────────────────────────

class WildCardEvent(BaseModel):
    id: str
    day: int
    time_slot: TimeSlot
    description: str
    schedule_overrides: dict[str, ScheduleEntry] = Field(default_factory=dict)  # char name -> override
    location_effects: dict[str, Any] = Field(default_factory=dict)  # location id -> effects


# ── Endings ────────────────────────────────────────────────────────────

class EndingCondition(BaseModel):
    type: EndingType
    required_interrupted_events: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    description: str = ""


# ── World State (generated once per game) ──────────────────────────────

class WorldState(BaseModel):
    catastrophe_description: str = ""
    catastrophe_day: int = 5
    catastrophe_slot: TimeSlot = TimeSlot.NIGHT
    camp_history: str = ""
    opening_narration: str = ""
    characters: list[Character] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    causal_chain: list[CausalChainEvent] = Field(default_factory=list)
    evidence_registry: list[Evidence] = Field(default_factory=list)
    intervention_tree: list[InterventionNode] = Field(default_factory=list)
    wild_cards: list[WildCardEvent] = Field(default_factory=list)
    ending_conditions: list[EndingCondition] = Field(default_factory=list)


# ── Rumor Mill ─────────────────────────────────────────────────────────

class Claim(BaseModel):
    """A piece of information circulating in the social network."""
    id: str                           # unique claim id
    source: str                       # who originated this (player name or NPC name)
    subject: str                      # who/what this claim is about
    content: str                      # the actual claim text
    slot_created: int                 # slot index when created
    is_true: bool = False             # does it match actual world state?
    heard_by: list[str] = Field(default_factory=list)  # NPCs who have heard this
    spread_count: int = 0             # how many times this has propagated


# ── Loop State (resets each loop) ──────────────────────────────────────

class LoopState(BaseModel):
    loop_number: int = 1
    current_day: int = 1
    current_slot: TimeSlot = TimeSlot.MORNING
    player_location: str = "main_lodge"
    character_trust: dict[str, int] = Field(default_factory=dict)
    interventions_made: list[str] = Field(default_factory=list)  # intervention ids
    schedule_modifications: dict[str, dict[str, ScheduleEntry]] = Field(default_factory=dict)
    events_triggered: list[str] = Field(default_factory=list)
    events_interrupted: list[str] = Field(default_factory=list)
    conversations_this_loop: dict[str, int] = Field(default_factory=dict)  # char name -> exchange count
    # Rumor Mill state
    active_claims: list[Claim] = Field(default_factory=list)
    npc_heard_claims: dict[str, list[str]] = Field(default_factory=dict)  # NPC name -> claim ids


# ── Persistent Knowledge (survives loops) ──────────────────────────────

class LoopSummary(BaseModel):
    loop_number: int
    days_survived: int = 5
    evidence_found: list[str] = Field(default_factory=list)
    interventions_attempted: list[str] = Field(default_factory=list)
    ending_reached: str = ""
    key_events: list[str] = Field(default_factory=list)


class EvidenceConnection(BaseModel):
    evidence_a: str
    evidence_b: str
    description: str = ""
    confirmed: bool = False


class PersistentKnowledge(BaseModel):
    evidence_discovered: list[str] = Field(default_factory=list)       # evidence ids
    character_schedules_known: dict[str, dict[str, str]] = Field(default_factory=dict)
    evidence_connections: list[EvidenceConnection] = Field(default_factory=list)
    theories: list[str] = Field(default_factory=list)
    loop_history: list[LoopSummary] = Field(default_factory=list)
    characters_met: list[str] = Field(default_factory=list)
    locations_searched: dict[str, list[int]] = Field(default_factory=dict)  # loc id -> slot indices
    pinned_characters: list[str] = Field(default_factory=list)
    conversation_journal: list[str] = Field(default_factory=list)
    npc_interaction_counts: dict[str, int] = Field(default_factory=dict)
    npc_previous_topics: dict[str, list[str]] = Field(default_factory=dict)


# ── Save Game ──────────────────────────────────────────────────────────

class SaveGame(BaseModel):
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    world_state: WorldState = Field(default_factory=WorldState)
    loop_state: LoopState = Field(default_factory=LoopState)
    knowledge: PersistentKnowledge = Field(default_factory=PersistentKnowledge)


# ── Conversation ───────────────────────────────────────────────────────

class ConversationExchange(BaseModel):
    player: str
    npc: str


class ConversationResult(BaseModel):
    trust_change: int = 0
    topics_discussed: list[str] = Field(default_factory=list)
    exchanges_count: int = 0
    summary: str = ""
    exchanges: list[ConversationExchange] = Field(default_factory=list)
    evidence_presented: list[str] = Field(default_factory=list)
