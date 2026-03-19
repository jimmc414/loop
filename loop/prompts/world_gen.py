"""Prompts for multi-step world generation."""

# ── Step 1: Catastrophe + causal chain + camp history + endings ────────

STEP1_CATASTROPHE = """\
You are a narrative designer creating the backstory for a time-loop mystery game \
set at Camp Pinehaven, a summer camp in the Pacific Northwest.

Design a CATASTROPHE that will occur on Day 5, Night slot unless the player intervenes.

Requirements:
- The catastrophe must be dramatic but grounded (fire, structural collapse, poisoning, \
explosion, flooding, chemical spill, etc.)
- It must be the result of a CAUSAL CHAIN of 8-12 events across 5 days
- Some events are deliberate (a character's plan), some are accidental (negligence, coincidence)
- At least 3 events must be interruptible by the player
- The chain must feel inevitable if no one intervenes, but preventable with the right knowledge

NARRATIVE QUALITY REQUIREMENTS:
- Include at least one mid-game twist: an event on Day 3 that recontextualizes earlier events
- Design at least 2-3 red herrings: events or character behaviors that COULD implicate the wrong person
- The deeper_truth ending should reveal a motivation that transforms understanding of the entire chain
- At least one character should have a secret that appears guilty but is actually innocent

Also create:
- A brief camp history (2-3 paragraphs) that contains clues to the catastrophe's roots
- 5 ending conditions (full_prevention, partial_prevention, exposure, failure, deeper_truth)

Return ONLY valid JSON matching this schema:
{
  "catastrophe_description": "What happens on Day 5 Night",
  "camp_history": "2-3 paragraph camp history with embedded clues",
  "causal_chain": [
    {
      "id": "event_01",
      "day": 1,
      "time_slot": "MORNING|AFTERNOON|EVENING|NIGHT",
      "character": "character name (use first names only)",
      "action": "what they do",
      "location": "location_id from: main_lodge, dining_hall, kitchen, campfire_circle, cabin_row, staff_quarters, bathhouse, arts_cabin, archery_range, trail_head, lake_shore, dock, old_boathouse, maintenance_shed, storage_cellar, directors_office, infirmary",
      "is_interruptible": true/false,
      "interrupt_method": "how the player could stop this (empty if not interruptible)",
      "downstream_effects": ["event_02"]
    }
  ],
  "ending_conditions": [
    {
      "type": "full_prevention|partial_prevention|exposure|failure|deeper_truth",
      "required_interrupted_events": ["event_ids"],
      "required_evidence": ["evidence_ids to be defined later"],
      "description": "What happens in this ending"
    }
  ]
}
"""

# ── Step 2: Characters ─────────────────────────────────────────────────

def step2_characters(catastrophe_json: str) -> str:
    return f"""\
You are designing characters for a time-loop mystery game at Camp Pinehaven.

Here is the catastrophe and causal chain:
{catastrophe_json}

Create exactly 8 characters. Requirements:
- 2 TIER1 characters: directly involved in causing the catastrophe
- 3 TIER2 characters: witnesses or unknowing participants
- 3 TIER3 characters: provide color, misdirection, or minor clues
- Each needs a distinct personality, speech pattern, and role at camp
- Ages 16-55 (mix of campers, counselors, staff)
- Relationships between characters should create tension and alliances
- Each character must have 1-3 secrets (some related to catastrophe, some personal)
- Use FIRST NAMES ONLY that match the names used in the causal chain

Return ONLY valid JSON:
{{
  "characters": [
    {{
      "name": "FirstName",
      "age": 25,
      "role": "camp role",
      "tier": "tier1|tier2|tier3",
      "personality": "paragraph describing personality",
      "speech_pattern": "how they talk - verbal tics, vocabulary level, etc.",
      "backstory": "brief backstory",
      "secrets": ["secret 1", "secret 2"],
      "trust_threshold": 40,
      "relationships": {{"OtherName": "relationship description"}}
    }}
  ]
}}
"""


# ── Step 3: Character schedules ────────────────────────────────────────

def step3_schedules(catastrophe_json: str, characters_json: str) -> str:
    return f"""\
You are creating detailed 5-day schedules for characters at Camp Pinehaven.

Catastrophe and causal chain:
{catastrophe_json}

Characters:
{characters_json}

For EACH character, create exactly 20 schedule entries (Day 1-5, slots: MORNING, AFTERNOON, EVENING, NIGHT).

Rules:
- Schedules must be consistent with the causal chain (characters must be at the right \
place at the right time for their causal chain events)
- Characters should have routines that feel natural (meals, activities, sleep)
- MORNING: most characters at activities or dining_hall
- NIGHT: most characters at cabin_row or campfire_circle (unless plot requires otherwise)
- Tier1 characters should have suspicious gaps or unusual movements
- Tier2 characters should be near key events (witnesses)
- Use ONLY these location IDs: main_lodge, dining_hall, kitchen, campfire_circle, cabin_row, \
staff_quarters, bathhouse, arts_cabin, archery_range, trail_head, lake_shore, dock, \
old_boathouse, maintenance_shed, storage_cellar, directors_office, infirmary

Return ONLY valid JSON:
{{
  "schedules": {{
    "CharacterName": [
      {{"location": "location_id", "activity": "what they're doing"}},
      ... (exactly 20 entries per character, in order: Day1 Morning, Day1 Afternoon, Day1 Evening, Day1 Night, Day2 Morning, ...)
    ]
  }}
}}
"""


# ── Step 4: Knowledge timelines ────────────────────────────────────────

def step4_knowledge(catastrophe_json: str, characters_json: str, schedules_json: str) -> str:
    return f"""\
You are creating knowledge timelines for characters at Camp Pinehaven. These define \
what each character knows and is willing to discuss at each time slot.

Catastrophe and causal chain:
{catastrophe_json}

Characters:
{characters_json}

Schedules:
{schedules_json}

For EACH character, create exactly 20 knowledge entries (Day 1-5, 4 slots each).

Rules:
- Characters accumulate knowledge over time (they learn things by witnessing events)
- available_topics: what they can discuss (increases as they witness/learn things)
- mood: reflects what's happening to them (neutral, anxious, angry, cheerful, suspicious, etc.)
- willingness: how open they are to talking (reluctant, normal, eager)
- Tier1 characters become more evasive as their plans progress
- Tier2 characters become more anxious as they notice odd things
- After witnessing a causal chain event, a character should gain related topics

Return ONLY valid JSON:
{{
  "knowledge_timelines": {{
    "CharacterName": [
      {{
        "available_topics": ["topic1", "topic2"],
        "mood": "neutral",
        "willingness": "normal"
      }},
      ... (exactly 20 entries per character)
    ]
  }}
}}
"""


# ── Step 5: Evidence + interventions + wild cards ──────────────────────

def step5_evidence(catastrophe_json: str, characters_json: str, schedules_json: str) -> str:
    return f"""\
You are creating the evidence registry, intervention tree, and wild card events for \
a time-loop mystery at Camp Pinehaven.

Catastrophe and causal chain:
{catastrophe_json}

Characters:
{characters_json}

Schedules:
{schedules_json}

Create:

1. EVIDENCE (15-25 pieces): things the player can discover through searching, observing, or talking.
   - Mix of testimony, observation, physical, behavioral types
   - Some require prerequisites (must find evidence A before evidence B is available)
   - Each links to other evidence via connects_to
   - Evidence must be discoverable at specific locations/times/characters
   - Prerequisite chains must be ACYCLIC and achievable within a single loop

EVIDENCE DESIGN REQUIREMENTS:
- Include at least 2-3 pieces of evidence that could plausibly implicate the wrong person (misdirection)
- At least one piece of early evidence (Day 1-2) should gain new meaning when combined with later evidence
- Create a "smoking gun" evidence piece that requires 2+ prerequisites to find

2. INTERVENTIONS (5-8): actions the player can take to interrupt causal chain events.
   - Each targets a specific causal chain event
   - Requires specific evidence, location, day/slot
   - Some require conversation (persuading a character) with minimum trust
   - Success changes character schedules and cascades to interrupt downstream events

3. WILD CARDS (2-4): unpredictable events that shake up schedules.
   - Occur at specific day/slot
   - Override some character schedules temporarily
   - Create new opportunities or complications

Use ONLY location IDs from: main_lodge, dining_hall, kitchen, campfire_circle, cabin_row, \
staff_quarters, bathhouse, arts_cabin, archery_range, trail_head, lake_shore, dock, \
old_boathouse, maintenance_shed, storage_cellar, directors_office, infirmary

Return ONLY valid JSON:
{{
  "evidence_registry": [
    {{
      "id": "evidence_01",
      "type": "testimony|observation|physical|behavioral",
      "description": "what the player finds/learns",
      "source_location": "location_id",
      "source_character": "CharacterName or empty",
      "available_day": 1,
      "available_slot": "MORNING|AFTERNOON|EVENING|NIGHT",
      "prerequisites": ["evidence_ids"],
      "significance": "why this matters",
      "connects_to": ["other evidence_ids"]
    }}
  ],
  "intervention_tree": [
    {{
      "id": "intervention_01",
      "causal_event_id": "event_id from causal chain",
      "required_evidence": ["evidence_ids"],
      "required_location": "location_id",
      "required_day": 1,
      "required_slot": "MORNING",
      "action_description": "what the player does",
      "requires_conversation": false,
      "trust_required": 0,
      "success_schedule_changes": {{
        "CharacterName": [
          {{"location": "location_id", "activity": "new activity"}}
        ]
      }},
      "cascade_interrupts": ["other causal_event_ids"]
    }}
  ],
  "wild_cards": [
    {{
      "id": "wildcard_01",
      "day": 2,
      "time_slot": "EVENING",
      "description": "what happens",
      "schedule_overrides": {{
        "CharacterName": {{"location": "location_id", "activity": "what they do instead"}}
      }},
      "location_effects": {{}}
    }}
  ]
}}
"""
