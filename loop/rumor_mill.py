"""The Rumor Mill — information propagation across the NPC social network.

Claims flow through the camp like this:
1. Player says something to an NPC → LLM extracts structured claims
2. When time advances, NPCs at the same location share claims (deterministic)
3. When the player talks to an NPC, the NPC knows what they've heard
4. State machine decides behavioral consequences (schedule shifts, trust changes)

The LLM generates dialogue. The state machine moves the information.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid

if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

from .config import TOTAL_SLOTS, slot_index
from .models import (
    Character,
    CharacterTier,
    Claim,
    LoopState,
    WorldState,
)

# ── Propagation rules (deterministic) ─────────────────────────────────

# Probability that NPC A shares a claim with NPC B when co-located
# Keyed by (relationship_type, claim_involves_them)
# "relationship_type" is derived from whether A and B have a defined relationship

SHARE_CHANCE_BASE = 0.60         # strangers / acquaintances
SHARE_CHANCE_CLOSE = 0.85        # characters with defined relationships
SHARE_CHANCE_ABOUT_LISTENER = 0.95  # "hey, someone said something about YOU"
SHARE_CHANCE_TIER1_SUPPRESSES = 0.25  # tier1 chars suppress incriminating info

# Claims decay: after spreading N times, they stop propagating
MAX_SPREAD_HOPS = 4

# Schedule disruption: if an NPC hears something alarming about themselves
# they may change their next schedule slot (deterministic override)
ALARM_KEYWORDS = {"danger", "killed", "stolen", "hiding", "weapon", "poison",
                  "sabotage", "destroy", "fire", "explosive", "break", "attack"}

EXTRACT_CLAIMS_PROMPT = """\
Extract factual claims, rumors, accusations, or notable information shared \
in this conversation. Focus on SPECIFIC claims about PEOPLE or EVENTS — \
skip small talk and pleasantries.

Conversation between the player and {char_name}:
{conversation_text}

Return a JSON array of claims. Each claim:
- "subject": who or what the claim is about (a character name, "the camp", or a specific thing)
- "content": the claim itself, stated as a fact (e.g., "Dave was seen near the generator at midnight")
- "source_is_player": true if the player stated/implied this, false if the NPC revealed it
- "is_accusation": true if this accuses someone of wrongdoing
- "severity": "low", "medium", or "high" — how alarming is this claim?

Return ONLY the JSON array. If no notable claims, return [].
Example: [{{"subject": "Dave", "content": "was seen near the generator at midnight", "source_is_player": true, "is_accusation": false, "severity": "medium"}}]
"""


async def extract_claims_from_conversation(
    char_name: str,
    exchanges: list[dict],
    loop_state: LoopState,
) -> list[Claim]:
    """Use LLM to extract structured claims from a conversation."""
    if not exchanges:
        return []

    conversation_text = "\n".join(
        f"Player: {e['player']}\n{char_name}: {e['npc']}" for e in exchanges
    )

    prompt = EXTRACT_CLAIMS_PROMPT.format(
        char_name=char_name,
        conversation_text=conversation_text,
    )

    from .llm import llm_query
    result_text = (await llm_query(prompt)).strip()
    if result_text.startswith("```"):
        lines = result_text.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        result_text = "\n".join(lines[start:end])

    try:
        raw_claims = json.loads(result_text)
    except json.JSONDecodeError:
        return []

    if not isinstance(raw_claims, list):
        return []

    claims = []
    slot_idx = slot_index(loop_state.current_day, loop_state.current_slot.value)

    for raw in raw_claims:
        if not isinstance(raw, dict):
            continue
        subject = raw.get("subject", "").strip()
        content = raw.get("content", "").strip()
        if not subject or not content:
            continue

        source = "player" if raw.get("source_is_player", True) else char_name

        claim = Claim(
            id=f"claim_{uuid.uuid4().hex[:8]}",
            source=source,
            subject=subject,
            content=content,
            slot_created=slot_idx,
            heard_by=[char_name],  # the NPC in conversation heard it
        )
        claims.append(claim)

    return claims


def propagate_claims(
    world: WorldState,
    loop_state: LoopState,
    location_occupants: dict[str, list[str]],
) -> list[dict]:
    """Deterministic claim propagation between co-located NPCs.

    Called during advance_time(). Returns list of propagation events for display.

    Args:
        world: The world state
        loop_state: Current loop state (contains active_claims)
        location_occupants: {location_id: [char_names]} — who is where right now
    """
    char_map = {c.name: c for c in world.characters}
    events = []

    for loc_id, occupants in location_occupants.items():
        if len(occupants) < 2:
            continue

        for claim in loop_state.active_claims:
            if claim.spread_count >= MAX_SPREAD_HOPS:
                continue

            # Find NPCs at this location who KNOW this claim
            knowers = [name for name in occupants if name in claim.heard_by]
            # Find NPCs who DON'T know it yet
            ignorant = [name for name in occupants if name not in claim.heard_by
                        and name != "player"]

            if not knowers or not ignorant:
                continue

            spread_this_step = False

            for sharer in knowers:
                sharer_char = char_map.get(sharer)
                if not sharer_char:
                    continue

                for listener in ignorant:
                    if listener in claim.heard_by:
                        continue  # already knows (may have learned from another sharer)

                    listener_char = char_map.get(listener)
                    if not listener_char:
                        continue

                    # Determine share probability
                    chance = _share_chance(sharer_char, listener_char, claim)

                    # Deterministic: use hashlib for cross-session stability
                    seed = int(hashlib.md5(repr((claim.id, sharer, listener, loop_state.current_day,
                                 loop_state.current_slot.value)).encode()).hexdigest(), 16)
                    rng = random.Random(seed)

                    if rng.random() < chance:
                        claim.heard_by.append(listener)
                        spread_this_step = True

                        # Track in loop state
                        loop_state.npc_heard_claims.setdefault(listener, [])
                        if claim.id not in loop_state.npc_heard_claims[listener]:
                            loop_state.npc_heard_claims[listener].append(claim.id)

                        events.append({
                            "type": "rumor_spread",
                            "claim_id": claim.id,
                            "from": sharer,
                            "to": listener,
                            "location": loc_id,
                            "content": claim.content,
                            "subject": claim.subject,
                        })

            # Increment spread_count once per time step, not per listener
            if spread_this_step:
                claim.spread_count += 1

    # Check for alarm reactions (schedule disruptions)
    alarm_events = _check_alarm_reactions(world, loop_state, char_map)
    events.extend(alarm_events)

    return events


def _share_chance(sharer: Character, listener: Character, claim: Claim) -> float:
    """Deterministic share probability based on relationships and claim content."""
    # Base chance
    chance = SHARE_CHANCE_BASE

    # Close relationship bonus
    if listener.name in sharer.relationships or sharer.name in listener.relationships:
        chance = SHARE_CHANCE_CLOSE

    # Claim is about the listener — almost always shared
    if claim.subject.lower() == listener.name.lower():
        chance = SHARE_CHANCE_ABOUT_LISTENER

    # Tier1 characters suppress incriminating claims about themselves
    if (sharer.tier == CharacterTier.TIER1
            and claim.subject.lower() == sharer.name.lower()):
        chance = SHARE_CHANCE_TIER1_SUPPRESSES

    # Tier1 characters suppress claims that could expose the plot
    if sharer.tier == CharacterTier.TIER1:
        alarm_words = set(claim.content.lower().split()) & ALARM_KEYWORDS
        if alarm_words:
            chance *= 0.3  # heavily suppress alarming claims

    return chance


def _check_alarm_reactions(
    world: WorldState,
    loop_state: LoopState,
    char_map: dict[str, Character],
) -> list[dict]:
    """Check if any NPC has heard something alarming about themselves.

    If so, they may change their next schedule slot (deterministic).
    Returns alarm event dicts for display.
    """
    events = []

    for claim in loop_state.active_claims:
        # Check if the subject of the claim has heard it
        subject_char = char_map.get(claim.subject)
        if not subject_char:
            continue
        if claim.subject not in claim.heard_by:
            continue

        # Is the claim alarming?
        alarm_words = set(claim.content.lower().split()) & ALARM_KEYWORDS
        if not alarm_words:
            continue

        # Has this alarm already been processed?
        alarm_key = f"alarm_{claim.id}_{claim.subject}"
        if alarm_key in loop_state.events_triggered:
            continue
        loop_state.events_triggered.append(alarm_key)

        # Determine reaction based on tier
        if subject_char.tier == CharacterTier.TIER1:
            # Tier1 characters get evasive — go somewhere private
            from .models import ScheduleEntry
            current_idx = slot_index(loop_state.current_day, loop_state.current_slot.value)
            next_idx = current_idx + 1
            if next_idx < TOTAL_SLOTS:
                # Move to a private location
                private_locs = ["old_boathouse", "storage_cellar", "maintenance_shed"]
                seed = int(hashlib.md5(repr((claim.id, claim.subject)).encode()).hexdigest(), 16)
                rng = random.Random(seed)
                new_loc = rng.choice(private_locs)

                loop_state.schedule_modifications.setdefault(claim.subject, {})
                loop_state.schedule_modifications[claim.subject][str(next_idx)] = ScheduleEntry(
                    location=new_loc,
                    activity="hiding / destroying evidence"
                )

                events.append({
                    "type": "alarm_reaction",
                    "character": claim.subject,
                    "claim_id": claim.id,
                    "reaction": "evasive",
                    "new_location": new_loc,
                    "detail": f"{claim.subject} heard the rumor and slipped away",
                })

        elif subject_char.tier == CharacterTier.TIER2:
            # Tier2 characters get anxious — seek authority figures
            from .models import ScheduleEntry
            current_idx = slot_index(loop_state.current_day, loop_state.current_slot.value)
            next_idx = current_idx + 1
            if next_idx < TOTAL_SLOTS:
                loop_state.schedule_modifications.setdefault(claim.subject, {})
                loop_state.schedule_modifications[claim.subject][str(next_idx)] = ScheduleEntry(
                    location="directors_office",
                    activity="seeking reassurance / reporting concerns"
                )

                events.append({
                    "type": "alarm_reaction",
                    "character": claim.subject,
                    "claim_id": claim.id,
                    "reaction": "anxious",
                    "new_location": "directors_office",
                    "detail": f"{claim.subject} heard something disturbing and went to the director",
                })

    return events


def get_claims_known_by(npc_name: str, loop_state: LoopState) -> list[Claim]:
    """Get all claims an NPC has heard, for injection into conversation prompts."""
    claim_map = {c.id: c for c in loop_state.active_claims}
    known_ids = loop_state.npc_heard_claims.get(npc_name, [])
    return [claim_map[cid] for cid in known_ids if cid in claim_map]


def format_claims_for_prompt(claims: list[Claim], npc_name: str) -> str:
    """Format claims into a section for the NPC's system prompt."""
    if not claims:
        return ""

    lines = ["\nTHINGS YOU'VE HEARD FROM OTHERS (gossip, rumors, reports):"]

    for claim in claims:
        source_label = "the new visitor" if claim.source == "player" else claim.source
        about_you = " [THIS IS ABOUT YOU]" if claim.subject.lower() == npc_name.lower() else ""

        lines.append(f"- {source_label} said: \"{claim.subject} {claim.content}\"{about_you}")

    lines.append("")
    lines.append(
        "React naturally to what you've heard. If someone told you something alarming, "
        "you'd be thinking about it. If it's about you, you might be defensive, "
        "curious how the player knows, or want to set the record straight. "
        "Don't dump all this info at once — bring it up if relevant to the conversation."
    )

    return "\n".join(lines)
