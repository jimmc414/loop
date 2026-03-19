"""Deterministic clockwork engine — no LLM calls, pure game logic."""

from __future__ import annotations

import random
from .config import (
    DAYS_PER_LOOP,
    FOLLOW_BASE_DETECTION,
    FOLLOW_HIGH_TRUST_REDUCTION,
    FOLLOW_TIER1_BONUS,
    MAX_LOOPS,
    SLOTS_PER_DAY,
    TOTAL_SLOTS,
    TRUST_FLOOR,
    slot_index,
)
from .models import (
    CausalChainEvent,
    Character,
    CharacterTier,
    EndingCondition,
    EndingType,
    Evidence,
    InterventionNode,
    Location,
    LoopState,
    PersistentKnowledge,
    ScheduleEntry,
    TimeSlot,
    WorldState,
)
from .rumor_mill import propagate_claims

SLOT_ORDER = [TimeSlot.MORNING, TimeSlot.AFTERNOON, TimeSlot.EVENING, TimeSlot.NIGHT]


class ClockworkEngine:
    """Pure-Python state machine driving all game logic."""

    def __init__(self, world: WorldState, loop: LoopState, knowledge: PersistentKnowledge):
        self.world = world
        self.loop = loop
        self.knowledge = knowledge
        self._location_map: dict[str, Location] = {loc.id: loc for loc in world.locations}
        self._character_map: dict[str, Character] = {c.name: c for c in world.characters}
        self._event_map: dict[str, CausalChainEvent] = {e.id: e for e in world.causal_chain}
        self._evidence_map: dict[str, Evidence] = {e.id: e for e in world.evidence_registry}
        self._intervention_map: dict[str, InterventionNode] = {i.id: i for i in world.intervention_tree}

    # ── Fast-forward ────────────────────────────────────────────────────

    def can_fast_forward(self) -> bool:
        """Player can fast-forward after loop 3."""
        return self.loop.loop_number >= 4

    def fast_forward_to(self, target_day: int, target_slot: TimeSlot) -> dict:
        """Skip ahead to a specific day/slot, consuming time slots.

        Returns dict with slots_consumed count and any events triggered.
        """
        slots_consumed = 0
        events_during = []

        while (self.loop.current_day < target_day or
               (self.loop.current_day == target_day and
                SLOT_ORDER.index(self.loop.current_slot) < SLOT_ORDER.index(target_slot))):
            result = self.advance_time()
            slots_consumed += 1

            if result["type"] == "loop_end":
                return {"slots_consumed": slots_consumed, "loop_ended": True, "result": result}

            # Collect rumor events
            events_during.extend(result.get("rumor_events", []))

        return {"slots_consumed": slots_consumed, "loop_ended": False, "events": events_during}

    # ── Schedule queries ───────────────────────────────────────────────

    def _current_slot_index(self) -> int:
        return slot_index(self.loop.current_day, self.loop.current_slot.value)

    def get_character_location(self, char_name: str, day: int | None = None,
                                slot: TimeSlot | None = None) -> ScheduleEntry | None:
        """Get where a character is. Checks modifications, wild cards, then default schedule."""
        d = day if day is not None else self.loop.current_day
        s = slot if slot is not None else self.loop.current_slot
        idx = slot_index(d, s.value)

        # Check schedule modifications (from interventions)
        if char_name in self.loop.schedule_modifications:
            if str(idx) in self.loop.schedule_modifications[char_name]:
                return self.loop.schedule_modifications[char_name][str(idx)]

        # Check wild card overrides
        for wc in self.world.wild_cards:
            if wc.day == d and wc.time_slot == s and char_name in wc.schedule_overrides:
                return wc.schedule_overrides[char_name]

        # Default schedule
        char = self._character_map.get(char_name)
        if char and 0 <= idx < len(char.schedule):
            return char.schedule[idx]
        return None

    def get_characters_at_location(self, location_id: str,
                                    day: int | None = None,
                                    slot: TimeSlot | None = None) -> list[str]:
        """Return character names present at a location."""
        result = []
        for char in self.world.characters:
            entry = self.get_character_location(char.name, day, slot)
            if entry and entry.location == location_id:
                result.append(char.name)
        return result

    def get_location(self, location_id: str) -> Location | None:
        return self._location_map.get(location_id)

    def get_adjacent_locations(self, location_id: str) -> list[Location]:
        loc = self._location_map.get(location_id)
        if not loc:
            return []
        return [self._location_map[adj] for adj in loc.adjacent_locations
                if adj in self._location_map]

    # ── Time advancement ───────────────────────────────────────────────

    def _get_location_occupants(self) -> dict[str, list[str]]:
        """Build a map of location_id -> [character names] for current time."""
        occupants: dict[str, list[str]] = {}
        for char in self.world.characters:
            entry = self.get_character_location(char.name)
            if entry:
                occupants.setdefault(entry.location, []).append(char.name)
        return occupants

    def advance_time(self) -> dict:
        """Advance to next slot. Returns status dict."""
        # Propagate rumors BEFORE advancing (NPCs share info while co-located)
        rumor_events = []
        if self.loop.active_claims:
            occupants = self._get_location_occupants()
            rumor_events = propagate_claims(self.world, self.loop, occupants)

        current_idx = SLOT_ORDER.index(self.loop.current_slot)

        if current_idx < len(SLOT_ORDER) - 1:
            # Next slot same day
            self.loop.current_slot = SLOT_ORDER[current_idx + 1]
            self._check_wild_cards()
            self._check_causal_events()
            return {"type": "slot_advance", "day": self.loop.current_day,
                    "slot": self.loop.current_slot, "rumor_events": rumor_events}

        elif self.loop.current_day < DAYS_PER_LOOP:
            # Next day
            self.loop.current_day += 1
            self.loop.current_slot = TimeSlot.MORNING
            self._check_wild_cards()
            self._check_causal_events()
            return {"type": "day_advance", "day": self.loop.current_day,
                    "slot": self.loop.current_slot, "rumor_events": rumor_events}

        else:
            # End of loop — check catastrophe
            catastrophe = self.check_catastrophe()
            return {"type": "loop_end", "catastrophe": catastrophe,
                    "rumor_events": rumor_events}

    def _check_wild_cards(self):
        """Trigger wild card events at current day/slot."""
        for wc in self.world.wild_cards:
            if wc.day == self.loop.current_day and wc.time_slot == self.loop.current_slot:
                if wc.id not in self.loop.events_triggered:
                    self.loop.events_triggered.append(wc.id)

    def _check_causal_events(self):
        """Mark causal events as triggered if they fire at current time and aren't interrupted."""
        for event in self.world.causal_chain:
            if (event.day == self.loop.current_day
                    and event.time_slot == self.loop.current_slot
                    and event.id not in self.loop.events_interrupted
                    and event.id not in self.loop.events_triggered):
                self.loop.events_triggered.append(event.id)

    # ── Catastrophe check ──────────────────────────────────────────────

    def check_catastrophe(self) -> bool:
        """Return True if catastrophe occurs (uninterrupted chain reaches Day 5)."""
        # Find the final event(s) in the chain
        final_events = [e for e in self.world.causal_chain
                        if e.day == self.world.catastrophe_day
                        and e.time_slot == self.world.catastrophe_slot]

        memo: dict[str, bool] = {}
        for event in final_events:
            if event.id not in self.loop.events_interrupted:
                # Check if the chain leading to this event is intact
                if self._chain_intact(event.id, memo):
                    return True
        return False

    def _chain_intact(self, event_id: str, memo: dict[str, bool]) -> bool:
        """Check if an event and all its upstream dependencies are uninterrupted."""
        if event_id in memo:
            return memo[event_id]
        if event_id in self.loop.events_interrupted:
            memo[event_id] = False
            return False
        # Mark as provisionally intact before recursing to prevent infinite loops on cycles
        memo[event_id] = True
        # Find events that feed into this one
        for event in self.world.causal_chain:
            if event_id in event.downstream_effects:
                if not self._chain_intact(event.id, memo):
                    memo[event_id] = False
                    return False
        return True

    # ── Available actions ──────────────────────────────────────────────

    def get_available_actions(self) -> list[dict]:
        """Return valid actions at current location/time."""
        actions = []
        loc_id = self.loop.player_location
        chars_here = self.get_characters_at_location(loc_id)

        # Talk to characters present
        for name in chars_here:
            actions.append({"type": "talk", "target": name,
                            "label": f"Talk to {name}"})

        # Observe the area
        actions.append({"type": "observe", "label": "Observe the area"})

        # Search — always available (evidence can be anywhere)
        actions.append({"type": "search", "label": "Search the area"})

        # Follow a character (if someone is leaving)
        for name in chars_here:
            actions.append({"type": "follow", "target": name,
                            "label": f"Follow {name}"})

        # Intervene (if conditions met)
        for intervention in self._get_available_interventions():
            actions.append({"type": "intervene", "target": intervention.id,
                            "label": f"Intervene: {intervention.action_description}"})

        # Travel to adjacent locations
        for adj in self.get_adjacent_locations(loc_id):
            actions.append({"type": "travel", "target": adj.id,
                            "label": f"Go to {adj.name}",
                            "costs_slot": adj.is_distant})

        # Fast-forward (available after loop 3)
        if self.can_fast_forward():
            actions.append({"type": "fast_forward", "label": "Fast-forward to specific time"})

        # Meta actions
        actions.append({"type": "evidence_board", "label": "Check evidence board"})
        actions.append({"type": "schedule_tracker", "label": "Check schedule tracker"})
        actions.append({"type": "map", "label": "View camp map"})
        actions.append({"type": "wait", "label": "Wait (advance time)"})

        return actions

    # ── Follow mechanic ────────────────────────────────────────────────

    def resolve_follow(self, char_name: str) -> dict:
        """Attempt to follow a character. Returns result dict."""
        char = self._character_map.get(char_name)
        if not char:
            return {"success": False, "reason": "Character not found"}

        # Detection chance
        detection = FOLLOW_BASE_DETECTION
        if char.tier == CharacterTier.TIER1:
            detection += FOLLOW_TIER1_BONUS
        trust = self.loop.character_trust.get(char_name, 0)
        if trust >= 40:
            detection -= FOLLOW_HIGH_TRUST_REDUCTION

        detected = random.random() < detection

        # Get their next location
        next_idx = self._current_slot_index() + 1
        if next_idx >= TOTAL_SLOTS:
            return {"success": False, "reason": "End of loop — nowhere to follow"}

        day, slot_name = next_idx // SLOTS_PER_DAY + 1, SLOT_ORDER[next_idx % SLOTS_PER_DAY]
        next_entry = self.get_character_location(char_name, day, slot_name)

        if not next_entry:
            return {"success": False, "reason": "Lost track of them"}

        if detected:
            # Trust penalty (clamped to trust floor)
            self.loop.character_trust[char_name] = max(TRUST_FLOOR, trust - 5)
            return {
                "success": False,
                "detected": True,
                "next_location": next_entry.location,
                "activity": next_entry.activity,
                "reason": f"{char_name} noticed you following them",
            }

        # Record observation
        self.knowledge.character_schedules_known.setdefault(char_name, {})[str(next_idx)] = next_entry.location

        return {
            "success": True,
            "detected": False,
            "next_location": next_entry.location,
            "activity": next_entry.activity,
        }

    # ── Search mechanic ────────────────────────────────────────────────

    def resolve_search(self, location_id: str) -> dict:
        """Search a location for evidence."""
        chars_here = self.get_characters_at_location(location_id)
        observers = [c for c in chars_here if self._character_map.get(c)
                     and self._character_map[c].tier in (CharacterTier.TIER1, CharacterTier.TIER2)]

        # Safety check — more observers = riskier
        is_risky = len(observers) > 0

        # Find available evidence at this location/time
        found = []
        idx = self._current_slot_index()
        for ev in self.world.evidence_registry:
            if ev.id in self.knowledge.evidence_discovered:
                continue
            if ev.source_location != location_id:
                continue
            ev_idx = slot_index(ev.available_day, ev.available_slot.value)
            if idx < ev_idx:
                continue
            # Check prerequisites
            if all(p in self.knowledge.evidence_discovered for p in ev.prerequisites):
                found.append(ev)

        # Record search
        self.knowledge.locations_searched.setdefault(location_id, []).append(idx)

        return {
            "found": found,
            "is_risky": is_risky,
            "observers": observers,
        }

    # ── Intervention ───────────────────────────────────────────────────

    def _get_available_interventions(self) -> list[InterventionNode]:
        """Interventions available at current state."""
        available = []
        for iv in self.world.intervention_tree:
            if iv.id in self.loop.interventions_made:
                continue
            if iv.required_location and iv.required_location != self.loop.player_location:
                continue
            if iv.required_day != self.loop.current_day:
                continue
            if iv.required_slot != self.loop.current_slot:
                continue
            if not all(e in self.knowledge.evidence_discovered for e in iv.required_evidence):
                continue
            if iv.trust_required > 0:
                # Find the character associated with the causal event
                event = self._event_map.get(iv.causal_event_id)
                if event:
                    trust = self.loop.character_trust.get(event.character, 0)
                    if trust < iv.trust_required:
                        continue
            available.append(iv)
        return available

    def get_near_miss_hints(self) -> list[str]:
        """Return hints for interventions the player is close to qualifying for."""
        hints = []
        for iv in self.world.intervention_tree:
            if iv.id in self.loop.interventions_made:
                continue

            # Count how many conditions are met
            conditions_met = 0
            conditions_total = 0
            unmet_reason = ""

            # Location check
            conditions_total += 1
            if not iv.required_location or iv.required_location == self.loop.player_location:
                conditions_met += 1
            else:
                unmet_reason = "location"

            # Day/slot check
            conditions_total += 1
            if iv.required_day == self.loop.current_day and iv.required_slot == self.loop.current_slot:
                conditions_met += 1
            else:
                if iv.required_day == self.loop.current_day:
                    unmet_reason = "time"
                else:
                    unmet_reason = "day"

            # Evidence check
            conditions_total += 1
            if all(e in self.knowledge.evidence_discovered for e in iv.required_evidence):
                conditions_met += 1
            else:
                unmet_reason = "evidence"

            # Trust check
            if iv.trust_required > 0:
                conditions_total += 1
                event = self._event_map.get(iv.causal_event_id)
                if event:
                    trust = self.loop.character_trust.get(event.character, 0)
                    if trust >= iv.trust_required:
                        conditions_met += 1
                    else:
                        unmet_reason = "trust"

            # Near miss: all but one condition met
            if conditions_met >= conditions_total - 1 and conditions_met < conditions_total:
                if unmet_reason == "location":
                    loc = self._location_map.get(iv.required_location)
                    loc_name = loc.name if loc else iv.required_location.replace('_', ' ').title()
                    hints.append(f"You sense something could happen at {loc_name}...")
                elif unmet_reason == "time":
                    from .config import SLOT_NAMES
                    hints.append(f"You feel like this place might be important at a different time today...")
                elif unmet_reason == "day":
                    hints.append(f"Your evidence might be useful here, but not today...")
                elif unmet_reason == "evidence":
                    hints.append(f"You feel close to understanding something, but you're missing a key piece...")
                elif unmet_reason == "trust":
                    event = self._event_map.get(iv.causal_event_id)
                    if event:
                        hints.append(f"You feel like {event.character} isn't ready to hear what you have to say yet...")

        return hints

    def apply_intervention(self, intervention_id: str) -> dict:
        """Execute an intervention. Returns result dict."""
        iv = self._intervention_map.get(intervention_id)
        if not iv:
            return {"success": False, "reason": "Intervention not found"}

        # Mark as done
        self.loop.interventions_made.append(iv.id)

        # Interrupt the causal event
        self.loop.events_interrupted.append(iv.causal_event_id)

        # Apply schedule changes
        for char_name, entries in iv.success_schedule_changes.items():
            if char_name not in self.loop.schedule_modifications:
                self.loop.schedule_modifications[char_name] = {}
            for i, entry in enumerate(entries):
                # Apply from current slot onward, clamped to valid range
                start_idx = self._current_slot_index()
                target_idx = start_idx + i
                if target_idx < TOTAL_SLOTS:
                    self.loop.schedule_modifications[char_name][str(target_idx)] = entry

        # Cascade interrupts
        for event_id in iv.cascade_interrupts:
            if event_id not in self.loop.events_interrupted:
                self.loop.events_interrupted.append(event_id)

        return {
            "success": True,
            "interrupted_event": iv.causal_event_id,
            "cascades": iv.cascade_interrupts,
        }

    # ── Travel ─────────────────────────────────────────────────────────

    def resolve_travel(self, destination_id: str) -> dict:
        """Move player to a new location."""
        dest = self._location_map.get(destination_id)
        if not dest:
            return {"success": False, "reason": "Unknown location"}

        current_loc = self._location_map.get(self.loop.player_location)
        if not current_loc:
            return {"success": False, "reason": "Current location invalid"}

        if destination_id not in current_loc.adjacent_locations:
            return {"success": False, "reason": "Not adjacent to current location"}

        if dest.locked:
            return {"success": False, "reason": f"Locked: {dest.lock_conditions}"}

        self.loop.player_location = destination_id

        return {
            "success": True,
            "costs_slot": dest.is_distant,
            "location": dest,
        }

    # ── Ending evaluation ──────────────────────────────────────────────

    def evaluate_ending(self) -> EndingCondition:
        """Determine which ending the player gets."""
        # Sort by priority: deeper_truth > full_prevention > partial > exposure > failure
        priority = [EndingType.DEEPER_TRUTH, EndingType.FULL_PREVENTION,
                    EndingType.PARTIAL_PREVENTION, EndingType.EXPOSURE, EndingType.FAILURE]

        for etype in priority:
            for cond in self.world.ending_conditions:
                if cond.type != etype:
                    continue
                # Check required interrupted events
                events_ok = all(e in self.loop.events_interrupted
                                for e in cond.required_interrupted_events)
                # Check required evidence
                evidence_ok = all(e in self.knowledge.evidence_discovered
                                  for e in cond.required_evidence)
                if events_ok and evidence_ok:
                    return cond

        # Default to failure
        return EndingCondition(type=EndingType.FAILURE, description="The catastrophe unfolds.")

    # ── Loop reset ─────────────────────────────────────────────────────

    def reset_loop(self) -> bool:
        """Reset for a new loop. Returns False if max loops exceeded."""
        if self.loop.loop_number >= MAX_LOOPS:
            return False

        # Record loop summary
        from .models import LoopSummary
        summary = LoopSummary(
            loop_number=self.loop.loop_number,
            evidence_found=list(self.knowledge.evidence_discovered),
            interventions_attempted=list(self.loop.interventions_made),
            ending_reached=self.evaluate_ending().type.value,
        )
        self.knowledge.loop_history.append(summary)

        # Reset loop state (keep knowledge)
        self.loop.loop_number += 1
        self.loop.current_day = 1
        self.loop.current_slot = TimeSlot.MORNING
        self.loop.player_location = "main_lodge"
        # Graduated trust carry-over: reward long-term relationship building
        previously_met = set(self.knowledge.characters_met)
        self.loop.character_trust.clear()
        if self.loop.loop_number <= 3:
            trust_bonus = 10
        elif self.loop.loop_number <= 5:
            trust_bonus = 15
        else:
            trust_bonus = 20
        for name in previously_met:
            base = trust_bonus
            if self.knowledge.npc_interaction_counts.get(name, 0) >= 15:
                base += 5
            self.loop.character_trust[name] = base
        self.loop.interventions_made.clear()
        self.loop.schedule_modifications.clear()
        self.loop.events_triggered.clear()
        self.loop.events_interrupted.clear()
        # Accumulate cross-loop NPC interaction counts (déjà vu tracking)
        for char_name, count in self.loop.conversations_this_loop.items():
            self.knowledge.npc_interaction_counts[char_name] = (
                self.knowledge.npc_interaction_counts.get(char_name, 0) + count
            )

        # Extract topics from conversation journal for current loop
        import re as _re
        journal_pattern = _re.compile(
            r"Loop\s+(\d+).*?Spoke with\s+([\w\s]+?)\s*\(.*?Topics?:\s*(.+?)\.?\s*$"
        )
        for entry in self.knowledge.conversation_journal:
            m = journal_pattern.search(entry)
            if m and int(m.group(1)) == self.loop.loop_number - 1:
                name = m.group(2).strip()
                topics = [t.strip() for t in m.group(3).split(",") if t.strip()]
                existing = self.knowledge.npc_previous_topics.get(name, [])
                for topic in topics:
                    if topic not in existing and len(existing) < 10:
                        existing.append(topic)
                self.knowledge.npc_previous_topics[name] = existing

        self.loop.conversations_this_loop.clear()
        self.loop.active_claims.clear()
        self.loop.npc_heard_claims.clear()

        return True
