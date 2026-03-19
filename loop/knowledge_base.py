"""Cross-loop knowledge persistence."""

from __future__ import annotations

from .models import (
    Evidence,
    EvidenceConnection,
    LoopSummary,
    PersistentKnowledge,
    WorldState,
)


class KnowledgeBase:
    """Manages persistent knowledge across loops."""

    def __init__(self, knowledge: PersistentKnowledge, world: WorldState):
        self.knowledge = knowledge
        self.world = world
        self._evidence_map = {e.id: e for e in world.evidence_registry}

    # ── Evidence ───────────────────────────────────────────────────────

    def discover_evidence(self, evidence_id: str) -> Evidence | None:
        """Record newly discovered evidence. Returns the Evidence or None if already known."""
        if evidence_id in self.knowledge.evidence_discovered:
            return None
        ev = self._evidence_map.get(evidence_id)
        if ev:
            self.knowledge.evidence_discovered.append(evidence_id)
        return ev

    def has_evidence(self, evidence_id: str) -> bool:
        return evidence_id in self.knowledge.evidence_discovered

    def get_discovered_evidence(self) -> list[Evidence]:
        return [self._evidence_map[eid] for eid in self.knowledge.evidence_discovered
                if eid in self._evidence_map]

    # ── Schedule observations ──────────────────────────────────────────

    def record_schedule_observation(self, char_name: str, slot_index: int, location_id: str):
        self.knowledge.character_schedules_known.setdefault(char_name, {})[str(slot_index)] = location_id

    # ── Character meetings ─────────────────────────────────────────────

    def record_character_met(self, char_name: str):
        if char_name not in self.knowledge.characters_met:
            self.knowledge.characters_met.append(char_name)

    # ── Evidence connections ───────────────────────────────────────────

    def add_connection(self, evidence_a: str, evidence_b: str, description: str = "") -> bool:
        """Add a player-drawn connection between two pieces of evidence."""
        # Reject self-connections
        if evidence_a == evidence_b:
            return False
        # Check both are discovered
        if evidence_a not in self.knowledge.evidence_discovered:
            return False
        if evidence_b not in self.knowledge.evidence_discovered:
            return False
        # Check not duplicate
        for conn in self.knowledge.evidence_connections:
            if ({conn.evidence_a, conn.evidence_b} == {evidence_a, evidence_b}):
                return False
        self.knowledge.evidence_connections.append(
            EvidenceConnection(evidence_a=evidence_a, evidence_b=evidence_b,
                               description=description)
        )
        return True

    def remove_connection(self, evidence_a: str, evidence_b: str) -> bool:
        for i, conn in enumerate(self.knowledge.evidence_connections):
            if {conn.evidence_a, conn.evidence_b} == {evidence_a, evidence_b}:
                self.knowledge.evidence_connections.pop(i)
                return True
        return False

    def confirm_connection(self, evidence_a: str, evidence_b: str) -> bool:
        """Mark a connection as confirmed (matches actual evidence links)."""
        for conn in self.knowledge.evidence_connections:
            if {conn.evidence_a, conn.evidence_b} == {evidence_a, evidence_b}:
                # Check if it matches actual connects_to in the evidence
                ev_a = self._evidence_map.get(conn.evidence_a)
                ev_b = self._evidence_map.get(conn.evidence_b)
                if ev_a and (conn.evidence_b in ev_a.connects_to):
                    conn.confirmed = True
                    return True
                if ev_b and (conn.evidence_a in ev_b.connects_to):
                    conn.confirmed = True
                    return True
        return False

    # ── Theories ───────────────────────────────────────────────────────

    def add_theory(self, theory: str):
        if theory not in self.knowledge.theories:
            self.knowledge.theories.append(theory)

    def remove_theory(self, index: int) -> bool:
        if 0 <= index < len(self.knowledge.theories):
            self.knowledge.theories.pop(index)
            return True
        return False

    # ── Pinning ────────────────────────────────────────────────────────

    def pin_character(self, char_name: str):
        if char_name not in self.knowledge.pinned_characters:
            self.knowledge.pinned_characters.append(char_name)

    def unpin_character(self, char_name: str):
        if char_name in self.knowledge.pinned_characters:
            self.knowledge.pinned_characters.remove(char_name)

    # ── Conversation journal ────────────────────────────────────────────

    def record_conversation_summary(self, loop_number: int, day: int, slot: str,
                                     char_name: str, exchanges: int, trust: int,
                                     topics: list[str]):
        """Record a conversation summary in the knowledge base."""
        summary = (f"Loop {loop_number}, Day {day} {slot}: Spoke with {char_name} "
                   f"({exchanges} exchanges). Trust: {trust}.")
        if topics:
            summary += f" Topics: {', '.join(topics[:3])}."
        self.knowledge.conversation_journal.append(summary)

    # ── Player knowledge flags (for conversation engine) ───────────────

    def get_player_knowledge_flags(self, char_name: str, current_slot_index: int) -> list[str]:
        """Facts the player knows that a character wouldn't expect at this point."""
        flags = []
        # Future schedule knowledge
        known = self.knowledge.character_schedules_known.get(char_name, {})
        for idx_str, loc in known.items():
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if idx > current_slot_index:
                flags.append(f"Knows {char_name} will be at {loc} later")
        return flags

    # ── Stats ──────────────────────────────────────────────────────────

    def get_knowledge_percentage(self) -> float:
        total = len(self.world.evidence_registry)
        if total == 0:
            return 0.0
        valid_count = sum(1 for e in self.knowledge.evidence_discovered if e in self._evidence_map)
        return valid_count / total

    # ── Loop completion ────────────────────────────────────────────────

    def complete_loop(self, loop_number: int, ending: str, interventions: list[str]) -> LoopSummary:
        summary = LoopSummary(
            loop_number=loop_number,
            evidence_found=list(self.knowledge.evidence_discovered),
            interventions_attempted=interventions,
            ending_reached=ending,
        )
        self.knowledge.loop_history.append(summary)
        return summary
