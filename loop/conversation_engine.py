"""LLM-powered NPC conversations with deterministic trust heuristics."""

from __future__ import annotations

import os
import re

if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

from .config import (
    CONFRONT_TRUST_THRESHOLD,
    MAX_CONVERSATION_EXCHANGES,
    MAX_EVIDENCE_PRESENTATIONS,
    TOTAL_SLOTS,
    TRUST_DELTA_ACCUSATION_HARSH,
    TRUST_DELTA_ACCUSATION_MILD,
    TRUST_DELTA_EMPATHY,
    TRUST_DELTA_IMPOSSIBLE_EXPLAINED,
    TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED,
    TRUST_DELTA_PRESSURE,
    TRUST_DELTA_REMEMBERED,
    TRUST_FLOOR,
    slot_index,
)
from .display import GameDisplay
from .models import (
    Character,
    CharacterTier,
    ConversationExchange,
    ConversationResult,
    Evidence,
    KnowledgeEntry,
    LoopState,
    PersistentKnowledge,
    ScheduleEntry,
    WorldState,
)
from .prompts.conversation import (
    build_conversation_prompt,
    build_evidence_confrontation_prompt,
    build_system_prompt,
)
from .rumor_mill import extract_claims_from_conversation, format_claims_for_prompt, get_claims_known_by


# ── Trust heuristic patterns ──────────────────────────────────────────

EMPATHY_PATTERNS = re.compile(
    r"\b(sorry|understand|how are you|must be hard|that sounds|i hear you|"
    r"are you okay|you alright|take your time|no rush)\b", re.IGNORECASE
)
REMEMBERED_THRESHOLD = 3  # min word length for echoed words

ACCUSATION_MILD_PATTERNS = re.compile(
    r"\b(admit it|confess|you(?:'re| are) hiding something|not telling me|what aren't you)\b", re.IGNORECASE
)
ACCUSATION_HARSH_PATTERNS = re.compile(
    r"\b(you(?:'re| are) (?:a )?(?:liar|guilty|murderer)|your fault|you did this|you caused|i blame you)\b", re.IGNORECASE
)
PRESSURE_PATTERNS = re.compile(
    r"\b(tell me now|you must|you have to|i demand|answer me|spit it out|"
    r"don't lie|stop lying)\b", re.IGNORECASE
)


class ConversationEngine:
    """Manages NPC conversations with LLM dialogue and deterministic trust."""

    def __init__(self, world: WorldState, display: GameDisplay):
        self.world = world
        self.display = display
        self._char_map = {c.name: c for c in world.characters}
        self._evidence_keywords: dict[str, list[str]] = {}
        self._build_evidence_keywords()

    def _build_evidence_keywords(self):
        """Build keyword lookup for evidence topic extraction."""
        for ev in self.world.evidence_registry:
            words = set(re.findall(r"\b\w{4,}\b", ev.description.lower()))
            self._evidence_keywords[ev.id] = list(words)

    # ── Evidence presentation methods ─────────────────────────────────

    def _classify_evidence_relevance(self, evidence: Evidence, character: Character) -> str:
        """Deterministic relevance classification of evidence to an NPC."""
        name_lower = character.name.lower()

        # source — evidence.source_character matches NPC name
        if evidence.source_character and evidence.source_character.lower() == name_lower:
            return "source"

        # subject — NPC name appears in evidence.description
        if name_lower in evidence.description.lower():
            return "subject"

        # connected — evidence.connects_to references evidence whose source is this NPC,
        # OR keyword overlap between evidence description and NPC's secrets
        ev_map = {e.id: e for e in self.world.evidence_registry}
        for connected_id in evidence.connects_to:
            connected_ev = ev_map.get(connected_id)
            if connected_ev and connected_ev.source_character.lower() == name_lower:
                return "connected"

        # Keyword overlap between evidence description and NPC secrets
        ev_words = set(re.findall(r"\b\w{4,}\b", evidence.description.lower()))
        for secret in character.secrets:
            secret_words = set(re.findall(r"\b\w{4,}\b", secret.lower()))
            if len(ev_words & secret_words) >= 2:
                return "connected"

        return "unrelated"

    def _compute_evidence_trust_delta(
        self, relevance: str, character: Character, trust: int,
    ) -> int:
        """Lookup trust delta from the confrontation matrix."""
        # Trust delta matrix: relevance x (tier, trust_bracket)
        matrix = {
            "source":    {CharacterTier.TIER1: -5, CharacterTier.TIER2: (-2, 3), CharacterTier.TIER3: 5},
            "subject":   {CharacterTier.TIER1: -3, CharacterTier.TIER2: (-1, 2), CharacterTier.TIER3: 4},
            "connected": {CharacterTier.TIER1: -2, CharacterTier.TIER2: (0, 1),  CharacterTier.TIER3: 3},
            "unrelated": {CharacterTier.TIER1: 0,  CharacterTier.TIER2: (0, 0),  CharacterTier.TIER3: 0},
        }
        row = matrix.get(relevance, matrix["unrelated"])
        value = row[character.tier]
        if isinstance(value, tuple):
            # TIER2: (low_trust, high_trust) split at threshold
            return value[1] if trust >= CONFRONT_TRUST_THRESHOLD else value[0]
        return value

    def _get_tier_guidance(self, character: Character, trust: int) -> str:
        """Behavioural instruction string for the LLM based on tier and trust bracket."""
        if trust >= 60:
            bracket = "high"
        elif trust >= CONFRONT_TRUST_THRESHOLD:
            bracket = "mid"
        else:
            bracket = "low"

        guidance = {
            (CharacterTier.TIER1, "low"):
                "You are defensive and evasive. Deny involvement, deflect blame, and try to discredit the evidence. You may become hostile.",
            (CharacterTier.TIER1, "mid"):
                "You are uneasy but try to maintain composure. Offer partial explanations while still hiding the full truth. Be guarded.",
            (CharacterTier.TIER1, "high"):
                "You are shaken — this person has earned some trust, and the evidence is hard to dismiss. Offer a deflection but let cracks show in your story.",
            (CharacterTier.TIER2, "low"):
                "You are nervous and reluctant. You don't trust the player enough to open up. Give vague, non-committal responses.",
            (CharacterTier.TIER2, "mid"):
                "You are conflicted. You want to help but are scared of consequences. Hint at what you know without fully committing.",
            (CharacterTier.TIER2, "high"):
                "The evidence and trust level break through your resistance. You crack and reveal what you know, possibly with emotion.",
            (CharacterTier.TIER3, "low"):
                "You have no real connection to this evidence. Respond with mild curiosity or confusion. You can't help much.",
            (CharacterTier.TIER3, "mid"):
                "You are intrigued by the evidence. Offer your honest observations, even if they are speculative.",
            (CharacterTier.TIER3, "high"):
                "You are engaged and helpful. Share any tangentially related observations freely and encourage the player's investigation.",
        }
        return guidance.get(
            (character.tier, bracket),
            "React naturally to the evidence being presented.",
        )

    async def _handle_evidence_presentation(
        self,
        evidence_id: str,
        character: Character,
        trust: int,
        loop_state: LoopState,
        knowledge: PersistentKnowledge,
        history: list[dict],
        sys_prompt: str,
    ) -> tuple[str, int, list[str]]:
        """Orchestrate the full evidence presentation flow.

        Returns (npc_response, trust_delta, newly_discovered_evidence_ids).
        """
        ev_map = {e.id: e for e in self.world.evidence_registry}
        evidence = ev_map.get(evidence_id)

        # Validate: evidence exists
        if not evidence:
            return f"(You don't have any evidence called '{evidence_id}'.)", 0, []

        # Validate: evidence is discovered
        if evidence_id not in knowledge.evidence_discovered:
            return "(You haven't discovered that evidence yet.)", 0, []

        # Classify and compute
        relevance = self._classify_evidence_relevance(evidence, character)
        delta = self._compute_evidence_trust_delta(relevance, character, trust)
        tier_guidance = self._get_tier_guidance(character, trust)

        # Build confrontation prompt and get LLM response
        confront_prompt = build_evidence_confrontation_prompt(
            character, evidence, trust, relevance, tier_guidance,
        )
        user_prompt = build_conversation_prompt(
            history, confront_prompt, character.name,
        )
        npc_response = await self._get_npc_response(sys_prompt, user_prompt)

        # Check for evidence discovery:
        # If relevance is source or subject AND trust >= threshold, reveal first
        # undiscovered entry in evidence.connects_to
        discovered = []
        if relevance in ("source", "subject") and trust >= CONFRONT_TRUST_THRESHOLD:
            for connected_id in evidence.connects_to:
                if connected_id not in knowledge.evidence_discovered:
                    discovered.append(connected_id)
                    break  # only first undiscovered

        # Schedule disruption: if NPC is TIER1 and relevance is source
        if character.tier == CharacterTier.TIER1 and relevance == "source":
            import hashlib
            import random as random_mod
            current_idx = slot_index(loop_state.current_day, loop_state.current_slot.value)
            next_idx = current_idx + 1
            if next_idx < TOTAL_SLOTS:
                private_locs = ["old_boathouse", "storage_cellar", "maintenance_shed"]
                seed = int(hashlib.md5(
                    repr((evidence_id, character.name)).encode()
                ).hexdigest(), 16)
                rng = random_mod.Random(seed)
                new_loc = rng.choice(private_locs)

                loop_state.schedule_modifications.setdefault(character.name, {})
                loop_state.schedule_modifications[character.name][str(next_idx)] = ScheduleEntry(
                    location=new_loc,
                    activity="hiding / destroying evidence",
                )

        return npc_response, delta, discovered

    async def run_conversation(
        self,
        char_name: str,
        loop_state: LoopState,
        knowledge: PersistentKnowledge,
    ) -> ConversationResult:
        """Full conversation loop with an NPC."""
        char = self._char_map.get(char_name)
        if not char:
            return ConversationResult(summary=f"{char_name} is not available.")

        trust = loop_state.character_trust.get(char_name, 0)
        idx = slot_index(loop_state.current_day, loop_state.current_slot.value)
        kentry = (char.knowledge_timeline[idx]
                  if idx < len(char.knowledge_timeline)
                  else KnowledgeEntry())

        # Build rumor context — what has this NPC heard through the grapevine?
        heard_claims = get_claims_known_by(char_name, loop_state)
        rumors_section = format_claims_for_prompt(heard_claims, char_name)

        # Get current location/activity for NPC context
        sched_entry = char.schedule[idx] if idx < len(char.schedule) else None
        npc_location = sched_entry.location if sched_entry else ""
        npc_activity = sched_entry.activity if sched_entry else ""

        # Build system prompt
        sys_prompt = build_system_prompt(
            character=char,
            knowledge=kentry,
            trust_level=trust,
            player_knowledge=knowledge,
            loop_number=loop_state.loop_number,
            day=loop_state.current_day,
            slot=loop_state.current_slot.value,
            heard_rumors=rumors_section,
            current_location=npc_location,
            current_activity=npc_activity,
        )

        history: list[dict] = []
        exchanges: list[ConversationExchange] = []
        total_trust_change = 0
        exchange_count = 0
        presentations_count = 0
        evidence_shown: list[str] = []

        # Determine next trust threshold
        next_threshold = 0
        if trust < 20:
            next_threshold = 20
        elif trust < 40:
            next_threshold = 40
        elif trust < 60:
            next_threshold = 60
        elif trust < 80:
            next_threshold = 80

        self.display.show_conversation_header(
            char_name, trust, loop_state,
            exchange_num=0, max_exchanges=MAX_CONVERSATION_EXCHANGES,
            trust_threshold=next_threshold,
        )

        while exchange_count < MAX_CONVERSATION_EXCHANGES:
            # Get player input
            player_msg = await self.display.get_input(f"\n  You: ")

            if not player_msg:
                continue
            if player_msg.lower() in ("[done]", "[leave]", "done", "leave", "quit", "exit"):
                break

            # ── Evidence presentation commands ────────────────────────
            msg_lower = player_msg.strip().lower()

            # "evidence" or bare "show" — list discovered evidence
            if msg_lower in ("evidence", "show"):
                discovered = [
                    e for e in self.world.evidence_registry
                    if e.id in knowledge.evidence_discovered
                ]
                if not discovered:
                    self.display.print("\n  [dim]You have no evidence to show.[/]")
                else:
                    self.display.print("\n  [bold]Your evidence:[/]")
                    for ev in discovered:
                        self.display.print(f"    [cyan]{ev.id}[/] — {ev.description}")
                    self.display.print(
                        f"\n  [dim]Type [show <id>] to present evidence "
                        f"({MAX_EVIDENCE_PRESENTATIONS - presentations_count} remaining)[/]"
                    )
                continue

            # "show <id>" — present evidence to NPC
            if msg_lower.startswith("show "):
                evidence_id = player_msg.strip()[5:].strip()
                if not evidence_id:
                    self.display.print("\n  [dim]Usage: show <evidence_id>[/]")
                    continue
                if presentations_count >= MAX_EVIDENCE_PRESENTATIONS:
                    self.display.print(
                        "\n  [yellow]You've already presented the maximum evidence "
                        "this conversation.[/]"
                    )
                    continue

                npc_response, delta, discovered_ids = await self._handle_evidence_presentation(
                    evidence_id, char, trust, loop_state, knowledge, history, sys_prompt,
                )

                # Check if it was a validation error (no trust change, no discoveries)
                if delta == 0 and not discovered_ids and npc_response.startswith("("):
                    self.display.print(f"\n  [yellow]{npc_response}[/]")
                    continue

                # Apply trust
                trust += delta
                trust = max(TRUST_FLOOR, min(100, trust))
                total_trust_change += delta
                loop_state.character_trust[char_name] = trust

                # Display dramatic separator and response
                self.display.show_evidence_presentation(char_name, evidence_id, delta)
                self.display.show_npc_response(char_name, npc_response)

                # Record exchange
                confront_text = f"[presents evidence: {evidence_id}]"
                history.append({"player": confront_text, "npc": npc_response})
                exchanges.append(ConversationExchange(player=confront_text, npc=npc_response))
                exchange_count += 1
                presentations_count += 1
                evidence_shown.append(evidence_id)

                # Display discovered evidence
                for disc_id in discovered_ids:
                    self.display.print(
                        f"\n  [bold green]New evidence revealed: {disc_id}[/]"
                    )

                continue

            # ── Normal conversation flow ──────────────────────────────
            self.display.show_player_message(player_msg)

            # Calculate trust change from player's message
            delta, reason = self._calculate_trust_delta(player_msg, char, history, knowledge)
            trust += delta
            trust = max(TRUST_FLOOR, min(100, trust))
            total_trust_change += delta

            if delta != 0:
                self.display.show_conversation_footer(delta, reason)

            # Build prompt and get LLM response
            user_prompt = build_conversation_prompt(history, player_msg, char_name)
            npc_response = await self._get_npc_response(sys_prompt, user_prompt)

            self.display.show_npc_response(char_name, npc_response)

            # Record exchange
            history.append({"player": player_msg, "npc": npc_response})
            exchanges.append(ConversationExchange(player=player_msg, npc=npc_response))
            exchange_count += 1

            # Update trust in loop state
            loop_state.character_trust[char_name] = trust

        # Extract topics discussed
        topics = self._extract_topics(history)

        # Extract claims for the Rumor Mill
        if history:
            new_claims = await extract_claims_from_conversation(
                char_name, history, loop_state,
            )
            for claim in new_claims:
                loop_state.active_claims.append(claim)
                # NPC heard these claims directly
                loop_state.npc_heard_claims.setdefault(char_name, [])
                if claim.id not in loop_state.npc_heard_claims[char_name]:
                    loop_state.npc_heard_claims[char_name].append(claim.id)

            if new_claims:
                self.display.print(
                    f"\n  [dim italic]{char_name} will remember what was discussed...[/]"
                )

        # Track meeting
        if char_name not in knowledge.characters_met:
            knowledge.characters_met.append(char_name)

        # Update conversation count
        loop_state.conversations_this_loop[char_name] = (
            loop_state.conversations_this_loop.get(char_name, 0) + exchange_count
        )

        # Record schedule observation (we now know where this character is)
        knowledge.character_schedules_known.setdefault(char_name, {})[str(idx)] = (
            char.schedule[idx].location if idx < len(char.schedule) else "unknown"
        )

        return ConversationResult(
            trust_change=total_trust_change,
            topics_discussed=topics,
            exchanges_count=exchange_count,
            summary=f"Spoke with {char_name} for {exchange_count} exchanges.",
            exchanges=exchanges,
            evidence_presented=evidence_shown,
        )

    async def _get_npc_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get LLM response for NPC dialogue."""
        from .llm import llm_query
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        result_text = (await llm_query(full_prompt)).strip()
        if len(result_text) > 500:
            sentences = re.split(r'(?<=[.!?])\s+', result_text)
            truncated = ""
            for s in sentences:
                if len(truncated) + len(s) + 1 > 500:
                    break
                truncated = f"{truncated} {s}" if truncated else s
            result_text = truncated or sentences[0]

        return result_text or "*says nothing*"

    def _calculate_trust_delta(
        self,
        player_msg: str,
        character: Character,
        history: list[dict],
        knowledge: PersistentKnowledge,
    ) -> tuple[int, str]:
        """Deterministic trust change based on pattern matching.

        Returns (delta, reason) tuple.
        """
        delta = 0
        reasons: list[str] = []

        # Harsh accusations (check first — strongest signal)
        if ACCUSATION_HARSH_PATTERNS.search(player_msg):
            return (TRUST_DELTA_ACCUSATION_HARSH, "harsh accusation")

        # Mild accusations
        if ACCUSATION_MILD_PATTERNS.search(player_msg):
            return (TRUST_DELTA_ACCUSATION_MILD, "accusation")

        # Pressure
        if PRESSURE_PATTERNS.search(player_msg):
            return (TRUST_DELTA_PRESSURE, "pressure")

        # Empathy
        if EMPATHY_PATTERNS.search(player_msg):
            delta += TRUST_DELTA_EMPATHY
            reasons.append("empathy")

        # Remembered details (echoing NPC's earlier words)
        if history:
            npc_words = set()
            for exchange in history:
                npc_words.update(
                    w.lower() for w in re.findall(r"\b\w+\b", exchange["npc"])
                    if len(w) >= REMEMBERED_THRESHOLD
                )
            player_words = set(
                w.lower() for w in re.findall(r"\b\w+\b", player_msg)
                if len(w) >= REMEMBERED_THRESHOLD
            )
            # If player echoes specific NPC words (not common words)
            common_words = {"that", "this", "with", "from", "have", "been", "were",
                           "what", "when", "where", "they", "them", "their", "about",
                           "would", "could", "should", "just", "like", "know", "think",
                           "here", "there", "some", "more", "very", "really"}
            echoed = player_words & npc_words - common_words
            if len(echoed) >= 2:
                delta += TRUST_DELTA_REMEMBERED
                reasons.append("you remembered a detail")

        # Impossible knowledge check
        # If player references something that hasn't happened yet in this loop
        # This is a simplified check — the conversation prompt handles the detailed version
        for secret in character.secrets:
            secret_words = set(w.lower() for w in re.findall(r"\b\w{5,}\b", secret))
            msg_words = set(w.lower() for w in re.findall(r"\b\w{5,}\b", player_msg))
            overlap = secret_words & msg_words
            if len(overlap) >= 2:
                # Player seems to know a secret
                if any(kw in player_msg.lower() for kw in
                       ("i heard", "someone told me", "i noticed", "i think", "maybe")):
                    delta += TRUST_DELTA_IMPOSSIBLE_EXPLAINED  # Explained
                    reasons.append("impossible knowledge explained")
                elif knowledge.npc_interaction_counts.get(character.name, 0) > 0:
                    delta += 0  # Déjà vu dampens suspicion
                    reasons.append("impossible knowledge (déjà vu dampened)")
                else:
                    delta += TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED  # Unexplained
                    reasons.append("unexplained knowledge")

        return (delta, " + ".join(reasons))

    def _extract_topics(self, history: list[dict]) -> list[str]:
        """Extract evidence-related topics from conversation via keyword matching."""
        topics = []
        all_text = " ".join(
            f"{e['player']} {e['npc']}" for e in history
        ).lower()

        for ev_id, keywords in self._evidence_keywords.items():
            matches = sum(1 for kw in keywords if kw in all_text)
            if matches >= 2:  # At least 2 keyword matches
                topics.append(ev_id)

        return topics
