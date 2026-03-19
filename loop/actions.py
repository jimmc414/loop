"""Player action resolution — dispatches to appropriate handlers."""

from __future__ import annotations

import os

if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

from claude_code_sdk import query, ClaudeCodeOptions

from .config import MODEL, slot_index
from .conversation_engine import ConversationEngine
from .display import GameDisplay
from .intervention import InterventionManager
from .knowledge_base import KnowledgeBase
from .models import LoopState, PersistentKnowledge, WorldState
from .state_machine import ClockworkEngine


class ActionResolver:
    """Dispatches player actions to appropriate handlers."""

    def __init__(
        self,
        world: WorldState,
        engine: ClockworkEngine,
        display: GameDisplay,
        conversation: ConversationEngine,
        kb: KnowledgeBase,
        intervention_mgr: InterventionManager,
    ):
        self.world = world
        self.engine = engine
        self.display = display
        self.conversation = conversation
        self.kb = kb
        self.intervention_mgr = intervention_mgr

    async def resolve(self, action: dict, loop_state: LoopState,
                      knowledge: PersistentKnowledge) -> dict:
        """Resolve a player action. Returns result dict with 'advance_time' flag."""
        atype = action["type"]

        if atype == "talk":
            return await self.handle_talk(action["target"], loop_state, knowledge)
        elif atype == "observe":
            return await self.handle_observe(loop_state)
        elif atype == "search":
            return await self.handle_search(loop_state, knowledge)
        elif atype == "follow":
            return await self.handle_follow(action["target"], loop_state, knowledge)
        elif atype == "intervene":
            return await self.handle_intervene(action["target"], loop_state, knowledge)
        elif atype == "travel":
            return self.handle_travel(action["target"], loop_state, knowledge)
        elif atype == "fast_forward":
            return await self.handle_fast_forward(loop_state)
        elif atype == "wait":
            return {"advance_time": True, "message": "Time passes..."}
        elif atype in ("evidence_board", "schedule_tracker", "map"):
            return {"advance_time": False, "meta_action": atype}
        else:
            return {"advance_time": False, "message": "Unknown action."}

    async def handle_talk(self, char_name: str, loop_state: LoopState,
                          knowledge: PersistentKnowledge) -> dict:
        """Launch conversation with NPC."""
        trust_before = loop_state.character_trust.get(char_name, 0)
        claims_before = len(loop_state.active_claims)

        result = await self.conversation.run_conversation(char_name, loop_state, knowledge)

        # Record in conversation journal
        self.kb.record_conversation_summary(
            loop_number=loop_state.loop_number,
            day=loop_state.current_day,
            slot=loop_state.current_slot.value,
            char_name=char_name,
            exchanges=result.exchanges_count,
            trust=loop_state.character_trust.get(char_name, 0),
            topics=result.topics_discussed,
        )

        # Check if conversation topics unlock any evidence
        evidence_learned = []
        for topic in result.topics_discussed:
            ev = self.kb.discover_evidence(topic)
            if ev:
                evidence_learned.append(ev.id)
                self.display.print(f"\n  [bold green]New evidence discovered: {ev.description}[/]")

        # Check if presented evidence unlocks connected evidence
        for ev_id in result.evidence_presented:
            ev = next((e for e in self.world.evidence_registry if e.id == ev_id), None)
            if ev:
                for connected_id in ev.connects_to:
                    disc = self.kb.discover_evidence(connected_id)
                    if disc:
                        evidence_learned.append(disc.id)
                        self.display.print(
                            f"\n  [bold green]New evidence discovered: {disc.description}[/]"
                        )

        # Post-conversation summary
        trust_after = loop_state.character_trust.get(char_name, 0)
        new_claims = len(loop_state.active_claims) - claims_before
        self.display.show_conversation_summary(
            char_name=char_name,
            exchanges=result.exchanges_count,
            trust_change=result.trust_change,
            trust_now=trust_after,
            evidence_learned=evidence_learned,
            rumors_planted=max(0, new_claims),
        )

        return {
            "advance_time": True,
            "message": result.summary,
            "trust_change": result.trust_change,
        }

    async def handle_fast_forward(self, loop_state: LoopState) -> dict:
        """Fast-forward to a specific day/slot."""
        from .config import DAYS_PER_LOOP, SLOT_NAMES

        self.display.print("\n  [bold]Fast-forward to when?[/]")
        self.display.print(f"  [dim]Current: Day {loop_state.current_day}, {loop_state.current_slot.value}[/]")
        self.display.print(f"  [dim]Format: <day> <slot> (e.g., '3 EVENING')[/]")
        self.display.print(f"  [dim]Days: 1-{DAYS_PER_LOOP}, Slots: {', '.join(SLOT_NAMES)}[/]")
        self.display.print(f"  [dim]Type 'cancel' to stay[/]")

        choice = await self.display.get_input("  Target> ")
        if not choice or choice.lower() == "cancel":
            return {"advance_time": False, "message": "Cancelled."}

        parts = choice.strip().split()
        if len(parts) != 2:
            self.display.print("  [yellow]Invalid format. Use: <day> <slot>[/]")
            return {"advance_time": False, "message": "Invalid input."}

        try:
            target_day = int(parts[0])
            target_slot_name = parts[1].upper()
            if target_slot_name not in SLOT_NAMES:
                self.display.print(f"  [yellow]Invalid slot. Choose from: {', '.join(SLOT_NAMES)}[/]")
                return {"advance_time": False, "message": "Invalid slot."}
            if target_day < 1 or target_day > DAYS_PER_LOOP:
                self.display.print(f"  [yellow]Day must be 1-{DAYS_PER_LOOP}.[/]")
                return {"advance_time": False, "message": "Invalid day."}
        except ValueError:
            self.display.print("  [yellow]Invalid day number.[/]")
            return {"advance_time": False, "message": "Invalid input."}

        from .models import TimeSlot
        target_slot = TimeSlot(target_slot_name)

        # Check it's in the future
        from .config import slot_index as _slot_idx
        current_idx = _slot_idx(loop_state.current_day, loop_state.current_slot.value)
        target_idx = _slot_idx(target_day, target_slot_name)
        if target_idx <= current_idx:
            self.display.print("  [yellow]Can only fast-forward to the future.[/]")
            return {"advance_time": False, "message": "Target is not in the future."}

        slots_to_skip = target_idx - current_idx
        self.display.print(f"\n  [dim]Fast-forwarding {slots_to_skip} time slot(s)...[/]")

        result = self.engine.fast_forward_to(target_day, target_slot)

        if result.get("loop_ended"):
            return {"advance_time": False, "message": "Loop ended during fast-forward.",
                    "fast_forward_loop_end": result["result"]}

        self.display.print(f"  [green]Arrived at Day {target_day}, {target_slot_name}.[/]")
        return {"advance_time": False, "message": f"Fast-forwarded {result['slots_consumed']} slots."}

    async def handle_observe(self, loop_state: LoopState) -> dict:
        """Observe the current location — LLM generates atmospheric description."""
        loc = self.engine.get_location(loop_state.player_location)
        chars = self.engine.get_characters_at_location(loop_state.player_location)

        # Record schedule observations for visible characters
        idx = slot_index(loop_state.current_day, loop_state.current_slot.value)
        for name in chars:
            self.kb.record_schedule_observation(name, idx, loop_state.player_location)
            self.kb.record_character_met(name)

        # Build observation prompt
        char_descriptions = []
        char_relationships = []
        for name in chars:
            char = next((c for c in self.world.characters if c.name == name), None)
            if char:
                entry = char.schedule[idx] if idx < len(char.schedule) else None
                activity = entry.activity if entry else "here"
                char_descriptions.append(f"- {name} ({char.role}): {activity}")
                # Add relationship context for multi-NPC observations
                for other_name in chars:
                    if other_name != name and other_name in char.relationships:
                        char_relationships.append(
                            f"- {name} and {other_name}: {char.relationships[other_name]}"
                        )

        prompt = (
            f"You are narrating a scene at {loc.name} in Camp Pinehaven.\n"
            f"Location: {loc.description}\n"
            f"Time: Day {loop_state.current_day}, {loop_state.current_slot.value}\n"
        )
        if char_descriptions:
            prompt += f"People present:\n" + "\n".join(char_descriptions) + "\n"
        if char_relationships:
            prompt += f"\nRelationships between people here:\n" + "\n".join(char_relationships) + "\n"
        prompt += (
            "\nWrite a brief atmospheric observation (2-3 sentences) of what the player "
            "sees and senses. Include subtle details about the characters' behavior if present. "
        )
        if len(chars) >= 2:
            prompt += (
                "Show how the characters interact with each other — body language, "
                "whispered conversations, tension or camaraderie. "
            )
        prompt += "Be evocative but concise."

        description = await self._llm_call(prompt)
        self.display.print(f"\n  [italic]{description}[/]")

        return {"advance_time": True, "message": "You observe the area."}

    async def handle_search(self, loop_state: LoopState,
                            knowledge: PersistentKnowledge) -> dict:
        """Search current location for evidence."""
        result = self.engine.resolve_search(loop_state.player_location)

        if result["is_risky"] and result["observers"]:
            self.display.print(
                f"\n  [yellow]You search carefully, aware that "
                f"{', '.join(result['observers'])} might notice...[/]"
            )

        if result["found"]:
            for ev in result["found"]:
                self.kb.discover_evidence(ev.id)
                # LLM narration of finding
                prompt = (
                    f"The player just found evidence while searching {loop_state.player_location.replace('_', ' ')}.\n"
                    f"Evidence: {ev.description}\n"
                    f"Type: {ev.type.value}\n"
                    f"Write a brief discovery narration (1-2 sentences). Make it feel like a revelation."
                )
                narration = await self._llm_call(prompt)
                self.display.print(f"\n  [bold green]EVIDENCE FOUND:[/] {ev.description}")
                self.display.print(f"  [italic]{narration}[/]")
        else:
            # Check why nothing was found
            loc_name = loop_state.player_location.replace('_', ' ').title()
            prev_searches = self.kb.knowledge.locations_searched.get(loop_state.player_location, [])
            if len(prev_searches) > 1:
                self.display.print(f"\n  [dim]You've already searched {loc_name} thoroughly this loop. Nothing new here.[/]")
            else:
                # Check if there's evidence here at a different time
                has_future_evidence = any(
                    ev for ev in self.world.evidence_registry
                    if ev.source_location == loop_state.player_location
                    and ev.id not in self.kb.knowledge.evidence_discovered
                )
                if has_future_evidence:
                    self.display.print(f"\n  [dim]Nothing catches your eye right now, but something about {loc_name} nags at you...[/]")
                else:
                    self.display.print(f"\n  [dim]You search {loc_name} thoroughly but find nothing of interest.[/]")

        return {
            "advance_time": True,
            "message": f"Found {len(result['found'])} piece(s) of evidence.",
            "evidence_found": [e.id for e in result["found"]],
        }

    async def handle_follow(self, char_name: str, loop_state: LoopState,
                            knowledge: PersistentKnowledge) -> dict:
        """Follow a character — costs 2 time slots."""
        result = self.engine.resolve_follow(char_name)

        if not result["success"]:
            if result.get("detected"):
                self.display.print(
                    f"\n  [red]{char_name} spotted you following them! "
                    f"They don't look pleased.[/]"
                )
                self.display.print(
                    f"  [dim]You saw them heading toward "
                    f"{result['next_location'].replace('_', ' ').title()}.[/]"
                )
            else:
                self.display.print(f"\n  [yellow]{result['reason']}[/]")
            return {"advance_time": True, "message": f"Failed to follow {char_name}."}

        loc_name = result["next_location"].replace("_", " ").title()
        self.display.print(
            f"\n  [green]You discreetly follow {char_name}.[/]"
        )
        self.display.print(
            f"  They head to [bold]{loc_name}[/] — {result['activity']}."
        )

        return {
            "advance_time": True,  # costs a slot
            "message": f"Followed {char_name} to {loc_name}.",
        }

    async def handle_intervene(self, intervention_id: str, loop_state: LoopState,
                               knowledge: PersistentKnowledge) -> dict:
        """Execute an intervention."""
        return await self.intervention_mgr.execute(intervention_id, loop_state, knowledge)

    def handle_travel(self, destination_id: str, loop_state: LoopState,
                      knowledge: PersistentKnowledge) -> dict:
        """Move to a new location."""
        result = self.engine.resolve_travel(destination_id)

        if not result["success"]:
            self.display.print(f"\n  [yellow]{result['reason']}[/]")
            return {"advance_time": False, "message": result["reason"]}

        loc = result["location"]
        self.display.print(f"\n  [green]You head to {loc.name}.[/]")

        # Record schedule observations at new location
        chars = self.engine.get_characters_at_location(destination_id)
        idx = slot_index(loop_state.current_day, loop_state.current_slot.value)
        for name in chars:
            self.kb.record_schedule_observation(name, idx, destination_id)
            self.kb.record_character_met(name)

        return {
            "advance_time": result["costs_slot"],
            "message": f"Arrived at {loc.name}.",
        }

    async def _llm_call(self, prompt: str) -> str:
        """Short LLM call for descriptions."""
        result_text = ""
        async for msg in query(
            prompt=prompt,
            options=ClaudeCodeOptions(
                model=MODEL,
                max_turns=1,
            ),
        ):
            if hasattr(msg, "content"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        result_text += block.text
        return result_text.strip() or "..."
