"""Intervention tree logic — executing player interventions against the causal chain."""

from __future__ import annotations

from .conversation_engine import ConversationEngine
from .display import GameDisplay
from .models import (
    InterventionNode,
    LoopState,
    PersistentKnowledge,
    WorldState,
)
from .state_machine import ClockworkEngine


class InterventionManager:
    """Manages intervention execution and causal chain recalculation."""

    def __init__(self, world: WorldState, engine: ClockworkEngine,
                 display: GameDisplay, conversation: ConversationEngine):
        self.world = world
        self.engine = engine
        self.display = display
        self.conversation = conversation
        self._intervention_map = {i.id: i for i in world.intervention_tree}
        self._event_map = {e.id: e for e in world.causal_chain}

    def get_available_interventions(self, loop_state: LoopState,
                                     knowledge: PersistentKnowledge) -> list[InterventionNode]:
        """Get interventions available at current state."""
        return self.engine._get_available_interventions()

    async def execute(self, intervention_id: str, loop_state: LoopState,
                      knowledge: PersistentKnowledge) -> dict:
        """Execute an intervention, possibly requiring conversation."""
        iv = self._intervention_map.get(intervention_id)
        if not iv:
            return {"advance_time": False, "message": "Intervention not found."}

        # Show pre-intervention stakes
        event = self._event_map.get(iv.causal_event_id)
        if event:
            self.display.print()
            self.display.print("[bold yellow]═══ INTERVENTION ═══[/]")
            self.display.print(f"  [bold]Target:[/] Prevent {event.character} from: {event.action}")
            self.display.print(f"  [bold]When:[/] Day {event.day}, {event.time_slot.value}")
            if iv.requires_conversation:
                char_name = event.character
                trust = loop_state.character_trust.get(char_name, 0)
                self.display.print(f"  [bold]Requires:[/] Convince {char_name} (trust: {trust}/{iv.trust_required})")
            self.display.print()

        # Check if conversation is required
        if iv.requires_conversation:
            event = self._event_map.get(iv.causal_event_id)
            if event:
                char_name = event.character
                trust = loop_state.character_trust.get(char_name, 0)
                if trust < iv.trust_required:
                    self.display.print(
                        f"\n  [yellow]You need more trust with {char_name} "
                        f"(current: {trust}, required: {iv.trust_required}).[/]"
                    )
                    self.display.print(
                        f"  [dim]Try talking to them more first.[/]"
                    )
                    return {"advance_time": False, "message": "Not enough trust."}

                # Run a persuasion conversation
                self.display.print(
                    f"\n  [bold]You need to convince {char_name}...[/]"
                )
                result = await self.conversation.run_conversation(
                    char_name, loop_state, knowledge
                )
                # Check if trust is now sufficient
                new_trust = loop_state.character_trust.get(char_name, 0)
                if new_trust < iv.trust_required:
                    self.display.print(
                        f"\n  [yellow]You weren't convincing enough. "
                        f"{char_name} won't cooperate.[/]"
                    )
                    return {"advance_time": True, "message": "Intervention failed — not convincing enough."}

        # Apply the intervention
        result = self.engine.apply_intervention(intervention_id)

        if result["success"]:
            event = self._event_map.get(iv.causal_event_id)
            # Dramatic intervention display
            self.display.print()
            self.display.print("[bold green]╔═══════════════════════════════════════════════╗[/]")
            self.display.print("[bold green]║                                               ║[/]")
            self.display.print("[bold green]║        I N T E R V E N T I O N                ║[/]")
            self.display.print("[bold green]║           S U C C E S S F U L                 ║[/]")
            self.display.print("[bold green]║                                               ║[/]")
            self.display.print("[bold green]╚═══════════════════════════════════════════════╝[/]")
            self.display.print()
            self.display.print(f"  [bold green]{iv.action_description}[/]")

            if event:
                self.display.print(f"\n  [green]Without this, {event.character}'s plan to "
                                  f"{event.action} will never happen.[/]")

            if result["cascades"]:
                self.display.print(f"\n  [bold]Causal chain disrupted:[/]")
                for cascade_id in result["cascades"]:
                    cascade_event = self._event_map.get(cascade_id)
                    if cascade_event:
                        self.display.print(
                            f"  [dim]→ {cascade_event.character}'s action "
                            f"'{cascade_event.action}' on Day {cascade_event.day} "
                            f"won't happen now[/]"
                        )
            self.display.print()

            # Recalculate what events are still active
            self.recalculate_causal_chain(loop_state)

            return {
                "advance_time": True,
                "message": f"Intervention successful: {iv.action_description}",
                "interrupted": iv.causal_event_id,
                "cascades": result["cascades"],
            }

        return {"advance_time": True, "message": "Intervention failed."}

    def recalculate_causal_chain(self, loop_state: LoopState):
        """Given current interruptions, mark downstream events as interrupted.

        Uses ANY-upstream semantics: if ANY upstream event is interrupted,
        the downstream event is also interrupted (matching _chain_intact logic).
        """
        changed = True
        while changed:
            changed = False
            for event in self.world.causal_chain:
                if event.id in loop_state.events_interrupted:
                    continue
                # Check if ANY upstream event is interrupted
                upstream = [e for e in self.world.causal_chain
                           if event.id in e.downstream_effects]
                if upstream and any(u.id in loop_state.events_interrupted for u in upstream):
                    loop_state.events_interrupted.append(event.id)
                    changed = True
