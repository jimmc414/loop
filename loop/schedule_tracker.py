"""Character schedule grid display and pin management."""

from __future__ import annotations

import re

from .config import DAYS_PER_LOOP
from .display import GameDisplay
from .models import LoopState, PersistentKnowledge, WorldState


class ScheduleTracker:
    """Interactive schedule tracker with pin/unpin."""

    def __init__(self, display: GameDisplay, world: WorldState):
        self.display = display
        self.world = world

    async def show(self, knowledge: PersistentKnowledge, loop_state: LoopState | None = None):
        """Display schedule tracker and handle pin/unpin/day commands."""
        # Default to current day view
        current_day = loop_state.current_day if loop_state else None
        view_day = current_day  # None means "all"

        while True:
            if view_day is not None:
                self.display.show_schedule_tracker_day(knowledge, self.world, view_day)
            else:
                self.display.show_schedule_tracker(knowledge, self.world)
            self.display.print()
            self.display.print("[bold]Commands:[/]")
            self.display.print("  [dim]pin <name> — track a character[/]")
            self.display.print("  [dim]unpin <name> — stop tracking[/]")
            self.display.print("  [dim]day <N> — show day N only[/]")
            self.display.print("  [dim]all — show full grid[/]")
            self.display.print("  [dim]back — return to game[/]")
            self.display.print()

            cmd = await self.display.get_input("Schedule> ")
            if not cmd or cmd.lower() in ("back", "exit", "quit", "done"):
                break

            parts = cmd.split(maxsplit=1)
            verb = parts[0].lower()

            if verb == "pin" and len(parts) >= 2:
                name = self._match_character_name(parts[1])
                if name:
                    if name not in knowledge.pinned_characters:
                        knowledge.pinned_characters.append(name)
                    self.display.print(f"[green]Now tracking {name}.[/]")
                else:
                    self.display.print("[yellow]Character not found.[/]")

            elif verb == "unpin" and len(parts) >= 2:
                name = self._match_character_name(parts[1])
                if name and name in knowledge.pinned_characters:
                    knowledge.pinned_characters.remove(name)
                    self.display.print(f"[green]Stopped tracking {name}.[/]")
                else:
                    self.display.print("[yellow]Character not pinned.[/]")

            elif verb == "all":
                view_day = None

            elif verb == "day" and len(parts) >= 2:
                match = re.match(r"(\d+)", parts[1].strip())
                if match:
                    d = int(match.group(1))
                    if 1 <= d <= DAYS_PER_LOOP:
                        view_day = d
                    else:
                        self.display.print(f"[yellow]Day must be 1-{DAYS_PER_LOOP}.[/]")
                else:
                    self.display.print("[yellow]Usage: day <N>[/]")

            elif re.match(r"d(\d+)$", verb):
                d = int(verb[1:])
                if 1 <= d <= DAYS_PER_LOOP:
                    view_day = d
                else:
                    self.display.print(f"[yellow]Day must be 1-{DAYS_PER_LOOP}.[/]")

            else:
                self.display.print("[dim]Unknown command.[/]")

    def _match_character_name(self, partial: str) -> str | None:
        """Fuzzy match a character name."""
        partial_lower = partial.lower()
        for char in self.world.characters:
            if char.name.lower() == partial_lower:
                return char.name
            if char.name.lower().startswith(partial_lower):
                return char.name
        return None
