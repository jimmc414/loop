"""Rich terminal UI for LOOP."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

from .config import (
    DAYS_PER_LOOP,
    MAX_LOOPS,
    SLOT_NAMES,
    TIME_COLORS,
    slot_index,
)
from .models import (
    Evidence,
    EvidenceType,
    Location,
    LoopState,
    PersistentKnowledge,
    TimeSlot,
    WorldState,
)

EVIDENCE_ICONS = {
    EvidenceType.TESTIMONY: "[bold cyan]T[/]",
    EvidenceType.OBSERVATION: "[bold yellow]O[/]",
    EvidenceType.PHYSICAL: "[bold red]P[/]",
    EvidenceType.BEHAVIORAL: "[bold magenta]B[/]",
}

TITLE_ART = r"""
[bold cyan]
  ██╗      ██████╗  ██████╗ ██████╗
  ██║     ██╔═══██╗██╔═══██╗██╔══██╗
  ██║     ██║   ██║██║   ██║██████╔╝
  ██║     ██║   ██║██║   ██║██╔═══╝
  ███████╗╚██████╔╝╚██████╔╝██║
  ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝
[/]
[dim]     Camp Pinehaven — A Time Loop Mystery[/]
"""


class GameDisplay:
    """All terminal rendering via Rich."""

    def __init__(self):
        self.console = Console()

    # ── Input helper ───────────────────────────────────────────────────

    async def get_input(self, prompt: str = "> ") -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input(prompt).strip())

    # ── Title screen ───────────────────────────────────────────────────

    def show_title_screen(self):
        self.console.clear()
        self.console.print(TITLE_ART)
        self.console.print()
        self.console.print("[dim]The loop begins at dawn. You have 7 chances.[/]")
        self.console.print("[dim]Knowledge persists. Everything else resets.[/]")
        self.console.print()
        self.console.print("[bold]Press ENTER to begin...[/]", end="")

    # ── HUD ────────────────────────────────────────────────────────────

    def _time_color(self, slot: TimeSlot | str) -> str:
        name = slot.value if isinstance(slot, TimeSlot) else slot
        return TIME_COLORS.get(name, "white")

    def _hud_text(self, loop_state: LoopState, actions_remaining: int | None = None) -> str:
        color = self._time_color(loop_state.current_slot)
        loop_urgency = ""
        if loop_state.loop_number >= 5:
            loop_urgency = " [bold red blink](!)[/]"
        elif loop_state.loop_number >= 3:
            loop_urgency = " [yellow](!)[/]"

        actions_text = ""
        if actions_remaining is not None:
            actions_text = f"  Actions: {actions_remaining}"

        return (
            f"[bold]Loop {loop_state.loop_number}/{MAX_LOOPS}[/]{loop_urgency}  "
            f"Day {loop_state.current_day}/{DAYS_PER_LOOP}  "
            f"[{color}]{loop_state.current_slot.value}[/]"
            f"{actions_text}"
        )

    # ── Location scene ─────────────────────────────────────────────────

    def show_location_scene(self, location: Location, characters_here: list[str],
                            loop_state: LoopState, description: str = "",
                            actions_remaining: int | None = None):
        color = self._time_color(loop_state.current_slot)
        hud = self._hud_text(loop_state, actions_remaining)

        body_parts = []
        if description:
            body_parts.append(description)
        else:
            # Dynamic description based on time of day
            time_flavor = {
                "MORNING": "Morning light filters through the trees.",
                "AFTERNOON": "The afternoon sun beats down warmly.",
                "EVENING": "Long shadows stretch across the ground as evening approaches.",
                "NIGHT": "Darkness has settled over the camp. Crickets chirp in the distance.",
            }
            slot_name = loop_state.current_slot.value if hasattr(loop_state, 'current_slot') else "MORNING"
            flavor = time_flavor.get(slot_name, "")
            body_parts.append(f"[dim]{location.description} {flavor}[/]")

        if characters_here:
            body_parts.append("")
            body_parts.append("[bold]People here:[/]")
            for name in characters_here:
                body_parts.append(f"  - {name}")
        else:
            body_parts.append("\n[dim]No one else is here.[/]")

        panel = Panel(
            "\n".join(body_parts),
            title=f"[bold {color}]{location.name}[/]",
            subtitle=hud,
            border_style=color,
            padding=(1, 2),
        )
        self.console.print(panel)

    # ── Action menu ────────────────────────────────────────────────────

    def _group_actions(self, actions: list[dict]) -> list[tuple[str, list[tuple[int, dict]]]]:
        """Group actions by category for display. Returns (group_name, [(index, action)]) pairs."""
        groups: dict[str, list[tuple[int, dict]]] = {}
        group_order = ["People", "Explore", "Intervene", "Tools", "Time"]

        for i, action in enumerate(actions):
            atype = action["type"]
            if atype in ("talk", "follow"):
                group = "People"
            elif atype in ("observe", "search", "travel"):
                group = "Explore"
            elif atype == "intervene":
                group = "Intervene"
            elif atype in ("evidence_board", "schedule_tracker", "map"):
                group = "Tools"
            elif atype in ("wait", "fast_forward"):
                group = "Time"
            else:
                group = "Other"
            groups.setdefault(group, []).append((i, action))

        # Return in defined order, skip empty groups
        result = []
        for name in group_order:
            if name in groups:
                result.append((name, groups[name]))
        if "Other" in groups:
            result.append(("Other", groups["Other"]))
        return result

    async def show_action_menu(self, actions: list[dict]) -> dict | None:
        self.console.print()
        grouped = self._group_actions(actions)

        for group_name, items in grouped:
            self.console.print(f"  [bold]── {group_name} {'─' * (18 - len(group_name))}[/]")
            for idx, action in items:
                label = action["label"]
                atype = action["type"]
                style = "dim" if atype in ("evidence_board", "schedule_tracker", "map") else ""
                cost = ""
                if action.get("costs_slot"):
                    cost = " [dim](costs time)[/]"
                self.console.print(f"  [{style}]{idx + 1:2}. {label}{cost}[/]")

        self.console.print()
        choice = await self.get_input("Choose action: ")

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(actions):
                return actions[idx]
        except ValueError:
            pass
        return None

    # ── Conversation ───────────────────────────────────────────────────

    def show_conversation_header(self, char_name: str, trust: int, loop_state: LoopState,
                                 exchange_num: int = 0, max_exchanges: int = 20,
                                 trust_threshold: int = 0):
        trust_bar = self._trust_bar(trust)
        hud = self._hud_text(loop_state)

        threshold_text = ""
        if trust_threshold > 0:
            threshold_text = f"\nNext secret at trust: {trust_threshold}"

        self.console.print()
        self.console.print(Panel(
            f"Trust: {trust_bar}  ({trust}){threshold_text}\n"
            f"Exchange: {exchange_num}/{max_exchanges}\n"
            f"[dim]Type [done] or [leave] to end conversation[/]\n"
            f"[dim]Type [evidence] to see your evidence, [show <id>] to present it[/]",
            title=f"[bold]Talking to {char_name}[/]",
            subtitle=hud,
            border_style="cyan",
        ))

    def show_evidence_presentation(self, char_name: str, evidence_id: str, trust_delta: int):
        """Dramatic separator when evidence is presented during conversation."""
        self.console.print()
        self.console.print("[bold yellow]═══════════════════════════════════════[/]")
        self.console.print(f"[bold yellow]       EVIDENCE PRESENTED[/]")
        self.console.print(f"[bold yellow]       [{evidence_id}] → {char_name}[/]")
        if trust_delta > 0:
            self.console.print(f"[bold yellow]       [green]+{trust_delta} trust[/][/]")
        elif trust_delta < 0:
            self.console.print(f"[bold yellow]       [red]{trust_delta} trust[/][/]")
        self.console.print("[bold yellow]═══════════════════════════════════════[/]")
        self.console.print()

    def show_conversation_footer(self, trust_change: int, reason: str = ""):
        if trust_change > 0:
            reason_text = f" ({reason})" if reason else ""
            self.console.print(f"  [green]+{trust_change} trust{reason_text}[/]")
        elif trust_change < 0:
            reason_text = f" ({reason})" if reason else ""
            self.console.print(f"  [red]{trust_change} trust{reason_text}[/]")

    def show_player_message(self, text: str):
        self.console.print(f"\n  [bold white]You:[/] {text}")

    def show_npc_response(self, name: str, text: str):
        self.console.print(f"  [bold cyan]{name}:[/] {text}")

    def show_conversation_summary(self, char_name: str, exchanges: int,
                                   trust_change: int, trust_now: int,
                                   evidence_learned: list[str],
                                   rumors_planted: int):
        """Show a compact summary panel after a conversation ends."""
        lines = [f"Spoke with {char_name} ({exchanges} exchanges)"]

        if trust_change >= 0:
            lines.append(f"Trust: [green]+{trust_change}[/] (now {trust_now})")
        else:
            lines.append(f"Trust: [red]{trust_change}[/] (now {trust_now})")

        if evidence_learned:
            lines.append(f"Learned: {', '.join(evidence_learned)}")

        if rumors_planted > 0:
            lines.append(f"Rumors planted: {rumors_planted}")

        self.console.print()
        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]Conversation Summary[/]",
            border_style="cyan",
            padding=(0, 2),
        ))

    def _trust_bar(self, trust: int, width: int = 20) -> str:
        filled = max(0, min(width, int(trust / 100 * width)))
        empty = width - filled
        if trust >= 60:
            color = "green"
        elif trust >= 20:
            color = "yellow"
        else:
            color = "red"
        return f"[{color}]{'█' * filled}{'░' * empty}[/]"

    # ── Evidence board ─────────────────────────────────────────────────

    def show_evidence_connection_confirmed(self, evidence_a: str, evidence_b: str):
        """Visual celebration when a connection is confirmed."""
        self.console.print()
        self.console.print("[bold green]╔═══════════════════════════════════════╗[/]")
        self.console.print("[bold green]║     CONNECTION CONFIRMED!             ║[/]")
        self.console.print(f"[bold green]║  {evidence_a} ←→ {evidence_b:<20}  ║[/]")
        self.console.print("[bold green]╚═══════════════════════════════════════╝[/]")
        self.console.print()

    def show_evidence_board(self, evidence: list[Evidence], knowledge: PersistentKnowledge):
        self.console.print()

        if not knowledge.evidence_discovered:
            self.console.print(Panel("[dim]No evidence discovered yet.[/]",
                                     title="[bold]Evidence Board[/]"))
            return

        tree = Tree("[bold]Evidence Board[/]")

        # Group by type
        by_type: dict[EvidenceType, list[Evidence]] = {}
        for ev in evidence:
            if ev.id in knowledge.evidence_discovered:
                by_type.setdefault(ev.type, []).append(ev)

        for etype, items in by_type.items():
            icon = EVIDENCE_ICONS.get(etype, "?")
            branch = tree.add(f"{icon} [bold]{etype.value.title()}[/]")
            for ev in items:
                connections = ""
                if ev.connects_to:
                    known_connections = [c for c in ev.connects_to if c in knowledge.evidence_discovered]
                    if known_connections:
                        connections = f" [dim]-> {', '.join(known_connections)}[/]"
                branch.add(f"[{ev.id}] {ev.description}{connections}")

        self.console.print(Panel(tree, border_style="cyan", padding=(1, 2)))

        # Theories
        if knowledge.theories:
            self.console.print("\n[bold]Your Theories:[/]")
            for i, theory in enumerate(knowledge.theories, 1):
                self.console.print(f"  {i}. {theory}")

        # Connections
        if knowledge.evidence_connections:
            self.console.print("\n[bold]Connections:[/]")
            for conn in knowledge.evidence_connections:
                status = "[green]confirmed[/]" if conn.confirmed else "[dim]unconfirmed[/]"
                self.console.print(f"  {conn.evidence_a} <-> {conn.evidence_b}: {conn.description} {status}")

    # ── Schedule tracker ───────────────────────────────────────────────

    def show_schedule_tracker(self, knowledge: PersistentKnowledge, world: WorldState):
        self.console.print()
        pinned = knowledge.pinned_characters or [c.name for c in world.characters[:4]]

        table = Table(title="Schedule Tracker", box=box.ROUNDED, show_lines=True)
        table.add_column("Character", style="bold")
        for day in range(1, DAYS_PER_LOOP + 1):
            for slot_name in SLOT_NAMES:
                color = TIME_COLORS.get(slot_name, "white")
                table.add_column(f"D{day} {slot_name[:3]}", style=color, max_width=12)

        for char_name in pinned:
            row = [char_name]
            known = knowledge.character_schedules_known.get(char_name, {})
            for day in range(1, DAYS_PER_LOOP + 1):
                for slot_name in SLOT_NAMES:
                    idx = slot_index(day, slot_name)
                    loc = known.get(str(idx)) or known.get(idx)
                    if loc:
                        # Shorten location name
                        loc_short = loc.replace("_", " ").title()[:10]
                        row.append(loc_short)
                    else:
                        row.append("[dim]?[/]")
            table.add_row(*row)

        self.console.print(table)
        self.console.print("[dim]Showing pinned characters. Use 'pin/unpin <name>' to change.[/]")

    def show_schedule_tracker_day(self, knowledge: PersistentKnowledge, world: WorldState, day: int):
        """Show schedule for a single day (4 columns)."""
        self.console.print()
        pinned = knowledge.pinned_characters or [c.name for c in world.characters[:4]]

        table = Table(title=f"Schedule — Day {day}", box=box.ROUNDED, show_lines=True)
        table.add_column("Character", style="bold")
        for slot_name in SLOT_NAMES:
            color = TIME_COLORS.get(slot_name, "white")
            table.add_column(slot_name[:3], style=color, max_width=20)

        for char_name in pinned:
            row = [char_name]
            known = knowledge.character_schedules_known.get(char_name, {})
            for slot_name in SLOT_NAMES:
                idx = slot_index(day, slot_name)
                loc = known.get(str(idx)) or known.get(idx)
                if loc:
                    loc_short = loc.replace("_", " ").title()[:20]
                    row.append(loc_short)
                else:
                    row.append("[dim]?[/]")
            table.add_row(*row)

        self.console.print(table)
        self.console.print("[dim]Showing pinned characters. Use 'pin/unpin <name>' to change.[/]")

    # ── Camp map ───────────────────────────────────────────────────────

    def show_map(self, player_location: str, known_positions: dict[str, str] | None = None):
        """Show ASCII camp map."""
        # Simple text map
        map_text = """
        [bold]╔═══════════════════════════════════════╗[/]
        [bold]║[/]        [bold]CAMP PINEHAVEN MAP[/]            [bold]║[/]
        [bold]╠═══════════════════════════════════════╣[/]
        [bold]║[/]                                       [bold]║[/]
        [bold]║[/]   [blue]~~~Lake Shore~~~[/]    [blue]Dock[/]          [bold]║[/]
        [bold]║[/]   [blue]~Old Boathouse~[/]                    [bold]║[/]
        [bold]║[/]         |                              [bold]║[/]
        [bold]║[/]    Trail Head --- Archery Range        [bold]║[/]
        [bold]║[/]         |              |               [bold]║[/]
        [bold]║[/]   Campfire Circle  Arts Cabin          [bold]║[/]
        [bold]║[/]      |    |           |                [bold]║[/]
        [bold]║[/]  Dining  Main Lodge--Infirmary         [bold]║[/]
        [bold]║[/]   Hall     |    |                      [bold]║[/]
        [bold]║[/]    |    Dir.Office                     [bold]║[/]
        [bold]║[/]  Kitchen                               [bold]║[/]
        [bold]║[/]                                        [bold]║[/]
        [bold]║[/]  Cabin Row --- Bathhouse               [bold]║[/]
        [bold]║[/]      |                                 [bold]║[/]
        [bold]║[/]  Staff Quarters                        [bold]║[/]
        [bold]║[/]      |                                 [bold]║[/]
        [bold]║[/]  Maintenance --- Storage Cellar        [bold]║[/]
        [bold]║[/]     Shed                               [bold]║[/]
        [bold]╚═══════════════════════════════════════╝[/]
"""
        self.console.print(map_text)

        loc_name = player_location.replace("_", " ").title()
        self.console.print(f"  [bold green]You are at: {loc_name}[/]")

        if known_positions:
            self.console.print("\n  [bold]Known character positions:[/]")
            for name, loc in known_positions.items():
                self.console.print(f"    {name} -> {loc.replace('_', ' ').title()}")

    # ── Loop reset ─────────────────────────────────────────────────────

    def show_loop_reset(self, loop_number: int, catastrophe: bool):
        self.console.print()
        if catastrophe:
            self.console.print("[bold red]═══════════════════════════════════════[/]")
            self.console.print("[bold red]  THE CATASTROPHE UNFOLDS.[/]")
            self.console.print("[bold red]═══════════════════════════════════════[/]")
            self.console.print()

        # Increasing urgency
        if loop_number >= 6:
            style = "bold red blink"
            msg = "THIS MIGHT BE YOUR LAST CHANCE."
        elif loop_number >= 4:
            style = "bold yellow"
            msg = "Time is running out."
        else:
            style = "bold cyan"
            msg = "You open your eyes. It's morning. Day one. Again."

        self.console.print(f"\n[{style}]  LOOP {loop_number} BEGINS[/]")
        self.console.print(f"  [{style}]{msg}[/]")
        self.console.print()
        self.console.print("  [dim]Your memories remain. Everything else has reset.[/]")
        self.console.print()

    # ── Catastrophe ────────────────────────────────────────────────────

    def show_catastrophe(self, description: str):
        self.console.print()
        self.console.print("[bold red]╔═══════════════════════════════════════════════╗[/]")
        self.console.print("[bold red]║                                               ║[/]")
        self.console.print("[bold red]║           C A T A S T R O P H E               ║[/]")
        self.console.print("[bold red]║                                               ║[/]")
        self.console.print("[bold red]╚═══════════════════════════════════════════════╝[/]")
        self.console.print()
        self.console.print(f"  [red]{description}[/]")
        self.console.print()

    # ── Post-game summary ──────────────────────────────────────────────

    def show_post_game_summary(self, world: WorldState, knowledge: PersistentKnowledge,
                                ending_description: str):
        self.console.print()
        self.console.print(Panel(
            ending_description,
            title="[bold]Ending[/]",
            border_style="green",
            padding=(1, 2),
        ))

        # Stats
        stats = Table(title="Game Statistics", box=box.ROUNDED)
        stats.add_column("Stat", style="bold")
        stats.add_column("Value")

        total_evidence = len(world.evidence_registry)
        found = len(knowledge.evidence_discovered)
        stats.add_row("Loops Used", str(len(knowledge.loop_history)))
        stats.add_row("Evidence Found", f"{found}/{total_evidence}")
        stats.add_row("Characters Met", str(len(knowledge.characters_met)))
        stats.add_row("Theories Written", str(len(knowledge.theories)))
        stats.add_row("Connections Made", str(len(knowledge.evidence_connections)))

        self.console.print(stats)

        # Missed evidence
        missed = [e for e in world.evidence_registry if e.id not in knowledge.evidence_discovered]
        if missed:
            self.console.print("\n[bold]Evidence You Missed:[/]")
            for ev in missed[:5]:
                self.console.print(f"  [dim]- {ev.description} (at {ev.source_location})[/]")
            if len(missed) > 5:
                self.console.print(f"  [dim]... and {len(missed) - 5} more[/]")

    # ── Rumor tracker ─────────────────────────────────────────────────

    def show_rumor_tracker(self, claims: list, characters: list[str]):
        """Show active rumors and who knows what."""
        from .models import Claim
        self.console.print()
        if not claims:
            self.console.print(Panel("[dim]No rumors circulating yet.[/]",
                                     title="[bold]Rumor Tracker[/]"))
            return

        lines = []
        for claim in claims:
            heard_count = len(claim.heard_by)
            lines.append(f"[bold]{claim.subject}[/]: {claim.content}")
            lines.append(f"  [dim]Source: {claim.source} | Known by {heard_count} people | "
                         f"Spread: {claim.spread_count}x[/]")

        self.console.print(Panel(
            "\n".join(lines),
            title="[bold]Rumor Tracker[/]",
            border_style="magenta",
            padding=(1, 2),
        ))

    # ── Utility ────────────────────────────────────────────────────────

    def print(self, text: str = "", **kwargs):
        self.console.print(text, **kwargs)

    def clear(self):
        self.console.clear()
