"""Interactive evidence board display and management."""

from __future__ import annotations

from .display import GameDisplay
from .knowledge_base import KnowledgeBase
from .models import PersistentKnowledge, WorldState


class EvidenceBoard:
    """Interactive evidence board for browsing, connecting, and theorizing."""

    def __init__(self, display: GameDisplay, kb: KnowledgeBase, world: WorldState):
        self.display = display
        self.kb = kb
        self.world = world

    def _get_clusters(self, knowledge: PersistentKnowledge) -> list[set[str]]:
        """Find connected clusters of evidence via confirmed connections."""
        # Build adjacency from confirmed connections
        adj: dict[str, set[str]] = {}
        for conn in knowledge.evidence_connections:
            if conn.confirmed:
                adj.setdefault(conn.evidence_a, set()).add(conn.evidence_b)
                adj.setdefault(conn.evidence_b, set()).add(conn.evidence_a)

        # Find connected components via BFS
        visited: set[str] = set()
        clusters: list[set[str]] = []
        for node in adj:
            if node in visited:
                continue
            cluster: set[str] = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.add(current)
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(cluster) >= 3:
                clusters.append(cluster)
        return clusters

    async def show(self, knowledge: PersistentKnowledge):
        """Show the evidence board and handle interaction."""
        while True:
            self.display.show_evidence_board(self.world.evidence_registry, knowledge)

            # Show clusters of connected evidence
            clusters = self._get_clusters(knowledge)
            if clusters:
                self.display.print("\n[bold cyan]Evidence Clusters:[/]")
                for i, cluster in enumerate(clusters, 1):
                    ev_descs = []
                    for ev_id in cluster:
                        ev = next((e for e in self.world.evidence_registry if e.id == ev_id), None)
                        if ev:
                            ev_descs.append(ev.description[:40])
                    self.display.print(f"  [cyan]Cluster {i}:[/] {', '.join(ev_descs)}")

            self.display.print()
            self.display.print("[bold]Evidence Board Commands:[/]")
            self.display.print("  [dim]connect <id1> <id2> [description] — draw a connection[/]")
            self.display.print("  [dim]disconnect <id1> <id2> — remove a connection[/]")
            self.display.print("  [dim]theory <text> — add a theory[/]")
            self.display.print("  [dim]remove theory <number> — remove a theory[/]")
            self.display.print("  [dim]back — return to game[/]")
            self.display.print()

            cmd = await self.display.get_input("Evidence board> ")
            if not cmd or cmd.lower() in ("back", "exit", "quit", "done"):
                break

            parts = cmd.split(maxsplit=3)
            verb = parts[0].lower() if parts else ""

            if verb == "connect" and len(parts) >= 3:
                desc = parts[3] if len(parts) > 3 else ""
                if self.kb.add_connection(parts[1], parts[2], desc):
                    confirmed = self.kb.confirm_connection(parts[1], parts[2])
                    if confirmed:
                        self.display.show_evidence_connection_confirmed(parts[1], parts[2])
                    else:
                        self.display.print("[green]Connection added.[/]")
                else:
                    self.display.print("[yellow]Could not add connection (already exists or evidence not found).[/]")

            elif verb == "disconnect" and len(parts) >= 3:
                if self.kb.remove_connection(parts[1], parts[2]):
                    self.display.print("[green]Connection removed.[/]")
                else:
                    self.display.print("[yellow]Connection not found.[/]")

            elif verb == "theory" and len(parts) >= 2:
                theory = " ".join(parts[1:])
                self.kb.add_theory(theory)
                self.display.print("[green]Theory added.[/]")

            elif verb == "remove" and len(parts) >= 3 and parts[1].lower() == "theory":
                try:
                    idx = int(parts[2]) - 1
                    if self.kb.remove_theory(idx):
                        self.display.print("[green]Theory removed.[/]")
                    else:
                        self.display.print("[yellow]Invalid theory number.[/]")
                except ValueError:
                    self.display.print("[yellow]Usage: remove theory <number>[/]")

            else:
                self.display.print("[dim]Unknown command. Type 'back' to return.[/]")
