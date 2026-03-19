"""Entry point for LOOP — the async game loop."""

from __future__ import annotations

import asyncio
import os
import sys

if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

from claude_code_sdk import query, ClaudeCodeOptions

from .actions import ActionResolver
from .config import MODEL, TOTAL_SLOTS, slot_index as _slot_index
from .conversation_engine import ConversationEngine
from .display import GameDisplay
from .evidence_board import EvidenceBoard
from .intervention import InterventionManager
from .knowledge_base import KnowledgeBase
from .models import LoopState, PersistentKnowledge, WorldState
from .prompts.summary import build_ending_prompt, build_summary_prompt
from .saves import SaveManager
from .schedule_tracker import ScheduleTracker
from .state_machine import ClockworkEngine
from .world_generator import generate_world, load_world


async def _llm_call(prompt: str) -> str:
    """Single LLM call for narration."""
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
    return result_text.strip()


async def main():
    display = GameDisplay()
    save_mgr = SaveManager()

    # ── Title screen ───────────────────────────────────────────────────
    display.show_title_screen()
    await display.get_input()

    # ── New game or load ───────────────────────────────────────────────
    display.clear()
    saves = save_mgr.list_saves()

    world: WorldState | None = None
    loop_state: LoopState | None = None
    knowledge: PersistentKnowledge | None = None

    if saves:
        display.print("[bold]Saved games found:[/]")
        for i, s in enumerate(saves, 1):
            display.print(f"  {i}. [{s['slot']}] Loop {s['loop']}, Day {s['day']} — {s['timestamp']}")
        display.print(f"  {len(saves) + 1}. New Game")
        display.print()

        choice = await display.get_input("Choose: ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(saves):
                save = save_mgr.load(saves[idx]["slot"])
                if save:
                    world = save.world_state
                    loop_state = save.loop_state
                    knowledge = save.knowledge
                    display.print(f"[green]Loaded save: Loop {loop_state.loop_number}, "
                                  f"Day {loop_state.current_day}[/]")
        except ValueError:
            pass

    # ── Generate or load world ─────────────────────────────────────────
    if not world:
        display.print("\n[bold]Starting new game...[/]")
        world = await load_world()
        if not world:
            world = await generate_world()
        loop_state = LoopState()
        knowledge = PersistentKnowledge()

        # Initialize trust for all characters
        for char in world.characters:
            loop_state.character_trust[char.name] = 0

    assert world and loop_state and knowledge

    # ── Initialize systems ─────────────────────────────────────────────
    engine = ClockworkEngine(world, loop_state, knowledge)
    conversation = ConversationEngine(world, display)
    kb = KnowledgeBase(knowledge, world)
    intervention_mgr = InterventionManager(world, engine, display, conversation)
    evidence_board = EvidenceBoard(display, kb, world)
    schedule_tracker = ScheduleTracker(display, world)
    action_resolver = ActionResolver(
        world, engine, display, conversation, kb, intervention_mgr
    )

    # ── Opening narration ──────────────────────────────────────────────
    if loop_state.loop_number == 1 and loop_state.current_day == 1:
        display.clear()
        display.show_loop_reset(1, False)

        opening = await _llm_call(
            f"You are narrating the opening of a mystery game at Camp Pinehaven.\n"
            f"Camp history: {world.camp_history}\n\n"
            f"Write a brief, atmospheric opening (2-3 sentences). "
            f"The player wakes up on their first morning at camp. "
            f"Something feels slightly off, but they can't place it. "
            f"End with them stepping out of their cabin."
        )
        display.print(f"\n  [italic]{opening}[/]\n")
        await display.get_input("[Press ENTER to continue]")

        # First-loop guidance
        display.print("\n  [bold cyan]Tip:[/] [dim]You hear raised voices from the Dining Hall. "
                      "Maybe someone there can tell you what's going on.[/]")
        display.print("  [dim]Try talking to people, searching locations, and observing "
                      "your surroundings.[/]")
        display.print("  [dim]Being empathetic builds trust. Knowledge persists between loops.[/]")
        display.print()

    # ── Main game loop ─────────────────────────────────────────────────
    game_over = False

    while not game_over:
        # Show current scene
        display.print(f"\n[dim]{'─' * 50}[/]\n")
        loc = engine.get_location(loop_state.player_location)
        chars_here = engine.get_characters_at_location(loop_state.player_location)

        if loc:
            current_idx = _slot_index(loop_state.current_day, loop_state.current_slot.value)
            actions_remaining = TOTAL_SLOTS - current_idx
            display.show_location_scene(loc, chars_here, loop_state, actions_remaining=actions_remaining)

        # Show near-miss intervention hints
        hints = engine.get_near_miss_hints()
        for hint in hints:
            display.print(f"  [yellow italic]{hint}[/]")

        # Get available actions
        actions = engine.get_available_actions()

        # Show menu and get choice
        action = await display.show_action_menu(actions)
        if not action:
            display.print("[dim]Invalid choice. Try again.[/]")
            continue

        # Handle meta actions (no time cost)
        if action["type"] == "evidence_board":
            await evidence_board.show(knowledge)
            continue
        elif action["type"] == "schedule_tracker":
            await schedule_tracker.show(knowledge, loop_state)
            continue
        elif action["type"] == "map":
            known_positions = {}
            for name in chars_here:
                known_positions[name] = loop_state.player_location
            display.show_map(loop_state.player_location, known_positions)
            await display.get_input("\n[Press ENTER to continue]")
            continue

        # Resolve action
        result = await action_resolver.resolve(action, loop_state, knowledge)

        if result.get("message"):
            display.print(f"\n  {result['message']}")

        # Handle fast-forward loop end
        if result.get("fast_forward_loop_end"):
            time_result = result["fast_forward_loop_end"]
            # Same loop-end handling as normal
            if time_result["catastrophe"]:
                display.show_catastrophe(world.catastrophe_description)
            save_mgr.auto_save(world, loop_state, knowledge)
            ending = engine.evaluate_ending()
            if not time_result["catastrophe"] and ending.type.value != "failure":
                game_over = True
                await _show_ending(display, world, knowledge, ending)
            elif not engine.reset_loop():
                game_over = True
                display.print("\n[bold red]You've exhausted all your loops.[/]")
                await _show_ending(display, world, knowledge, ending)
            else:
                engine = ClockworkEngine(world, loop_state, knowledge)
                display.show_loop_reset(loop_state.loop_number, time_result["catastrophe"])
                await _between_loop_screen(display, evidence_board, schedule_tracker,
                                            knowledge, kb, loop_state, world)

        # Advance time if action costs it
        if result.get("advance_time"):
            time_result = engine.advance_time()

            if time_result["type"] == "loop_end":
                # End of loop
                if time_result["catastrophe"]:
                    # Generate visceral catastrophe narration
                    chars_known = ", ".join(knowledge.characters_met[:5]) if knowledge.characters_met else "the people here"
                    catastrophe_narration = await _llm_call(
                        f"Narrate the catastrophe at Camp Pinehaven in 3-4 vivid sentences.\n"
                        f"The catastrophe: {world.catastrophe_description}\n"
                        f"Characters the player knows: {chars_known}\n"
                        f"This is loop {loop_state.loop_number}.\n\n"
                        f"Make it visceral and personal — reference the characters the player has met. "
                        f"Show the moment of realization, the chaos, and the inevitable reset. "
                        f"End with the world going dark as the loop resets."
                    )
                    display.show_catastrophe(world.catastrophe_description)
                    display.print(f"  [italic red]{catastrophe_narration}[/]\n")
                    await display.get_input("[Press ENTER]")

                # Auto-save
                save_mgr.auto_save(world, loop_state, knowledge)

                # Check if game is over
                ending = engine.evaluate_ending()

                if not time_result["catastrophe"] and ending.type.value != "failure":
                    # Player prevented catastrophe — game over with good ending
                    game_over = True
                    await _show_ending(display, world, knowledge, ending)
                elif not engine.reset_loop():
                    # Max loops reached
                    game_over = True
                    display.print("\n[bold red]You've exhausted all your loops.[/]")
                    await _show_ending(display, world, knowledge, ending)
                else:
                    # Reset for new loop
                    # Rebuild engine with fresh loop state
                    engine = ClockworkEngine(world, loop_state, knowledge)

                    display.show_loop_reset(loop_state.loop_number, time_result["catastrophe"])

                    # Generate per-loop opening narration (loops 2+)
                    loop_num = loop_state.loop_number
                    chars_met_count = len(knowledge.characters_met)
                    evidence_count = len(knowledge.evidence_discovered)
                    loop_narration = await _llm_call(
                        f"You are narrating the start of loop {loop_num} in a time-loop mystery at Camp Pinehaven.\n"
                        f"The player has lived through this before — {loop_num - 1} time(s).\n"
                        f"They've met {chars_met_count} people and found {evidence_count} pieces of evidence.\n"
                        f"Camp setting: {world.camp_history[:200]}\n\n"
                        f"Write 2-3 sentences of the player waking up again. Emphasize the growing "
                        f"sense of deja vu and determination. Reference specific sensory details — "
                        f"the bugle, the morning light, the smell of pine. Each loop should feel "
                        f"slightly different as the player's knowledge grows."
                    )
                    display.print(f"\n  [italic]{loop_narration}[/]\n")
                    await display.get_input("[Press ENTER to continue]")

                    # Between-loop screen
                    await _between_loop_screen(display, evidence_board, schedule_tracker,
                                                knowledge, kb, loop_state, world)

            elif time_result["type"] in ("slot_advance", "day_advance"):
                # Show rumor propagation events
                for rev in time_result.get("rumor_events", []):
                    if rev["type"] == "rumor_spread":
                        display.print(
                            f"  [dim italic]Elsewhere, {rev['from']} mentions "
                            f"something to {rev['to']}...[/]"
                        )
                    elif rev["type"] == "alarm_reaction":
                        display.print(
                            f"  [yellow italic]{rev['detail']}.[/]"
                        )

                if time_result["type"] == "day_advance":
                    display.print(
                        f"\n  [bold]A new day dawns. Day {time_result['day']}.[/]"
                    )

        # Auto-save periodically
        if loop_state.current_slot.value == "NIGHT":
            save_mgr.auto_save(world, loop_state, knowledge)

    display.print("\n[bold]Thank you for playing LOOP.[/]\n")


async def _between_loop_screen(
    display: GameDisplay,
    evidence_board: EvidenceBoard,
    schedule_tracker: ScheduleTracker,
    knowledge: PersistentKnowledge,
    kb: KnowledgeBase,
    loop_state: LoopState | None = None,
    world: WorldState | None = None,
):
    """Between-loop review screen."""
    display.print("\n[bold cyan]═══ BETWEEN LOOPS ═══[/]")
    display.print("[dim]Review your findings before the next loop begins.[/]\n")

    # Show missed opportunities
    if world:
        missed_opps = []
        # Closest unfound evidence
        undiscovered = [e for e in world.evidence_registry if e.id not in knowledge.evidence_discovered]
        if undiscovered:
            # Find evidence with fewest unmet prerequisites
            best = min(undiscovered, key=lambda e: sum(1 for p in e.prerequisites if p not in knowledge.evidence_discovered))
            unmet = sum(1 for p in best.prerequisites if p not in knowledge.evidence_discovered)
            if unmet == 0:
                missed_opps.append(f"There's evidence at {best.source_location.replace('_', ' ').title()} you haven't found yet")
            elif unmet == 1:
                missed_opps.append(f"You're one discovery away from unlocking something important")

        # Available interventions not taken
        for iv in world.intervention_tree:
            if iv.id not in (loop_state.interventions_made if loop_state else []):
                has_evidence = all(e in knowledge.evidence_discovered for e in iv.required_evidence)
                if has_evidence:
                    missed_opps.append(f"You had the evidence for an intervention but didn't use it in time")
                    break

        if missed_opps:
            display.print("[bold yellow]Missed Opportunities:[/]")
            for opp in missed_opps[:3]:
                display.print(f"  [yellow]- {opp}[/]")
            display.print()

    while True:
        display.print("[bold]Options:[/]")
        display.print("  1. Evidence Board")
        display.print("  2. Schedule Tracker")
        display.print("  3. Write/Review Theories")
        display.print("  4. View Loop History")
        display.print("  5. Conversation Journal")
        display.print("  6. Begin Next Loop")
        display.print()

        choice = await display.get_input("Choose: ")

        if choice == "1":
            await evidence_board.show(knowledge)
        elif choice == "2":
            await schedule_tracker.show(knowledge, loop_state)
        elif choice == "3":
            display.print("\n[bold]Current Theories:[/]")
            if knowledge.theories:
                for i, t in enumerate(knowledge.theories, 1):
                    display.print(f"  {i}. {t}")
            else:
                display.print("  [dim]No theories yet.[/]")
            display.print()
            display.print("  [dim]Type a new theory, 'remove <number>', or 'back'[/]")
            cmd = await display.get_input("Theory> ")
            if cmd and cmd.lower() not in ("back", "done"):
                if cmd.lower().startswith("remove "):
                    try:
                        idx = int(cmd.split()[1]) - 1
                        kb.remove_theory(idx)
                    except (ValueError, IndexError):
                        pass
                else:
                    kb.add_theory(cmd)
        elif choice == "4":
            if knowledge.loop_history:
                for loop_sum in knowledge.loop_history:
                    display.print(
                        f"\n  [bold]Loop {loop_sum.loop_number}[/]: "
                        f"Found {len(loop_sum.evidence_found)} evidence, "
                        f"attempted {len(loop_sum.interventions_attempted)} interventions"
                    )
                    if loop_sum.ending_reached:
                        display.print(f"    Ending: {loop_sum.ending_reached}")
            else:
                display.print("  [dim]No loop history yet.[/]")
            await display.get_input("\n[Press ENTER]")
        elif choice == "5":
            if knowledge.conversation_journal:
                display.print("\n[bold]Conversation Journal:[/]")
                for entry in knowledge.conversation_journal:
                    display.print(f"  [dim]{entry}[/]")
            else:
                display.print("  [dim]No conversations recorded yet.[/]")
            await display.get_input("\n[Press ENTER]")
        elif choice == "6":
            break


async def _show_ending(display: GameDisplay, world: WorldState,
                       knowledge: PersistentKnowledge, ending):
    """Show the game ending with LLM narration."""
    display.clear()

    # Generate ending narration
    ending_prompt = build_ending_prompt(world, ending, knowledge)
    narration = await _llm_call(ending_prompt)

    display.print(f"\n[bold]{'═' * 50}[/]")
    display.print(f"[bold]  {ending.type.value.replace('_', ' ').upper()}[/]")
    display.print(f"[bold]{'═' * 50}[/]")
    display.print()
    display.print(f"  {narration}")
    display.print()

    await display.get_input("[Press ENTER to see your summary]")

    # Generate summary
    summary_prompt = build_summary_prompt(world, knowledge)
    summary = await _llm_call(summary_prompt)

    display.show_post_game_summary(world, knowledge, summary)

    await display.get_input("\n[Press ENTER to exit]")


def run():
    """Synchronous entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
