"""Ending and summary prompts."""

from __future__ import annotations

from ..models import EndingCondition, PersistentKnowledge, WorldState


def build_ending_prompt(
    world: WorldState,
    ending: EndingCondition,
    knowledge: PersistentKnowledge,
) -> str:
    """Build prompt for LLM-generated ending narration."""
    # Gather evidence descriptions
    discovered = []
    for ev_id in knowledge.evidence_discovered:
        ev = next((e for e in world.evidence_registry if e.id == ev_id), None)
        if ev:
            discovered.append(ev.description)

    # Characters met
    chars_met = knowledge.characters_met

    # Interventions from final loop
    interventions = []
    if knowledge.loop_history:
        last_loop = knowledge.loop_history[-1]
        for iv_id in last_loop.interventions_attempted:
            iv = next((i for i in world.intervention_tree if i.id == iv_id), None)
            if iv:
                interventions.append(iv.action_description)

    return f"""\
You are the narrator of a time-loop mystery game at Camp Pinehaven.

The player has reached the {ending.type.value.replace('_', ' ').upper()} ending.

ENDING DESCRIPTION: {ending.description}

CAMP HISTORY: {world.camp_history}

CATASTROPHE: {world.catastrophe_description}

EVIDENCE THE PLAYER DISCOVERED:
{chr(10).join(f'- {d}' for d in discovered) if discovered else '- None'}

CHARACTERS THE PLAYER MET: {', '.join(chars_met) if chars_met else 'None'}

INTERVENTIONS MADE:
{chr(10).join(f'- {i}' for i in interventions) if interventions else '- None'}

LOOPS USED: {len(knowledge.loop_history)}

Write a compelling ending narration (3-5 paragraphs):
- Describe the immediate aftermath of the player's actions (or inactions)
- Reveal the full truth behind the catastrophe
- Show what happens to key characters
- If the player prevented the catastrophe, show the moment they realize they've broken the loop
- If they failed, show the catastrophe and the haunting restart
- End with a reflective final line

Tone: Literary, atmospheric, satisfying. The player should feel the weight of their journey.
"""


def build_summary_prompt(
    world: WorldState,
    knowledge: PersistentKnowledge,
) -> str:
    """Build prompt for post-game analysis."""
    total_evidence = len(world.evidence_registry)
    found = len(knowledge.evidence_discovered)
    missed = total_evidence - found

    return f"""\
Write a brief post-game analysis (2-3 paragraphs) for a time-loop mystery game.

The player used {len(knowledge.loop_history)} out of 7 loops.
They found {found} of {total_evidence} pieces of evidence ({missed} missed).
They met {len(knowledge.characters_met)} of {len(world.characters)} characters.
They wrote {len(knowledge.theories)} theories and made {len(knowledge.evidence_connections)} connections.

Comment on their investigative approach. Were they thorough? Did they miss key connections?
What might they discover on a replay?

Keep it brief and encouraging — make them want to play again.
"""
