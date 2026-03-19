"""Multi-step LLM world generation using Claude Agent SDK."""

from __future__ import annotations

import json
import os

if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

from .config import DATA_DIR, LOCATIONS_TEMPLATE, TOTAL_SLOTS
from .models import (
    CausalChainEvent,
    Character,
    EndingCondition,
    EndingType,
    Evidence,
    InterventionNode,
    KnowledgeEntry,
    Location,
    ScheduleEntry,
    WildCardEvent,
    WorldState,
)
from .prompts.world_gen import (
    STEP1_CATASTROPHE,
    step2_characters,
    step3_schedules,
    step4_knowledge,
    step5_evidence,
)


async def _llm_call(prompt: str) -> str:
    """Single LLM call returning text content."""
    from .llm import llm_query
    return await llm_query(prompt)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end])
    return json.loads(text)


def _build_locations() -> list[Location]:
    """Build Location models from the template."""
    locations = []
    for tmpl in LOCATIONS_TEMPLATE:
        locations.append(Location(
            name=tmpl["name"],
            id=tmpl["id"],
            area=tmpl["area"],
            description=tmpl["description"],
            adjacent_locations=tmpl["adjacent"],
            is_distant=tmpl["is_distant"],
        ))
    return locations


def _validate_world(world: WorldState) -> list[str]:
    """Validate world state consistency. Returns list of issues."""
    issues = []
    location_ids = {loc.id for loc in world.locations}
    char_names = {c.name for c in world.characters}

    # Check schedule lengths
    for char in world.characters:
        if len(char.schedule) != TOTAL_SLOTS:
            issues.append(f"{char.name} has {len(char.schedule)} schedule entries, expected {TOTAL_SLOTS}")

    # Check knowledge timeline lengths
    for char in world.characters:
        if len(char.knowledge_timeline) != TOTAL_SLOTS:
            issues.append(f"{char.name} has {len(char.knowledge_timeline)} knowledge entries, expected {TOTAL_SLOTS}")

    # Check location references in schedules
    for char in world.characters:
        for i, entry in enumerate(char.schedule):
            if entry.location not in location_ids:
                issues.append(f"{char.name} schedule[{i}] references unknown location: {entry.location}")

    # Check causal chain references
    for event in world.causal_chain:
        if event.location not in location_ids:
            issues.append(f"Causal event {event.id} references unknown location: {event.location}")
        if event.character not in char_names:
            issues.append(f"Causal event {event.id} references unknown character: {event.character}")

    # Check causal chain chronological order
    for i in range(len(world.causal_chain) - 1):
        a = world.causal_chain[i]
        b = world.causal_chain[i + 1]
        slot_order = {"MORNING": 0, "AFTERNOON": 1, "EVENING": 2, "NIGHT": 3}
        a_val = a.day * 4 + slot_order[a.time_slot.value]
        b_val = b.day * 4 + slot_order[b.time_slot.value]
        if a_val > b_val:
            issues.append(f"Causal chain out of order: {a.id} (day{a.day} {a.time_slot}) > {b.id} (day{b.day} {b.time_slot})")

    # Check evidence prerequisite acyclicity
    evidence_ids = {e.id for e in world.evidence_registry}
    ev_map = {e.id: e for e in world.evidence_registry}
    for ev in world.evidence_registry:
        for prereq in ev.prerequisites:
            if prereq not in evidence_ids:
                issues.append(f"Evidence {ev.id} has unknown prerequisite: {prereq}")
        # Per-path cycle detection (handles diamond DAGs correctly)
        def has_cycle(node_id: str, path: set[str]) -> bool:
            if node_id in path:
                return True
            node = ev_map.get(node_id)
            if not node:
                return False
            for prereq in node.prerequisites:
                if has_cycle(prereq, path | {node_id}):
                    return True
            return False
        if has_cycle(ev.id, set()):
            issues.append(f"Cycle detected in evidence prerequisites involving: {ev.id}")

    # Check at least one ending is achievable
    if not world.ending_conditions:
        issues.append("No ending conditions defined")

    return issues


def _validate_solvability(world: WorldState) -> list[str]:
    """Check if an oracle player could complete the mystery in a single loop.

    An 'oracle' player knows everything and plays optimally.
    Returns list of solvability issues.
    """
    issues = []

    # Check: all evidence is discoverable
    # Build prerequisite graph and verify all evidence is reachable
    ev_map = {e.id: e for e in world.evidence_registry}

    def can_discover(ev_id: str, discovered: set[str], depth: int = 0) -> bool:
        if depth > 50:  # prevent infinite recursion
            return False
        if ev_id in discovered:
            return True
        ev = ev_map.get(ev_id)
        if not ev:
            return False
        # Check all prerequisites are discoverable
        for prereq in ev.prerequisites:
            if not can_discover(prereq, discovered, depth + 1):
                return False
        return True

    undiscoverable = []
    for ev in world.evidence_registry:
        if not can_discover(ev.id, set()):
            undiscoverable.append(ev.id)

    if undiscoverable:
        issues.append(f"Undiscoverable evidence (circular prerequisites): {undiscoverable}")

    # Check: at least one non-failure ending is achievable
    achievable_endings = []
    all_evidence_ids = {e.id for e in world.evidence_registry}
    all_event_ids = {e.id for e in world.causal_chain}
    interruptible_ids = {e.id for e in world.causal_chain if e.is_interruptible}

    for ending in world.ending_conditions:
        if ending.type == EndingType.FAILURE:
            continue
        # Check if all required evidence exists
        evidence_ok = all(eid in all_evidence_ids or eid in all_evidence_ids
                         for eid in ending.required_evidence)
        # Check if required interruptions are possible
        events_ok = all(eid in interruptible_ids or eid in all_event_ids
                       for eid in ending.required_interrupted_events)
        if evidence_ok and events_ok:
            achievable_endings.append(ending.type.value)

    if not achievable_endings:
        issues.append("No non-failure ending appears achievable")

    # Check: interventions reference valid causal events
    causal_ids = {e.id for e in world.causal_chain}
    for iv in world.intervention_tree:
        if iv.causal_event_id not in causal_ids:
            issues.append(f"Intervention {iv.id} references non-existent causal event: {iv.causal_event_id}")
        for req_ev in iv.required_evidence:
            if req_ev not in all_evidence_ids:
                issues.append(f"Intervention {iv.id} requires non-existent evidence: {req_ev}")

    return issues


async def generate_world(max_retries: int = 3) -> WorldState:
    """Generate a complete world through 5 sequential LLM calls."""
    from rich.console import Console
    console = Console()

    for attempt in range(max_retries):
        try:
            console.print(f"\n[bold cyan]Generating world (attempt {attempt + 1}/{max_retries})...[/]")

            # Step 1: Catastrophe + causal chain
            console.print("  [dim]Step 1/5: Catastrophe & causal chain...[/]")
            step1_raw = await _llm_call(STEP1_CATASTROPHE)
            step1 = _extract_json(step1_raw)

            catastrophe_json = json.dumps(step1, indent=2)

            # Step 2: Characters
            console.print("  [dim]Step 2/5: Characters...[/]")
            step2_raw = await _llm_call(step2_characters(catastrophe_json))
            step2 = _extract_json(step2_raw)

            # Inter-step validation: verify character names match causal chain
            causal_names = {e["character"] for e in step1["causal_chain"]}
            char_names_step2 = {c["name"] for c in step2["characters"]}
            missing_names = causal_names - char_names_step2
            if missing_names:
                console.print(f"  [yellow]Warning: Causal chain references characters not in Step 2: {missing_names}[/]")
                # Add missing characters as TIER3 placeholders rather than failing
                for name in missing_names:
                    step2["characters"].append({
                        "name": name,
                        "age": 30,
                        "role": "camp staff",
                        "tier": "tier3",
                        "personality": "Reserved and quiet.",
                        "speech_pattern": "Speaks plainly.",
                        "backstory": "",
                        "secrets": [],
                        "trust_threshold": 40,
                        "relationships": {},
                    })

            characters_json = json.dumps(step2, indent=2)

            # Step 3: Schedules
            console.print("  [dim]Step 3/5: Schedules...[/]")
            step3_raw = await _llm_call(step3_schedules(catastrophe_json, characters_json))
            step3 = _extract_json(step3_raw)

            schedules_json = json.dumps(step3, indent=2)

            # Step 4: Knowledge timelines
            console.print("  [dim]Step 4/5: Knowledge timelines...[/]")
            step4_raw = await _llm_call(step4_knowledge(catastrophe_json, characters_json, schedules_json))
            step4 = _extract_json(step4_raw)

            # Step 5: Evidence + interventions + wild cards
            console.print("  [dim]Step 5/5: Evidence & interventions...[/]")
            step5_raw = await _llm_call(step5_evidence(catastrophe_json, characters_json, schedules_json))
            step5 = _extract_json(step5_raw)

            # ── Assemble WorldState ────────────────────────────────────

            # Build locations
            locations = _build_locations()

            # Build characters with schedules and knowledge
            characters = []
            for char_data in step2["characters"]:
                name = char_data["name"]
                schedule_data = step3.get("schedules", {}).get(name, [])
                knowledge_data = step4.get("knowledge_timelines", {}).get(name, [])

                schedule = [ScheduleEntry(**s) for s in schedule_data]
                knowledge = [KnowledgeEntry(**k) for k in knowledge_data]

                # Pad if short
                while len(schedule) < TOTAL_SLOTS:
                    schedule.append(ScheduleEntry(location="cabin_row", activity="resting"))
                while len(knowledge) < TOTAL_SLOTS:
                    knowledge.append(KnowledgeEntry())

                characters.append(Character(
                    name=name,
                    age=char_data["age"],
                    role=char_data["role"],
                    tier=char_data["tier"],
                    personality=char_data["personality"],
                    speech_pattern=char_data["speech_pattern"],
                    backstory=char_data.get("backstory", ""),
                    schedule=schedule[:TOTAL_SLOTS],
                    secrets=char_data.get("secrets", []),
                    knowledge_timeline=knowledge[:TOTAL_SLOTS],
                    trust_threshold=char_data.get("trust_threshold", 40),
                    relationships=char_data.get("relationships", {}),
                ))

            # Build causal chain
            causal_chain = [CausalChainEvent(**e) for e in step1["causal_chain"]]

            # Build evidence
            evidence = [Evidence(**e) for e in step5.get("evidence_registry", [])]

            # Build interventions
            interventions = [InterventionNode(**i) for i in step5.get("intervention_tree", [])]

            # Build wild cards
            wild_cards = [WildCardEvent(**w) for w in step5.get("wild_cards", [])]

            # Build ending conditions
            endings = [EndingCondition(**e) for e in step1.get("ending_conditions", [])]

            # Reconcile ending conditions: remove evidence IDs that don't exist
            # (Step 1 may reference evidence IDs before Step 5 defines them)
            valid_evidence_ids = {e.id for e in evidence}
            for ec in endings:
                ec.required_evidence = [
                    eid for eid in ec.required_evidence if eid in valid_evidence_ids
                ]

            world = WorldState(
                catastrophe_description=step1["catastrophe_description"],
                camp_history=step1.get("camp_history", ""),
                characters=characters,
                locations=locations,
                causal_chain=causal_chain,
                evidence_registry=evidence,
                intervention_tree=interventions,
                wild_cards=wild_cards,
                ending_conditions=endings,
            )

            # Validate
            issues = _validate_world(world)
            solvability_issues = _validate_solvability(world)
            issues.extend(solvability_issues)
            if issues:
                console.print(f"[yellow]  Validation issues ({len(issues)}):[/]")
                for issue in issues[:10]:
                    console.print(f"    [yellow]- {issue}[/]")
                if len(issues) > 10:
                    console.print(f"    [yellow]... and {len(issues) - 10} more[/]")
                if attempt < max_retries - 1:
                    console.print("  [yellow]Retrying...[/]")
                    continue
                else:
                    console.print("  [yellow]Proceeding with warnings.[/]")

            # Save
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            world_path = DATA_DIR / "world_state.json"
            world_path.write_text(world.model_dump_json(indent=2))
            console.print(f"[green]  World saved to {world_path}[/]")

            return world

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            console.print(f"[red]  Error in generation: {e}[/]")
            if attempt < max_retries - 1:
                console.print("  [yellow]Retrying...[/]")
            else:
                raise RuntimeError(f"World generation failed after {max_retries} attempts: {e}")

    raise RuntimeError("World generation failed")


async def load_world() -> WorldState | None:
    """Load saved world state if it exists."""
    world_path = DATA_DIR / "world_state.json"
    if world_path.exists():
        data = json.loads(world_path.read_text())
        return WorldState(**data)
    return None
