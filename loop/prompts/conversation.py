"""NPC conversation prompt builders."""

from __future__ import annotations

from ..models import Character, CharacterTier, Evidence, KnowledgeEntry, PersistentKnowledge


def build_system_prompt(
    character: Character,
    knowledge: KnowledgeEntry,
    trust_level: int,
    player_knowledge: PersistentKnowledge,
    loop_number: int,
    day: int,
    slot: str,
    heard_rumors: str = "",
    current_location: str = "",
    current_activity: str = "",
) -> str:
    """Build the system prompt for an NPC conversation."""

    # Trust tier label
    if trust_level >= 80:
        trust_label = "VULNERABLE — will share deep secrets and fears"
    elif trust_level >= 60:
        trust_label = "CONFIDING — will share personal concerns"
    elif trust_level >= 40:
        trust_label = "FRIENDLY — open to real conversation"
    elif trust_level >= 20:
        trust_label = "CASUAL — polite but guarded"
    else:
        trust_label = "STRANGER — minimal, wary"

    # What the character is willing to discuss
    topics = ", ".join(knowledge.available_topics) if knowledge.available_topics else "general camp small talk"

    # Secrets they might reveal based on trust
    revealable_secrets = []
    if trust_level >= 60 and character.secrets:
        revealable_secrets.append(character.secrets[0])
    if trust_level >= 80 and len(character.secrets) > 1:
        revealable_secrets.extend(character.secrets[1:])

    secrets_section = ""
    if revealable_secrets:
        secrets_section = f"""
SECRETS YOU MAY HINT AT OR REVEAL (if the conversation naturally goes there):
{chr(10).join(f'- {s}' for s in revealable_secrets)}
"""

    # What the player knows that the character might find suspicious
    impossible_knowledge_flags = _get_impossible_knowledge_flags(character, player_knowledge, day, slot)
    impossible_section = ""
    if impossible_knowledge_flags:
        interaction_count = player_knowledge.npc_interaction_counts.get(character.name, 0)
        deja_vu_active = loop_number >= 3 and interaction_count > 0
        if deja_vu_active and character.tier == CharacterTier.TIER3:
            suspicion_guidance = (
                "You are LESS suspicious of their knowledge. Accept it with mild curiosity — "
                "something about this person feels familiar, like a half-remembered dream."
            )
        elif deja_vu_active and character.tier == CharacterTier.TIER2:
            suspicion_guidance = (
                "You feel torn about their knowledge. Part of you wants to question it, "
                "but another part feels like you already knew. Reduce your suspicion."
            )
        else:
            suspicion_guidance = (
                "If the player mentions these without explanation, react with suspicion or confusion.\n"
                "If they provide a plausible cover story, accept it cautiously."
            )
        impossible_section = f"""
WARNING — The player may reference knowledge they shouldn't have yet:
{chr(10).join(f'- {f}' for f in impossible_knowledge_flags)}
{suspicion_guidance}
"""

    # Relationships context
    rel_section = ""
    if character.relationships:
        rels = "\n".join(f"- {name}: {desc}" for name, desc in character.relationships.items())
        rel_section = f"\nYOUR RELATIONSHIPS:\n{rels}\n"

    return f"""\
You are {character.name}, a {character.age}-year-old {character.role} at Camp Pinehaven.

PERSONALITY: {character.personality}

SPEECH PATTERN: {character.speech_pattern}
Stay in character at all times. Use this speech pattern consistently.

BACKSTORY: {character.backstory}
{rel_section}
CURRENT STATE:
- Day {day}, {slot}
- Location: {current_location.replace('_', ' ').title() if current_location else 'somewhere in camp'}
- Activity: {current_activity if current_activity else 'going about their day'}
- Mood: {knowledge.mood}
- Willingness to talk: {knowledge.willingness}
- Trust level with player: {trust_level} ({trust_label})

TOPICS YOU CAN DISCUSS: {topics}
You should naturally steer toward these topics but respond to what the player says.
If asked about things you wouldn't know, say so honestly.

{secrets_section}{impossible_section}{heard_rumors}\
{_build_deja_vu_section(character, player_knowledge, loop_number)}
RULES:
- Stay in character. Never break the fourth wall.
- Keep responses to 2-4 sentences. This is dialogue, not monologue.
- React to the player's tone. Empathy earns trust, aggression loses it.
- If your willingness is "reluctant", give shorter responses and resist probing.
- If your mood is negative, it should color your dialogue.
- {_build_time_loop_rule(character, player_knowledge, loop_number)}
- NEVER reveal all your secrets at once, even at high trust. Dole them out naturally.
- If the player tries to rush or pressure you, become more guarded.
"""


def build_conversation_prompt(
    history: list[dict],
    new_message: str,
    character_name: str,
) -> str:
    """Build the user prompt with conversation history."""
    parts = []

    if history:
        parts.append("--- CONVERSATION SO FAR ---")
        for exchange in history:
            parts.append(f"Player: {exchange['player']}")
            parts.append(f"{character_name}: {exchange['npc']}")
        parts.append("--- END ---\n")

    parts.append(f"Player: {new_message}")
    parts.append(f"Respond as {character_name}:")

    return "\n".join(parts)


def _get_impossible_knowledge_flags(
    character: Character,
    player_knowledge: PersistentKnowledge,
    day: int,
    slot: str,
) -> list[str]:
    """Identify knowledge the player has that they shouldn't at this point."""
    flags = []

    # Check if player knows things about this character's future schedule
    if character.name in player_knowledge.character_schedules_known:
        known = player_knowledge.character_schedules_known[character.name]
        from ..config import slot_index, SLOT_NAMES
        current_idx = slot_index(day, slot)
        for idx_str, loc in known.items():
            idx = int(idx_str) if isinstance(idx_str, str) else idx_str
            if idx > current_idx:
                d, s = idx // 4 + 1, SLOT_NAMES[idx % 4]
                flags.append(f"Player knows {character.name} will be at {loc} on Day {d} {s}")

    # Check if player has evidence that relates to this character's secrets
    # Build a quick lookup of discovered evidence descriptions
    # (We don't have the world object here, so we check evidence IDs against secrets via keyword overlap)
    for secret in character.secrets:
        secret_words = set(w.lower() for w in secret.split() if len(w) >= 5)
        if not secret_words:
            continue
        for ev_id in player_knowledge.evidence_discovered:
            # If the evidence ID references this character (heuristic: character name in ID)
            if character.name.lower() in ev_id.lower():
                flags.append(f"Player has evidence ({ev_id}) that may relate to: {secret[:60]}...")
                break
            # Check keyword overlap between evidence ID and secret
            ev_words = set(w.lower() for w in ev_id.replace("_", " ").split() if len(w) >= 4)
            if len(secret_words & ev_words) >= 1:
                flags.append(f"Player has evidence ({ev_id}) touching on: {secret[:60]}...")
                break

    return flags


# ── Déjà Vu system ────────────────────────────────────────────────────


def _build_deja_vu_section(
    character: Character,
    player_knowledge: PersistentKnowledge,
    loop_number: int,
) -> str:
    """Build the déjà vu prompt section based on loop number and prior interactions."""
    interaction_count = player_knowledge.npc_interaction_counts.get(character.name, 0)

    # No déjà vu in loops 1-2, or if NPC has never been interacted with
    if loop_number <= 2 or interaction_count == 0:
        return ""

    # Gather previous topics for this NPC
    previous_topics = player_knowledge.npc_previous_topics.get(character.name, [])

    if loop_number == 3:
        return _tier_subtle(character, interaction_count, previous_topics)
    elif loop_number <= 5:
        return _tier_growing(character, interaction_count, previous_topics)
    else:
        return _tier_fracture(character, interaction_count, previous_topics)


def _tier_subtle(
    character: Character,
    interaction_count: int,
    previous_topics: list[str],
) -> str:
    """Loop 3: Rare flickers of familiarity."""
    topic_hint = ""
    if previous_topics:
        topic_hint = (
            f" When topics like {', '.join(previous_topics[:3])} come up, "
            f"you feel a faint chill — like you've discussed this before."
        )

    return f"""
DEJA VU — SUBTLE:
Something about this person feels vaguely familiar, though you can't place why.
Occasionally pause mid-sentence as if reaching for a memory that isn't there.
You might say things like "Have we met before?" or "That's funny, I was just thinking about that."
Do NOT overdo it — these are rare flickers, not constant commentary.{topic_hint}
Interaction intensity: {interaction_count} prior exchanges across loops (you don't know this number).
"""


def _tier_growing(
    character: Character,
    interaction_count: int,
    previous_topics: list[str],
) -> str:
    """Loops 4-5: Dreams, premonitions, tier-specific behavior."""
    topic_section = ""
    if previous_topics:
        topic_section = (
            f"\nTopic-specific déjà vu triggers: {', '.join(previous_topics[:5])}. "
            f"When these come up, you experience a stronger flash of recognition."
        )

    intensity = "moderate" if interaction_count < 10 else "strong"

    if character.tier == CharacterTier.TIER3:
        behavior = (
            "You feel OPEN and drawn to this person. You've been having dreams about a stranger "
            "who feels exactly like them. You share more freely than you normally would — "
            "their presence feels safe and familiar. You may anticipate what they're about to say."
        )
    elif character.tier == CharacterTier.TIER2:
        behavior = (
            "You feel CONFLICTED. Words bubble up that you didn't plan to say — "
            "you blurt things then catch yourself. Part of you trusts this person deeply "
            "for reasons you can't explain; another part is terrified of what that means."
        )
    else:  # TIER1
        behavior = (
            "You feel an urge to SUPPRESS. This person makes you uneasy in a way that goes "
            "beyond normal wariness. You're more guarded than usual. You actively resist "
            "sharing information and may try to steer the conversation away from sensitive topics."
        )

    return f"""
DEJA VU — GROWING ({intensity}):
{behavior}{topic_section}
You've had recurring dreams about this person. You can't explain it rationally.
Interaction intensity: {interaction_count} prior exchanges (you sense this as {intensity} familiarity).
"""


def _tier_fracture(
    character: Character,
    interaction_count: int,
    previous_topics: list[str],
) -> str:
    """Loops 6-7: Specific echoes, potential direct acknowledgment."""
    topic_section = ""
    if previous_topics:
        topic_section = (
            f"\nYou have specific echoes of past conversations about: {', '.join(previous_topics[:7])}. "
            f"You may reference these as 'dreams' or 'feelings' you've had."
        )

    high_interaction = interaction_count >= 15

    if character.tier == CharacterTier.TIER3:
        if high_interaction:
            behavior = (
                "You are an ALLY. You feel a deep, inexplicable bond with this person. "
                "You may predict their questions before they ask them. You actively help "
                "their investigation — you don't fully understand why, but you know it matters. "
                "You sense that something is deeply wrong with time itself."
            )
        else:
            behavior = (
                "You feel a strong connection to this person. You're open and trusting, "
                "and you sometimes finish their thoughts. Dreams about them feel more "
                "like memories now."
            )
    elif character.tier == CharacterTier.TIER2:
        behavior = (
            "You are TERRIFIED. The familiarity with this person has become overwhelming. "
            "You oscillate between wanting to confess everything and wanting to flee. "
            "Your hands shake. You may start sentences you can't finish. "
            "The weight of something you can't name is crushing you."
        )
    else:  # TIER1
        if high_interaction:
            behavior = (
                "You are PARANOID. This person has haunted your dreams. You feel pre-caught — "
                "as if judgment has already been passed and you're just waiting for the sentence. "
                "You are dangerous: cornered and desperate. However, if the player shows "
                "genuine empathy, there is a rare crack — a moment where the weight of "
                "all these loops might break through your defenses."
            )
        else:
            behavior = (
                "You are deeply unsettled by this person. Your instinct is to shut down "
                "the conversation and create distance. Everything they say feels like "
                "an accusation you've already heard."
            )

    return f"""
DEJA VU — FRACTURE:
{behavior}{topic_section}
Interaction intensity: {interaction_count} prior exchanges (the weight of these is {
    'overwhelming' if high_interaction else 'significant'}).
"""


def _build_time_loop_rule(
    character: Character,
    player_knowledge: PersistentKnowledge,
    loop_number: int,
) -> str:
    """Build the time-loop awareness rule, which may change at high loops for Tier3."""
    interaction_count = player_knowledge.npc_interaction_counts.get(character.name, 0)
    if (loop_number >= 6
            and character.tier == CharacterTier.TIER3
            and interaction_count >= 15):
        return (
            "Something is deeply wrong with time. You feel it but can't explain it. "
            "The player is just a fellow camper/visitor, but you sense a connection "
            "that transcends normal experience."
        )
    return "You do NOT know this is a time loop. The player is just a fellow camper/visitor."


def build_evidence_confrontation_prompt(
    character: Character,
    evidence: Evidence,
    trust_level: int,
    relevance: str,
    tier_guidance: str,
) -> str:
    """Build a prompt fragment for presenting evidence to an NPC during conversation.

    Returns a focused prompt injected into the user message when the player
    presents evidence.  Contains the evidence description, relevance tag,
    tier-based behavioural directive, and instruction to respond in 2-4
    sentences in character.
    """
    trust_context = (
        f"high (trust {trust_level})" if trust_level >= 40
        else f"low (trust {trust_level})"
    )

    return (
        f"[EVIDENCE PRESENTED]\n"
        f"The player shows you the following evidence: {evidence.description}\n"
        f"Evidence type: {evidence.type.value}\n"
        f"Relevance to you: {relevance}\n"
        f"Your current trust with the player: {trust_context}\n\n"
        f"BEHAVIORAL DIRECTIVE:\n{tier_guidance}\n\n"
        f"Respond in 2-4 sentences, staying fully in character. "
        f"React to being confronted with this evidence. "
        f"Your reaction should reflect both the relevance of the evidence to you "
        f"and your current trust level with the player."
    )
