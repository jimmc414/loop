# LOOP

A time-loop investigation game where every NPC conversation is powered by an LLM.

You wake up at Camp Pinehaven. Something terrible is going to happen in five days. When it does, time resets and you wake up again — but you remember everything. The NPCs don't. Or at least, they shouldn't.

## What makes this different

Most time-loop games use scripted dialogue trees. LOOP uses Claude as the dialogue engine. NPCs respond to what you actually say — you can lie, empathize, confront, deflect. There are no dialogue options. You type what you want to say.

The game also has a system called **Deja Vu**: as you interact with the same NPCs across multiple loops, they develop fragmented awareness of previous iterations. Not full memories — dreams, premonitions, a sense that you've met before. Innocent NPCs become allies. Guilty ones become paranoid. This isn't cosmetic; it affects trust mechanics, knowledge penalties, and the strategic calculus of who you invest time in.

## How it works

The game is split into two layers:

**Deterministic clockwork engine** — All game state (time, location, trust scores, evidence, causal chains, interventions) is managed by pure Python. No LLM calls for game logic. Trust changes are pattern-matched from player input. Evidence discovery is prerequisite-gated. The causal chain leading to the catastrophe is a directed graph that you interrupt by gathering evidence and building trust.

**LLM dialogue layer** — NPC conversations use Claude with carefully constructed system prompts that encode personality, speech patterns, mood, trust level, knowledge state, rumor awareness, and deja vu intensity. The prompts are the game design.

World generation is also LLM-powered. Each new game creates a unique catastrophe, cast of characters, evidence chain, and intervention tree. Two playthroughs won't have the same mystery.

## Architecture

```
loop/
  main.py              # async game loop
  state_machine.py     # deterministic clockwork engine (no LLM)
  conversation_engine.py  # LLM dialogue + trust heuristics
  models.py            # Pydantic v2 data models
  world_generator.py   # multi-step LLM world generation
  config.py            # constants and tuning knobs
  display.py           # Rich terminal UI
  game_logger.py       # session logging (plain-text capture)
  rumor_mill.py        # NPC-to-NPC information propagation
  evidence_board.py    # player evidence tracking
  knowledge_base.py    # persistent cross-loop knowledge
  prompts/
    conversation.py    # NPC system prompt construction
    world_gen.py       # world generation prompts
    summary.py         # ending narration prompts
tests/
  conftest.py          # shared test fixtures
  test_state_machine.py
  test_conversation_trust.py
  test_deja_vu.py
  ...                  # 360 tests total
```

## Mechanics

- **7 loops**, 5 days each, 4 time slots per day (140 max actions)
- **Trust system**: empathy builds trust, pressure loses it. NPCs have thresholds that gate secret revelation
- **Evidence chain**: physical evidence, testimony, observations — connected in a prerequisite graph
- **Causal chain**: the catastrophe is the end of a multi-step chain. Interrupt any link to change the outcome
- **Rumor mill**: NPCs share information when co-located. Tell one NPC something and it propagates
- **Character tiers**: Tier 1 (guilty), Tier 2 (witnesses), Tier 3 (innocents) — each behaves differently under pressure
- **Deja vu**: cross-loop NPC awareness that escalates from subtle flickers (loop 3) to fracture-level recognition (loops 6-7)
- **Five endings**: full prevention, partial prevention, exposure, failure, and a deeper truth

## Requirements

- Python 3.11+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed and authenticated
- A Claude Max subscription (uses OAuth credentials from `~/.claude/.credentials.json`)

## Install

```bash
git clone https://github.com/jimmc414/loop.git
cd loop
pip install pydantic rich claude-code-sdk
```

## Run

```bash
python -m loop.main
```

First run generates a new world (takes ~60 seconds). Subsequent runs load the saved world.

### Session logging

Record the entire game session to a plain-text file for review or analysis:

```bash
python -m loop.main --log                      # auto-generates timestamped file in loop/data/logs/
python -m loop.main --log=my_session.log       # custom path
```

The log captures all terminal output (Rich markup stripped) and player input, bookended with timestamps.

## Run tests

```bash
pip install pytest
pytest tests/ -q
```

## Design notes

The `loop_number` parameter was threaded through `build_system_prompt()` from the start but never used — the architecture anticipated cross-loop NPC awareness before the feature existed. The deja vu system activates it.

Trust deltas are deterministic pattern matches, not LLM judgments. This means the game is testable and reproducible — the LLM handles flavor, the engine handles state.

The rumor mill uses a co-location propagation model: NPCs sharing a location during time advancement exchange claims. This means the camp's social network is emergent from schedules, not hardcoded.

## License

MIT
