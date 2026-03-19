"""Save/load system for LOOP."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import SAVE_DIR
from .models import LoopState, PersistentKnowledge, SaveGame, WorldState


class SaveManager:
    """Manages save/load of game state."""

    def __init__(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, world: WorldState, loop_state: LoopState,
             knowledge: PersistentKnowledge, slot: str = "auto") -> Path:
        """Save game state to a file."""
        save = SaveGame(
            timestamp=datetime.now(),
            world_state=world,
            loop_state=loop_state,
            knowledge=knowledge,
        )
        path = SAVE_DIR / f"save_{slot}.json"
        path.write_text(save.model_dump_json(indent=2))
        return path

    def load(self, slot: str = "auto") -> SaveGame | None:
        """Load game state from a file."""
        path = SAVE_DIR / f"save_{slot}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return SaveGame(**data)

    def list_saves(self) -> list[dict]:
        """List available save files."""
        saves = []
        for path in sorted(SAVE_DIR.glob("save_*.json")):
            try:
                data = json.loads(path.read_text())
                save = SaveGame(**data)
                slot_name = path.stem.replace("save_", "")
                saves.append({
                    "slot": slot_name,
                    "path": path,
                    "timestamp": save.timestamp.isoformat(),
                    "loop": save.loop_state.loop_number,
                    "day": save.loop_state.current_day,
                })
            except (json.JSONDecodeError, ValueError):
                continue
        return saves

    def delete_save(self, slot: str) -> bool:
        """Delete a save file."""
        path = SAVE_DIR / f"save_{slot}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def auto_save(self, world: WorldState, loop_state: LoopState,
                  knowledge: PersistentKnowledge) -> Path:
        """Auto-save between loops."""
        return self.save(world, loop_state, knowledge, slot="auto")
