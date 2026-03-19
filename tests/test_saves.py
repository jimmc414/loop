"""Tests for SaveManager (loop/saves.py)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from loop.models import (
    LoopState,
    PersistentKnowledge,
    SaveGame,
    ScheduleEntry,
    TimeSlot,
    WorldState,
)
from loop.saves import SaveManager
from tests.conftest import make_knowledge, make_loop_state, make_world


@pytest.fixture
def save_dir(tmp_path):
    """Patch SAVE_DIR to use a temp directory."""
    with patch("loop.saves.SAVE_DIR", tmp_path):
        yield tmp_path


@pytest.fixture
def manager(save_dir):
    return SaveManager()


class TestSaveLoad:
    def test_full_roundtrip(self, manager, save_dir):
        world = make_world(catastrophe_description="Test catastrophe")
        loop = make_loop_state(
            current_day=3,
            current_slot=TimeSlot.EVENING,
            character_trust={"Alice": 42},
            schedule_modifications={
                "Alice": {"5": ScheduleEntry(location="kitchen", activity="cooking")}
            },
        )
        knowledge = make_knowledge(
            evidence_discovered=["ev1", "ev2"],
            characters_met=["Alice", "Bob"],
        )

        path = manager.save(world, loop, knowledge, slot="test1")
        assert path.exists()

        loaded = manager.load(slot="test1")
        assert loaded is not None
        assert loaded.world_state.catastrophe_description == "Test catastrophe"
        assert loaded.loop_state.current_day == 3
        assert loaded.loop_state.current_slot == TimeSlot.EVENING
        assert loaded.loop_state.character_trust["Alice"] == 42
        assert "ev1" in loaded.knowledge.evidence_discovered
        assert "Alice" in loaded.knowledge.characters_met
        # Verify str keys in schedule_modifications survive
        mods = loaded.loop_state.schedule_modifications
        assert "Alice" in mods
        assert "5" in mods["Alice"]

    def test_load_nonexistent_returns_none(self, manager, save_dir):
        result = manager.load(slot="nonexistent")
        assert result is None

    def test_save_creates_file(self, manager, save_dir):
        manager.save(make_world(), make_loop_state(), make_knowledge(), slot="s1")
        assert (save_dir / "save_s1.json").exists()

    def test_list_saves(self, manager, save_dir):
        manager.save(make_world(), make_loop_state(), make_knowledge(), slot="a")
        manager.save(
            make_world(),
            make_loop_state(current_day=3, loop_number=2),
            make_knowledge(),
            slot="b",
        )
        saves = manager.list_saves()
        assert len(saves) == 2
        slots = {s["slot"] for s in saves}
        assert "a" in slots
        assert "b" in slots

    def test_list_saves_metadata(self, manager, save_dir):
        manager.save(
            make_world(),
            make_loop_state(loop_number=3, current_day=4),
            make_knowledge(),
            slot="meta",
        )
        saves = manager.list_saves()
        assert len(saves) == 1
        meta = saves[0]
        assert meta["loop"] == 3
        assert meta["day"] == 4
        assert "timestamp" in meta

    def test_delete_save(self, manager, save_dir):
        manager.save(make_world(), make_loop_state(), make_knowledge(), slot="del")
        assert (save_dir / "save_del.json").exists()
        result = manager.delete_save("del")
        assert result is True
        assert not (save_dir / "save_del.json").exists()

    def test_delete_nonexistent(self, manager, save_dir):
        result = manager.delete_save("nope")
        assert result is False

    def test_auto_save(self, manager, save_dir):
        path = manager.auto_save(make_world(), make_loop_state(), make_knowledge())
        assert path.exists()
        assert "auto" in path.name
