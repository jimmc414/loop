"""Tests for action menu grouping (loop/display.py)."""

import pytest

from loop.display import GameDisplay


@pytest.fixture
def display():
    return GameDisplay()


def _make_actions(types_and_labels: list[tuple[str, str]], **extras) -> list[dict]:
    """Helper to build action dicts."""
    actions = []
    for atype, label in types_and_labels:
        d = {"type": atype, "label": label}
        d.update(extras)
        actions.append(d)
    return actions


class TestGroupActions:
    def test_talk_follow_grouped_under_people(self, display):
        actions = _make_actions([
            ("talk", "Talk to Alice"),
            ("follow", "Follow Alice"),
            ("observe", "Observe the area"),
            ("wait", "Wait"),
        ])
        grouped = display._group_actions(actions)
        group_names = [name for name, _ in grouped]
        assert "People" in group_names
        people_items = next(items for name, items in grouped if name == "People")
        people_types = [a["type"] for _, a in people_items]
        assert "talk" in people_types
        assert "follow" in people_types

    def test_travel_observe_search_grouped_under_explore(self, display):
        actions = _make_actions([
            ("observe", "Observe the area"),
            ("search", "Search the area"),
            ("travel", "Go to Dining Hall"),
            ("wait", "Wait"),
        ])
        grouped = display._group_actions(actions)
        explore_items = next(items for name, items in grouped if name == "Explore")
        explore_types = [a["type"] for _, a in explore_items]
        assert set(explore_types) == {"observe", "search", "travel"}

    def test_meta_actions_grouped_under_tools(self, display):
        actions = _make_actions([
            ("evidence_board", "Check evidence board"),
            ("schedule_tracker", "Check schedule tracker"),
            ("map", "View camp map"),
            ("wait", "Wait"),
        ])
        grouped = display._group_actions(actions)
        tools_items = next(items for name, items in grouped if name == "Tools")
        tools_types = [a["type"] for _, a in tools_items]
        assert set(tools_types) == {"evidence_board", "schedule_tracker", "map"}

    def test_interventions_get_own_group(self, display):
        actions = _make_actions([
            ("talk", "Talk to Alice"),
            ("intervene", "Intervene: Block the gate"),
            ("wait", "Wait"),
        ])
        grouped = display._group_actions(actions)
        group_names = [name for name, _ in grouped]
        assert "Intervene" in group_names
        intervene_items = next(items for name, items in grouped if name == "Intervene")
        assert len(intervene_items) == 1
        assert intervene_items[0][1]["type"] == "intervene"

    def test_empty_groups_omitted(self, display):
        actions = _make_actions([
            ("observe", "Observe the area"),
            ("wait", "Wait"),
        ])
        grouped = display._group_actions(actions)
        group_names = [name for name, _ in grouped]
        assert "People" not in group_names
        assert "Intervene" not in group_names
        assert "Tools" not in group_names
        assert "Explore" in group_names
        assert "Time" in group_names

    def test_indices_preserved_for_flat_selection(self, display):
        """Original indices must be preserved so numeric input still works."""
        actions = _make_actions([
            ("talk", "Talk to Alice"),
            ("observe", "Observe"),
            ("evidence_board", "Evidence board"),
            ("wait", "Wait"),
        ])
        grouped = display._group_actions(actions)
        all_indices = []
        for _, items in grouped:
            for idx, _ in items:
                all_indices.append(idx)
        # Indices should be 0, 1, 2, 3 (original positions)
        assert sorted(all_indices) == [0, 1, 2, 3]

    def test_group_ordering(self, display):
        """Groups should appear in defined order: People, Explore, Intervene, Tools, Time."""
        actions = _make_actions([
            ("wait", "Wait"),
            ("map", "View map"),
            ("intervene", "Intervene: X"),
            ("talk", "Talk to Alice"),
            ("observe", "Observe"),
        ])
        grouped = display._group_actions(actions)
        group_names = [name for name, _ in grouped]
        expected_order = ["People", "Explore", "Intervene", "Tools", "Time"]
        # Filter to only groups present
        expected = [g for g in expected_order if g in group_names]
        assert group_names == expected
