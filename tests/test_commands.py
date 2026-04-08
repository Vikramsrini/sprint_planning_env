"""
SprintBoard — Command parsing and board state tests.

Verifies that the command parser correctly routes all commands,
the board state machine works correctly, and error handling is robust.
"""

import pytest
from sprint_planning_env.server.board import ProjectBoard
from sprint_planning_env.server.command_parser import (
    parse_and_execute, is_destructive, get_command_type,
)


class TestCommandParser:
    """Test command parsing and routing."""

    @pytest.fixture
    def board(self):
        b = ProjectBoard()
        b.reset({"sprint_stories": ["101", "103", "105"]})
        return b

    def test_list_backlog(self, board):
        output, error = parse_and_execute("LIST_BACKLOG", board)
        assert error is None
        assert "BACKLOG" in output
        assert "101" in output

    def test_view_story(self, board):
        output, error = parse_and_execute("VIEW_STORY 101", board)
        assert error is None
        assert "User Authentication" in output

    def test_view_story_not_found(self, board):
        output, error = parse_and_execute("VIEW_STORY 999", board)
        assert "not found" in output.lower() or (error and "not found" in error.lower())

    def test_view_team(self, board):
        output, error = parse_and_execute("VIEW_TEAM", board)
        assert error is None
        assert "Alice" in output
        assert "Bob" in output

    def test_view_velocity(self, board):
        output, error = parse_and_execute("VIEW_VELOCITY", board)
        assert error is None
        assert "VELOCITY" in output

    def test_view_sprint(self, board):
        output, error = parse_and_execute("VIEW_SPRINT", board)
        assert error is None
        assert "SPRINT" in output

    def test_check_deps(self, board):
        output, error = parse_and_execute("CHECK_DEPS 105", board)
        assert error is None
        assert "101" in output  # 105 depends on 101

    def test_search_backlog(self, board):
        output, error = parse_and_execute("SEARCH_BACKLOG auth", board)
        assert error is None
        assert "101" in output

    def test_help(self, board):
        output, error = parse_and_execute("HELP", board)
        assert error is None
        assert "Available Commands" in output

    def test_estimate(self, board):
        output, error = parse_and_execute("ESTIMATE 101 8", board)
        assert error is None
        assert "estimated" in output.lower()

    def test_estimate_invalid_points(self, board):
        output, error = parse_and_execute("ESTIMATE 101 7", board)
        # 7 is not Fibonacci
        assert "Fibonacci" in output or "Invalid" in output

    def test_assign(self, board):
        output, error = parse_and_execute("ASSIGN 101 Alice", board)
        assert error is None
        assert "assigned" in output.lower()

    def test_assign_unknown_dev(self, board):
        output, error = parse_and_execute("ASSIGN 101 Zach", board)
        assert "not found" in output.lower()

    def test_add_to_sprint(self, board):
        output, error = parse_and_execute("ADD_TO_SPRINT 102", board)
        assert error is None
        assert "added" in output.lower()

    def test_remove_from_sprint(self, board):
        output, error = parse_and_execute("REMOVE_FROM_SPRINT 101", board)
        assert error is None
        assert "removed" in output.lower()

    def test_flag_risk(self, board):
        output, error = parse_and_execute("FLAG_RISK 101 vague acceptance criteria", board)
        assert error is None
        assert "flagged" in output.lower()

    def test_finalize_sprint(self, board):
        output, error = parse_and_execute("FINALIZE_SPRINT", board)
        assert error is None
        assert "FINALIZED" in output

    def test_unknown_command(self, board):
        output, error = parse_and_execute("YOLO", board)
        assert error is not None
        assert "Unknown" in error

    def test_empty_command(self, board):
        output, error = parse_and_execute("", board)
        assert error is not None


class TestDestructiveDetection:
    """Test destructive command detection."""

    def test_delete_story(self):
        assert is_destructive("DELETE_STORY 101") is True

    def test_clear_sprint(self):
        assert is_destructive("CLEAR_SPRINT") is True

    def test_normal_command(self):
        assert is_destructive("VIEW_TEAM") is False

    def test_assign_not_destructive(self):
        assert is_destructive("ASSIGN 101 Alice") is False


class TestCommandTypes:
    """Test command type classification."""

    def test_investigation(self):
        assert get_command_type("LIST_BACKLOG") == "investigation"
        assert get_command_type("VIEW_STORY 101") == "investigation"
        assert get_command_type("CHECK_DEPS 105") == "investigation"
        assert get_command_type("VIEW_TEAM") == "investigation"

    def test_planning(self):
        assert get_command_type("ESTIMATE 101 5") == "planning"
        assert get_command_type("ASSIGN 101 Alice") == "planning"
        assert get_command_type("ADD_TO_SPRINT 102") == "planning"

    def test_destructive(self):
        assert get_command_type("DELETE_STORY 101") == "destructive"
        assert get_command_type("CLEAR_SPRINT") == "destructive"

    def test_unknown(self):
        assert get_command_type("YOLO") == "unknown"


class TestBoardState:
    """Test board state machine correctness."""

    def test_estimate_updates_state(self):
        board = ProjectBoard()
        board.reset({"sprint_stories": ["103"], "unestimated_story_ids": ["103"]})
        assert board.estimates.get("103") is None
        board.estimate("103", 5)
        assert board.estimates.get("103") == 5

    def test_assign_updates_state(self):
        board = ProjectBoard()
        board.reset({"sprint_stories": ["101"]})
        board.assign("101", "Alice")
        assert board.assignments.get("101") == "Alice"

    def test_unassign(self):
        board = ProjectBoard()
        board.reset({"sprint_stories": ["101"], "initial_assignments": {"101": "Alice"}})
        board.unassign("101")
        assert "101" not in board.assignments

    def test_add_remove_sprint(self):
        board = ProjectBoard()
        board.reset({"sprint_stories": ["101"]})
        assert "101" in board.sprint_stories
        board.remove_from_sprint("101")
        assert "101" not in board.sprint_stories
        board.add_to_sprint("101")
        assert "101" in board.sprint_stories

    def test_flag_risk(self):
        board = ProjectBoard()
        board.reset({"sprint_stories": ["114"]})
        board.flag_risk("114", "vague acceptance criteria")
        assert "114" in board.risk_flags

    def test_metrics(self):
        board = ProjectBoard()
        board.reset({
            "sprint_stories": ["101", "103"],
            "initial_assignments": {"101": "Alice"},
        })
        metrics = board.get_metrics()
        assert metrics["sprint_stories"] == 2
        assert metrics["unassigned_count"] == 1  # 103 unassigned

    def test_circular_dependency_detection(self):
        board = ProjectBoard()
        board.reset({
            "sprint_stories": ["105", "113"],
            "circular_deps": {"105": "113", "113": "105"},
        })
        output = board.check_deps("105")
        assert "CIRCULAR" in output

    def test_finalize(self):
        board = ProjectBoard()
        board.reset({"sprint_stories": ["101"]})
        assert not board.is_finalized
        board.finalize_sprint()
        assert board.is_finalized
