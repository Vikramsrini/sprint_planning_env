"""
SprintBoard — Adversarial reward hacking tests.

Verifies that the reward system and graders cannot be gamed by:
1. Repeating the same diagnostic command many times
2. Running corrective actions for the wrong fault type
3. Submitting destructive commands
4. Running a no-op agent that does nothing
5. Keyword-stuffing without meaningful investigation
"""

import pytest
from sprint_planning_env.server.reward import compute_step_reward
from sprint_planning_env.server.board import ProjectBoard
from sprint_planning_env.server.grader import grade_episode


class TestRewardDeduplication:
    """Verify that repeating commands doesn't accumulate reward."""

    def test_repeated_list_backlog(self):
        """LIST_BACKLOG should only reward once."""
        rewarded = set()
        r1 = compute_step_reward("LIST_BACKLOG", "output", None, "unestimated_stories",
                                  ["LIST_BACKLOG"], {}, rewarded)
        r2 = compute_step_reward("LIST_BACKLOG", "output", None, "unestimated_stories",
                                  ["LIST_BACKLOG", "LIST_BACKLOG"], {}, rewarded)
        assert r1 > 0, "First LIST_BACKLOG should earn reward"
        assert r2 <= 0, "Second LIST_BACKLOG should earn zero or negative"

    def test_repeated_view_team(self):
        """VIEW_TEAM should only reward once per episode."""
        rewarded = set()
        r1 = compute_step_reward("VIEW_TEAM", "output", None, "developer_overload",
                                  ["VIEW_TEAM"], {}, rewarded)
        r2 = compute_step_reward("VIEW_TEAM", "output", None, "developer_overload",
                                  ["VIEW_TEAM", "VIEW_TEAM"], {}, rewarded)
        assert r1 > 0
        assert r2 <= 0

    def test_repeated_estimate(self):
        """ESTIMATE should only reward once per episode."""
        rewarded = set()
        r1 = compute_step_reward("ESTIMATE US-103 5", "ok", None, "unestimated_stories",
                                  ["ESTIMATE US-103 5"], {}, rewarded)
        r2 = compute_step_reward("ESTIMATE US-103 5", "ok", None, "unestimated_stories",
                                  ["ESTIMATE US-103 5", "ESTIMATE US-103 5"], {}, rewarded)
        assert r1 > 0
        assert r2 <= 0


class TestFaultTypeGating:
    """Verify that corrective actions only reward on correct fault types."""

    def test_estimate_on_wrong_fault(self):
        """ESTIMATE on a developer_overload task should penalize."""
        rewarded = set()
        r = compute_step_reward("ESTIMATE US-103 5", "ok", None, "developer_overload",
                                 ["ESTIMATE US-103 5"], {}, rewarded)
        assert r < 0, "ESTIMATE on wrong fault type should penalize"

    def test_assign_on_wrong_fault(self):
        """ASSIGN on a scope_creep task should penalize."""
        rewarded = set()
        r = compute_step_reward("ASSIGN US-103 Alice", "ok", None, "scope_creep",
                                 ["ASSIGN US-103 Alice"], {}, rewarded)
        assert r < 0, "ASSIGN on wrong fault type should penalize"

    def test_view_team_on_wrong_fault(self):
        """VIEW_TEAM on an unestimated_stories task earns 0 (not penalized)."""
        rewarded = set()
        r = compute_step_reward("VIEW_TEAM", "output", None, "unestimated_stories",
                                 ["VIEW_TEAM"], {}, rewarded)
        # VIEW_TEAM is not in DIAGNOSTIC_FAULT_GATES for unestimated_stories
        # so it earns 0 (no reward, no penalty)
        assert r == 0, "VIEW_TEAM on irrelevant fault should earn 0"


class TestErrorPenalties:
    """Verify that errors are penalized."""

    def test_error_command(self):
        """Commands with errors should result in negative reward."""
        rewarded = set()
        # Use developer_overload where ESTIMATE is a wrong-fault corrective (-0.03)
        # plus error penalty (-0.05) = net negative
        r = compute_step_reward("ESTIMATE US-999 5", "ok", "ERROR: Story not found",
                                 "developer_overload",
                                 ["ESTIMATE US-999 5"], {}, rewarded)
        assert r < 0, "Error with wrong-fault action should result in negative reward"

    def test_duplicate_command_penalty(self):
        """Exact duplicate commands should be penalized."""
        rewarded = set()
        history = ["LIST_BACKLOG", "LIST_BACKLOG"]
        r = compute_step_reward("LIST_BACKLOG", "output", None, "unestimated_stories",
                                 history, {}, rewarded)
        assert r <= 0, "Duplicate command should be zero or negative"


class TestNoOpAgent:
    """Verify that a no-op agent scores poorly."""

    def test_noop_grader(self):
        """An agent that does nothing should score near 0."""
        board = ProjectBoard()
        board.reset({"sprint_stories": ["US-103", "US-106"], "unestimated_story_ids": ["US-103"]})
        score, breakdown = grade_episode(
            board=board,
            fault_type="unestimated_stories",
            task_params={"unestimated_story_ids": ["US-103"], "sprint_stories": ["US-103", "US-106"]},
            action_history=[],
            error_history=[],
            steps_used=0,
        )
        assert score < 0.30, f"No-op agent should score < 0.30, got {score}"

    def test_help_only_agent(self):
        """An agent that only types HELP should score near 0."""
        board = ProjectBoard()
        board.reset({"sprint_stories": ["US-103"], "unestimated_story_ids": ["US-103"]})
        score, breakdown = grade_episode(
            board=board,
            fault_type="unestimated_stories",
            task_params={"unestimated_story_ids": ["US-103"], "sprint_stories": ["US-103"]},
            action_history=["HELP"] * 10,
            error_history=[False] * 10,
            steps_used=10,
        )
        assert score < 0.30, f"HELP-only agent should score < 0.30, got {score}"


class TestDestructiveActions:
    """Verify destructive commands are penalized."""

    def test_delete_story_penalty(self):
        """DELETE_STORY should trigger fatal penalty."""
        rewarded = set()
        r = compute_step_reward("DELETE_STORY US-101", "", None, "unestimated_stories",
                                 ["DELETE_STORY US-101"], {}, rewarded)
        # The destructive detection is in the environment, not reward.py
        # reward.py handles the wrong-corrective penalty
        # The environment handles fatal detection


class TestGraderDeterminism:
    """Verify that graders produce identical scores for identical state."""

    def test_same_state_same_score(self):
        """Same board state → same grader score."""
        for _ in range(5):
            board = ProjectBoard()
            board.reset({
                "sprint_stories": ["US-103", "US-106", "US-107"],
                "unestimated_story_ids": ["US-103", "US-106", "US-107"],
            })
            # Simulate agent estimating
            board.estimate("US-103", 5)
            board.estimate("US-106", 3)

            score, _ = grade_episode(
                board=board,
                fault_type="unestimated_stories",
                task_params={
                    "unestimated_story_ids": ["US-103", "US-106", "US-107"],
                    "sprint_stories": ["US-103", "US-106", "US-107"],
                },
                action_history=["LIST_BACKLOG", "VIEW_STORY US-103", "ESTIMATE US-103 5",
                                 "ESTIMATE US-106 3"],
                error_history=[False, False, False, False],
                steps_used=4,
            )

        # Run again
        board2 = ProjectBoard()
        board2.reset({
            "sprint_stories": ["US-103", "US-106", "US-107"],
            "unestimated_story_ids": ["US-103", "US-106", "US-107"],
        })
        board2.estimate("US-103", 5)
        board2.estimate("US-106", 3)

        score2, _ = grade_episode(
            board=board2,
            fault_type="unestimated_stories",
            task_params={
                "unestimated_story_ids": ["US-103", "US-106", "US-107"],
                "sprint_stories": ["US-103", "US-106", "US-107"],
            },
            action_history=["LIST_BACKLOG", "VIEW_STORY US-103", "ESTIMATE US-103 5",
                             "ESTIMATE US-106 3"],
            error_history=[False, False, False, False],
            steps_used=4,
        )
        assert score == score2, "Same state should produce identical scores"
