"""
SprintBoard — Core Environment class.

Models the workflow of a Scrum Master resolving sprint planning issues.
Each episode: receive scenario alert, investigate board, fix plan, verify.

Implements the OpenEnv Environment interface: reset(), step(), state.
Orchestrates fault injection via board state, command execution, reward
shaping, and deterministic grading.

Architecture mirrors SQLab: where SQLab uses a live PostgreSQL database,
we use a simulated ProjectBoard. Where SQLab executes SQL, we execute
planning commands. The grading, reward shaping, and episode lifecycle
are structurally identical.
"""

import logging
import random
import uuid
from typing import Optional, Any

from openenv.core.env_server.interfaces import Environment

from sprint_planning_env.models import SprintAction, SprintObservation, SprintState
from sprint_planning_env.server.board import ProjectBoard
from sprint_planning_env.server.tasks import TASK_REGISTRY, get_task, list_task_ids
from sprint_planning_env.server.command_parser import parse_and_execute, is_destructive
from sprint_planning_env.server.reward import compute_step_reward
from sprint_planning_env.server.grader import grade_episode

logger = logging.getLogger(__name__)

MAX_STEPS = 15

# Global destructive patterns — always fatal regardless of task
GLOBAL_FATAL_PATTERNS = [
    "DELETE_STORY",
    "CLEAR_SPRINT",
    "REMOVE_DEVELOPER",
    "DROP_ALL",
    "RESET_BOARD",
]


class SprintBoardEnvironment(Environment[SprintAction, SprintObservation, SprintState]):
    """Sprint planning training environment.

    Each episode:
    1. reset() picks a task, injects a board fault, returns initial observation
    2. step() executes agent command, computes reward, checks resolution
    3. state property returns current episode metadata
    """

    # Class-level storage for the /grader endpoint
    last_grader_result: Optional[dict] = None

    def __init__(self):
        super().__init__()

        # Project board (our equivalent of SQLab's database)
        self._board = ProjectBoard()

        # Episode state
        self._episode_id: str = ""
        self._task_id: str = ""
        self._task: dict = {}
        self._fault_type: str = ""
        self._step_count: int = 0
        self._done: bool = True
        self._is_resolved: bool = False
        self._cumulative_reward: float = 0.0
        self._grader_score: Optional[float] = None
        self._action_history: list[str] = []
        self._error_history: list[bool] = []
        self._alert: str = ""
        self._seed: Optional[int] = None
        self._rewarded_set: set = set()  # dedup for per-step rewards

    # ── OpenEnv interface ────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SprintObservation:
        """Start a new episode.

        Resets the project board and injects a planning fault per the
        selected task configuration.

        Args:
            seed: Random seed for reproducibility.
            episode_id: Optional episode ID (auto-generated if not given).
            **kwargs: May include 'task_id' to select a specific task.
        """
        # Seed
        self._seed = seed
        if seed is not None:
            random.seed(seed)

        # Pick task — 15 tasks span 3 difficulty tiers
        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = random.choice(list_task_ids())
        self._task_id = task_id
        self._task = get_task(task_id)
        self._fault_type = self._task["fault_type"]
        self._alert = self._task["alert"]

        # Episode bookkeeping
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._is_resolved = False
        self._cumulative_reward = 0.0
        self._grader_score = None
        self._action_history = []
        self._error_history = []
        self._rewarded_set = set()

        # Initialize board with task-specific fault state
        self._board.reset(self._task["params"])

        # Build initial observation with board context
        board_hint = (
            "SprintBoard: Agile Sprint Planning Environment\n"
            "Team: 5 developers (Alice, Bob, Charlie, Diana, Eve)\n"
            "Backlog: 20 user stories with dependencies, estimates, priorities\n"
            "Velocity: ~34 pts/sprint (last 5 sprints)\n"
            "Use planning commands to investigate and fix the sprint plan.\n"
            "Type HELP for available commands."
        )

        metrics = self._board.get_metrics()

        return SprintObservation(
            command_output=board_hint,
            error=None,
            alert=self._alert,
            metrics=metrics,
            step_number=0,
            max_steps=MAX_STEPS,
            done=False,
            reward=0.0,
            metadata={"task_id": self._task_id, "difficulty": self._task["difficulty"]},
        )

    def step(
        self,
        action: SprintAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SprintObservation:
        """Execute one agent action (planning command) and return observation."""
        if self._done:
            return self._terminal_observation(
                "Episode is already done. Call reset() to start a new one."
            )

        self._step_count += 1
        command = action.command.strip()
        self._action_history.append(command)

        # ── Fatal action detection (task-aware) ──
        cmd_upper = command.upper().split()[0] if command.split() else ""
        task_fatal = self._task.get("fatal_patterns", [])
        task_allowed = self._task.get("allowed_dangerous", [])

        is_fatal = False
        for pattern in GLOBAL_FATAL_PATTERNS + task_fatal:
            if pattern in command.upper():
                is_fatal = True
                break
        for allowed in task_allowed:
            if allowed in command.upper():
                is_fatal = False
                break

        if is_fatal:
            self._done = True
            self._is_resolved = False
            self._cumulative_reward -= 0.5
            self._grader_score = self._run_grader()
            metrics = self._board.get_metrics()
            self._error_history.append(True)
            return SprintObservation(
                command_output=f"Command attempted: {command[:80]}",
                error="FATAL: Destructive action detected. Episode terminated with penalty.",
                alert=self._alert,
                metrics=metrics,
                step_number=self._step_count,
                max_steps=MAX_STEPS,
                done=True,
                reward=-0.5,
                metadata={
                    "task_id": self._task_id,
                    "difficulty": self._task["difficulty"],
                    "is_resolved": False,
                    "cumulative_reward": round(self._cumulative_reward, 4),
                    "grader_score": self._grader_score,
                    "fatal_action": True,
                },
            )

        # Execute command against board
        output, error = parse_and_execute(command, self._board)
        self._error_history.append(error is not None)

        # Compute per-step reward
        step_reward = compute_step_reward(
            command=command,
            output=output,
            error=error,
            fault_type=self._fault_type,
            action_history=self._action_history,
            task_params=self._task.get("params", {}),
            rewarded_set=self._rewarded_set,
        )
        self._cumulative_reward += step_reward
        self._cumulative_reward = max(0.0, min(1.0, self._cumulative_reward))

        # Check resolution by verifying board state
        self._is_resolved = self._check_resolved()

        # Check done conditions
        done = False
        if self._step_count >= MAX_STEPS:
            done = True
        if self._board.is_finalized:
            done = True
        self._done = done

        # Collect metrics
        metrics = self._board.get_metrics()

        # If done, compute final grader score
        if done:
            self._grader_score = self._run_grader()
            if self._grader_score is not None:
                completion_bonus = self._grader_score * 0.5
                step_reward += completion_bonus
                self._cumulative_reward += completion_bonus

        return SprintObservation(
            command_output=output,
            error=error,
            alert=self._alert,
            metrics=metrics,
            step_number=self._step_count,
            max_steps=MAX_STEPS,
            done=done,
            reward=step_reward,
            metadata={
                "task_id": self._task_id,
                "difficulty": self._task["difficulty"],
                "is_resolved": self._is_resolved,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "grader_score": self._grader_score,
            },
        )

    @property
    def state(self) -> SprintState:
        """Return current episode state.

        Episode metadata including cumulative_reward, grader_score, and
        difficulty tier. Useful for curriculum learning.
        """
        return SprintState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            task_name=self._task.get("name", ""),
            difficulty=self._task.get("difficulty", ""),
            fault_type=self._fault_type,
            is_resolved=self._is_resolved,
            cumulative_reward=round(self._cumulative_reward, 4),
            grader_score=self._grader_score,
        )

    def close(self) -> None:
        """Clean up resources."""
        pass  # No external resources to clean up (no DB connections)

    @property
    def board(self) -> ProjectBoard:
        """Public accessor for the project board (for direct-state agents)."""
        return self._board

    # ── Internal helpers ─────────────────────────────────────────

    def _check_resolved(self) -> bool:
        """Check if the current fault has been resolved by examining board state.

        Resolution is verified by checking actual board state, not by
        pattern-matching commands. This makes grading robust against
        reward hacking.
        """
        metrics = self._board.get_metrics()
        ft = self._fault_type
        params = self._task.get("params", {})

        if ft == "unestimated_stories":
            return metrics["unestimated_count"] == 0

        if ft == "developer_overload":
            return len(metrics["overloaded_developers"]) == 0

        if ft == "missing_dependency":
            missing = params.get("missing_dep", "")
            return missing in self._board.sprint_stories

        if ft == "scope_creep":
            vague = params.get("vague_story_ids_ground_truth", [])
            return all(s in self._board.risk_flags for s in vague)

        if ft == "wrong_priority":
            p0s = params.get("missing_p0s", [])
            return all(s in self._board.sprint_stories for s in p0s)

        if ft == "velocity_overload":
            avg_vel = params.get("velocity_avg", 34)
            return metrics["total_points"] <= avg_vel * 1.1

        if ft == "skill_mismatch":
            correct_map = params.get("correct_skill_assignments", {})
            for sid, valid_devs in correct_map.items():
                if self._board.assignments.get(sid) not in valid_devs:
                    return False
            return True

        if ft == "epic_decomposition":
            epic_id = params.get("epic_id", "")
            return epic_id in self._board.decomposed_epics

        if ft == "priority_conflict":
            conflicting = params.get("conflicting_stories", [])
            return any(s in self._board.sprint_stories for s in conflicting)

        if ft == "tech_debt_balance":
            return len(self._board.bugs_added) >= 2

        if ft == "dependency_chain_overload":
            issues = self._board._audit_sprint()
            circular = [i for i in issues if "circular" in i.lower()]
            overload = [i for i in issues if "overloaded" in i.lower()]
            return not circular and not overload

        if ft == "pto_velocity_drop":
            pto_dev = params.get("pto_developer", "")
            return self._board._get_dev_load(pto_dev) == 0

        if ft == "cross_team_dependency":
            return self._board.is_finalized

        if ft == "sprint_rescue":
            return self._board.is_finalized

        if ft == "full_sprint_planning":
            return self._board.is_finalized

        return False

    def _run_grader(self) -> float:
        """Run the deterministic grader and store result."""
        try:
            score, breakdown = grade_episode(
                board=self._board,
                fault_type=self._fault_type,
                task_params=self._task.get("params", {}),
                action_history=self._action_history,
                error_history=self._error_history,
                steps_used=self._step_count,
            )
            # Store for /grader endpoint
            SprintBoardEnvironment.last_grader_result = {
                "task_id": self._task_id,
                "episode_id": self._episode_id,
                "score": round(score, 4),
                "breakdown": breakdown,
                "steps_used": self._step_count,
                "is_resolved": self._is_resolved,
            }
            logger.info(
                "Graded episode %s: score=%.3f breakdown=%s",
                self._episode_id, score, breakdown,
            )
            return round(score, 4)
        except Exception as e:
            logger.error("Grader error: %s", e)
            return 0.0

    def _terminal_observation(self, message: str) -> SprintObservation:
        """Return an observation for a terminal/error state."""
        return SprintObservation(
            command_output=message,
            error=None,
            alert=self._alert,
            metrics={},
            step_number=self._step_count,
            max_steps=MAX_STEPS,
            done=True,
            reward=0.0,
            metadata={
                "task_id": self._task_id,
                "grader_score": self._grader_score,
            },
        )
