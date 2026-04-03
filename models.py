"""
SprintBoard — Pydantic models for Action, Observation, and State.

These define the typed interface between the agent and the environment.

The action space is intentionally open-ended: agents submit free-form planning
commands (ASSIGN, ESTIMATE, CHECK_DEPS, VIEW_VELOCITY, etc.), mirroring how
a real Scrum Master interacts with a project board during sprint planning.
This contrasts with constrained action spaces — the agent must compose valid
commands from scratch, making the problem closer to real sprint planning
than to a multiple-choice exercise.

The environment ships 15 sprint-planning tasks across three difficulty tiers
(easy / medium / hard), each scored by a deterministic three-section grader
(investigation 30% | planning quality 50% | process 20%). Observations
surface the same signals a human SM would see: a planning scenario alert,
live board metrics, and formatted command output.

Why this matters for the RL/agent community: sprint planning is a high-value,
under-served domain — no existing RL benchmark exercises project management
decision-making with deterministic grading. SprintBoard fills that gap with
a reproducible, Docker-containerised environment that any researcher can
spin up in minutes for agent evaluation or GRPO fine-tuning.
"""

from typing import Optional, Dict, Any, List
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class SprintAction(Action):
    """Agent submits a planning command to investigate or modify the sprint plan.

    The unbounded string action space is a deliberate design choice: frontier
    models can generate valid planning commands, so restricting them to a
    dropdown of pre-authored actions would trivialise the hard tasks and
    remove the compositional reasoning challenge.
    """
    # Open action space: any valid planning command is accepted,
    # from VIEW_VELOCITY to ASSIGN to FINALIZE_SPRINT.
    # This matches real Scrum Master workflow — no artificial action discretisation.
    command: str = Field(
        ...,
        min_length=1,
        description="Planning command to execute against the project board"
    )


class SprintObservation(Observation):
    """What the agent sees after each action.

    Inherits from Observation which provides:
        - done: bool (whether episode has terminated)
        - reward: Optional[float] (reward signal from last action)
        - metadata: Dict[str, Any]
    """
    # Formatted output from the last command — tables, lists, dependency graphs.
    # Structured for LLM readability with clear formatting.
    command_output: str = Field(
        default="",
        description="Formatted output from the planning command"
    )
    # Command errors are surfaced clearly so agents can learn from mistakes.
    error: Optional[str] = Field(
        default=None,
        description="Error message if the planning command failed"
    )
    # Persistent scenario alert mirrors a real sprint planning scenario —
    # the agent sees it on every step, like a meeting facilitator's agenda.
    alert: str = Field(
        default="",
        description="The sprint planning scenario description"
    )
    # Real-time board metrics matching what a PM dashboard would show:
    # total points, assigned vs unassigned, capacity usage, dependency count.
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Board health metrics (total_points, assigned_count, capacity_used, etc.)"
    )
    # Step budget creates a tight episode horizon (15 steps), forcing efficient
    # planning. Human SMs typically plan a sprint in 10-15 decisions; 15 steps
    # gives enough room for investigation while penalising aimless wandering.
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=15, description="Maximum steps allowed per episode")


class SprintState(State):
    """Episode metadata exposed to training harnesses and curriculum schedulers.

    Inherits from State which provides:
        - episode_id: Optional[str]
        - step_count: int

    cumulative_reward and grader_score are surfaced here so RL training loops
    (e.g. TRL's GRPO) can build curriculum strategies — for instance, promoting
    tasks where the agent consistently scores below 0.5 into more frequent
    sampling.
    """
    task_id: str = Field(default="", description="Identifier for the current task")
    task_name: str = Field(default="", description="Human-readable task name")
    # Three-tier difficulty enables curriculum learning: start on easy single-fault
    # tasks, graduate to hard compound faults that require multi-step reasoning.
    difficulty: str = Field(default="", description="Task difficulty: easy, medium, hard")
    fault_type: str = Field(default="", description="Type of planning fault injected")
    is_resolved: bool = Field(default=False, description="Whether the fault has been resolved")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated this episode")
    grader_score: Optional[float] = Field(
        default=None,
        description="Final grader score (0.0-1.0), set at end of episode"
    )
