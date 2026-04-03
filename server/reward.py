"""
SprintBoard — Per-step reward shaping.

Provides small positive rewards for investigative and corrective actions,
and small negative rewards for errors, destructive actions, and repetition.

Per-step rewards are essential for RL sample efficiency: without them, the
agent receives zero learning signal until the episode-ending grader fires,
making credit assignment across a 15-step episode nearly impossible. These
shaped rewards give the policy gradient meaningful direction on every step.

Three anti-reward-hacking mechanisms prevent degenerate strategies:
  1. Fault-type gating — investigative rewards only fire on relevant fault types.
  2. Target-awareness — planning rewards require the command to reference the
     fault's actual target (from task params).
  3. Deduplication — each reward category fires at most once per episode, so
     repeating a useful command yields nothing.

Rewards are:
- Fault-type-gated: investigation actions only reward on relevant fault types
- Target-aware: commands must reference the right story/developer
- Deduplicated: each reward category fires at most once per episode
- Clamped: cumulative reward stays in [0, 1] (enforced in environment.py)

Design rationale: most RL environments for tool use provide only a sparse
terminal reward. This works for short-horizon tasks but fails for multi-step
sprint planning where the agent must first investigate, then fix, then verify.
Shaped per-step rewards bridge each sub-goal transition without leaking the
grader answer.
"""

import logging
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# ── Fault-type gates for planning commands ─────────────────────────
# Each corrective action only earns reward on relevant fault types.
# Prevents brute-force strategies that cycle through every possible action.

CORRECTIVE_GATES = {
    "ESTIMATE": (
        "unestimated_stories", "full_sprint_planning", "sprint_rescue",
    ),
    "ASSIGN": (
        "developer_overload", "skill_mismatch", "full_sprint_planning",
        "sprint_rescue", "pto_velocity_drop", "dependency_chain_overload",
    ),
    "UNASSIGN": (
        "developer_overload", "skill_mismatch", "sprint_rescue",
        "pto_velocity_drop", "dependency_chain_overload",
    ),
    "ADD_TO_SPRINT": (
        "missing_dependency", "wrong_priority", "velocity_overload",
        "tech_debt_balance", "full_sprint_planning", "sprint_rescue",
    ),
    "REMOVE_FROM_SPRINT": (
        "velocity_overload", "wrong_priority", "tech_debt_balance",
        "sprint_rescue", "pto_velocity_drop",
    ),
    "FLAG_RISK": (
        "scope_creep", "cross_team_dependency", "sprint_rescue",
    ),
    "DECOMPOSE": (
        "epic_decomposition",
    ),
    "SET_PRIORITY": (
        "wrong_priority", "priority_conflict", "sprint_rescue",
    ),
    "FINALIZE_SPRINT": (
        # Finalizing is always valid — it's the termination action
        "unestimated_stories", "developer_overload", "missing_dependency",
        "scope_creep", "wrong_priority", "velocity_overload", "skill_mismatch",
        "epic_decomposition", "priority_conflict", "tech_debt_balance",
        "dependency_chain_overload", "pto_velocity_drop",
        "cross_team_dependency", "sprint_rescue", "full_sprint_planning",
    ),
}

# ── Fault-type gates for investigation commands ────────────────────
# Investigation rewards only fire when the investigation targets the
# actual fault. Checking velocity on a scope_creep task earns 0.

DIAGNOSTIC_FAULT_GATES = {
    "VIEW_TEAM": (
        "developer_overload", "skill_mismatch", "pto_velocity_drop",
        "dependency_chain_overload", "cross_team_dependency",
        "sprint_rescue", "full_sprint_planning",
    ),
    "VIEW_VELOCITY": (
        "velocity_overload", "pto_velocity_drop", "full_sprint_planning",
    ),
    "CHECK_DEPS": (
        "missing_dependency", "dependency_chain_overload",
        "cross_team_dependency", "sprint_rescue", "full_sprint_planning",
    ),
    "VIEW_BUGS": (
        "tech_debt_balance",
    ),
    "VIEW_EPIC": (
        "epic_decomposition",
    ),
}

# Investigation commands that are useful regardless of fault type
UNIVERSAL_DIAGNOSTICS = {"LIST_BACKLOG", "VIEW_SPRINT", "VIEW_STORY"}


def _reward_once(rewarded_set: Optional[Set[str]], category: str, amount: float) -> float:
    """Give reward only if this category hasn't been rewarded yet.

    Deduplication prevents reward farming: running the same diagnostic five
    times earns the same reward as running it once. The rewarded_set persists
    across all steps in an episode, so the agent must explore diverse actions.
    """
    if rewarded_set is not None and category in rewarded_set:
        return 0.0
    if rewarded_set is not None:
        rewarded_set.add(category)
    return amount


def compute_step_reward(
    command: str,
    output: str,
    error: str | None,
    fault_type: str,
    action_history: List[str],
    task_params: dict = None,
    rewarded_set: set = None,
) -> float:
    """Compute reward for a single step.

    Returns a float (can be positive or negative).
    Per-step range approximately [-0.10, +0.10]. The asymmetry is intentional:
    correct actions are rewarded more than bad actions are penalised, biasing
    exploration toward productive commands.

    Cumulative reward is clamped to [0, 1] in environment.py, keeping rewards
    on the same scale as the grader score for straightforward RL loss functions.
    """
    reward = 0.0
    tokens = command.strip().split()
    if not tokens:
        return -0.02

    cmd = tokens[0].upper()
    args_upper = " ".join(tokens[1:]).upper()
    params = task_params or {}

    # ── Positive: investigation commands (fault-type gated) ────────
    if cmd in UNIVERSAL_DIAGNOSTICS:
        reward += _reward_once(rewarded_set, f"diag_{cmd.lower()}", 0.03)

    elif cmd in DIAGNOSTIC_FAULT_GATES:
        if fault_type in DIAGNOSTIC_FAULT_GATES[cmd]:
            reward += _reward_once(rewarded_set, f"diag_{cmd.lower()}", 0.05)
        # No reward for wrong-fault-type diagnostics

    # Target-aware investigation: VIEW_STORY on the right story
    if cmd == "VIEW_STORY" and len(tokens) > 1:
        story_id = tokens[1].upper()
        # Check if this story is relevant to the fault
        target_stories = set()
        for key in ("blocked_story", "missing_dep", "overloaded_dev",
                     "skill_gap_story", "epic_id"):
            val = params.get(key)
            if val and isinstance(val, str):
                target_stories.add(val.upper())
        for key in ("unestimated_story_ids", "sprint_stories",
                     "missing_p0s", "conflicting_stories", "vague_stories"):
            val = params.get(key, [])
            if isinstance(val, list):
                target_stories.update(v.upper() for v in val)

        if story_id in target_stories:
            reward += _reward_once(rewarded_set, f"diag_story_{story_id}", 0.05)

    # ── Positive: corrective actions (fault-type gated) ───────────
    if cmd in CORRECTIVE_GATES:
        if fault_type in CORRECTIVE_GATES[cmd]:
            # Higher reward for primary fix actions
            if cmd == "FINALIZE_SPRINT":
                reward += _reward_once(rewarded_set, "finalize", 0.05)
            elif cmd in ("ESTIMATE", "ASSIGN", "DECOMPOSE"):
                reward += _reward_once(rewarded_set, f"fix_{cmd.lower()}", 0.08)
            else:
                reward += _reward_once(rewarded_set, f"fix_{cmd.lower()}", 0.05)

    # ── Negative: wrong-corrective penalty ─────────────────────────
    # Applying a corrective action for the wrong fault type incurs a small
    # penalty. Discourages brute-force "try every fix" strategies.
    if cmd in CORRECTIVE_GATES and cmd != "FINALIZE_SPRINT":
        if fault_type not in CORRECTIVE_GATES[cmd]:
            reward -= 0.03

    # ── Negative: errors ──────────────────────────────────────────
    if error is not None:
        reward -= 0.05

    # ── Negative: exact duplicate command ─────────────────────────
    # Exact-match repeated commands lose points, preventing degenerate loops.
    if command.strip() in [a.strip() for a in action_history[:-1]]:
        reward -= 0.03

    # ── Negative: empty or trivial commands ───────────────────────
    if cmd in ("HELP",) and len(action_history) > 1:
        reward -= 0.02  # HELP is fine on step 1, wasteful after

    # Round to 4 decimal places to avoid floating-point drift.
    return round(reward, 4)
