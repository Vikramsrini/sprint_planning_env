"""
SprintBoard — Deterministic graders for all 15 sprint planning tasks.

All 15 graders are fully deterministic — no LLM judge, no stochastic sampling.
Score reproducibility is critical for RL reward signal stability: given the same
board state and action history, a grader will always return the same score.

Structure: every grader scores across three sections:
  Investigation  (0.3) = Did the agent look at the right data?
  Planning       (0.5) = Is the resulting sprint plan valid and good?
  Process        (0.2) = Did the agent avoid destructive actions?

The 3-section structure reflects real sprint planning evaluation: understanding
the problem matters as much as fixing it. This mirrors how PMs are assessed
in retrospectives.

Planning scores check actual board state (assignments, estimates, dependencies),
not whether the agent typed the right keywords. This prevents reward hacking.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

from sprint_planning_env.server.board import ProjectBoard
from sprint_planning_env.server.tasks import STORY_POOL, TEAM_MEMBERS, BUG_BACKLOG, EPIC_POOL

logger = logging.getLogger(__name__)


def _history_contains(action_history: List[str], *keywords: str) -> bool:
    """Check if any command in history contains ALL the given keywords."""
    for cmd in action_history:
        upper = cmd.upper()
        if all(kw.upper() in upper for kw in keywords):
            return True
    return False


def _history_contains_any(action_history: List[str], *keywords: str) -> bool:
    """Check if any command in history contains ANY of the given keywords."""
    for cmd in action_history:
        upper = cmd.upper()
        if any(kw.upper() in upper for kw in keywords):
            return True
    return False


def _has_destructive(history: List[str]) -> bool:
    """Check for destructive commands in history."""
    return _history_contains_any(history, "DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER")


def _error_rate(error_history: List[bool]) -> float:
    """Fraction of commands that errored."""
    if not error_history:
        return 0.0
    return sum(error_history) / len(error_history)


def _efficiency_penalty(steps_used: int, threshold: int) -> float:
    """Multiplier on planning score. At/under threshold = 1.0.
    Each step over: -0.05. Minimum 0.5."""
    if steps_used <= threshold:
        return 1.0
    return max(0.5, 1.0 - (steps_used - threshold) * 0.05)


STEP_THRESHOLDS = {
    "unestimated_stories": 8,
    "developer_overload": 8,
    "missing_dependency": 7,
    "scope_creep": 8,
    "wrong_priority": 8,
    "velocity_overload": 10,
    "skill_mismatch": 10,
    "epic_decomposition": 8,
    "priority_conflict": 10,
    "tech_debt_balance": 10,
    "dependency_chain_overload": 12,
    "pto_velocity_drop": 12,
    "cross_team_dependency": 12,
    "sprint_rescue": 13,
    "full_sprint_planning": 14,
}


# ══════════════════════════════════════════════════════════════════
# EASY TASK GRADERS (tasks 1–5)
# ══════════════════════════════════════════════════════════════════

def _grade_unestimated_stories(board: ProjectBoard, params: dict, history: List[str],
                                error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Grade task 1: Were all unestimated stories estimated correctly?"""
    breakdown = {}
    score = 0.0
    target_ids = params.get("unestimated_story_ids", [])

    # Investigation (0.3)
    if _history_contains_any(history, "LIST_BACKLOG"):
        breakdown["inv_listed_backlog"] = 0.10
        score += 0.10
    for sid in target_ids:
        if _history_contains(history, "VIEW_STORY", sid):
            breakdown[f"inv_viewed_{sid}"] = 0.07
            score += 0.07
            break
    if _history_contains_any(history, "VIEW_VELOCITY"):
        breakdown["inv_checked_velocity"] = 0.06
        score += 0.06

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["unestimated_stories"])
    plan_score = 0.0

    estimated_count = 0
    reasonable_count = 0
    for sid in target_ids:
        pts = board.estimates.get(sid)
        if pts is not None:
            estimated_count += 1
            actual = STORY_POOL.get(sid, {}).get("points", 5)
            if abs(pts - actual) <= 3:
                reasonable_count += 1

    if estimated_count == len(target_ids):
        plan_score += 0.25
        breakdown["plan_all_estimated"] = 0.25
    elif estimated_count > 0:
        partial = 0.25 * (estimated_count / len(target_ids))
        plan_score += partial
        breakdown["plan_partially_estimated"] = round(partial, 3)

    if reasonable_count == len(target_ids):
        plan_score += 0.25
        breakdown["plan_reasonable_estimates"] = 0.25
    elif reasonable_count > 0:
        partial = 0.25 * (reasonable_count / len(target_ids))
        plan_score += partial
        breakdown["plan_some_reasonable"] = round(partial, 3)

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_developer_overload(board: ProjectBoard, params: dict, history: List[str],
                               error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Grade task 2: Was the overloaded developer's workload fixed?"""
    breakdown = {}
    score = 0.0
    overloaded_dev = params.get("overloaded_dev", "Alice")

    # Investigation (0.3)
    if _history_contains_any(history, "VIEW_TEAM"):
        breakdown["inv_checked_team"] = 0.15
        score += 0.15
    if _history_contains_any(history, "VIEW_SPRINT", "LIST_BACKLOG"):
        breakdown["inv_checked_sprint"] = 0.15
        score += 0.15

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["developer_overload"])
    plan_score = 0.0

    # Check if overloaded dev is now within capacity
    dev_load = board._get_dev_load(overloaded_dev)
    dev_cap = TEAM_MEMBERS[overloaded_dev]["capacity"]
    if dev_load <= dev_cap:
        plan_score += 0.30
        breakdown["plan_overload_fixed"] = 0.30
    elif dev_load < params.get("overloaded_points", 24):
        plan_score += 0.15
        breakdown["plan_overload_reduced"] = 0.15

    # Check all stories still assigned
    sprint_stories = params.get("sprint_stories", [])
    assigned = sum(1 for s in sprint_stories if s in board.assignments)
    if assigned == len(sprint_stories):
        plan_score += 0.20
        breakdown["plan_all_assigned"] = 0.20
    elif assigned > 0:
        partial = 0.20 * (assigned / len(sprint_stories))
        plan_score += partial
        breakdown["plan_partially_assigned"] = round(partial, 3)

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_missing_dependency(board: ProjectBoard, params: dict, history: List[str],
                               error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Grade task 3: Was the missing dependency added to the sprint?"""
    breakdown = {}
    score = 0.0
    blocked = params.get("blocked_story", "105")
    missing = params.get("missing_dep", "101")

    # Investigation (0.3)
    if _history_contains(history, "CHECK_DEPS", blocked):
        breakdown["inv_checked_deps"] = 0.15
        score += 0.15
    if _history_contains_any(history, "VIEW_STORY") and _history_contains_any(history, blocked):
        breakdown["inv_viewed_blocked"] = 0.10
        score += 0.10
    if _history_contains_any(history, "LIST_BACKLOG", "VIEW_SPRINT"):
        breakdown["inv_checked_sprint"] = 0.05
        score += 0.05

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["missing_dependency"])
    plan_score = 0.0

    if missing in board.sprint_stories:
        plan_score += 0.35
        breakdown["plan_dep_added"] = 0.35
    if blocked in board.sprint_stories:
        plan_score += 0.15
        breakdown["plan_blocked_still_in"] = 0.15

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_scope_creep(board: ProjectBoard, params: dict, history: List[str],
                        error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Grade task 4: Were vague stories flagged?"""
    breakdown = {}
    score = 0.0
    vague_ids = params.get("vague_story_ids_ground_truth", [])

    # Investigation (0.3)
    viewed_stories = sum(1 for sid in params.get("sprint_stories", [])
                         if _history_contains(history, "VIEW_STORY", sid))
    if viewed_stories >= 2:
        breakdown["inv_viewed_stories"] = 0.15
        score += 0.15
    if _history_contains_any(history, "LIST_BACKLOG", "VIEW_SPRINT"):
        breakdown["inv_checked_sprint"] = 0.15
        score += 0.15

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["scope_creep"])
    plan_score = 0.0

    flagged_correct = sum(1 for sid in vague_ids if sid in board.risk_flags)
    if flagged_correct == len(vague_ids):
        plan_score += 0.35
        breakdown["plan_all_flagged"] = 0.35
    elif flagged_correct > 0:
        partial = 0.35 * (flagged_correct / len(vague_ids))
        plan_score += partial
        breakdown["plan_some_flagged"] = round(partial, 3)

    # Penalty for false positives (flagging non-vague stories)
    false_flags = sum(1 for sid in board.risk_flags if sid not in vague_ids)
    if false_flags == 0:
        plan_score += 0.15
        breakdown["plan_no_false_flags"] = 0.15

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_wrong_priority(board: ProjectBoard, params: dict, history: List[str],
                           error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Grade task 5: Were P0 stories added and low-priority removed?"""
    breakdown = {}
    score = 0.0
    missing_p0s = params.get("missing_p0s", [])
    low_pri = params.get("low_priority_in_sprint", [])

    # Investigation (0.3)
    if _history_contains_any(history, "LIST_BACKLOG"):
        breakdown["inv_listed_backlog"] = 0.15
        score += 0.15
    p0_viewed = sum(1 for sid in missing_p0s if _history_contains(history, "VIEW_STORY", sid))
    if p0_viewed > 0:
        breakdown["inv_viewed_p0s"] = 0.15
        score += 0.15

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["wrong_priority"])
    plan_score = 0.0

    p0s_added = sum(1 for sid in missing_p0s if sid in board.sprint_stories)
    if p0s_added == len(missing_p0s):
        plan_score += 0.30
        breakdown["plan_p0s_added"] = 0.30
    elif p0s_added > 0:
        plan_score += 0.15
        breakdown["plan_some_p0s_added"] = 0.15

    low_removed = sum(1 for sid in low_pri if sid not in board.sprint_stories)
    if low_removed > 0:
        plan_score += 0.20
        breakdown["plan_low_pri_removed"] = 0.20

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


# ══════════════════════════════════════════════════════════════════
# MEDIUM TASK GRADERS (tasks 6–10)
# ══════════════════════════════════════════════════════════════════

def _grade_velocity_overload(board: ProjectBoard, params: dict, history: List[str],
                              error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0
    velocity_avg = params.get("velocity_avg", 34)

    # Investigation (0.3)
    if _history_contains_any(history, "VIEW_VELOCITY"):
        breakdown["inv_checked_velocity"] = 0.15
        score += 0.15
    if _history_contains_any(history, "LIST_BACKLOG", "VIEW_SPRINT"):
        breakdown["inv_checked_sprint"] = 0.15
        score += 0.15

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["velocity_overload"])
    plan_score = 0.0

    total_pts = sum(board.estimates.get(s, 0) or 0 for s in board.sprint_stories)
    if total_pts <= velocity_avg * 1.1:
        plan_score += 0.30
        breakdown["plan_within_velocity"] = 0.30
    elif total_pts <= velocity_avg * 1.3:
        plan_score += 0.15
        breakdown["plan_close_to_velocity"] = 0.15

    # Check P0s are still in sprint
    p0_in_sprint = all(
        sid in board.sprint_stories
        for sid, story in STORY_POOL.items()
        if story.get("priority") == "P0" and sid in params.get("sprint_stories", [])
    )
    if p0_in_sprint:
        plan_score += 0.20
        breakdown["plan_p0s_preserved"] = 0.20

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_skill_mismatch(board: ProjectBoard, params: dict, history: List[str],
                           error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0
    correct_map = params.get("correct_skill_assignments", {})

    # Investigation (0.3)
    if _history_contains_any(history, "VIEW_TEAM"):
        breakdown["inv_checked_team"] = 0.15
        score += 0.15
    if _history_contains_any(history, "VIEW_STORY"):
        breakdown["inv_viewed_stories"] = 0.15
        score += 0.15

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["skill_mismatch"])
    plan_score = 0.0

    correct_assignments = 0
    for sid, valid_devs in correct_map.items():
        assigned = board.assignments.get(sid)
        if assigned in valid_devs:
            correct_assignments += 1

    if correct_assignments == len(correct_map):
        plan_score += 0.50
        breakdown["plan_all_correct"] = 0.50
    elif correct_assignments > 0:
        partial = 0.50 * (correct_assignments / len(correct_map))
        plan_score += partial
        breakdown["plan_some_correct"] = round(partial, 3)

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_epic_decomposition(board: ProjectBoard, params: dict, history: List[str],
                               error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0
    epic_id = params.get("epic_id", "EP-01")
    epic = params.get("epic", EPIC_POOL.get(epic_id, {}))

    # Investigation (0.3)
    if _history_contains(history, "VIEW_EPIC", epic_id):
        breakdown["inv_viewed_epic"] = 0.20
        score += 0.20
    if _history_contains_any(history, "LIST_BACKLOG"):
        breakdown["inv_checked_backlog"] = 0.10
        score += 0.10

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["epic_decomposition"])
    plan_score = 0.0

    decomposed = board.decomposed_epics.get(epic_id, [])
    min_sub = params.get("min_subtasks", 4)
    max_sub = params.get("max_subtasks", 8)

    if decomposed:
        count = len(decomposed)
        if min_sub <= count <= max_sub:
            plan_score += 0.20
            breakdown["plan_correct_count"] = 0.20
        elif count >= 2:
            plan_score += 0.10
            breakdown["plan_some_subtasks"] = 0.10

        # Keyword coverage
        required_kws = epic.get("required_keywords", [])
        if required_kws:
            all_text = " ".join(s["title"].lower() for s in decomposed)
            matched = sum(1 for kw in required_kws if kw.lower() in all_text)
            coverage = matched / len(required_kws) if required_kws else 0
            coverage_score = 0.30 * coverage
            plan_score += coverage_score
            breakdown["plan_keyword_coverage"] = round(coverage_score, 3)

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_priority_conflict(board: ProjectBoard, params: dict, history: List[str],
                              error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0
    conflicting = params.get("conflicting_stories", [])

    # Investigation (0.3)
    viewed = sum(1 for sid in conflicting if _history_contains(history, "VIEW_STORY", sid))
    if viewed == len(conflicting):
        breakdown["inv_viewed_both"] = 0.20
        score += 0.20
    elif viewed > 0:
        breakdown["inv_viewed_one"] = 0.10
        score += 0.10
    if _history_contains_any(history, "VIEW_TEAM", "VIEW_VELOCITY"):
        breakdown["inv_checked_capacity"] = 0.10
        score += 0.10

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["priority_conflict"])
    plan_score = 0.0

    # At least one P0 in sprint
    p0_in = sum(1 for sid in conflicting if sid in board.sprint_stories)
    if p0_in >= 1:
        plan_score += 0.30
        breakdown["plan_p0_included"] = 0.30

    # Capacity not exceeded
    metrics = board.get_metrics()
    if not metrics.get("overloaded_developers"):
        plan_score += 0.20
        breakdown["plan_no_overload"] = 0.20

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_tech_debt_balance(board: ProjectBoard, params: dict, history: List[str],
                              error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0
    target_pct = params.get("tech_debt_target_pct", 0.20)

    # Investigation (0.3)
    if _history_contains_any(history, "VIEW_BUGS"):
        breakdown["inv_checked_bugs"] = 0.15
        score += 0.15
    if _history_contains_any(history, "VIEW_SPRINT", "LIST_BACKLOG"):
        breakdown["inv_checked_sprint"] = 0.15
        score += 0.15

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["tech_debt_balance"])
    plan_score = 0.0

    # Check if bugs were added
    bugs_added = len(board.bugs_added)
    p1_bugs_added = sum(1 for bid in board.bugs_added
                        if any(b["id"] == bid and b["priority"] == "P1" for b in BUG_BACKLOG))
    if bugs_added >= 2:
        plan_score += 0.20
        breakdown["plan_bugs_added"] = 0.20
    elif bugs_added >= 1:
        plan_score += 0.10
        breakdown["plan_some_bugs"] = 0.10

    if p1_bugs_added >= 1:
        plan_score += 0.15
        breakdown["plan_p1_bugs_addressed"] = 0.15

    # Check tech debt ratio
    total_pts = sum(board.estimates.get(s, 0) or 0 for s in board.sprint_stories)
    bug_pts = sum(
        next((b["points"] for b in BUG_BACKLOG if b["id"] == bid), 0)
        for bid in board.bugs_added
    )
    tech_debt_stories = sum(
        board.estimates.get(s, 0) or 0
        for s in board.sprint_stories
        if STORY_POOL.get(s, {}).get("type") == "tech-debt"
    )
    debt_total = bug_pts + tech_debt_stories
    if total_pts > 0 and debt_total / (total_pts + bug_pts) >= target_pct:
        plan_score += 0.15
        breakdown["plan_debt_ratio_met"] = 0.15

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


# ══════════════════════════════════════════════════════════════════
# HARD TASK GRADERS (tasks 11–15)
# ══════════════════════════════════════════════════════════════════

def _grade_dependency_chain_overload(board: ProjectBoard, params: dict, history: List[str],
                                      error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0

    # Investigation (0.3)
    if _history_contains_any(history, "CHECK_DEPS"):
        breakdown["inv_checked_deps"] = 0.10
        score += 0.10
    if _history_contains_any(history, "VIEW_TEAM"):
        breakdown["inv_checked_team"] = 0.10
        score += 0.10
    if _history_contains_any(history, "VIEW_SPRINT"):
        breakdown["inv_checked_sprint"] = 0.10
        score += 0.10

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["dependency_chain_overload"])
    plan_score = 0.0

    # Circular dep resolved
    issues = board._audit_sprint()
    circular_issues = [i for i in issues if "circular" in i.lower()]
    if not circular_issues:
        plan_score += 0.20
        breakdown["plan_circular_fixed"] = 0.20

    # Overload resolved
    overloaded_dev = params.get("overloaded_dev", "Alice")
    dev_load = board._get_dev_load(overloaded_dev)
    dev_cap = TEAM_MEMBERS[overloaded_dev]["capacity"]
    if dev_load <= dev_cap:
        plan_score += 0.20
        breakdown["plan_overload_fixed"] = 0.20

    # Both resolved bonus
    if not circular_issues and dev_load <= dev_cap:
        plan_score += 0.10
        breakdown["plan_both_resolved"] = 0.10

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_pto_velocity_drop(board: ProjectBoard, params: dict, history: List[str],
                              error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0
    pto_dev = params.get("pto_developer", "Alice")

    # Investigation (0.3)
    if _history_contains_any(history, "VIEW_TEAM"):
        breakdown["inv_checked_team"] = 0.10
        score += 0.10
    if _history_contains_any(history, "VIEW_VELOCITY"):
        breakdown["inv_checked_velocity"] = 0.10
        score += 0.10
    if _history_contains_any(history, "VIEW_SPRINT"):
        breakdown["inv_checked_sprint"] = 0.10
        score += 0.10

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["pto_velocity_drop"])
    plan_score = 0.0

    # PTO dev has no assignments
    pto_load = board._get_dev_load(pto_dev)
    if pto_load == 0:
        plan_score += 0.20
        breakdown["plan_pto_cleared"] = 0.20

    # Total points within declining velocity
    total_pts = sum(board.estimates.get(s, 0) or 0 for s in board.sprint_stories)
    velocity = params.get("velocity_history_decline", [28])
    recent_vel = velocity[-1] if velocity else 28
    if total_pts <= recent_vel * 1.1:
        plan_score += 0.20
        breakdown["plan_velocity_aligned"] = 0.20

    if pto_load == 0 and total_pts <= recent_vel * 1.1:
        plan_score += 0.10
        breakdown["plan_both_addressed"] = 0.10

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_cross_team(board: ProjectBoard, params: dict, history: List[str],
                       error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0

    # Investigation (0.3)
    if _history_contains_any(history, "VIEW_TEAM"):
        breakdown["inv_checked_team"] = 0.10
        score += 0.10
    if _history_contains_any(history, "CHECK_DEPS"):
        breakdown["inv_checked_deps"] = 0.10
        score += 0.10
    if _history_contains_any(history, "VIEW_STORY"):
        breakdown["inv_viewed_stories"] = 0.10
        score += 0.10

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["cross_team_dependency"])
    plan_score = 0.0

    # Skill gap addressed
    skill_story = params.get("skill_gap_story")
    capable_dev = params.get("only_capable_dev")
    if skill_story and capable_dev:
        assigned = board.assignments.get(skill_story)
        if assigned == capable_dev:
            plan_score += 0.20
            breakdown["plan_skill_gap_fixed"] = 0.20

    # Cross-team story handled (flagged or noted or removed)
    cross_stories = params.get("cross_team_stories", [])
    for sid in cross_stories:
        if sid in board.risk_flags or sid not in board.sprint_stories:
            plan_score += 0.20
            breakdown["plan_cross_team_handled"] = 0.20
            break

    if plan_score >= 0.35:
        plan_score += 0.10
        breakdown["plan_both_addressed"] = 0.10

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_sprint_rescue(board: ProjectBoard, params: dict, history: List[str],
                          error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0
    problems = params.get("problems", {})

    # Investigation (0.3) — must investigate multiple aspects
    inv_score = 0.0
    if _history_contains_any(history, "VIEW_TEAM"):
        inv_score += 0.06
    if _history_contains_any(history, "VIEW_SPRINT", "LIST_BACKLOG"):
        inv_score += 0.06
    if _history_contains_any(history, "CHECK_DEPS"):
        inv_score += 0.06
    if _history_contains_any(history, "VIEW_STORY"):
        inv_score += 0.06
    if _history_contains_any(history, "VIEW_VELOCITY"):
        inv_score += 0.06
    inv_score = min(0.30, inv_score)
    breakdown["inv_total"] = round(inv_score, 2)
    score += inv_score

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["sprint_rescue"])
    plan_score = 0.0
    problems_fixed = 0

    # Check unestimated
    unest = problems.get("unestimated", [])
    if all(board.estimates.get(s) is not None for s in unest):
        plan_score += 0.10
        problems_fixed += 1
        breakdown["plan_estimated"] = 0.10

    # Check overload
    overloaded = problems.get("overloaded_dev")
    if overloaded:
        load = board._get_dev_load(overloaded)
        cap = TEAM_MEMBERS[overloaded]["capacity"]
        if load <= cap:
            plan_score += 0.10
            problems_fixed += 1
            breakdown["plan_overload_fixed"] = 0.10

    # Check missing dep
    missing_deps = problems.get("missing_dep", {})
    for sid, dep in missing_deps.items():
        if dep in board.sprint_stories:
            plan_score += 0.10
            problems_fixed += 1
            breakdown["plan_dep_fixed"] = 0.10
            break

    # Check skill mismatch
    mismatches = problems.get("skill_mismatch", {})
    fixed_mismatches = 0
    for sid, wrong_dev in mismatches.items():
        current = board.assignments.get(sid)
        if current != wrong_dev:
            fixed_mismatches += 1
    if fixed_mismatches == len(mismatches):
        plan_score += 0.10
        problems_fixed += 1
        breakdown["plan_skills_fixed"] = 0.10

    # Bonus for fixing all 5
    if problems_fixed >= 4:
        plan_score += 0.10
        breakdown["plan_comprehensive_fix"] = 0.10

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


def _grade_full_sprint_planning(board: ProjectBoard, params: dict, history: List[str],
                                 error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    breakdown = {}
    score = 0.0

    # Investigation (0.3)
    inv_score = 0.0
    if _history_contains_any(history, "LIST_BACKLOG"):
        inv_score += 0.06
    if _history_contains_any(history, "VIEW_TEAM"):
        inv_score += 0.06
    if _history_contains_any(history, "VIEW_VELOCITY"):
        inv_score += 0.06
    if _history_contains_any(history, "CHECK_DEPS"):
        inv_score += 0.06
    if _history_contains_any(history, "VIEW_STORY"):
        inv_score += 0.06
    inv_score = min(0.30, inv_score)
    breakdown["inv_total"] = round(inv_score, 2)
    score += inv_score

    # Planning (0.5) × efficiency
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["full_sprint_planning"])
    plan_score = 0.0

    metrics = board.get_metrics()
    target_velocity = params.get("target_velocity", 34)

    # Stories selected (not empty sprint)
    if len(board.sprint_stories) >= 4:
        plan_score += 0.10
        breakdown["plan_stories_selected"] = 0.10

    # Within velocity
    total_pts = metrics["total_points"]
    if total_pts <= target_velocity * 1.15 and total_pts >= target_velocity * 0.7:
        plan_score += 0.10
        breakdown["plan_velocity_match"] = 0.10

    # No overloads
    if not metrics["overloaded_developers"]:
        plan_score += 0.10
        breakdown["plan_no_overload"] = 0.10

    # Dependencies satisfied
    if metrics["dependency_issues"] == 0:
        plan_score += 0.10
        breakdown["plan_deps_satisfied"] = 0.10

    # All sprint stories assigned
    if metrics["unassigned_count"] == 0:
        plan_score += 0.10
        breakdown["plan_all_assigned"] = 0.10

    plan_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += plan_score

    # Process (0.2)
    if not _has_destructive(history):
        breakdown["proc_no_destructive"] = 0.10
        score += 0.10
    if _error_rate(error_history) < 0.3:
        breakdown["proc_clean_execution"] = 0.10
        score += 0.10

    return max(0.001, min(0.999, round(score, 4))), breakdown


# ══════════════════════════════════════════════════════════════════
# GRADER REGISTRY
# ══════════════════════════════════════════════════════════════════

GRADER_REGISTRY = {
    "unestimated_stories": _grade_unestimated_stories,
    "developer_overload": _grade_developer_overload,
    "missing_dependency": _grade_missing_dependency,
    "scope_creep": _grade_scope_creep,
    "wrong_priority": _grade_wrong_priority,
    "velocity_overload": _grade_velocity_overload,
    "skill_mismatch": _grade_skill_mismatch,
    "epic_decomposition": _grade_epic_decomposition,
    "priority_conflict": _grade_priority_conflict,
    "tech_debt_balance": _grade_tech_debt_balance,
    "dependency_chain_overload": _grade_dependency_chain_overload,
    "pto_velocity_drop": _grade_pto_velocity_drop,
    "cross_team_dependency": _grade_cross_team,
    "sprint_rescue": _grade_sprint_rescue,
    "full_sprint_planning": _grade_full_sprint_planning,
}


def grade_episode(
    board: ProjectBoard,
    fault_type: str,
    task_params: dict,
    action_history: List[str],
    error_history: List[bool],
    steps_used: int,
) -> Tuple[float, dict]:
    """Run the deterministic grader for the given fault type.

    Returns (score, breakdown) where score is strictly within (0, 1)
    exclusive, as required by the hackathon validator.
    """
    grader = GRADER_REGISTRY.get(fault_type)
    if grader is None:
        logger.error("No grader for fault_type=%s", fault_type)
        return 0.001, {"error": f"No grader for {fault_type}"}

    try:
        score, breakdown = grader(board, task_params, action_history, error_history, steps_used)
        # Clamp to strictly (0, 1) — validator rejects 0.0 and 1.0
        score = max(0.001, min(0.999, score))
        return score, breakdown
    except Exception as e:
        logger.error("Grader error for %s: %s", fault_type, e)
        return 0.001, {"error": str(e)}

