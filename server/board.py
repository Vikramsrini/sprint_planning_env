"""
SprintBoard — Simulated project board state machine.

This is SprintBoard's equivalent of SQLab's live PostgreSQL database.
Where SQLab agents query pg_stat_activity and pg_indexes, SprintBoard
agents query the board for stories, team capacity, velocity history,
and dependency graphs.

The board maintains mutable state that the agent modifies through
planning commands (ASSIGN, ESTIMATE, ADD_TO_SPRINT, etc.). Each task's
fault injector pre-breaks the board state; the agent must investigate
and fix it.

All data is deterministic (seeded) for reproducible RL training.
"""

import copy
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

try:
    from sprint_planning_env.server.tasks import (
        STORY_POOL, TEAM_MEMBERS, VELOCITY_HISTORY, EPIC_POOL, BUG_BACKLOG,
    )
except ModuleNotFoundError:
    from server.tasks import (
        STORY_POOL, TEAM_MEMBERS, VELOCITY_HISTORY, EPIC_POOL, BUG_BACKLOG,
    )

logger = logging.getLogger(__name__)


class ProjectBoard:
    """Stateful simulation of an Agile project management board.

    Maintains:
    - Sprint backlog (which stories are in the current sprint)
    - Assignments (story → developer mapping)
    - Estimates (story → points mapping, some may be None)
    - Team roster with capacities and skills
    - Velocity history
    - Dependency graph
    - Risk flags
    - Board metrics for observation
    """

    def __init__(self):
        self._stories: Dict[str, Dict[str, Any]] = {}
        self._sprint_stories: Set[str] = set()
        self._assignments: Dict[str, str] = {}  # story_id → developer_name
        self._estimates: Dict[str, Optional[int]] = {}  # story_id → points or None
        self._team: Dict[str, Dict[str, Any]] = {}
        self._velocity_history: List[int] = []
        self._risk_flags: Dict[str, str] = {}  # story_id → risk_reason
        self._decomposed_epics: Dict[str, List[Dict[str, str]]] = {}
        self._priority_overrides: Dict[str, str] = {}  # story_id → new priority
        self._bugs_added: List[str] = []
        self._pto_developers: Set[str] = set()
        self._cross_team_notes: Dict[str, str] = {}
        self._finalized: bool = False

    def reset(self, task_params: Dict[str, Any]) -> None:
        """Initialize board state from task parameters."""
        # Load full story pool
        self._stories = copy.deepcopy(STORY_POOL)
        self._team = copy.deepcopy(TEAM_MEMBERS)
        self._velocity_history = list(VELOCITY_HISTORY)

        # Set sprint stories from task
        sprint_ids = task_params.get("sprint_stories", [])
        backlog_ids = task_params.get("backlog_stories", sprint_ids)
        self._sprint_stories = set(sprint_ids)

        # Set initial estimates (from story pool defaults)
        self._estimates = {}
        for sid, story in self._stories.items():
            self._estimates[sid] = story.get("points")

        # Clear unestimated stories if specified
        for sid in task_params.get("unestimated_story_ids", []):
            self._estimates[sid] = None

        # Set initial assignments
        self._assignments = dict(task_params.get("initial_assignments", {}))

        # Handle PTO
        pto_dev = task_params.get("pto_developer")
        self._pto_developers = {pto_dev} if pto_dev else set()

        # Override velocity if task specifies declining velocity
        if "velocity_history_decline" in task_params:
            self._velocity_history = list(task_params["velocity_history_decline"])

        # Inject circular dependencies if specified
        circular_deps = task_params.get("circular_deps", {})
        for from_id, to_id in circular_deps.items():
            if from_id in self._stories:
                self._stories[from_id].setdefault("dependencies", [])
                if to_id not in self._stories[from_id]["dependencies"]:
                    self._stories[from_id]["dependencies"].append(to_id)

        # Reset other state
        self._risk_flags = {}
        self._decomposed_epics = {}
        self._priority_overrides = {}
        self._bugs_added = []
        self._cross_team_notes = {}
        self._finalized = False

    # ── Query commands ────────────────────────────────────────────

    def list_backlog(self, sort_by: str = "priority") -> str:
        """List all stories in the backlog with their status."""
        stories = []
        for sid, story in self._stories.items():
            in_sprint = "✓" if sid in self._sprint_stories else " "
            pts = self._estimates.get(sid)
            pts_str = f"{pts}pts" if pts is not None else "UNESTIMATED"
            pri = self._priority_overrides.get(sid, story.get("priority", "P2"))
            assigned = self._assignments.get(sid, "unassigned")
            stype = story.get("type", "feature")
            stories.append((pri, sid, story["title"], pts_str, assigned, in_sprint, stype))

        # Sort
        pri_order = {"P0": 0, "P1": 1, "P2": 2}
        if sort_by == "priority":
            stories.sort(key=lambda x: (pri_order.get(x[0], 9), x[1]))
        elif sort_by == "points":
            stories.sort(key=lambda x: (self._estimates.get(x[1]) or 99, x[1]))
        else:
            stories.sort(key=lambda x: x[1])

        lines = ["BACKLOG — All Stories", "=" * 80]
        lines.append(f"{'':2} {'ID':<8} {'Title':<35} {'Pts':<14} {'Assigned':<12} {'Pri':<4} {'Type':<10}")
        lines.append("-" * 80)
        for pri, sid, title, pts_str, assigned, in_sprint, stype in stories:
            title_short = title[:33] + ".." if len(title) > 35 else title
            lines.append(f"[{in_sprint}] {sid:<8} {title_short:<35} {pts_str:<14} {assigned:<12} {pri:<4} {stype}")

        # Summary
        sprint_pts = sum(self._estimates.get(s, 0) or 0 for s in self._sprint_stories)
        lines.append("-" * 80)
        lines.append(f"Sprint: {len(self._sprint_stories)} stories, {sprint_pts} points")
        lines.append(f"Backlog: {len(self._stories)} total stories")
        return "\n".join(lines)

    def view_story(self, story_id: str) -> str:
        """Show detailed info about a story."""
        story = self._stories.get(story_id)
        if not story:
            return f"ERROR: Story {story_id} not found."

        pts = self._estimates.get(story_id)
        pri = self._priority_overrides.get(story_id, story.get("priority", "P2"))
        assigned = self._assignments.get(story_id, "unassigned")
        in_sprint = story_id in self._sprint_stories
        deps = story.get("dependencies", [])
        skills = story.get("skills_required", [])
        acs = story.get("acceptance_criteria", [])
        risk = self._risk_flags.get(story_id)

        lines = [f"STORY: {story_id}", "=" * 60]
        lines.append(f"Title:       {story['title']}")
        lines.append(f"Description: {story['description']}")
        lines.append(f"Points:      {pts if pts is not None else 'UNESTIMATED'}")
        lines.append(f"Priority:    {pri}")
        lines.append(f"Type:        {story.get('type', 'feature')}")
        lines.append(f"Assigned:    {assigned}")
        lines.append(f"In Sprint:   {'Yes' if in_sprint else 'No'}")
        lines.append(f"Skills:      {', '.join(skills)}")
        if deps:
            dep_status = []
            for d in deps:
                in_s = "in sprint" if d in self._sprint_stories else "NOT in sprint"
                dep_status.append(f"{d} ({in_s})")
            lines.append(f"Dependencies: {', '.join(dep_status)}")
        else:
            lines.append("Dependencies: None")
        if acs:
            lines.append("Acceptance Criteria:")
            for i, ac in enumerate(acs, 1):
                lines.append(f"  {i}. {ac}")
        if risk:
            lines.append(f"⚠ RISK FLAG: {risk}")
        return "\n".join(lines)

    def check_deps(self, story_id: str) -> str:
        """Show dependency graph for a story."""
        story = self._stories.get(story_id)
        if not story:
            return f"ERROR: Story {story_id} not found."

        deps = story.get("dependencies", [])
        lines = [f"DEPENDENCY CHECK: {story_id} ({story['title']})", "=" * 60]

        if not deps:
            lines.append("No dependencies. This story can be worked on independently.")
            return "\n".join(lines)

        all_clear = True
        for dep_id in deps:
            dep_story = self._stories.get(dep_id, {})
            in_sprint = dep_id in self._sprint_stories
            status = "✓ IN SPRINT" if in_sprint else "✗ NOT IN SPRINT — BLOCKER"
            if not in_sprint:
                all_clear = False
            lines.append(f"  → {dep_id} ({dep_story.get('title', 'unknown')}): {status}")

        # Check for circular dependencies
        visited = set()
        def find_circular(sid, path):
            if sid in path:
                return path[path.index(sid):] + [sid]
            if sid in visited:
                return None
            visited.add(sid)
            s = self._stories.get(sid, {})
            for d in s.get("dependencies", []):
                result = find_circular(d, path + [sid])
                if result:
                    return result
            return None

        circular = find_circular(story_id, [])
        if circular:
            lines.append(f"\n⚠ CIRCULAR DEPENDENCY DETECTED: {' → '.join(circular)}")
            all_clear = False

        if all_clear:
            lines.append("\n✓ All dependencies satisfied.")
        else:
            lines.append("\n✗ Dependency issues found. Sprint plan is invalid.")

        return "\n".join(lines)

    def view_team(self) -> str:
        """Show team members with skills, capacity, and current load."""
        lines = ["TEAM ROSTER", "=" * 80]
        lines.append(f"{'Name':<12} {'Role':<30} {'Cap':>4} {'Load':>5} {'Avail':>6}  Skills")
        lines.append("-" * 80)

        for name, info in self._team.items():
            load = self._get_dev_load(name)
            cap = info["capacity"]
            avail = cap - load
            pto = " [PTO]" if name in self._pto_developers else ""
            status = " ⚠OVER" if load > cap else ""
            skills = ", ".join(info["skills"][:5])
            lines.append(
                f"{name:<12} {info['role']:<30} {cap:>4} {load:>5} {avail:>6}  {skills}{pto}{status}"
            )

        total_cap = sum(
            i["capacity"] for n, i in self._team.items()
            if n not in self._pto_developers
        )
        total_load = sum(self._get_dev_load(n) for n in self._team if n not in self._pto_developers)
        lines.append("-" * 80)
        lines.append(f"Total capacity: {total_cap} | Total assigned: {total_load} | Available: {total_cap - total_load}")
        if self._pto_developers:
            lines.append(f"On PTO: {', '.join(self._pto_developers)}")
        return "\n".join(lines)

    def view_velocity(self, last_n: int = 5) -> str:
        """Show historical sprint velocity."""
        history = self._velocity_history[-last_n:]
        avg = sum(history) / len(history) if history else 0
        lines = [f"VELOCITY HISTORY (last {len(history)} sprints)", "=" * 50]
        for i, v in enumerate(history):
            sprint_num = len(self._velocity_history) - len(history) + i + 1
            bar = "█" * (v // 2)
            lines.append(f"  Sprint {sprint_num:>2}: {v:>3} pts  {bar}")

        lines.append("-" * 50)
        lines.append(f"Average: {avg:.1f} pts")

        # Trend
        if len(history) >= 3:
            recent = history[-3:]
            if recent[-1] < recent[0] - 3:
                lines.append("Trend: ↓ DECLINING")
            elif recent[-1] > recent[0] + 3:
                lines.append("Trend: ↑ IMPROVING")
            else:
                lines.append("Trend: → STABLE")

        return "\n".join(lines)

    def view_sprint(self) -> str:
        """Show current sprint plan summary."""
        lines = ["CURRENT SPRINT PLAN", "=" * 80]
        lines.append(f"{'ID':<8} {'Title':<30} {'Pts':>4} {'Assigned':<12} {'Pri':<4} {'Deps'}")
        lines.append("-" * 80)

        total_pts = 0
        issues = []
        for sid in sorted(self._sprint_stories):
            story = self._stories.get(sid, {})
            pts = self._estimates.get(sid)
            pts_val = pts if pts is not None else 0
            total_pts += pts_val
            pts_str = str(pts) if pts is not None else "???"
            assigned = self._assignments.get(sid, "—")
            pri = self._priority_overrides.get(sid, story.get("priority", "P2"))
            deps = story.get("dependencies", [])
            dep_str = ", ".join(deps) if deps else "none"
            title_short = story.get("title", "")[:28]
            lines.append(f"{sid:<8} {title_short:<30} {pts_str:>4} {assigned:<12} {pri:<4} {dep_str}")

        lines.append("-" * 80)
        lines.append(f"Total: {len(self._sprint_stories)} stories, {total_pts} points")

        # Check for issues
        issues = self._audit_sprint()
        if issues:
            lines.append(f"\n⚠ {len(issues)} ISSUE(S) DETECTED:")
            for issue in issues:
                lines.append(f"  • {issue}")
        else:
            lines.append("\n✓ Sprint plan looks valid.")

        return "\n".join(lines)

    def search_backlog(self, keyword: str) -> str:
        """Search stories by keyword."""
        keyword_lower = keyword.lower()
        matches = []
        for sid, story in self._stories.items():
            searchable = f"{story['title']} {story['description']}".lower()
            if keyword_lower in searchable:
                pts = self._estimates.get(sid)
                pts_str = f"{pts}pts" if pts is not None else "UNEST"
                in_sprint = "✓" if sid in self._sprint_stories else " "
                matches.append(f"[{in_sprint}] {sid} {story['title'][:40]} ({pts_str})")

        if not matches:
            return f"No stories found matching '{keyword}'."
        return f"SEARCH RESULTS for '{keyword}' ({len(matches)} found):\n" + "\n".join(matches)

    def view_bugs(self) -> str:
        """Show the bug backlog."""
        lines = ["BUG BACKLOG", "=" * 60]
        lines.append(f"{'ID':<8} {'Title':<35} {'Pri':<4} {'Pts':>4}")
        lines.append("-" * 60)
        for bug in BUG_BACKLOG:
            in_sprint = "✓" if bug["id"] in self._bugs_added else " "
            lines.append(f"[{in_sprint}] {bug['id']:<6} {bug['title']:<35} {bug['priority']:<4} {bug['points']:>4}")
        p1_count = sum(1 for b in BUG_BACKLOG if b["priority"] == "P1")
        lines.append(f"\nTotal: {len(BUG_BACKLOG)} bugs ({p1_count} P1)")
        return "\n".join(lines)

    def view_epic(self, epic_id: str) -> str:
        """Show epic details."""
        epic = EPIC_POOL.get(epic_id)
        if not epic:
            return f"ERROR: Epic {epic_id} not found."
        lines = [f"EPIC: {epic_id}", "=" * 60]
        lines.append(f"Title: {epic['title']}")
        lines.append(f"Description: {epic['description']}")
        lines.append(f"Estimated Size: ~{epic['total_points']} points")
        if epic_id in self._decomposed_epics:
            lines.append(f"\nDecomposed into {len(self._decomposed_epics[epic_id])} subtasks:")
            for i, sub in enumerate(self._decomposed_epics[epic_id], 1):
                lines.append(f"  {i}. {sub['title']}")
        else:
            lines.append("\nNot yet decomposed.")
        return "\n".join(lines)

    # ── Mutation commands ─────────────────────────────────────────

    def estimate(self, story_id: str, points: int) -> str:
        """Set story point estimate."""
        if story_id not in self._stories:
            return f"ERROR: Story {story_id} not found."
        valid_points = [1, 2, 3, 5, 8, 13, 21]
        if points not in valid_points:
            return f"ERROR: Invalid points {points}. Use Fibonacci: {valid_points}"
        self._estimates[story_id] = points
        return f"✓ {story_id} estimated at {points} points."

    def assign(self, story_id: str, developer: str) -> str:
        """Assign story to developer."""
        if story_id not in self._stories:
            return f"ERROR: Story {story_id} not found."
        if developer not in self._team:
            return f"ERROR: Developer '{developer}' not found. Team: {list(self._team.keys())}"
        if developer in self._pto_developers:
            return f"WARNING: {developer} is on PTO this sprint. Assignment saved but risky."
        if story_id not in self._sprint_stories:
            return f"WARNING: {story_id} is not in the sprint yet. Use ADD_TO_SPRINT first."

        self._assignments[story_id] = developer
        load = self._get_dev_load(developer)
        cap = self._team[developer]["capacity"]
        status = f" (load: {load}/{cap})"
        if load > cap:
            status += " ⚠ OVER CAPACITY"
        return f"✓ {story_id} assigned to {developer}.{status}"

    def unassign(self, story_id: str) -> str:
        """Remove assignment."""
        if story_id not in self._assignments:
            return f"ERROR: {story_id} is not assigned to anyone."
        dev = self._assignments.pop(story_id)
        return f"✓ {story_id} unassigned from {dev}."

    def add_to_sprint(self, story_id: str) -> str:
        """Add story to sprint."""
        if story_id not in self._stories:
            # Check if it's a bug
            bug = next((b for b in BUG_BACKLOG if b["id"] == story_id), None)
            if bug:
                self._bugs_added.append(story_id)
                return f"✓ Bug {story_id} ({bug['title']}) added to sprint ({bug['points']} pts)."
            return f"ERROR: Story {story_id} not found."
        if story_id in self._sprint_stories:
            return f"INFO: {story_id} is already in the sprint."
        self._sprint_stories.add(story_id)
        return f"✓ {story_id} added to sprint."

    def remove_from_sprint(self, story_id: str) -> str:
        """Remove story from sprint."""
        if story_id not in self._sprint_stories:
            return f"ERROR: {story_id} is not in the sprint."
        self._sprint_stories.discard(story_id)
        if story_id in self._assignments:
            del self._assignments[story_id]
        return f"✓ {story_id} removed from sprint."

    def flag_risk(self, story_id: str, reason: str) -> str:
        """Flag a story as risky."""
        if story_id not in self._stories:
            return f"ERROR: Story {story_id} not found."
        self._risk_flags[story_id] = reason
        return f"✓ {story_id} flagged: {reason}"

    def set_priority(self, story_id: str, priority: str) -> str:
        """Override story priority."""
        if story_id not in self._stories:
            return f"ERROR: Story {story_id} not found."
        if priority not in ("P0", "P1", "P2"):
            return f"ERROR: Invalid priority '{priority}'. Use P0, P1, or P2."
        self._priority_overrides[story_id] = priority
        return f"✓ {story_id} priority set to {priority}."

    def decompose(self, epic_id: str, subtask_titles: List[str]) -> str:
        """Decompose an epic into subtasks."""
        epic = EPIC_POOL.get(epic_id)
        if not epic:
            return f"ERROR: Epic {epic_id} not found."
        if len(subtask_titles) < 2:
            return "ERROR: Must provide at least 2 subtasks."

        subtasks = [{"title": t, "id": f"{epic_id}-SUB-{i+1}"} for i, t in enumerate(subtask_titles)]
        self._decomposed_epics[epic_id] = subtasks
        lines = [f"✓ {epic_id} decomposed into {len(subtasks)} subtasks:"]
        for sub in subtasks:
            lines.append(f"  • {sub['id']}: {sub['title']}")
        return "\n".join(lines)

    def add_note(self, story_id: str, note: str) -> str:
        """Add a cross-team coordination note."""
        self._cross_team_notes[story_id] = note
        return f"✓ Note added to {story_id}: {note}"

    def finalize_sprint(self) -> str:
        """Submit the sprint plan for grading."""
        self._finalized = True
        issues = self._audit_sprint()
        lines = ["SPRINT FINALIZED", "=" * 60]
        lines.append(f"Stories: {len(self._sprint_stories)}")
        total_pts = sum(self._estimates.get(s, 0) or 0 for s in self._sprint_stories)
        lines.append(f"Total Points: {total_pts}")
        assigned_count = sum(1 for s in self._sprint_stories if s in self._assignments)
        lines.append(f"Assigned: {assigned_count}/{len(self._sprint_stories)}")
        if issues:
            lines.append(f"\n⚠ {len(issues)} remaining issues:")
            for issue in issues:
                lines.append(f"  • {issue}")
        else:
            lines.append("\n✓ Sprint plan is valid!")
        return "\n".join(lines)

    # ── Metrics ───────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return board health metrics for observation."""
        total_pts = sum(self._estimates.get(s, 0) or 0 for s in self._sprint_stories)
        avg_velocity = sum(self._velocity_history) / len(self._velocity_history) if self._velocity_history else 0
        unestimated = sum(1 for s in self._sprint_stories if self._estimates.get(s) is None)
        unassigned = sum(1 for s in self._sprint_stories if s not in self._assignments)
        overloaded = []
        for name in self._team:
            if name not in self._pto_developers:
                load = self._get_dev_load(name)
                if load > self._team[name]["capacity"]:
                    overloaded.append(name)

        dep_issues = 0
        for sid in self._sprint_stories:
            story = self._stories.get(sid, {})
            for dep in story.get("dependencies", []):
                if dep not in self._sprint_stories:
                    dep_issues += 1

        return {
            "sprint_stories": len(self._sprint_stories),
            "total_points": total_pts,
            "avg_velocity": round(avg_velocity, 1),
            "velocity_ratio": round(total_pts / avg_velocity, 2) if avg_velocity else 0,
            "unestimated_count": unestimated,
            "unassigned_count": unassigned,
            "overloaded_developers": overloaded,
            "dependency_issues": dep_issues,
            "risk_flags": len(self._risk_flags),
            "finalized": self._finalized,
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _get_dev_load(self, developer: str) -> int:
        """Calculate total points assigned to a developer."""
        total = 0
        for sid, dev in self._assignments.items():
            if dev == developer and sid in self._sprint_stories:
                pts = self._estimates.get(sid, 0) or 0
                total += pts
        return total

    def _audit_sprint(self) -> List[str]:
        """Check sprint for common issues."""
        issues = []

        # Unestimated stories
        for sid in self._sprint_stories:
            if self._estimates.get(sid) is None:
                issues.append(f"{sid}: No estimate")

        # Capacity overloads
        for name, info in self._team.items():
            if name in self._pto_developers:
                # Check if PTO dev has assignments
                load = self._get_dev_load(name)
                if load > 0:
                    issues.append(f"{name}: On PTO but assigned {load} pts")
                continue
            load = self._get_dev_load(name)
            if load > info["capacity"]:
                issues.append(f"{name}: Overloaded ({load}/{info['capacity']} pts)")

        # Missing dependencies
        for sid in self._sprint_stories:
            story = self._stories.get(sid, {})
            for dep in story.get("dependencies", []):
                if dep not in self._sprint_stories:
                    issues.append(f"{sid}: Dependency {dep} not in sprint")

        # Circular dependencies
        for sid in self._sprint_stories:
            visited = set()
            def has_cycle(s, path):
                if s in path:
                    return True
                if s in visited:
                    return False
                visited.add(s)
                story = self._stories.get(s, {})
                for d in story.get("dependencies", []):
                    if d in self._sprint_stories and has_cycle(d, path | {s}):
                        return True
                return False
            if has_cycle(sid, set()):
                issues.append(f"{sid}: Part of circular dependency chain")

        return issues

    @property
    def sprint_stories(self) -> Set[str]:
        return self._sprint_stories

    @property
    def assignments(self) -> Dict[str, str]:
        return self._assignments

    @property
    def estimates(self) -> Dict[str, Optional[int]]:
        return self._estimates

    @property
    def risk_flags(self) -> Dict[str, str]:
        return self._risk_flags

    @property
    def decomposed_epics(self) -> Dict[str, List[Dict[str, str]]]:
        return self._decomposed_epics

    @property
    def stories(self) -> Dict[str, Dict[str, Any]]:
        return self._stories

    @property
    def team(self) -> Dict[str, Dict[str, Any]]:
        return self._team

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    @property
    def pto_developers(self) -> Set[str]:
        return self._pto_developers

    @property
    def bugs_added(self) -> List[str]:
        return self._bugs_added
