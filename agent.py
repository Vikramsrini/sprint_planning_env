"""
agent.py — Elite Skill-Aware SprintBot.

Heuristic agent that solves the 4 'Hard' fault types:
  1. Skill Mismatch: Matches dev skills to story requirements.
  2. Dependencies: Pulls blockers into the sprint.
  3. PTO: Avoids assigning work to developers on leave.
  4. Velocity: Balances points to hit 0.9x-1.1x ratio.
"""

import os
import re
import time

VALID_PREFIXES = [
    "LIST_BACKLOG", "VIEW_STORY", "CHECK_DEPS", "VIEW_TEAM",
    "VIEW_VELOCITY", "VIEW_SPRINT", "VIEW_BUGS", "VIEW_EPIC",
    "SEARCH_BACKLOG", "ESTIMATE", "ASSIGN", "UNASSIGN",
    "ADD_TO_SPRINT", "REMOVE_FROM_SPRINT", "FLAG_RISK",
    "SET_PRIORITY", "FINALIZE_SPRINT", "HELP",
]

# ── Elite Heuristic Bot ───────────────────────────────────────────────────────

class EliteSprintBot:
    def __init__(self, board):
        self.board = board
        self._phase = "INVESTIGATE"
        self._queue: list[str] = ["LIST_BACKLOG", "VIEW_TEAM", "VIEW_SPRINT"]
        self._done = False

    def _get_avg_velocity(self):
        h = self.board._velocity_history
        return sum(h)/len(h) if h else 34.0

    def _find_best_dev(self, story_id, skip_overloaded=True):
        """Find a dev whose skills match the story and isn't on PTO."""
        story = self.board._stories.get(story_id, {})
        req_skills = set(story.get("skills_required", []))
        
        candidates = []
        for name, info in self.board._team.items():
            if name in self.board._pto_developers: continue
            
            # Score the dev based on skill match
            dev_skills = set(info.get("skills", []))
            match_score = len(req_skills.intersection(dev_skills))
            
            load = self.board._get_dev_load(name)
            cap = info["capacity"]
            
            if skip_overloaded and load >= cap: continue
            candidates.append((match_score, cap - load, name))
        
        if not candidates: return "Alice" # Fallback
        # Sort by match score (desc) then by available capacity (desc)
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        return candidates[0][2]

    def next_command(self) -> str | None:
        if self._done: return None
        if self._queue: return self._queue.pop(0)

        sprint = self.board._sprint_stories
        avg = self._get_avg_velocity()

        # ── Phase Transitions ────────────────────────────────────────────────
        if self._phase == "INVESTIGATE":
            self._phase = "FIX_DEPS"
            for sid in sorted(sprint):
                for dep in self.board._stories.get(sid, {}).get("dependencies", []):
                    if dep not in sprint:
                        self._queue.append(f"ADD_TO_SPRINT {dep}")
            return self.next_command()

        if self._phase == "FIX_DEPS":
            self._phase = "ESTIMATE"
            for sid in sorted(sprint):
                if self.board._estimates.get(sid) is None:
                    self._queue.append(f"ESTIMATE {sid} 5")
            return self.next_command()

        if self._phase == "ESTIMATE":
            self._phase = "SKILL_CHECK"
            # Crucial for Task 7: Unassign if there is a skill mismatch
            for sid in sorted(sprint):
                dev_name = self.board._assignments.get(sid)
                if dev_name:
                    story = self.board._stories.get(sid, {})
                    req_skills = set(story.get("skills_required", []))
                    dev_skills = set(self.board._team.get(dev_name, {}).get("skills", []))
                    if req_skills and not req_skills.intersection(dev_skills):
                        self._queue.append(f"UNASSIGN {sid}")
            return self.next_command()

        if self._phase == "SKILL_CHECK":
            self._phase = "ASSIGN"
            # Assign anything unassigned using skill matching
            for sid in sorted(sprint):
                if sid not in self.board._assignments:
                    best_dev = self._find_best_dev(sid)
                    self._queue.append(f"ASSIGN {sid} {best_dev}")
            return self.next_command()

        if self._phase == "ASSIGN":
            self._phase = "BALANCE"
            pts = sum(self.board._estimates.get(s, 0) or 0 for s in sprint)
            if pts > avg * 1.1:
                for sid in sorted(sprint, reverse=True):
                    if pts <= avg * 1.05: break
                    pts -= (self.board._estimates.get(sid, 0) or 0)
                    self._queue.append(f"REMOVE_FROM_SPRINT {sid}")
            elif pts < avg * 0.9:
                backlog = [s for s in self.board._stories if s not in sprint]
                for sid in sorted(backlog):
                    if pts >= avg * 0.95: break
                    self._queue.append(f"ADD_TO_SPRINT {sid}")
                    pts += (self.board._estimates.get(sid, 0) or 5)
            # If we added/removed anything, we need to re-assign/unassign in next step
            if self._queue: 
                self._phase = "SKILL_CHECK" # Loop back to ensure new stories get assigned
            return self.next_command()

        if self._phase == "BALANCE":
            self._phase = "FINISH"
            return "FINALIZE_SPRINT"

        self._done = True
        return None


def run_agent(env, task_id: str, max_steps: int = 15):
    from sprint_planning_env.server.tasks import TASK_REGISTRY
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.app import _format_metrics, _format_score, _format_score_initial

    task = TASK_REGISTRY[task_id]
    env.reset(task_id=task_id)
    bot = EliteSprintBot(env.board)
    
    yield ("🤖 Elite SprintBot starting...\n", _format_metrics(env.board.get_metrics()), _format_score_initial(), "Step 0/15")

    for step in range(1, max_steps + 1):
        cmd = bot.next_command() or "FINALIZE_SPRINT"
        obs = env.step(SprintAction(command=cmd))
        
        log = f"\n{'-'*40}\n$ {cmd}\n{obs.command_output or obs.error or ''}\n"
        if obs.done:
            grade = obs.metadata.get("grader_score", 0.0) or 0.0
            log += f"\n🏆 FINAL SCORE: {int(grade*100)}%\n"
            yield (log, _format_metrics(obs.metrics), _format_score(obs), f"Step {step}/{max_steps}")
            break
        
        yield (log, _format_metrics(obs.metrics), _format_score(obs), f"Step {step}/{max_steps}")
        time.sleep(0.5)
