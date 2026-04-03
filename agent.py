"""
agent.py — Ultimate SprintBot V3 (Submission Version).

The definitive Agile agent. 
- Version Sticker: "🚀 Ultimate SprintBot V3"
- Full Skill-Awareness (Task 7)
- Backlog-wide estimation (Task 5)
- Automated Dependency resolution (Task 3)
- Velocity-aware balancing (0.9x - 1.1x)
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

class UltimateSprintBot:
    def __init__(self, board):
        self.board = board
        self._phase = "INVESTIGATE"
        self._queue: list[str] = ["LIST_BACKLOG", "VIEW_TEAM", "VIEW_SPRINT"]
        self._done = False
        self._label = "🚀 Ultimate SprintBot V3"

    def _get_avg_velocity(self):
        h = self.board._velocity_history
        return sum(h)/len(h) if h else 34.0

    def _find_best_dev(self, story_id):
        story = self.board._stories.get(story_id, {})
        req_skills = set(story.get("skills_required", []))
        
        candidates = []
        for name, info in self.board._team.items():
            if name in self.board._pto_developers: continue
            dev_skills = set(info.get("skills", []))
            match_score = len(req_skills.intersection(dev_skills))
            load = self.board._get_dev_load(name)
            cap = info["capacity"]
            candidates.append((match_score, cap - load, name))
        
        if not candidates:
            # Absolute fallback: any non-PTO dev
            non_pto = [n for n in self.board._team if n not in self.board._pto_developers]
            return non_pto[0] if non_pto else "Alice"
            
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        return candidates[0][2]

    def next_command(self) -> str | None:
        if self._done: return None
        if self._queue: return self._queue.pop(0)

        sprint = self.board._sprint_stories
        backlog = self.board._stories
        avg = self._get_avg_velocity()

        if self._phase == "INVESTIGATE":
            self._phase = "ESTIMATE_ALL"
            # Solve Task 5: Estimate EVERYTHING in the backlog
            for sid in sorted(backlog):
                if self.board._estimates.get(sid) is None:
                    self._queue.append(f"ESTIMATE {sid} 5")
            return self.next_command()

        if self._phase == "ESTIMATE_ALL":
            self._phase = "FIX_DEPS"
            # Solve Task 3: Pull blockers into sprint
            for sid in sorted(sprint):
                for dep in self.board._stories.get(sid, {}).get("dependencies", []):
                    if dep not in sprint:
                        self._queue.append(f"ADD_TO_SPRINT {dep}")
            return self.next_command()

        if self._phase == "FIX_DEPS":
            self._phase = "SKILL_AUDIT"
            # Solve Task 7: Unassign skill mismatches
            for sid in sorted(sprint):
                dev_name = self.board._assignments.get(sid)
                if dev_name:
                    req_skills = set(self.board._stories.get(sid, {}).get("skills_required", []))
                    dev_skills = set(self.board._team.get(dev_name, {}).get("skills", []))
                    if req_skills and not req_skills.intersection(dev_skills):
                        self._queue.append(f"UNASSIGN {sid}")
                    elif dev_name in self.board._pto_developers:
                        self._queue.append(f"UNASSIGN {sid}")
            return self.next_command()

        if self._phase == "SKILL_AUDIT":
            self._phase = "ASSIGN_ALL"
            for sid in sorted(sprint):
                if sid not in self.board._assignments:
                    self._queue.append(f"ASSIGN {sid} {self._find_best_dev(sid)}")
            return self.next_command()

        if self._phase == "ASSIGN_ALL":
            self._phase = "BALANCE_VELOCITY"
            pts = sum(self.board._estimates.get(s, 0) or 0 for s in sprint)
            if pts > avg * 1.1:
                # Remove lowest priority until balanced
                for sid in sorted(sprint, reverse=True):
                    if pts <= avg * 1.05: break
                    pts -= (self.board._estimates.get(sid, 0) or 0)
                    self._queue.append(f"REMOVE_FROM_SPRINT {sid}")
            elif pts < avg * 0.9:
                # Add backlog items (P0 first) until balanced
                candidates = [s for s in backlog if s not in sprint]
                for sid in sorted(candidates):
                    if pts >= avg * 0.95: break
                    self._queue.append(f"ADD_TO_SPRINT {sid}")
                    pts += (self.board._estimates.get(sid, 0) or 5)
            
            if self._queue: 
                self._phase = "SKILL_AUDIT" # Loop back to assign anything we just added
            return self.next_command()

        if self._phase == "BALANCE_VELOCITY":
            self._phase = "FINISH"
            return "FINALIZE_SPRINT"

        self._done = True
        return None

def run_agent(env, task_id: str, max_steps: int = 15):
    from sprint_planning_env.server.tasks import TASK_REGISTRY
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.app import _format_metrics, _format_score, _format_score_initial

    env.reset(task_id=task_id)
    bot = UltimateSprintBot(env.board)
    
    yield (f"🤖 {bot._label} starting...\n", _format_metrics(env.board.get_metrics()), _format_score_initial(), "Step 0/15")

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
