"""
agent.py — Ultimate SprintBot OMEGA (V4.1).

Refined state machine:
- Ensuring the action queue is fully drained before phase transitions.
- Improved 'Assign' logic specifically for stories moving from Backlog to Sprint.
- Hard capacity limits preserved.
"""

import os
import re
import time

class OmegaSprintBot:
    def __init__(self, board):
        self.board = board
        self._phase = "INVESTIGATE"
        self._queue: list[str] = ["LIST_BACKLOG", "VIEW_TEAM", "VIEW_VELOCITY", "VIEW_SPRINT"]
        self._done = False
        self._label = "🚀 Ultimate SprintBot OMEGA V4.1"

    def _get_avg_velocity(self):
        h = self.board._velocity_history
        return sum(h)/len(h) if h else 34.0

    def _find_best_dev(self, story_id, target_load_check=True):
        story = self.board._stories.get(story_id, {})
        req_skills = set(story.get("skills_required", []))
        story_pts = self.board._estimates.get(story_id, 0) or 5
        
        candidates = []
        for name, info in self.board._team.items():
            if name in self.board._pto_developers: continue
            dev_skills = set(info.get("skills", []))
            match_score = len(req_skills.intersection(dev_skills))
            load = self.board._get_dev_load(name)
            cap = info["capacity"]
            
            if target_load_check and (load + story_pts > cap): continue
            candidates.append((match_score, cap - load, name))
        
        if not candidates and target_load_check:
            return self._find_best_dev(story_id, target_load_check=False)
            
        if not candidates: 
            return [n for n in self.board._team if n not in self.board._pto_developers][0]
            
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        return candidates[0][2]

    def next_command(self) -> str | None:
        if self._done: return None
        if self._queue: return self._queue.pop(0)

        sprint = self.board._sprint_stories
        backlog = self.board._stories
        avg = self._get_avg_velocity()

        # ── State Machine ────────────────────────────────────────────────────
        
        if self._phase == "INVESTIGATE":
            # 1. Estimate everything
            for sid in sorted(backlog):
                if self.board._estimates.get(sid) is None:
                    self._queue.append(f"ESTIMATE {sid} 5")
            self._phase = "FIX_DEPS"
            return self.next_command()

        if self._phase == "FIX_DEPS":
            # 2. Fix dependencies
            for sid in sorted(sprint):
                for dep in self.board._stories.get(sid, {}).get("dependencies", []):
                    if dep not in sprint:
                        self._queue.append(f"ADD_TO_SPRINT {dep}")
            self._phase = "SKILL_AUDIT"
            return self.next_command()

        if self._phase == "SKILL_AUDIT":
            # 3. Unassign invalid/overloaded
            for sid in sorted(sprint):
                dev_name = self.board._assignments.get(sid)
                if dev_name:
                    dev_info = self.board._team.get(dev_name, {})
                    load = self.board._get_dev_load(dev_name)
                    cap = dev_info.get("capacity", 10)
                    if dev_name in self.board._pto_developers or load > cap:
                        self._queue.append(f"UNASSIGN {sid}")
            self._phase = "ASSIGN_ALL"
            return self.next_command()

        if self._phase == "ASSIGN_ALL":
            # 4. Assign stories with best dev
            for sid in sorted(sprint):
                if sid not in self.board._assignments:
                    best = self._find_best_dev(sid)
                    self._queue.append(f"ASSIGN {sid} {best}")
            self._phase = "BALANCE_VELOCITY"
            return self.next_command()

        if self._phase == "BALANCE_VELOCITY":
            # 5. Velocity Check
            pts = sum(self.board._estimates.get(s, 0) or 0 for s in sprint)
            if pts > avg * 1.1:
                for sid in sorted(sprint, reverse=True):
                    if pts <= avg * 1.05: break
                    pts -= (self.board._estimates.get(sid, 0) or 0)
                    self._queue.append(f"REMOVE_FROM_SPRINT {sid}")
            elif pts < avg * 0.9:
                candidates = [s for s in sorted(backlog) if s not in sprint]
                for sid in candidates:
                    if pts >= avg * 0.95: break
                    self._queue.append(f"ADD_TO_SPRINT {sid}")
                    pts += (self.board._estimates.get(sid, 0) or 5)
            
            if self._queue: 
                # If we added/removed anything, we MUST jump back to assignment
                self._phase = "SKILL_AUDIT"
            else:
                self._phase = "FINISH"
            return self.next_command()

        if self._phase == "FINISH":
            self._done = True
            return "FINALIZE_SPRINT"

        return None

def run_agent(env, task_id: str, max_steps: int = 15):
    from sprint_planning_env.server.tasks import TASK_REGISTRY
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.app import _format_metrics, _format_score, _format_score_initial

    env.reset(task_id=task_id)
    bot = OmegaSprintBot(env.board)

    # Accumulate the full terminal log so each yield shows the complete history
    accumulated_log = f"🤖 {bot._label} starting...\n"
    yield (accumulated_log, _format_metrics(env.board.get_metrics()), _format_score_initial(), "Step 0/15")

    _obs = None
    _step = 0
    for step in range(1, max_steps + 1):
        cmd = bot.next_command()
        if not cmd:
            break

        obs = env.step(SprintAction(command=cmd))
        _obs = obs
        _step = step
        entry = f"\n{'─'*40}\n$ {cmd}\n{obs.command_output or obs.error or ''}\n"

        if obs.done:
            grade = obs.metadata.get("grader_score", 0.0) or 0.0
            pct = int(grade * 100)
            entry += (
                f"\n{'═'*40}\n"
                f"  EPISODE COMPLETE\n"
                f"  🏆 FINAL SCORE: {pct}%\n"
                f"{'═'*40}\n"
            )
            accumulated_log += entry
            yield (accumulated_log, _format_metrics(obs.metrics), _format_score(obs), f"Step {step}/{max_steps}")
            return  # already finalized — skip the forced step below

        accumulated_log += entry
        yield (accumulated_log, _format_metrics(obs.metrics), _format_score(obs), f"Step {step}/{max_steps}")
        time.sleep(0.4)

    # ── Force FINALIZE_SPRINT so the summary is always shown ──────────────────
    time.sleep(0.3)
    obs = env.step(SprintAction(command="FINALIZE_SPRINT"))
    grade = obs.metadata.get("grader_score", 0.0) or 0.0
    pct = int(grade * 100)
    entry = (
        f"\n{'─'*40}\n$ FINALIZE_SPRINT\n"
        f"{obs.command_output or obs.error or ''}\n"
        f"\n{'═'*40}\n"
        f"  EPISODE COMPLETE\n"
        f"  🏆 FINAL SCORE: {pct}%\n"
        f"{'═'*40}\n"
    )
    accumulated_log += entry
    yield (accumulated_log, _format_metrics(obs.metrics), _format_score(obs), f"Step {_step + 1}/{max_steps}")
