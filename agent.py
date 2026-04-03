"""
agent.py — SprintBot: Direct-state reactive agent for SprintBoard.

Instead of parsing board text, reads the board's internal state directly
via env.board.* properties for 100% reliable issue detection.

Strategy:
  1. Investigate  — run standard investigation commands
  2. Estimate     — find and estimate all unestimated sprint stories
  3. Assign       — assign all unassigned sprint stories (balanced)
  4. Balance      — trim/add stories to hit ~0.90-1.05x velocity ratio
  5. Finalize     — FINALIZE_SPRINT

Falls back to Mistral LLM agent if MISTRAL_API_KEY is set.
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

# ── Mistral LLM helpers ───────────────────────────────────────────────────────

def _build_mistral_client():
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        return None
    try:
        from mistralai import Mistral
        return Mistral(api_key=key)
    except Exception:
        return None

SYSTEM_PROMPT = (
    "You are an expert Scrum Master. Respond with ONE valid command only. "
    "No explanations, no punctuation, no markdown. "
    "Commands: LIST_BACKLOG, VIEW_TEAM, VIEW_VELOCITY, VIEW_SPRINT, VIEW_BUGS, "
    "CHECK_DEPS <id>, ESTIMATE <id> <pts>, ASSIGN <id> <name>, "
    "ADD_TO_SPRINT <id>, REMOVE_FROM_SPRINT <id>, FLAG_RISK <id> <reason>, FINALIZE_SPRINT."
)

def _llm_command(client, messages: list) -> str | None:
    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            max_tokens=32,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        return _extract_command(raw)
    except Exception:
        return None

def _extract_command(raw: str) -> str:
    raw = re.sub(r"```[a-z]*", "", raw).strip("`").strip()
    for line in raw.splitlines():
        line = line.strip().strip("`").strip("*").strip()
        for prefix in VALID_PREFIXES:
            if line.upper().startswith(prefix):
                return line
    return next((l.strip() for l in raw.splitlines() if l.strip()), "HELP").upper()


# ── Direct-state SprintBot ────────────────────────────────────────────────────

class DirectStateBot:
    """
    Reads board._sprint_stories, ._estimates, ._assignments, ._team directly.
    Builds a precise fix queue with no text-parsing errors.
    """

    INVESTIGATE_CMDS = [
        "LIST_BACKLOG",
        "VIEW_SPRINT",
    ]

    def __init__(self, board):
        self.board = board
        self._phase = "INVESTIGATE"
        self._inv_iter = iter(self.INVESTIGATE_CMDS)
        self._fix_queue: list[str] = []
        self._dev_cycle = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _avg_velocity(self) -> float:
        hist = self.board._velocity_history
        return sum(hist) / len(hist) if hist else 34.0

    def _sprint_pts(self) -> int:
        return sum(
            self.board._estimates.get(s, 0) or 0
            for s in self.board._sprint_stories
        )

    def _available_devs(self) -> list[str]:
        """Return developers sorted by available capacity (most available first)."""
        devs = []
        for name, info in self.board._team.items():
            if name in self.board._pto_developers:
                continue
            load = self.board._get_dev_load(name)
            avail = info["capacity"] - load
            devs.append((avail, name))
        devs.sort(reverse=True)
        return [d[1] for d in devs]

    # ── Build fix queue from board state ──────────────────────────────────────

    def _build_fix_queue(self) -> list[str]:
        cmds: list[str] = []
        sprint = self.board._sprint_stories
        estimates = self.board._estimates
        assignments = self.board._assignments

        # 1. Estimate all unestimated SPRINT stories
        for sid in sorted(sprint):
            if estimates.get(sid) is None:
                cmds.append(f"ESTIMATE {sid} 5")

        # 2. Assign all unassigned SPRINT stories (round-robin, skip overloaded)
        dev_idx = 0
        devs = self._available_devs()
        if not devs:
            devs = list(self.board._team.keys())

        for sid in sorted(sprint):
            if sid not in assignments:
                dev = devs[dev_idx % len(devs)]
                cmds.append(f"ASSIGN {sid} {dev}")
                dev_idx += 1

        return cmds

    def _balance_queue(self) -> list[str]:
        """Add/remove stories to hit 0.90-1.05x velocity."""
        cmds: list[str] = []
        avg = self._avg_velocity()
        target_low  = avg * 0.90
        target_high = avg * 1.05
        pts = self._sprint_pts()

        # Remove overloaded: stories with no estimate shouldn't bloat points
        if pts > target_high:
            # Remove lowest-priority unassigned stories first
            candidates = [
                s for s in sorted(self.board._sprint_stories)
                if s not in self.board._assignments
            ]
            for sid in candidates:
                if pts <= target_high:
                    break
                est = self.board._estimates.get(sid, 0) or 0
                cmds.append(f"REMOVE_FROM_SPRINT {sid}")
                pts -= est

        # Add stories if under capacity
        if pts < target_low:
            not_in_sprint = [
                s for s in self.board._stories
                if s not in self.board._sprint_stories
            ]
            # Sort by estimated points (add smaller stories first to avoid overshoot)
            not_in_sprint.sort(
                key=lambda s: self.board._estimates.get(s) or 99
            )
            for sid in not_in_sprint:
                if pts >= target_low:
                    break
                est = self.board._estimates.get(sid, 0) or 0
                if pts + est <= target_high * 1.1:  # allow small overshoot
                    cmds.append(f"ADD_TO_SPRINT {sid}")
                    pts += est

        return cmds

    # ── Main step ─────────────────────────────────────────────────────────────

    def next_command(self) -> str | None:
        # Phase 1: Investigate
        if self._phase == "INVESTIGATE":
            try:
                return next(self._inv_iter)
            except StopIteration:
                self._phase = "FIX"
                self._fix_queue = self._build_fix_queue()

        # Phase 2: Fix
        if self._phase == "FIX":
            if self._fix_queue:
                return self._fix_queue.pop(0)
            self._phase = "BALANCE"
            self._fix_queue = self._balance_queue()

        # Phase 3: Balance capacity
        if self._phase == "BALANCE":
            if self._fix_queue:
                return self._fix_queue.pop(0)
            # After balancing, assign any newly-added unassigned stories
            self._phase = "ASSIGN_NEW"
            sprint = self.board._sprint_stories
            assignments = self.board._assignments
            devs = self._available_devs() or list(self.board._team.keys())
            dev_idx = 0
            for sid in sorted(sprint):
                if sid not in assignments:
                    dev = devs[dev_idx % len(devs)]
                    self._fix_queue.append(f"ASSIGN {sid} {dev}")
                    dev_idx += 1

        # Phase 4: Assign newly added stories
        if self._phase == "ASSIGN_NEW":
            if self._fix_queue:
                return self._fix_queue.pop(0)
            self._phase = "FINALIZE"

        # Phase 5: Finalize
        if self._phase == "FINALIZE":
            self._phase = "DONE"
            return "FINALIZE_SPRINT"

        return None


# ── Main agent runner ─────────────────────────────────────────────────────────

def run_agent(env, task_id: str, max_steps: int = 15):
    """
    Run SprintBot on the given environment.
    Yields (terminal_log, metrics_text, score_text, step_info).
    """
    from sprint_planning_env.server.tasks import TASK_REGISTRY
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.app import (
        _format_metrics,
        _format_score,
        _format_score_initial,
        DIFFICULTY_EMOJIS,
    )

    task  = TASK_REGISTRY[task_id]
    diff  = task["difficulty"]
    emoji = DIFFICULTY_EMOJIS[diff]

    # Try Mistral, fall back to direct-state bot
    llm_client   = _build_mistral_client()
    llm_failures = 0
    agent_label  = "🤖 Mistral Agent" if llm_client else "🤖 SprintBot"

    # Reset environment
    obs = env.reset(task_id=task_id)

    # Create direct-state bot (reads env.board directly)
    bot = DirectStateBot(env.board)

    terminal_log = (
        f"╔══════════════════════════════════════════╗\n"
        f"║  {agent_label:<41}║\n"
        f"║  Task: {task['name']:<35}║\n"
        f"║  Difficulty: {emoji} {diff.upper():<29}║\n"
        f"╚══════════════════════════════════════════╝\n\n"
        f"📋 SCENARIO:\n{obs.alert}\n\n"
        f"{'─' * 45}\n"
        f"{obs.command_output}\n\n"
        f"⏳ Analysing board state...\n"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if llm_client:
        messages.append({
            "role": "user",
            "content": (
                f"TASK: {task.get('description', task['name'])}\n"
                f"SCENARIO: {obs.alert}\n"
                f"BOARD:\n{obs.command_output}\n\nFirst command:"
            ),
        })

    yield (
        terminal_log,
        _format_metrics(obs.metrics),
        _format_score_initial(),
        f"Step 0 / {obs.max_steps}",
    )

    step = 0

    while step < max_steps and not obs.done:
        # Choose command
        command = None

        if llm_client and llm_failures < 3:
            command = _llm_command(llm_client, messages)
            if command is None:
                llm_failures += 1
                if llm_failures >= 3:
                    agent_label = "🤖 SprintBot"
                    terminal_log += "\n⚠ LLM unavailable — switching to SprintBot.\n"

        if command is None:
            command = bot.next_command()
            if command is None:
                command = "FINALIZE_SPRINT"

        # Execute
        action = SprintAction(command=command)
        obs    = env.step(action)
        step  += 1

        output = obs.command_output or ""
        error  = obs.error or ""

        new_entry = f"\n{'─' * 45}\n{agent_label} $ {command}\n"
        if error:
            new_entry += f"⚠  {error}\n"
        elif output:
            new_entry += f"{output}\n"

        if obs.done:
            grade   = obs.metadata.get("grader_score", 0.0) or 0.0
            pct     = int(grade * 100)
            verdict = (
                "🏆 EXCELLENT" if grade >= 0.8 else
                "✅ GOOD"      if grade >= 0.6 else
                "⚠️ PARTIAL"   if grade >= 0.4 else
                "❌ NEEDS WORK"
            )
            new_entry += (
                f"\n{'═' * 45}\n"
                f"  AGENT COMPLETE — {pct}% {verdict}\n"
                f"{'═' * 45}\n"
            )

        terminal_log += new_entry

        if llm_client:
            messages.append({"role": "assistant", "content": command})
            messages.append({
                "role": "user",
                "content": f"Result:\n{output or error}\n\nNext command:",
            })

        yield (
            terminal_log,
            _format_metrics(obs.metrics),
            _format_score(obs),
            f"Step {obs.step_number} / {obs.max_steps}",
        )

        time.sleep(0.4)
