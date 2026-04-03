"""
agent.py — SprintBot reactive agent for SprintBoard.

Investigation-first strategy:
  1. Gather board state (backlog, team, velocity, sprint)
  2. Parse real story IDs and issues from the board output
  3. Execute targeted fixes in the correct order
  4. Finalize sprint

Falls back to Mistral LLM if MISTRAL_API_KEY is set.
"""

import os
import re
import time

# ── Valid commands ────────────────────────────────────────────────────────────
VALID_PREFIXES = [
    "LIST_BACKLOG", "VIEW_STORY", "CHECK_DEPS", "VIEW_TEAM",
    "VIEW_VELOCITY", "VIEW_SPRINT", "VIEW_BUGS", "VIEW_EPIC",
    "SEARCH_BACKLOG", "ESTIMATE", "ASSIGN", "UNASSIGN",
    "ADD_TO_SPRINT", "REMOVE_FROM_SPRINT", "FLAG_RISK",
    "SET_PRIORITY", "FINALIZE_SPRINT", "HELP",
]

# ── Board output parsers ──────────────────────────────────────────────────────

def _parse_story_ids(text: str) -> list[str]:
    """Extract all US-XXX story IDs from board output."""
    return list(dict.fromkeys(re.findall(r"US-\d+", text)))


def _parse_unestimated(text: str) -> list[str]:
    """Find story IDs that have 0 pts or ? pts."""
    ids = []
    for line in text.splitlines():
        if re.search(r"\b0\s*pts?\b|\?\s*pts?", line, re.IGNORECASE):
            m = re.search(r"US-\d+", line)
            if m:
                ids.append(m.group())
    return list(dict.fromkeys(ids))


def _parse_unassigned(text: str) -> list[str]:
    """Find story IDs marked Unassigned."""
    ids = []
    for line in text.splitlines():
        if "unassigned" in line.lower():
            m = re.search(r"US-\d+", line)
            if m:
                ids.append(m.group())
    return list(dict.fromkeys(ids))


def _parse_sprint_ids(text: str) -> list[str]:
    """Find story IDs that are currently in the sprint."""
    ids = []
    for line in text.splitlines():
        if "[in_sprint]" in line.lower() or "sprint" in line.lower():
            m = re.search(r"US-\d+", line)
            if m:
                ids.append(m.group())
    return list(dict.fromkeys(ids))


def _parse_developers(text: str) -> list[str]:
    """Extract developer names from VIEW_TEAM output."""
    names = []
    for line in text.splitlines():
        # lines like "  Alice       : 8/34 pts"
        m = re.match(r"\s+([A-Z][a-z]+)\s*:", line)
        if m:
            names.append(m.group(1))
    return names if names else ["Alice", "Bob", "Charlie", "Diana", "Eve"]


def _parse_overloaded(text: str) -> list[str]:
    """Find developers listed as overloaded."""
    names = []
    for line in text.splitlines():
        if "overload" in line.lower() or "over capacity" in line.lower():
            m = re.search(r"\b([A-Z][a-z]+)\b", line)
            if m:
                names.append(m.group(1))
    return list(dict.fromkeys(names))


# ── Reactive heuristic planner ────────────────────────────────────────────────

class ReactiveSprintBot:
    """
    Stateful heuristic agent that reads board output before acting.

    Phases:
      INVESTIGATE → ESTIMATE → ASSIGN → BALANCE → FINALIZE
    """

    def __init__(self):
        self.phase = "INVESTIGATE"
        self.investigate_cmds = iter([
            "VIEW_VELOCITY",
            "VIEW_TEAM",
            "LIST_BACKLOG",
            "VIEW_SPRINT",
        ])
        self.fix_queue: list[str] = []
        self.developers: list[str] = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        self.dev_idx = 0
        self.sprint_ids: set[str] = set()
        self.all_ids: list[str] = []
        self.done = False

    def observe(self, output: str):
        """Update internal state from board output."""
        if not output:
            return

        # Learn developer names
        devs = _parse_developers(output)
        if devs:
            self.developers = devs

        # Accumulate sprint story IDs
        sprint = _parse_sprint_ids(output)
        self.sprint_ids.update(sprint)

        # Accumulate all story IDs seen
        all_ids = _parse_story_ids(output)
        for sid in all_ids:
            if sid not in self.all_ids:
                self.all_ids.append(sid)

        # Build fix queue from unestimated / unassigned stories
        unestimated = _parse_unestimated(output)
        unassigned   = _parse_unassigned(output)
        overloaded   = _parse_overloaded(output)

        for sid in unestimated:
            cmd = f"ESTIMATE {sid} 5"
            if cmd not in self.fix_queue:
                self.fix_queue.append(cmd)

        for sid in unassigned:
            if sid in self.sprint_ids:
                dev = self._next_dev(overloaded)
                cmd = f"ASSIGN {sid} {dev}"
                if cmd not in self.fix_queue:
                    self.fix_queue.append(cmd)

    def _next_dev(self, skip: list[str] = []) -> str:
        available = [d for d in self.developers if d not in skip]
        devs = available if available else self.developers
        dev = devs[self.dev_idx % len(devs)]
        self.dev_idx += 1
        return dev

    def next_command(self, last_output: str = "") -> str | None:
        """Return the next command to execute, or None if done."""
        if self.done:
            return None

        # Phase 1: Investigate
        if self.phase == "INVESTIGATE":
            try:
                return next(self.investigate_cmds)
            except StopIteration:
                self.phase = "FIX"

        # After investigating, parse what we've learned
        if last_output:
            self.observe(last_output)

        # Phase 2: Execute fix queue
        if self.phase == "FIX":
            if self.fix_queue:
                return self.fix_queue.pop(0)
            # If no fixes found, try adding backlog stories to fill capacity
            self.phase = "FILL"

        # Phase 3: Fill sprint capacity (add stories not yet in sprint)
        if self.phase == "FILL":
            candidates = [
                sid for sid in self.all_ids
                if sid not in self.sprint_ids
            ]
            if candidates:
                sid = candidates[0]
                self.sprint_ids.add(sid)
                return f"ADD_TO_SPRINT {sid}"
            self.phase = "FINALIZE"

        # Phase 4: Finalize
        if self.phase == "FINALIZE":
            self.done = True
            return "FINALIZE_SPRINT"

        return None


# ── Mistral LLM client ────────────────────────────────────────────────────────

def _build_mistral_client():
    """Build Mistral client from MISTRAL_API_KEY secret. Returns None if unset."""
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
    "Valid commands: LIST_BACKLOG, VIEW_TEAM, VIEW_VELOCITY, VIEW_SPRINT, "
    "VIEW_BUGS, CHECK_DEPS <id>, VIEW_STORY <id>, "
    "ESTIMATE <id> <pts>, ASSIGN <id> <name>, ADD_TO_SPRINT <id>, "
    "REMOVE_FROM_SPRINT <id>, FLAG_RISK <id> <reason>, FINALIZE_SPRINT."
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


# ── Main agent runner ─────────────────────────────────────────────────────────

def run_agent(env, task_id: str, max_steps: int = 15):
    """
    Run the SprintBot agent step-by-step.
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

    # Try Mistral first, fall back to heuristic
    llm_client   = _build_mistral_client()
    agent_label  = "🤖 Mistral Agent" if llm_client else "🤖 SprintBot"
    bot          = ReactiveSprintBot()
    llm_failures = 0

    # LLM conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Reset environment
    obs = env.reset(task_id=task_id)

    terminal_log = (
        f"╔══════════════════════════════════════════╗\n"
        f"║  {agent_label:<41}║\n"
        f"║  Task: {task['name']:<35}║\n"
        f"║  Difficulty: {emoji} {diff.upper():<29}║\n"
        f"╚══════════════════════════════════════════╝\n\n"
        f"📋 SCENARIO:\n{obs.alert}\n\n"
        f"{'─' * 45}\n"
        f"{obs.command_output}\n\n"
        f"⏳ Analysing board...\n"
    )

    # Let bot observe the initial state
    bot.observe(obs.command_output or "")

    # Seed LLM conversation
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
        # ── Choose next command ────────────────────────────────────────────────
        command = None

        if llm_client and llm_failures < 3:
            command = _llm_command(llm_client, messages)
            if command is None:
                llm_failures += 1
                if llm_failures >= 3:
                    agent_label = "🤖 SprintBot"
                    terminal_log += "\n⚠ LLM unavailable — switching to SprintBot.\n"

        if command is None:
            command = bot.next_command(obs.command_output or "")
            if command is None:
                command = "FINALIZE_SPRINT"

        # ── Execute ────────────────────────────────────────────────────────────
        action = SprintAction(command=command)
        obs    = env.step(action)
        step  += 1

        # Let reactive bot learn from this output
        bot.observe(obs.command_output or "")

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
