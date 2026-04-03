"""
agent.py — LLM Auto-Solve agent for SprintBoard.

Uses the HF Inference API (free serverless endpoint).
HF_TOKEN is read from environment variables / HF Spaces secrets.
NEVER hard-code any token here.
"""

import os
import re
import time

from huggingface_hub import InferenceClient

# ── Model ─────────────────────────────────────────────────────────────────────
# Qwen2.5-72B is available for free on HF serverless inference.
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# ── Valid commands for extraction fallback ─────────────────────────────────────
VALID_PREFIXES = [
    "LIST_BACKLOG", "VIEW_STORY", "CHECK_DEPS", "VIEW_TEAM",
    "VIEW_VELOCITY", "VIEW_SPRINT", "VIEW_BUGS", "VIEW_EPIC",
    "SEARCH_BACKLOG", "ESTIMATE", "ASSIGN", "UNASSIGN",
    "ADD_TO_SPRINT", "REMOVE_FROM_SPRINT", "FLAG_RISK",
    "SET_PRIORITY", "FINALIZE_SPRINT", "HELP",
]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Scrum Master solving a sprint planning task.

STRICT RULES:
- Respond with ONLY a single command, nothing else.
- No explanations, no punctuation, no markdown.
- Valid commands you may use:
  LIST_BACKLOG
  VIEW_STORY <id>
  CHECK_DEPS <id>
  VIEW_TEAM
  VIEW_VELOCITY
  VIEW_SPRINT
  ESTIMATE <id> <points>   [points must be: 1, 2, 3, 5, 8, or 13]
  ASSIGN <id> <name>
  ADD_TO_SPRINT <id>
  REMOVE_FROM_SPRINT <id>
  FLAG_RISK <id> <reason>
  FINALIZE_SPRINT

Strategy:
1. First investigate: LIST_BACKLOG, VIEW_TEAM, VIEW_VELOCITY.
2. Fix issues: estimate stories, assign developers, balance workload.
3. Only call FINALIZE_SPRINT when the sprint is healthy.

Output ONE command and nothing else."""


def _build_client() -> InferenceClient | None:
    """Build HF InferenceClient from the HF_TOKEN secret."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    return InferenceClient(token=token)


def _extract_command(raw: str) -> str:
    """
    Robustly extract a valid command from LLM output.
    Handles markdown, explanations, leading symbols, etc.
    """
    # Strip markdown code fences
    raw = re.sub(r"```[a-z]*", "", raw).strip("`").strip()

    # Try each line from the top, find the first that starts with a known prefix
    for line in raw.splitlines():
        line = line.strip().strip("`").strip("*").strip()
        for prefix in VALID_PREFIXES:
            if line.upper().startswith(prefix):
                return line.upper() if " " not in line else (
                    prefix + line[len(prefix):]  # preserve args as-is
                )

    # Last resort: return first non-empty line uppercased
    first = next((l.strip() for l in raw.splitlines() if l.strip()), "HELP")
    return first.upper()


def run_agent(env, task_id: str, max_steps: int = 15):
    """
    Run the LLM agent step-by-step.
    Yields (terminal_log, metrics_text, score_text, step_info) after each step.
    """
    from sprint_planning_env.server.tasks import TASK_REGISTRY
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.app import (
        _format_metrics,
        _format_score,
        _format_score_initial,
        DIFFICULTY_EMOJIS,
    )

    client = _build_client()
    if client is None:
        yield (
            "⚠  HF_TOKEN not set.\n\n"
            "Add it in:\n"
            "HF Spaces → Settings → Variables and Secrets → New Secret\n"
            "Name: HF_TOKEN",
            "Agent not started.",
            _format_score_initial(),
            "Step 0 / 0",
        )
        return

    # ── Reset env ─────────────────────────────────────────────────────────────
    obs = env.reset(task_id=task_id)
    task = TASK_REGISTRY[task_id]
    diff = task["difficulty"]
    emoji = DIFFICULTY_EMOJIS[diff]

    terminal_log = (
        f"╔══════════════════════════════════════════╗\n"
        f"║  🤖 AUTO-SOLVE  ·  {task['name']:<24}║\n"
        f"║  Difficulty: {emoji} {diff.upper():<29}║\n"
        f"╚══════════════════════════════════════════╝\n\n"
        f"📋 SCENARIO:\n{obs.alert}\n\n"
        f"{'─' * 45}\n"
        f"{obs.command_output}\n\n"
        f"🤖 Agent is thinking...\n"
    )

    yield (
        terminal_log,
        _format_metrics(obs.metrics),
        _format_score_initial(),
        f"Step 0 / {obs.max_steps}",
    )

    # ── Conversation history ───────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user",
        "content": (
            f"TASK GOAL: {task.get('description', task['name'])}\n\n"
            f"BOARD STATE:\n{obs.command_output}\n\n"
            f"SCENARIO: {obs.alert}\n\n"
            f"Issue your first command:"
        ),
    })

    step = 0
    consecutive_errors = 0

    while step < max_steps and not obs.done:
        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=32,
                temperature=0.1,
            )
            raw_cmd = response.choices[0].message.content.strip()
            command = _extract_command(raw_cmd)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            error_msg = str(e)[:120]
            terminal_log += f"\n⚠ API error ({consecutive_errors}): {error_msg}\n"
            if consecutive_errors >= 3:
                terminal_log += "\n❌ Too many API errors. Stopping agent.\n"
                yield (
                    terminal_log,
                    _format_metrics(obs.metrics),
                    _format_score(obs),
                    f"Step {step} / {obs.max_steps}",
                )
                return
            time.sleep(1)
            continue

        # ── Execute in environment ─────────────────────────────────────────────
        action = SprintAction(command=command)
        obs = env.step(action)
        step += 1

        output = obs.command_output or ""
        error  = obs.error or ""

        new_entry = f"\n{'─' * 45}\n🤖 $ {command}\n"
        if error:
            new_entry += f"⚠  {error}\n"
        elif output:
            new_entry += f"{output}\n"

        if obs.done:
            grade = obs.metadata.get("grader_score", 0.0) or 0.0
            pct   = int(grade * 100)
            verdict = (
                "🏆 EXCELLENT" if grade >= 0.8 else
                "✅ GOOD"      if grade >= 0.6 else
                "⚠️ PARTIAL"   if grade >= 0.4 else
                "❌ NEEDS WORK"
            )
            new_entry += (
                f"\n{'═' * 45}\n"
                f"  🤖 AGENT COMPLETE\n"
                f"  Final Score: {pct}%  {verdict}\n"
                f"{'═' * 45}\n"
            )

        terminal_log += new_entry

        # Feed result back to LLM
        messages.append({"role": "assistant", "content": command})
        messages.append({
            "role": "user",
            "content": (
                f"Result:\n{output or error}\n\n"
                f"Issue the next command:"
            ),
        })

        yield (
            terminal_log,
            _format_metrics(obs.metrics),
            _format_score(obs),
            f"Step {obs.step_number} / {obs.max_steps}",
        )

        time.sleep(0.4)
