"""
agent.py — LLM Auto-Solve agent for SprintBoard.

Uses the HF Inference API (free, no OpenAI key needed).
The HF_TOKEN is loaded from environment variables / HF Spaces secrets.
NEVER hard-code a token here.
"""

import os
import re
import time
from typing import Generator

from huggingface_hub import InferenceClient

# ── Model ────────────────────────────────────────────────────────────────────
# Use a fast, free instruction-following model available on HF Inference API.
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Scrum Master and Agile Coach.
You are solving a sprint planning task inside the SprintBoard RL environment.

RULES:
1. Issue ONE command per response, nothing else.
2. Use ONLY these valid commands:
   - LIST_BACKLOG
   - VIEW_STORY <id>
   - CHECK_DEPS <id>
   - VIEW_TEAM
   - VIEW_VELOCITY
   - VIEW_SPRINT
   - VIEW_BUGS
   - SEARCH_BACKLOG <keyword>
   - ESTIMATE <id> <points>   (points: 1,2,3,5,8,13)
   - ASSIGN <id> <developer>
   - ADD_TO_SPRINT <id>
   - REMOVE_FROM_SPRINT <id>
   - FLAG_RISK <id> <reason>
   - SET_PRIORITY <id> <P0|P1|P2>
   - FINALIZE_SPRINT
3. Start by investigating (LIST_BACKLOG, VIEW_TEAM, VIEW_VELOCITY).
4. Fix problems found (estimate stories, assign developers, balance load).
5. End with FINALIZE_SPRINT only after the sprint looks healthy.
6. Output ONLY the command, no explanation.
"""


def _build_client() -> InferenceClient | None:
    """Build HF InferenceClient using the HF_TOKEN secret."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    return InferenceClient(token=token)


def run_agent(
    env,
    task_id: str,
    max_steps: int = 15,
) -> Generator[tuple[str, str, str, str], None, None]:
    """
    Run the LLM agent on the given environment for up to max_steps.

    Yields tuples of:
        (terminal_log, metrics_text, score_text, step_info)

    The caller (Gradio) should update the UI with each yielded value.
    """
    from sprint_planning_env.server.tasks import TASK_REGISTRY
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.app import (
        _format_metrics,
        _format_score,
        _format_score_initial,
        start_task,
        DIFFICULTY_EMOJIS,
    )

    client = _build_client()
    if client is None:
        yield (
            "⚠ HF_TOKEN not set.\n\nPlease add your HF_TOKEN as a Secret in\nHugging Face Spaces → Settings → Secrets.",
            "No metrics — agent not started.",
            _format_score_initial(),
            "Step 0 / 0",
        )
        return

    # ── Reset environment ─────────────────────────────────────────────────────
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

    # ── Conversation history for the LLM ─────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user",
        "content": (
            f"TASK: {task['description']}\n\n"
            f"CURRENT BOARD STATE:\n{obs.command_output}\n\n"
            f"SCENARIO: {obs.alert}\n\n"
            f"Issue your first command:"
        ),
    })

    step = 0
    while step < max_steps and not obs.done:
        # ── Ask LLM for next command ──────────────────────────────────────────
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=64,
                temperature=0.2,
            )
            raw_cmd = response.choices[0].message.content.strip()
        except Exception as e:
            raw_cmd = f"# LLM error: {e}"

        # Extract only the first line (command), strip markdown fences
        command = raw_cmd.splitlines()[0].strip().strip("`").strip()

        # ── Execute in environment ────────────────────────────────────────────
        action = SprintAction(command=command)
        obs = env.step(action)
        step += 1

        output = obs.command_output or ""
        error  = obs.error or ""

        separator = "─" * 45
        new_entry = f"\n{separator}\n🤖 $ {command}\n"
        if error:
            new_entry += f"⚠  {error}\n"
        elif output:
            new_entry += f"{output}\n"

        if obs.done:
            grade = obs.metadata.get("grader_score", 0.0) or 0.0
            pct   = int(grade * 100)
            if grade >= 0.8:
                verdict = "🏆 EXCELLENT"
            elif grade >= 0.6:
                verdict = "✅ GOOD"
            elif grade >= 0.4:
                verdict = "⚠️ PARTIAL"
            else:
                verdict = "❌ NEEDS WORK"

            new_entry += (
                f"\n{'═' * 45}\n"
                f"  🤖 AGENT COMPLETE\n"
                f"  Final Score: {pct}%  {verdict}\n"
                f"{'═' * 45}\n"
            )

        terminal_log += new_entry

        # Add assistant turn + new board state for next iteration
        messages.append({"role": "assistant", "content": command})
        messages.append({
            "role": "user",
            "content": (
                f"Result:\n{output or error}\n\n"
                f"Continue. Issue the next command:"
            ),
        })

        yield (
            terminal_log,
            _format_metrics(obs.metrics),
            _format_score(obs),
            f"Step {obs.step_number} / {obs.max_steps}",
        )

        # Small delay so the UI feels like it's thinking in real time
        time.sleep(0.3)
