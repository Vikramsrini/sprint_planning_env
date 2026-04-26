"""
SprintBoard — Gradio Playground for Hugging Face Spaces.

This file is the HF Spaces entry point. It creates a professional
interactive UI where anyone can select a sprint planning task,
type planning commands, and see the board respond in real time —
no local setup required.

Layout:
  - Left panel:  Task selector, scenario alert, live board metrics
  - Center:      Terminal-style command log (all commands + outputs)
  - Right panel: Score tracker with 3-axis grading breakdown

The environment runs in-process (no HTTP server) for simplicity.
"""

import sys
import os
import json
import threading
import time
from huggingface_hub import HfApi, SpaceHardware

# Make sure the sprint_planning_env package is importable
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from typing import Optional

# In-process environment (no HTTP server needed for UI)
from sprint_planning_env.server.environment import SprintBoardEnvironment
from sprint_planning_env.server.tasks import TASK_REGISTRY
from sprint_planning_env.models import SprintAction

# ── Colour palette ──────────────────────────────────────────────────
DIFFICULTY_COLOURS = {
    "easy":   "#22c55e",   # green
    "medium": "#f59e0b",   # amber
    "hard":   "#ef4444",   # red
}

DIFFICULTY_EMOJIS = {
    "easy": "🟢",
    "medium": "🟡",
    "hard": "🔴",
}

# ── Custom CSS ──────────────────────────────────────────────────────
CSS = """
/* ── Global Dark Background ── */
body, .gradio-container {
    background: #0b0d14 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e2e8f0 !important;
}

.gradio-container .main {
    background: #0b0d14 !important;
}

/* -- Remove all default Gradio panel backgrounds -- */
.gradio-container .contain,
.gradio-container .block,
.gradio-container .wrap {
    background: transparent !important;
}

/* ── Header ── */
#header-bar {
    background: linear-gradient(135deg, #0f1729 0%, #13152a 50%, #0d1836 100%);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
#header-bar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(124, 58, 237, 0.06) 0%, transparent 70%);
    pointer-events: none;
}
#header-bar h1 {
    color: #c4b5fd;
    font-family: 'Inter', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
#header-bar p {
    color: #64748b;
    font-family: 'Inter', sans-serif;
    margin: 6px 0 0 0;
    font-size: 0.82rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* ── Glass Cards (panels) ── */
.glass-card {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 14px;
    padding: 20px;
}

/* ── Section Headers ── */
.section-header {
    color: #a78bfa;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(124, 58, 237, 0.15);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── All labels – hide default "Textbox" labels ── */
.gradio-container label span {
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Task Dropdown ── */
#task-select {
    margin-bottom: 8px;
}
#task-select select,
#task-select input,
#task-select .wrap,
#task-select .secondary-wrap {
    background: #131a2e !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(124, 58, 237, 0.25) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Terminal log ── */
#terminal-log {
    flex-grow: 1;
}
#terminal-log textarea {
    background: #080a12 !important;
    color: #67e8f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
    border: 1px solid rgba(30, 41, 59, 0.8) !important;
    border-radius: 10px !important;
    padding: 14px !important;
    line-height: 1.55 !important;
    letter-spacing: 0.01em;
}

/* ── Command input ── */
#cmd-input textarea {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    border: 2px solid rgba(124, 58, 237, 0.5) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 10px 14px !important;
    transition: all 0.25s ease !important;
}
#cmd-input textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15), 0 0 20px rgba(124, 58, 237, 0.08) !important;
}
#cmd-input textarea::placeholder {
    color: #475569 !important;
}

/* ── Buttons – Primary (Execute / Start) ── */
#btn-start, #btn-execute {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.25) !important;
    letter-spacing: 0.02em;
    min-height: 42px !important;
}
#btn-start:hover, #btn-execute:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35) !important;
}

/* ── Button – Autosolve ── */
#btn-autosolve {
    background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(8, 145, 178, 0.2) !important;
    min-height: 42px !important;
}
#btn-autosolve:hover {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(8, 145, 178, 0.3) !important;
}

/* ── Button – Secondary (Quick cmds, Reset) ── */
#btn-reset {
    background: rgba(30, 41, 59, 0.6) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(51, 65, 85, 0.6) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    min-height: 38px !important;
}
#btn-reset:hover {
    background: rgba(51, 65, 85, 0.6) !important;
    color: #e2e8f0 !important;
    border-color: rgba(124, 58, 237, 0.3) !important;
}

.quick-btn {
    background: rgba(30, 41, 59, 0.5) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(51, 65, 85, 0.5) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease !important;
    min-height: 34px !important;
}
.quick-btn:hover {
    background: rgba(124, 58, 237, 0.15) !important;
    color: #c4b5fd !important;
    border-color: rgba(124, 58, 237, 0.35) !important;
}

/* ── Score / Metrics textboxes ── */
#score-box textarea,
#metrics-box textarea,
#alert-box textarea {
    background: rgba(8, 10, 18, 0.7) !important;
    color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
    border: 1px solid rgba(30, 41, 59, 0.6) !important;
    border-radius: 10px !important;
    padding: 12px !important;
    line-height: 1.6 !important;
}

/* ── Step counter ── */
#step-counter {
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    text-align: right;
    padding: 2px 8px;
}
#step-counter textarea {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    text-align: right !important;
    font-size: 0.72rem !important;
    padding: 2px !important;
    min-height: 0 !important;
    height: auto !important;
}

/* ── Command Reference in right panel ── */
.cmd-ref-block {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(51, 65, 85, 0.4);
    border-radius: 10px;
    padding: 14px;
    margin-top: 12px;
}
.cmd-ref-block code {
    display: block !important;
    background: rgba(124, 58, 237, 0.08) !important;
    color: #c4b5fd !important;
    padding: 6px 10px !important;
    margin: 4px 0 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    line-height: 1.4 !important;
    border: 1px solid rgba(124, 58, 237, 0.1) !important;
}
.cmd-ref-block .cmd-category {
    color: #64748b;
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 12px;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(51, 65, 85, 0.3);
}
.cmd-ref-block .cmd-category:first-child {
    margin-top: 0;
}

/* ── Sprint Manifest ── */
#sprint-manifest {
    margin-top: 12px;
}

/* ── Markdown styling ── */
.gradio-container .prose h3,
.gradio-container .markdown-text h3 {
    color: #c4b5fd !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
    display: none !important;  /* hide default section headers — we use HTML instead */
}

/* ── Scrollbar styling ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0b0d14;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* ── Footer ── */
#footer-bar {
    text-align: center;
    padding: 14px;
    color: #334155;
    font-size: 0.72rem;
    border-top: 1px solid rgba(30, 41, 59, 0.5);
    margin-top: 16px;
    font-family: 'Inter', sans-serif;
}
#footer-bar a {
    color: #7c3aed;
    text-decoration: none;
}
#footer-bar a:hover {
    color: #a78bfa;
    text-decoration: underline;
}

/* ── Accordion in right panel ── */
.gradio-container .gradio-accordion {
    background: transparent !important;
    border: 1px solid rgba(51, 65, 85, 0.4) !important;
    border-radius: 10px !important;
}
.gradio-container .gradio-accordion > .label-wrap {
    background: rgba(15, 23, 42, 0.5) !important;
    color: #a78bfa !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}

/* ── Hide Gradio default elements we don't need ── */
.gradio-container footer {
    display: none !important;
}

/* ── Responsive adjustments ── */
@media (max-width: 768px) {
    #header-bar {
        padding: 16px 20px;
    }
    #header-bar h1 {
        font-size: 1.3rem;
    }
}
"""

# ── State management ─────────────────────────────────────────────────────────
# Each Gradio session holds its own environment instance via gr.State.

def _make_env():
    """Create a fresh environment instance per session."""
    return SprintBoardEnvironment()

def _task_choices():
    """Build dropdown choices: 'task_1 — Unestimated Stories [🟢 Easy]'"""
    choices = []
    for tid, task in TASK_REGISTRY.items():
        emoji = DIFFICULTY_EMOJIS[task["difficulty"]]
        label = f"{tid} — {task['name']} [{emoji} {task['difficulty'].capitalize()}]"
        choices.append((label, tid))
    return choices


class TrainingManager:
    """Manage non-blocking training runs from the app UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._log_lines = []
        self._status = "idle"
        self._mode = "none"
        self._job_id = None
        self._job_url = None
        self._started_at = None
        self._last_job_payload = None

    def _append(self, line: str) -> None:
        self._log_lines.append(line)
        # Keep memory bounded.
        if len(self._log_lines) > 400:
            self._log_lines = self._log_lines[-400:]

    def _status_header(self) -> str:
        started = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._started_at))
            if self._started_at
            else "n/a"
        )
        return (
            f"STATUS: {self._status}\n"
            f"MODE: {self._mode}\n"
            f"STARTED: {started}\n"
            f"JOB ID: {self._job_id or '-'}\n"
            f"JOB URL: {self._job_url or '-'}\n"
            f"{'-'*52}"
        )

    def render(self) -> str:
        body = "\n".join(self._log_lines[-40:]) if self._log_lines else "No training logs yet."
        payload = ""
        if self._last_job_payload is not None:
            payload = "\n" + json.dumps(self._last_job_payload, indent=2, default=str)
        return f"{self._status_header()}\n{body}{payload}"

    def _start_hf_job(
        self,
        model_id: str,
        epochs: int,
        max_samples: int,
        output_dir: str,
        hardware: str,
    ) -> tuple[bool, str]:
        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            return False, "HF_TOKEN is missing; cannot submit HF Job."

        api = HfApi(token=token)
        repo_id = os.getenv("HF_JOB_REPO_ID", "vikramsrini/sprint_planning_env").strip()
        command = [
            "bash",
            "-lc",
            (
                "set -euo pipefail && "
                f"git clone --depth 1 https://huggingface.co/spaces/{repo_id} sprint_planning_env && "
                "cd sprint_planning_env && "
                "python -m pip install -r requirements-train.txt pydantic && "
                f"python train_grpo.py --model-id {model_id} --output-dir {output_dir} "
                f"--epochs {epochs} --max-samples {max_samples} "
                "--max-completion-length 96 --num-generations 4 --learning-rate 3e-6"
            ),
        ]

        try:
            # Use official Hub Jobs API instead of raw HTTP endpoint.
            job = api.run_job(
                image="python:3.11",
                command=command,
                flavor=SpaceHardware(hardware),
                env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return False, f"HF Job create failed: {exc}"

        self._job_id = str(getattr(job, "id", "") or "")
        self._job_url = getattr(job, "url", None) or (f"https://huggingface.co/jobs/{self._job_id}" if self._job_id else None)
        self._mode = "hf_jobs"
        self._status = str(getattr(job, "status", "submitted"))
        self._started_at = time.time()
        self._last_job_payload = job.__dict__
        self._append("Submitted HF Job successfully.")
        return True, self.render()

    def start(
        self,
        model_id: str,
        epochs: int,
        max_samples: int,
        output_dir: str,
        hardware: str,
    ) -> str:
        with self._lock:
            if self._status in {"running", "submitted"}:
                return self.render()
            self._log_lines = []
            self._job_id = None
            self._job_url = None
            self._status = "starting"
            self._mode = "none"
            self._started_at = time.time()
            self._last_job_payload = None

            ok, msg = self._start_hf_job(model_id, epochs, max_samples, output_dir, hardware)
            if ok:
                return msg
            self._append(msg)
            self._status = "failed(hf_job_submit)"
            self._mode = "hf_jobs"
            self._append("HF job submission failed. Local training is disabled.")
            return self.render()

    def refresh(self) -> str:
        with self._lock:
            # HF job state.
            if self._job_id:
                token = os.getenv("HF_TOKEN", "").strip()
                if token:
                    try:
                        api = HfApi(token=token)
                        data = api.inspect_job(job_id=self._job_id)
                        new_status = getattr(data, "status", None)
                        if new_status:
                            self._status = str(new_status)
                        self._last_job_payload = data.__dict__
                    except Exception as exc:  # pragma: no cover - network/runtime dependent
                        self._append(f"HF status refresh warning: {exc}")
            return self.render()

    def stop(self) -> str:
        with self._lock:
            if self._job_id:
                self._append("HF job stop is not wired in this app yet; stop from HF UI.")
            else:
                self._append("No active training run.")
            return self.render()


TRAINING_MANAGER = TrainingManager()

# ── Core logic ────────────────────────────────────────────────────────────────

def start_task(task_id: str, env: SprintBoardEnvironment):
    """Reset environment and start a new task episode."""
    obs = env.reset(task_id=task_id)
    task = TASK_REGISTRY[task_id]
    diff = task["difficulty"]
    colour = DIFFICULTY_COLOURS[diff]
    emoji = DIFFICULTY_EMOJIS[diff]

    terminal_log = (
        f"╔══════════════════════════════════════════╗\n"
        f"║  SprintBoard  ·  {task['name']:<26}║\n"
        f"║  Difficulty: {emoji} {diff.upper():<30}║\n"
        f"╚══════════════════════════════════════════╝\n\n"
        f"📋 SCENARIO:\n{obs.alert}\n\n"
        f"─────────────────────────────────────────────\n"
        f"{obs.command_output}\n\n"
        f"Type your first command below ↓\n"
    )

    metrics_text = _format_metrics(obs.metrics)
    score_text = _format_score_initial()
    step_info = f"Step 0 / {obs.max_steps}"

    return (
        terminal_log,
        metrics_text,
        score_text,
        step_info,
        obs.alert,  # store alert in state
    )


def execute_command(
    command: str,
    terminal_log: str,
    env: SprintBoardEnvironment,
):
    """Execute a planning command and update the UI."""
    command = command.strip()
    if not command:
        return terminal_log, "", "", "⚠ Please enter a command.", ""

    action = SprintAction(command=command)
    obs = env.step(action)

    # Build terminal entry
    separator = "─" * 45
    cmd_line   = f"$ {command}"
    output     = obs.command_output or ""
    error      = obs.error or ""

    new_entry = f"\n{separator}\n{cmd_line}\n"
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
            f"  EPISODE COMPLETE\n"
            f"  Final Score: {pct}%  {verdict}\n"
            f"{'═' * 45}\n"
        )

    updated_log = terminal_log + new_entry
    metrics_text = _format_metrics(obs.metrics)
    score_text = _format_score(obs)
    step_info = f"Step {obs.step_number} / {obs.max_steps}"

    # Clear input after execution
    cleared_cmd = ""
    return updated_log, metrics_text, score_text, step_info, cleared_cmd


def _format_metrics(metrics: dict) -> str:
    """Format board metrics as a readable string."""
    if not metrics:
        return "Initialize task to see metrics..."

    total_pts  = metrics.get("total_points", 0)
    avg_vel    = metrics.get("avg_velocity", 0)
    ratio      = metrics.get("velocity_ratio", 0)
    stories    = metrics.get("sprint_stories", 0)
    unest      = metrics.get("unestimated_count", 0)
    unassigned = metrics.get("unassigned_count", 0)
    overloaded = metrics.get("overloaded_developers", [])
    dep_issues = metrics.get("dependency_issues", 0)
    risks      = metrics.get("risk_flags", 0)
    finalized  = metrics.get("finalized", False)

    vel_icon = "🔴" if ratio > 1.15 else ("🟡" if ratio > 0.95 else "🟢")

    lines = [
        f"■ Sprint Stories : {stories}",
        f"■ Total Points   : {total_pts} pts",
        f"♦ Velocity Ratio : {ratio:.2f}x  {vel_icon}",
        f"",
        f"  Unestimated    : {unest}",
        f"  Unassigned     : {unassigned}",
        f"⚠ Dep Issues     : {dep_issues}",
        f"⚠ Risk Flags     : {risks}",
    ]

    if overloaded:
        lines.append(f"🔴 Overloaded    : {', '.join(overloaded)}")
    if finalized:
        lines.append(f"\n✅ Sprint FINALIZED")

    return "\n".join(lines)


def _format_score(obs) -> str:
    """Format score display after each step."""
    meta = obs.metadata or {}
    cum_reward = meta.get("cumulative_reward", 0.0)
    grader_score = meta.get("grader_score")
    is_resolved = meta.get("is_resolved", False)

    lines = [
        f"CUMULATIVE REWARD : {cum_reward:.3f}",
        f"STEP REWARD       : {obs.reward:.3f}" if obs.reward else "",
        "",
    ]

    if grader_score is not None:
        pct = int(grader_score * 100)
        bar_filled = "█" * (pct // 5)
        bar_empty  = "░" * (20 - pct // 5)
        lines += [
            f"FINAL SCORE: {pct}%",
            f"[{bar_filled}{bar_empty}]",
            "",
            f"Investigation  (30%): {'●' * int(grader_score * 3)}",
            f"Planning Q.   (50%): {'●' * int(grader_score * 5)}",
            f"Process       (20%): {'●' * int(grader_score * 2)}",
            "",
            "✅ RESOLVED" if is_resolved else "🔄 IN PROGRESS",
        ]
    else:
        lines += [
            "Awaiting final grade...",
        ]

    return "\n".join(line for line in lines if line is not None)


def _format_score_initial() -> str:
    return (
        "CUMULATIVE REWARD : 0.000\n\n"
        "Awaiting final grade..."
    )


def _format_sprint_manifest(board, is_done: bool = False) -> str:
    """Create a beautiful HTML manifest of the final sprint plan."""
    is_finalized = getattr(board, "is_finalized", False)

    if not is_finalized and not is_done:
        return """
        <div style="
            color: #475569;
            text-align: center;
            padding: 30px 20px;
            border: 1px dashed rgba(71, 85, 105, 0.3);
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.3);
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            line-height: 1.6;
        ">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">📜</div>
            Sprint manifest will appear after finalization.
        </div>
        """

    stories = board._sprint_stories
    team = board._team
    assignments = board._assignments

    html = "<div style='display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:12px;'>"

    for name, info in team.items():
        dev_stories = [s for s in stories if assignments.get(s) == name]
        total_pts = sum(board._estimates.get(s, 0) or 0 for s in dev_stories)
        cap = info["capacity"]
        load_pct = (total_pts / cap * 100) if cap > 0 else 0
        border_color = "#22c55e" if load_pct <= 100 else "#ef4444"
        pto_tag = "<span style='background:#ef4444;color:white;padding:2px 6px;border-radius:4px;font-size:10px;margin-left:8px;'>PTO</span>" if name in board._pto_developers else ""

        html += f"""
        <div style="background:rgba(30,27,75,0.6);border:1px solid {border_color}33;border-top:3px solid {border_color};padding:12px;border-radius:10px;">
            <div style="font-weight:bold;color:#f1f5f9;display:flex;align-items:center;font-family:'Inter',sans-serif;font-size:0.85rem;">
                👤 {name} {pto_tag}
            </div>
            <div style="font-size:0.7rem;color:#94a3b8;margin:4px 0 8px 0;font-family:'JetBrains Mono',monospace;">
                Load: {total_pts}/{cap} pts
            </div>
            <div style="display:flex;flex-direction:column;gap:4px;">
        """
        for sid in dev_stories:
            story = board._stories.get(sid, {})
            title = story.get("title", sid)
            pts = board._estimates.get(sid, 0) or "?"
            html += f"""
                <div style="background:rgba(15,23,42,0.6);padding:6px 8px;border-radius:6px;font-size:0.7rem;color:#cbd5e1;font-family:'JetBrains Mono',monospace;">
                    <span style="color:#a78bfa;font-weight:bold;">{sid}</span> ({pts} pts)
                    <div style="color:#64748b;margin-top:2px;font-size:0.65rem;">{title[:28]}...</div>
                </div>
            """
        if not dev_stories:
            html += "<div style='color:#475569;font-size:0.7rem;font-style:italic;font-family:Inter,sans-serif;'>No stories assigned.</div>"

        html += "</div></div>"

    html += "</div>"
    return html


# ── Build the Gradio UI ───────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    # Dark background overrides
    body_background_fill="#0b0d14",
    body_background_fill_dark="#0b0d14",
    block_background_fill="rgba(15, 23, 42, 0.65)",
    block_background_fill_dark="rgba(15, 23, 42, 0.65)",
    block_border_color="rgba(124, 58, 237, 0.12)",
    block_border_color_dark="rgba(124, 58, 237, 0.12)",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_label_border_color="transparent",
    block_label_border_color_dark="transparent",
    block_radius="14px",
    block_shadow="0 4px 24px rgba(0, 0, 0, 0.2)",
    block_title_text_color="#c4b5fd",
    block_title_text_color_dark="#c4b5fd",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    border_color_primary="rgba(124, 58, 237, 0.2)",
    border_color_primary_dark="rgba(124, 58, 237, 0.2)",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed, #6d28d9)",
    button_primary_background_fill_dark="linear-gradient(135deg, #7c3aed, #6d28d9)",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="rgba(30, 41, 59, 0.6)",
    button_secondary_background_fill_dark="rgba(30, 41, 59, 0.6)",
    button_secondary_text_color="#94a3b8",
    button_secondary_text_color_dark="#94a3b8",
    button_secondary_border_color="rgba(51, 65, 85, 0.5)",
    button_secondary_border_color_dark="rgba(51, 65, 85, 0.5)",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="rgba(124, 58, 237, 0.3)",
    input_border_color_dark="rgba(124, 58, 237, 0.3)",
    input_radius="10px",
    input_placeholder_color="#475569",
    input_placeholder_color_dark="#475569",
    panel_background_fill="rgba(15, 23, 42, 0.5)",
    panel_background_fill_dark="rgba(15, 23, 42, 0.5)",
    panel_border_color="rgba(124, 58, 237, 0.1)",
    panel_border_color_dark="rgba(124, 58, 237, 0.1)",
    shadow_drop="0 4px 14px rgba(0, 0, 0, 0.15)",
    shadow_drop_lg="0 8px 28px rgba(0, 0, 0, 0.2)",
)


def build_ui():
    with gr.Blocks(title="SprintBoard — Sprint Planning RL Environment") as demo:

        # ── Font loader + CSS fallback ──
        gr.HTML("""
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap">
        """)

        # ── Header ──
        gr.HTML("""
        <div id="header-bar">
            <h1>⚡ SprintBoard</h1>
            <p>Sprint Planning & Backlog Management · Auto-Solve runs <strong>Qwen2.5 + your LoRA</strong> (see <code>SPRINTBOARD_ADAPTER_ID</code>)</p>
        </div>
        """)

        # ── Per-session state ──
        env_state = gr.State(_make_env)

        with gr.Row(equal_height=False):
            # ── LEFT PANEL ────────────────────────────────
            with gr.Column(scale=1, min_width=260):

                # -- Session Control --
                gr.HTML("<div class='section-header'>🎯 Session Control</div>")

                task_selector = gr.Dropdown(
                    choices=_task_choices(),
                    value="task_1",
                    label="Select Task",
                    elem_id="task-select",
                    interactive=True,
                )

                btn_start = gr.Button(
                    "▶  START TASK",
                    variant="primary",
                    elem_id="btn-start",
                )

                # -- Scenario Alert --
                gr.HTML("<div class='section-header' style='margin-top:16px;'>📋 Scenario Alert</div>")
                alert_display = gr.Textbox(
                    value="Initialize to see scenario.",
                    label="",
                    lines=5,
                    interactive=False,
                    show_label=False,
                    elem_id="alert-box",
                )

                # -- Board Metrics --
                gr.HTML("<div class='section-header' style='margin-top:16px;'>📊 Board Metrics</div>")
                metrics_display = gr.Textbox(
                    value="Initialize task to see metrics...",
                    label="",
                    lines=10,
                    interactive=False,
                    show_label=False,
                    elem_id="metrics-box",
                )

            # ── CENTER PANEL ───────────────────────────────
            with gr.Column(scale=3, min_width=480):

                gr.HTML("<div class='section-header'>🖥️ Command Terminal</div>")

                terminal_log = gr.Textbox(
                    value=(
                        "Welcome to SprintBoard.\n"
                        "Select a task and click Start Task to begin.\n"
                    ),
                    label="",
                    lines=20,
                    max_lines=20,
                    interactive=False,
                    elem_id="terminal-log",
                    show_label=False,
                    autoscroll=True,
                )

                # Command input row
                with gr.Row():
                    cmd_input = gr.Textbox(
                        placeholder="Enter planning command...",
                        label="",
                        lines=1,
                        max_lines=1,
                        scale=4,
                        interactive=True,
                        elem_id="cmd-input",
                        show_label=False,
                    )
                    btn_exec = gr.Button(
                        "EXECUTE ▶",
                        scale=1,
                        variant="primary",
                        elem_id="btn-execute",
                    )
                    btn_autosolve = gr.Button(
                        "🤖 Auto-Solve (Qwen+LoRA)",
                        scale=1,
                        variant="primary",
                        elem_id="btn-autosolve",
                    )

                # Quick command buttons
                with gr.Row():
                    q_btn_backlog = gr.Button("📋 Backlog", size="sm", variant="secondary", elem_classes=["quick-btn"])
                    q_btn_team    = gr.Button("👥 Team", size="sm", variant="secondary", elem_classes=["quick-btn"])
                    q_btn_vel     = gr.Button("📈 Velocity", size="sm", variant="secondary", elem_classes=["quick-btn"])
                    q_btn_sprint  = gr.Button("🔍 Sprint", size="sm", variant="secondary", elem_classes=["quick-btn"])
                    q_btn_bugs    = gr.Button("🐛 Bugs", size="sm", variant="secondary", elem_classes=["quick-btn"])

                # Step counter + Reset
                with gr.Row():
                    btn_reset = gr.Button(
                        "↺ RESET EPISODE",
                        variant="secondary",
                        elem_id="btn-reset",
                        scale=2,
                    )
                    step_display = gr.Textbox(
                        value="Step 0 / 15",
                        label="",
                        interactive=False,
                        max_lines=1,
                        show_label=False,
                        elem_id="step-counter",
                        scale=1,
                    )

                # Sprint Manifest
                gr.HTML("<div class='section-header' style='margin-top:16px;'>📜 Sprint Manifest</div>")
                sprint_manifest = gr.HTML(
                    value=_format_sprint_manifest(_make_env().board, is_done=False),
                    elem_id="sprint-manifest",
                )

                # Command Reference
                gr.HTML("""
                <div class='section-header' style='margin-top:16px;'>📖 Command Reference</div>
                <div class='cmd-ref-block'>
                    <div class='cmd-category'>Investigation</div>
                    <div><code>LIST_BACKLOG</code></div>
                    <div><code>VIEW_STORY  &lt;id&gt;</code></div>
                    <div><code>CHECK_DEPS  &lt;id&gt;</code></div>
                    <div><code>VIEW_TEAM</code></div>
                    <div><code>VIEW_VELOCITY</code></div>
                    <div><code>VIEW_SPRINT</code></div>
                    <div><code>VIEW_BUGS</code></div>
                    <div><code>VIEW_EPIC   &lt;id&gt;</code></div>
                    <div><code>SEARCH_BACKLOG &lt;kw&gt;</code></div>
                    <div class='cmd-category' style='margin-top:16px;'>Planning</div>
                    <div><code>ESTIMATE &lt;id&gt; &lt;pts&gt;</code></div>
                    <div><code>ASSIGN   &lt;id&gt; &lt;name&gt;</code></div>
                    <div><code>UNASSIGN &lt;id&gt;</code></div>
                    <div><code>ADD_TO_SPRINT &lt;id&gt;</code></div>
                    <div><code>REMOVE_FROM_SPRINT &lt;id&gt;</code></div>
                    <div><code>SET_PRIORITY &lt;id&gt; P0|P1|P2</code></div>
                    <div><code>FLAG_RISK &lt;id&gt; &lt;reason&gt;</code></div>
                    <div><code>FINALIZE_SPRINT</code></div>
                </div>
                """)

            # ── RIGHT PANEL ────────────────────────────────
            with gr.Column(scale=1, min_width=260):

                gr.HTML("<div class='section-header'>🏆 Score Tracker</div>")
                score_display = gr.Textbox(
                    value=_format_score_initial(),
                    label="",
                    lines=10,
                    interactive=False,
                    elem_id="score-box",
                    show_label=False,
                    autoscroll=True,
                )

                gr.HTML("<div class='section-header' style='margin-top:16px;'>🧪 Training (HF Jobs only)</div>")
                train_model_id = gr.Textbox(
                    value="Qwen/Qwen2.5-1.5B-Instruct",
                    label="Model ID",
                    lines=1,
                )
                with gr.Row():
                    train_epochs = gr.Number(value=3, precision=0, label="Epochs")
                    train_samples = gr.Number(value=60, precision=0, label="Max Samples")
                train_output_dir = gr.Textbox(
                    value="runs/sprintboard-grpo",
                    label="Output Dir",
                    lines=1,
                )
                train_hardware = gr.Dropdown(
                    choices=["cpu-basic", "t4-small", "a10g-small", "a100-large"],
                    value="t4-small",
                    label="HF Hardware",
                )
                with gr.Row():
                    btn_train_start = gr.Button("▶ Start Training", variant="primary")
                    btn_train_refresh = gr.Button("↻ Refresh", variant="secondary")
                    btn_train_stop = gr.Button("■ Stop", variant="secondary")
                train_status = gr.Textbox(
                    value=TRAINING_MANAGER.render(),
                    label="Training Status",
                    lines=10,
                    interactive=False,
                    autoscroll=True,
                )

        # ── Footer ──
        gr.HTML("""
        <div id="footer-bar">
            SprintBoard · OpenEnv RL Environment · 15 Tasks · 3-Axis Deterministic Grading ·
            <a href="https://github.com/meta-pytorch/OpenEnv" target="_blank">OpenEnv Spec</a>
        </div>
        """)

        # ── Event wiring ──────────────────────────────────────────────────────

        # Start Button
        def on_start(task_id, env):
            log, metrics, score, step, alert = start_task(task_id, env)
            return log, metrics, score, step, alert, _format_sprint_manifest(env.board, is_done=False)

        btn_start.click(
            fn=on_start,
            inputs=[task_selector, env_state],
            outputs=[terminal_log, metrics_display, score_display, step_display, alert_display, sprint_manifest],
        )

        # Execute Action
        def on_action_execute(command, log, env):
            updated_log, metrics, score, step, _ = execute_command(command, log, env)
            is_done = "EPISODE COMPLETE" in updated_log
            return updated_log, metrics, score, step, _format_sprint_manifest(env.board, is_done=is_done)

        btn_exec.click(
            fn=on_action_execute,
            inputs=[cmd_input, terminal_log, env_state],
            outputs=[terminal_log, metrics_display, score_display, step_display, sprint_manifest],
        )
        cmd_input.submit(
            fn=on_action_execute,
            inputs=[cmd_input, terminal_log, env_state],
            outputs=[terminal_log, metrics_display, score_display, step_display, sprint_manifest],
        )

        # Quick investigation buttons
        def on_quick_cmd(cmd, log, env):
            updated_log, metrics, score, step, _ = execute_command(cmd, log, env)
            is_done = "EPISODE COMPLETE" in updated_log
            return updated_log, metrics, score, step, _format_sprint_manifest(env.board, is_done=is_done)

        outputs_list = [terminal_log, metrics_display, score_display, step_display, sprint_manifest]
        q_btn_backlog.click(fn=lambda l, e: on_quick_cmd("LIST_BACKLOG", l, e), inputs=[terminal_log, env_state], outputs=outputs_list)
        q_btn_team.click(fn=lambda l, e: on_quick_cmd("VIEW_TEAM", l, e) , inputs=[terminal_log, env_state], outputs=outputs_list)
        q_btn_vel.click(fn=lambda l, e: on_quick_cmd("VIEW_VELOCITY", l, e) , inputs=[terminal_log, env_state], outputs=outputs_list)
        q_btn_sprint.click(fn=lambda l, e: on_quick_cmd("VIEW_SPRINT", l, e) , inputs=[terminal_log, env_state], outputs=outputs_list)
        q_btn_bugs.click(fn=lambda l, e: on_quick_cmd("VIEW_BUGS", l, e) , inputs=[terminal_log, env_state], outputs=outputs_list)

        # Reset button
        def on_reset(task_id):
            fresh_env = _make_env()
            log, metrics, score, step, alert = start_task(task_id, fresh_env)
            return log, metrics, score, step, alert, fresh_env, _format_sprint_manifest(fresh_env.board, is_done=False)

        btn_reset.click(
            fn=on_reset,
            inputs=[task_selector],
            outputs=[terminal_log, metrics_display, score_display, step_display, alert_display, env_state, sprint_manifest],
        )

        # Auto-Solve — Qwen2.5 + LoRA (`llm_autosolve`) or heuristic fallback
        def on_autosolve(task_id, env):
            from sprint_planning_env.llm_autosolve import run_llm_agent

            fresh_env = _make_env()
            for log, metrics, score, step, is_terminal in run_llm_agent(
                fresh_env,
                task_id,
                _format_metrics,
                _format_score,
                _format_score_initial,
            ):
                yield log, metrics, score, step, fresh_env, _format_sprint_manifest(
                    fresh_env.board, is_done=is_terminal
                )

        btn_autosolve.click(
            fn=on_autosolve,
            inputs=[task_selector, env_state],
            outputs=[terminal_log, metrics_display, score_display, step_display, env_state, sprint_manifest],
        )

        # Training controls
        def on_train_start(model_id, epochs, samples, output_dir, hardware):
            return TRAINING_MANAGER.start(
                model_id=model_id.strip(),
                epochs=int(epochs),
                max_samples=int(samples),
                output_dir=output_dir.strip(),
                hardware=str(hardware),
            )

        btn_train_start.click(
            fn=on_train_start,
            inputs=[train_model_id, train_epochs, train_samples, train_output_dir, train_hardware],
            outputs=[train_status],
        )
        btn_train_refresh.click(fn=lambda: TRAINING_MANAGER.refresh(), inputs=[], outputs=[train_status])
        btn_train_stop.click(fn=lambda: TRAINING_MANAGER.stop(), inputs=[], outputs=[train_status])


    return demo



# ── Entry point ───────────────────────────────────────────────────────────────

# Import the OpenEnv API app (has /reset, /step, /ws, /tasks, /grader endpoints)
from sprint_planning_env.server.app import app

# Build and mount Gradio UI onto the same FastAPI server
demo = build_ui()
app = gr.mount_gradio_app(app, demo, path="/", theme=THEME, css=CSS)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
