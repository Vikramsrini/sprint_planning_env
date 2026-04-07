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
/* ── Global ── */
body, .gradio-container {
    background: #0f0f13 !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
}

/* ── Header ── */
#header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-bottom: 1px solid #7c3aed44;
    padding: 20px 32px;
    border-radius: 12px 12px 0 0;
}
#header h1 {
    color: #a78bfa;
    font-size: 2rem;
    margin: 0;
    letter-spacing: -0.5px;
}
#header p {
    color: #64748b;
    margin: 4px 0 0 0;
    font-size: 0.85rem;
}

/* ── Scenario alert box ── */
.alert-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #1a1a2e 100%);
    border: 1px solid #7c3aed55;
    border-left: 4px solid #7c3aed;
    border-radius: 8px;
    padding: 14px 16px;
    color: #c4b5fd;
    font-size: 0.82rem;
    line-height: 1.6;
}

/* ── Terminal log ── */
#terminal-log textarea {
    background: #0a0a0f !important;
    color: #22d3ee !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    padding: 12px !important;
    line-height: 1.5 !important;
}

/* ── Command input ── */
#cmd-input textarea {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    border: 2px solid #7c3aed !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
}
#cmd-input textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px #7c3aed33 !important;
}

/* ── Buttons ── */
#btn-execute {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
#btn-execute:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
    transform: translateY(-1px) !important;
}
#btn-reset {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}
#btn-reset:hover {
    background: #334155 !important;
    color: #e2e8f0 !important;
}

/* ── Score card ── */
#score-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
}

/* ── Metrics ── */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    color: #94a3b8;
    font-size: 0.78rem;
}

/* ── Task dropdown ── */
#task-select select {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* ── Step counter ── */
#step-counter {
    color: #64748b;
    font-size: 0.75rem;
    text-align: right;
    padding: 4px 8px;
}

/* ── Labels ── */
label {
    color: #7c3aed !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Diff colours in terminal ── */
.green { color: #22c55e; }
.red   { color: #ef4444; }
.blue  { color: #3b82f6; }
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
        return "No metrics yet. Start a task to see live board data."

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

    vel_bar = "█" * int(min(ratio, 2) * 10) if avg_vel else ""
    vel_icon = "🔴" if ratio > 1.15 else ("🟡" if ratio > 0.95 else "🟢")

    lines = [
        f"📊 BOARD METRICS",
        f"",
        f"Sprint Stories    : {stories}",
        f"Total Points      : {total_pts} pts",
        f"Avg Velocity      : {avg_vel} pts/sprint",
        f"Velocity Ratio    : {ratio:.2f}x  {vel_icon}",
        f"",
        f"⚠ Unestimated    : {unest}",
        f"⚠ Unassigned     : {unassigned}",
        f"⚠ Dep Issues     : {dep_issues}",
        f"⚠ Risk Flags     : {risks}",
    ]

    if overloaded:
        lines.append(f"🔴 Overloaded      : {', '.join(overloaded)}")
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
        "🏆 SCORE TRACKER",
        "",
        f"Cumulative Reward : {cum_reward:.3f}",
        f"Step Reward       : {obs.reward:.3f}" if obs.reward else "",
        "",
    ]

    if grader_score is not None:
        pct = int(grader_score * 100)
        bar_filled = "█" * (pct // 5)
        bar_empty  = "░" * (20 - pct // 5)
        lines += [
            "─── FINAL GRADE ───",
            f"Score: {pct}%",
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
            "─── GRADING ───",
            "Complete the task to",
            "see your final score.",
            "",
            "Investigation  (30%)",
            "Planning Q.   (50%)",
            "Process       (20%)",
        ]

    return "\n".join(line for line in lines if line is not None)


def _format_score_initial() -> str:
    return (
        "🏆 SCORE TRACKER\n\n"
        "Cumulative Reward : 0.000\n\n"
        "─── GRADING ───\n"
        "Complete the task to\n"
        "see your final score.\n\n"
        "Investigation  (30%)\n"
        "Planning Q.   (50%)\n"
        "Process       (20%)"
    )


def _format_sprint_manifest(board, is_done: bool = False) -> str:
    """Create a beautiful HTML manifest of the final sprint plan."""
    # Show manifest if finalized OR if the episode is done (auto-resolved)
    is_finalized = getattr(board, "is_finalized", False)
    
    if not is_finalized and not is_done:
        return "<div style='color:#64748b;text-align:center;padding:40px;'>Sprint not finalized yet. Complete your planning and click FINALIZE_SPRINT.</div>"

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
        <div style="background:#1e1b4b;border:1px solid {border_color}33;border-top:4px solid {border_color};padding:12px;border-radius:8px;">
            <div style="font-weight:bold;color:#f1f5f9;display:flex;align-items:center;">
                👤 {name} {pto_tag}
            </div>
            <div style="font-size:11px;color:#94a3b8;margin:4px 0 8px 0;">
                Points: {total_pts}/{cap} ({load_pct:.0f}%)
            </div>
            <div style="display:flex;flex-direction:column;gap:4px;">
        """
        for sid in dev_stories:
            story = board._stories.get(sid, {})
            title = story.get("title", sid)
            pts = board._estimates.get(sid, 0) or "?"
            html += f"""
                <div style="background:#0f172a;padding:6px;border-radius:4px;font-size:11px;color:#cbd5e1;">
                    <span style="color:#7c3aed;font-weight:bold;">{sid}</span> ({pts} pts)
                    <div style="color:#64748b;margin-top:2px;">{title[:25]}...</div>
                </div>
            """
        if not dev_stories:
            html += "<div style='color:#475569;font-size:11px;font-style:italic;'>No stories assigned.</div>"
        
        html += "</div></div>"
    
    html += "</div>"
    return html


# ── Build the Gradio UI ───────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)


def build_ui():
    with gr.Blocks(title="SprintBoard — Sprint Planning RL Environment") as demo:

        # ── Header ──
        gr.HTML("""
        <div id="header">
            <h1>⚡ SprintBoard</h1>
            <p>Sprint Planning & Backlog Grooming · Training Environment for LLM Agents · OpenEnv Compliant</p>
        </div>
        """)

        # ── Per-session state ──
        env_state = gr.State(_make_env)

        with gr.Row():
            # ── LEFT PANEL ────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 🎯 Select Task")

                task_selector = gr.Dropdown(
                    choices=_task_choices(),
                    value="task_1",
                    label="Task",
                    elem_id="task-select",
                    interactive=True,
                )

                btn_start = gr.Button(
                    "▶  Start Task",
                    variant="primary",
                    elem_id="btn-execute",
                )

                gr.Markdown("### 📋 Scenario")
                alert_display = gr.Textbox(
                    value="Select a task and click Start to begin.",
                    label="Current Scenario",
                    lines=7,
                    interactive=False,
                    show_label=True,
                )

                gr.Markdown("### 📊 Live Metrics")
                metrics_display = gr.Textbox(
                    value="No task started yet.",
                    label="Board Health",
                    lines=14,
                    interactive=False,
                    show_label=True,
                )

            # ── CENTER PANEL ───────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown("### 🖥️ Command Terminal")

                step_display = gr.Textbox(
                    value="Step 0 / 15",
                    label="",
                    interactive=False,
                    max_lines=1,
                    show_label=False,
                    elem_id="step-counter",
                )

                terminal_log = gr.Textbox(
                    value=(
                        "Welcome to SprintBoard!\n\n"
                        "→ Select a task from the left panel\n"
                        "→ Click 'Start Task' to begin an episode\n"
                        "→ Type planning commands below and press Execute\n\n"
                        "Example commands:\n"
                        "  LIST_BACKLOG\n"
                        "  VIEW_TEAM\n"
                        "  VIEW_VELOCITY\n"
                        "  ASSIGN US-101 Alice\n"
                        "  ESTIMATE US-103 5\n"
                        "  FINALIZE_SPRINT\n"
                    ),
                    label="Terminal Output",
                    lines=22,
                    max_lines=22,
                    interactive=False,
                    elem_id="terminal-log",
                    show_label=True,
                    autoscroll=True,
                )

                gr.Markdown("### ⌨️ Enter Command")
                with gr.Row():
                    cmd_input = gr.Textbox(
                        placeholder="e.g.  VIEW_TEAM  or  ASSIGN 101 Alice  or  HELP",
                        label="",
                        lines=1,
                        max_lines=1,
                        scale=4,
                        interactive=True,
                        elem_id="cmd-input",
                        show_label=False,
                    )
                    btn_exec = gr.Button(
                        "Execute ▶",
                        scale=1,
                        variant="primary",
                        elem_id="btn-execute",
                    )

                # Quick command buttons (one-click investigation)
                gr.Markdown("**⚡ Quick Investigation:**")
                with gr.Row():
                    q_btn_backlog = gr.Button("📋 Backlog", size="sm", variant="secondary")
                    q_btn_team    = gr.Button("👥 Team", size="sm", variant="secondary")
                    q_btn_vel     = gr.Button("📈 Velocity", size="sm", variant="secondary")
                    q_btn_sprint  = gr.Button("🔍 Sprint", size="sm", variant="secondary")
                    q_btn_bugs    = gr.Button("🐛 Bugs", size="sm", variant="secondary")

                # AI Agent row
                gr.Markdown("**🤖 AI Agent:**")
                with gr.Row():
                    btn_autosolve = gr.Button(
                        "🤖  Auto-Solve with AI",
                        variant="primary",
                        size="lg",
                        elem_id="btn-autosolve",
                    )

                # Visual Manifest Tab
                gr.Markdown("### 📜 Final Sprint Manifest")
                sprint_manifest = gr.HTML(
                    value=_format_sprint_manifest(_make_env().board, is_done=False),
                    elem_id="sprint-manifest",
                )

            # ── RIGHT PANEL ────────────────────────────────
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### 🏆 Score")
                score_display = gr.Textbox(
                    value=_format_score_initial(),
                    label="Grader",
                    lines=18,
                    interactive=False,
                    elem_id="score-box",
                    show_label=True,
                    autoscroll=True,
                )

                gr.Markdown("### 📖 Command Reference")
                gr.Markdown("""
**Investigation:**
```
LIST_BACKLOG
VIEW_STORY <id>
CHECK_DEPS <id>
VIEW_TEAM
VIEW_VELOCITY
VIEW_SPRINT
VIEW_BUGS
VIEW_EPIC <id>
SEARCH_BACKLOG <kw>
```
**Planning:**
```
ESTIMATE <id> <pts>
ASSIGN <id> <name>
UNASSIGN <id>
ADD_TO_SPRINT <id>
REMOVE_FROM_SPRINT <id>
FLAG_RISK <id> <reason>
SET_PRIORITY <id> P0|P1|P2
FINALIZE_SPRINT
```
""")

                btn_reset = gr.Button(
                    "↺  New Episode",
                    variant="secondary",
                    elem_id="btn-reset",
                )

        # ── Footer ──
        gr.HTML("""
        <div style="text-align:center;padding:16px;color:#334155;font-size:0.75rem;border-top:1px solid #1e293b;margin-top:16px;">
            SprintBoard · OpenEnv RL Environment · 15 Tasks · 3-Axis Deterministic Grading ·
            <a href="https://github.com/meta-pytorch/OpenEnv" target="_blank" style="color:#7c3aed;">OpenEnv Spec</a>
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
            is_done = "EPISODE COMPLETE" in updated_log  # heuristic for execute_command results
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

        # Auto-Solve
        def on_autosolve(task_id, env):
            from sprint_planning_env.agent import run_agent
            fresh_env = _make_env()
            for log, metrics, score, step, is_terminal in run_agent(fresh_env, task_id):
                yield log, metrics, score, step, fresh_env, _format_sprint_manifest(fresh_env.board, is_done=is_terminal)

        btn_autosolve.click(
            fn=on_autosolve,
            inputs=[task_selector, env_state],
            outputs=[terminal_log, metrics_display, score_display, step_display, env_state, sprint_manifest],
        )


    return demo



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    
    import uvicorn
    from sprint_planning_env.server.app import app as openenv_app
    
    app = gr.mount_gradio_app(openenv_app, demo, path="/")
    
    uvicorn.run(app, host="0.0.0.0", port=7860)
