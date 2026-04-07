"""
SprintBoard — Gradio Playground for Hugging Face Spaces.

This file is the HF Spaces entry point. It creates a professional
interactive UI where anyone can select a sprint planning task,
type planning commands, and see the board respond in real time.
"""

import sys
import os

# Make sure the sprint_planning_env package is importable
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from typing import Optional

# In-process environment
from sprint_planning_env.server.environment import SprintBoardEnvironment
from sprint_planning_env.server.tasks import TASK_REGISTRY
from sprint_planning_env.models import SprintAction

# ── Metadata ────────────────────────────────────────────────────────

DIFFICULTY_EMOJIS = {
    "easy": "🟢",
    "medium": "🟡",
    "hard": "🔴",
}

# ── Custom CSS ──────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --primary: #5eead4;
    --primary-dim: #115e59;
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --border: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --terminal-bg: #000000;
}

body, .gradio-container {
    background: var(--bg-dark) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-main) !important;
}

.crime-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

#header {
    text-align: center;
    padding: 24px 0 32px 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 32px;
}
#header h1 {
    color: #fff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Space Grotesk', sans-serif;
}

#terminal-log-wrapper {
    height: 520px !important;
    max-height: 520px !important;
    overflow-y: auto !important;
    background: #111827 !important;
    border: 1px solid var(--border);
    border-radius: 4px !important;
    margin-bottom: 16px;
}
#terminal-log-wrapper pre {
    margin: 0;
    padding: 20px;
    color: #d1d5db;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.6;
    white-space: pre !important; 
    overflow-x: auto !important;
}

#cmd-input textarea {
    background: #000000 !important;
    color: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
}

.btn-crime {
    background: #334155 !important;
    color: #fff !important;
    border: 1px solid #475569 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 11px !important;
    transition: all 0.2s ease !important;
}
.btn-crime:hover {
    background: #475569 !important;
    border-color: var(--primary) !important;
}

.btn-action {
    background: var(--primary-dim) !important;
    color: var(--primary) !important;
    border: 1px solid var(--primary) !important;
    border-radius: 4px !important;
    height: 48px !important;
}

.metric-box {
    background: rgba(0, 0, 0, 0.2);
    border-left: 3px solid var(--primary);
    padding: 12px;
    margin-bottom: 8px;
}

label {
    color: var(--primary) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 8px !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
"""

# ── State management ─────────────────────────────────────────────────────────

def _make_env():
    return SprintBoardEnvironment()

def _task_choices():
    choices = []
    for tid, task in TASK_REGISTRY.items():
        emoji = DIFFICULTY_EMOJIS[task["difficulty"]]
        label = f"{tid} — {task['name']} [{emoji} {task['difficulty'].capitalize()}]"
        choices.append((label, tid))
    return choices

# ── Formatting functions ──────────────────────────────────────────────────────

def _render_terminal(text: str) -> str:
    """Wraps text in a pre tag for terminal-style rendering."""
    import html
    escaped = html.escape(text)
    return f'''
    <div id="terminal-log-wrapper">
        <pre id="terminal-content">{escaped}</pre>
        <script>
            (function() {{
                var el = document.getElementById("terminal-log-wrapper");
                if (el) {{
                    // Use a small delay to ensure DOM is ready
                    setTimeout(function() {{
                        el.scrollTop = el.scrollHeight;
                    }}, 10);
                }}
            }})();
        </script>
    </div>
    '''

def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return "<div style='color:#64748b;font-style:italic;padding:12px;'>Initialize task to see metrics...</div>"

    total_pts  = metrics.get('total_points', 0)
    ratio      = metrics.get('velocity_ratio', 0)
    stories    = metrics.get('sprint_stories', 0)
    unest      = metrics.get('unestimated_count', 0)
    unassigned = metrics.get('unassigned_count', 0)
    dep_issues = metrics.get('dependency_issues', 0)
    risks      = metrics.get('risk_flags', 0)

    def metric_item(label, value, warning=False):
        color = "#e2e8f0"
        if warning and value > 0: color = "#f43f5e"
        return f'''
        <div class="metric-box">
            <div style="font-size: 10px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">{label}</div>
            <div style="font-size: 16px; color: {color}; font-weight: 700;">{value}</div>
        </div>
        '''

    return f'''
    <div style="display: flex; flex-direction: column; gap: 8px;">
        {metric_item('Assigned Stories', stories)}
        {metric_item('Total Points', total_pts)}
        {metric_item('Velocity Ratio', f"{ratio:.2f}x")}
        {metric_item('Unestimated', unest, True)}
        {metric_item('Unassigned', unassigned, True)}
        {metric_item('Dep. Issues', dep_issues, True)}
        {metric_item('Risk Flags', risks, True)}
    </div>
    '''

def _format_score(obs) -> str:
    meta = getattr(obs, 'metadata', {}) or {}
    cum_reward = meta.get('cumulative_reward', 0.0)
    score = meta.get('grader_score')
    is_resolved = meta.get('is_resolved', False)

    html = f'''
    <div style="padding: 16px; text-align: center; border-bottom: 1px solid #334155; margin-bottom: 16px;">
        <div style="font-size: 11px; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">Reward</div>
        <div style="font-size: 24px; color: #fff; font-weight: 700;">{cum_reward:.3f}</div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 12px; padding: 0 16px;">
    '''
    if score is not None:
        pct = int(score * 100)
        color = '#5eead4' if score >= 0.7 else ('#fbbf24' if score >= 0.4 else '#f87171')
        html += f'''
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12px; color: #94a3b8;">Grader Score</span>
                <span style="color: {color}; font-weight: 700;">{pct}%</span>
            </div>
            <div style="background: #0f172a; height: 12px; border: 1px solid #334155; border-radius: 2px;">
                <div style="background: {color}; width: {pct}%; height: 100%;"></div>
            </div>
            <div style="text-align: center; margin-top: 8px;">
                <span style="font-size: 11px; font-weight: 700; color: {color if is_resolved else '#94a3b8'};">
                    { "[ RESOLVED ]" if is_resolved else "[ IN PROGRESS ]" }
                </span>
            </div>
        '''
    else:
        html += '<div style="color: #475569; font-size: 12px; text-align: center;">Wait for episode finish...</div>'
    html += "</div>"
    return html

def _format_score_initial() -> str:
    return _format_score(type('obj', (object,), {'metadata': None, 'reward': 0.0})())

def _format_sprint_manifest(board, is_done: bool = False) -> str:
    is_finalized = getattr(board, 'is_finalized', False)
    if not is_finalized and not is_done:
        return "<div style='color:#71717a;text-align:center;padding:40px;background:rgba(24,24,27,0.4);border:1px dashed #3f3f46;border-radius:12px;'>Sprint manifest will appear after finalization.</div>"

    stories = board._sprint_stories
    team = board._team
    assignments = board._assignments
    html = "<div style='display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:12px;'>"
    for name, info in team.items():
        dev_stories = [s for s in stories if assignments.get(s) == name]
        total_pts = sum(board._estimates.get(s, 0) or 0 for s in dev_stories)
        cap = info['capacity']
        load_pct = (total_pts / cap * 100) if cap > 0 else 0
        color = '#10b981' if load_pct <= 100 else '#ef4444'
        pto_tag = "<span style='background:#ef4444;color:white;padding:2px 4px;border-radius:3px;font-size:9px;margin-left:6px;'>PTO</span>" if name in board._pto_developers else ""
        html += f'''
        <div style="background:rgba(39,39,42,0.3); border:1px solid {color}33; border-top:4px solid {color}; padding: 12px; border-radius: 10px;">
            <div style="font-weight:800; color:#fff; font-size:13px; margin-bottom:8px;">👤 {name} {pto_tag}</div>
            <div style="font-size:10px; color:#a1a1aa; margin-bottom:10px;">Load: {total_pts}/{cap} pts</div>
            <div style="display:flex; flex-direction:column; gap:4px;">'''
        for sid in dev_stories:
            title = board._stories.get(sid, {}).get('title', sid)
            html += f'''
                <div style="background:rgba(9,9,11,0.4); padding:6px; border-radius:6px; font-size:10px; color:#e4e4e7; border:1px solid #27272a;">
                    <span style="color:#8b5cf6; font-weight:800;">{sid}</span>: {title[:20]}...
                </div>'''
        if not dev_stories:
            html += "<div style='color:#52525b; font-size:10px; font-style:italic;'>Unassigned</div>"
        html += "</div></div>"
    html += "</div>"
    return html

# ── Core Logic ────────────────────────────────────────────────────────────────

def start_task(task_id: str, env: SprintBoardEnvironment):
    obs = env.reset(task_id=task_id)
    task = TASK_REGISTRY[task_id]
    diff = task["difficulty"]
    emoji = DIFFICULTY_EMOJIS[diff]

    terminal_log = (
        f"WELCOME TO SPRINTBOARD v2.0\n"
        f"Session: Planning Episode\n"
        f"Task   : {task['name']}\n"
        f"Target : {diff.upper()} difficulty\n"
        f"─────────────────────────────────────────────\n\n"
        f"📋 SCENARIO ALERT:\n{obs.alert}\n\n"
        f"─────────────────────────────────────────────\n"
        f"{obs.command_output}\n\n"
        f"Ready for input...\n"
    )

    return (
        _render_terminal(terminal_log),
        _format_metrics(obs.metrics),
        _format_score_initial(),
        f"Step 0 / {obs.max_steps}",
        obs.alert,
        terminal_log,
    )

def execute_command(command: str, terminal_log: str, env: SprintBoardEnvironment):
    command = command.strip()
    if not command:
        return _render_terminal(terminal_log), _format_metrics(env.board.get_metrics()), _format_score_initial(), "", "", terminal_log

    action = SprintAction(command=command)
    obs = env.step(action)

    separator = "─" * 45
    new_entry = f"\n{separator}\n$ {command}\n"
    if obs.error:
        new_entry += f"⚠ ERROR: {obs.error}\n"
    elif obs.command_output:
        new_entry += f"{obs.command_output}\n"

    if obs.done:
        grade = obs.metadata.get("grader_score", 0.0) or 0.0
        pct = int(grade * 100)
        verdict = "🏆 EXCELLENT" if grade >= 0.8 else ("✅ GOOD" if grade >= 0.6 else "⚠️ NEEDS WORK")
        new_entry += f"\n{'═' * 45}\n  EPISODE COMPLETE\n  Final Score: {pct}%  {verdict}\n{'═' * 45}\n"

    updated_log = terminal_log + new_entry
    return (
        _render_terminal(updated_log),
        _format_metrics(obs.metrics),
        _format_score(obs),
        f"Step {obs.step_number} / {obs.max_steps}",
        "",
        updated_log,
    )

# ── Gradio UI Build ───────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="zinc",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

def build_ui():
    with gr.Blocks(title="SprintBoard — Detective Agent", theme=THEME, css=CSS) as demo:
        gr.HTML("""
        <div id="header">
            <h1>🔍 SPRINTBOARD — DETECTIVE AGENT</h1>
        </div>
        """)

        env_state = gr.State(_make_env)
        raw_terminal_state = gr.State("")

        with gr.Row():
            # Left Column: Search & Case Info
            with gr.Column(scale=1, min_width=300):
                with gr.Column(elem_classes=["crime-card"]):
                    gr.Markdown("### 🗃 Select Task File")
                    task_selector = gr.Dropdown(choices=_task_choices(), value="task_1", show_label=False, interactive=True)
                    btn_start = gr.Button("Initialize Task", variant="primary", elem_classes=["btn-action"])
                
                gr.Markdown("#### 📂 Task Briefing")
                alert_display = gr.Textbox(value="Select a task to see tactical overview...", show_label=False, lines=6, interactive=False)
                
                with gr.Column(elem_classes=["crime-card"]):
                    gr.Markdown("#### 🏆 Score Tracker")
                    score_display = gr.HTML(value=_format_score_initial())
                    
                gr.Markdown("#### 📈 Metrics")
                metrics_display = gr.HTML(value=_format_metrics(None))

            # Center Column: Terminal & Controls
            with gr.Column(scale=2):
                gr.Markdown("#### 🖥 Textbox")
                terminal_log = gr.HTML(
                    value=_render_terminal("Welcome to SprintBoard Lab.\nWaiting for task initialization...\n"),
                    elem_id="terminal-outer"
                )
                
                cmd_input = gr.Textbox(placeholder="Enter command here...", show_label=False, interactive=True, elem_id="cmd-input")
                
                with gr.Row():
                    btn_exec = gr.Button("Execute Command", variant="primary", elem_classes=["btn-action"])
                    btn_autosolve = gr.Button("Auto-Solve with AI 🤖", variant="primary", elem_classes=["btn-action"])
                
                with gr.Row():
                    q_btn_backlog = gr.Button("📋 Backlog", elem_classes=["btn-crime"])
                    q_btn_team    = gr.Button("👥 Team", elem_classes=["btn-crime"])
                    q_btn_vel     = gr.Button("📈 Velocity", elem_classes=["btn-crime"])
                    q_btn_sprint  = gr.Button("🔍 Sprint", elem_classes=["btn-crime"])

                with gr.Row():
                    btn_reset = gr.Button("↺ RESET EPISODE", elem_classes=["btn-crime"])
                    step_display = gr.HTML(value="<div style='color:#64748b;font-size:12px;padding:8px;font-family:\"JetBrains Mono\"'>Step 0 / 15</div>")

                with gr.Accordion("📖 Command Reference", open=False):
                    gr.Markdown("""
| Investigation Commands | Planning Commands |
| :--- | :--- |
| `LIST_BACKLOG` | `ESTIMATE <id> <points>` |
| `VIEW_STORY <id>` | `ASSIGN <id> <developer>` |
| `VIEW_TEAM` | `ADD_TO_SPRINT <id>` |
| `VIEW_VELOCITY` | `REMOVE_FROM_SPRINT <id>` |
| `CHECK_DEPS <id>` | `FINALIZE_SPRINT` |
""")

            # Right Column: Manifest Wall
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("#### 📋 Sprint Manifest")
                sprint_manifest = gr.HTML(value=_format_sprint_manifest(_make_env().board))

        # ── Event Wiring ──────────────────────────────────────────────────────

        def on_start(task_id, env):
            log_html, met, sco, stp, alrt, raw_log = start_task(task_id, env)
            return log_html, met, sco, stp, alrt, _format_sprint_manifest(env.board), raw_log

        btn_start.click(fn=on_start, inputs=[task_selector, env_state], outputs=[terminal_log, metrics_display, score_display, step_display, alert_display, sprint_manifest, raw_terminal_state])

        def on_action(cmd, raw_log, env):
            log_html, met, sco, stp, _, updated_raw_log = execute_command(cmd, raw_log, env)
            is_done = "EPISODE COMPLETE" in updated_raw_log
            return log_html, met, sco, stp, _format_sprint_manifest(env.board, is_done=is_done), "", updated_raw_log

        btn_exec.click(fn=on_action, inputs=[cmd_input, raw_terminal_state, env_state], outputs=[terminal_log, metrics_display, score_display, step_display, sprint_manifest, cmd_input, raw_terminal_state])
        cmd_input.submit(fn=on_action, inputs=[cmd_input, raw_terminal_state, env_state], outputs=[terminal_log, metrics_display, score_display, step_display, sprint_manifest, cmd_input, raw_terminal_state])

        q_btn_backlog.click(fn=lambda l, e: on_action("LIST_BACKLOG", l, e), inputs=[raw_terminal_state, env_state], outputs=[terminal_log, metrics_display, score_display, step_display, sprint_manifest, cmd_input, raw_terminal_state])
        q_btn_team.click(fn=lambda l, e: on_action("VIEW_TEAM", l, e), inputs=[raw_terminal_state, env_state], outputs=[terminal_log, metrics_display, score_display, step_display, sprint_manifest, cmd_input, raw_terminal_state])
        q_btn_vel.click(fn=lambda l, e: on_action("VIEW_VELOCITY", l, e), inputs=[raw_terminal_state, env_state], outputs=[terminal_log, metrics_display, score_display, step_display, sprint_manifest, cmd_input, raw_terminal_state])
        q_btn_sprint.click(fn=lambda l, e: on_action("VIEW_SPRINT", l, e), inputs=[raw_terminal_state, env_state], outputs=[terminal_log, metrics_display, score_display, step_display, sprint_manifest, cmd_input, raw_terminal_state])

        def on_reset(task_id):
            env = _make_env()
            log_html, met, sco, stp, alrt, raw_log = start_task(task_id, env)
            return log_html, met, sco, stp, alrt, env, _format_sprint_manifest(env.board), raw_log

        btn_reset.click(fn=on_reset, inputs=[task_selector], outputs=[terminal_log, metrics_display, score_display, step_display, alert_display, env_state, sprint_manifest, raw_terminal_state])

        def on_autosolve(task_id, env):
            from sprint_planning_env.agent import run_agent
            fresh_env = _make_env()
            for raw_log, met, sco, stp, is_terminal in run_agent(fresh_env, task_id, _format_metrics, _format_score, _format_score_initial):
                yield _render_terminal(raw_log), met, sco, stp, fresh_env, _format_sprint_manifest(fresh_env.board, is_done=is_terminal), raw_log

        btn_autosolve.click(fn=on_autosolve, inputs=[task_selector, env_state], outputs=[terminal_log, metrics_display, score_display, step_display, env_state, sprint_manifest, raw_terminal_state])

    return demo

# ── Entry Point ───────────────────────────────────────────────────────────────

import uvicorn
from sprint_planning_env.server.app import app as openenv_app

demo = build_ui()
app = gr.mount_gradio_app(openenv_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
