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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --primary: #8b5cf6;
    --primary-hover: #7c3aed;
    --bg-dark: #09090b;
    --card-bg: rgba(24, 24, 27, 0.7);
    --border: rgba(39, 39, 42, 0.8);
    --text-main: #f4f4f5;
    --text-muted: #a1a1aa;
    --terminal-bg: #000000;
    --terminal-cyan: #22d3ee;
}

body, .gradio-container {
    background: var(--bg-dark) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main) !important;
}

.panel-glass {
    background: var(--card-bg) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

#header {
    background: linear-gradient(135deg, #1e1b4b 0%, #09090b 100%);
    border: 1px solid rgba(124, 58, 237, 0.2);
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 24px;
    text-align: left;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
#header h1 {
    color: #fff;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -2px;
    background: linear-gradient(to right, #ffffff 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#header p {
    color: var(--text-muted);
    margin: 8px 0 0 0;
    font-size: 1.1rem;
    font-weight: 500;
}

.terminal-window {
    background: var(--terminal-bg) !important;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #3f3f46;
    display: flex;
    flex-direction: column;
}
.terminal-header {
    background: #18181b;
    padding: 10px 16px;
    display: flex;
    gap: 8px;
    align-items: center;
    border-bottom: 1px solid #27272a;
}
.dot { width: 12px; height: 12px; border-radius: 50%; }
.dot.red { background: #ff5f56; }
.dot.amber { background: #ffbd2e; }
.dot.green { background: #27c93f; }
.terminal-title {
    color: #71717a;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-left: 4px;
}

#terminal-log-wrapper {
    height: 520px !important;
    overflow-y: auto !important;
    padding: 0 !important;
    background: #000 !important;
    scrollbar-width: thin;
    scrollbar-color: #27272a transparent;
}
#terminal-log-wrapper pre {
    margin: 0;
    padding: 24px;
    color: var(--terminal-cyan);
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.6;
    white-space: pre !important;
    overflow-x: auto;
    background: transparent !important;
}

#cmd-input textarea {
    background: #18181b !important;
    color: #fff !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 14px 18px !important;
    font-size: 14px !important;
    transition: all 0.2s ease;
}
#cmd-input textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
}

.btn-primary {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3) !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    background: var(--primary-hover) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
}
.btn-secondary {
    background: #18181b !important;
    color: var(--text-main) !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
.btn-secondary:hover {
    background: #27272a !important;
    border-color: #52525b !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #27272a; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #3f3f46; }

label {
    color: var(--primary) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 8px !important;
    display: block !important;
}
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
                    el.scrollTop = el.scrollHeight;
                    // Double check after a frame for layout changes
                    requestAnimationFrame(() => {{ el.scrollTop = el.scrollHeight; }});
                }}
            }})();
        </script>
    </div>
    '''

def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return "<div style='color:#71717a;font-style:italic;padding:20px;text-align:center;'>Start a task to see metrics...</div>"

    total_pts  = metrics.get('total_points', 0)
    ratio      = metrics.get('velocity_ratio', 0)
    stories    = metrics.get('sprint_stories', 0)
    unest      = metrics.get('unestimated_count', 0)
    unassigned = metrics.get('unassigned_count', 0)
    dep_issues = metrics.get('dependency_issues', 0)
    risks      = metrics.get('risk_flags', 0)
    overloaded = metrics.get('overloaded_developers', [])
    finalized  = metrics.get('finalized', False)

    vel_color = '#22c55e' if ratio <= 1.0 else ('#f59e0b' if ratio <= 1.15 else '#ef4444')
    
    def metric_pill(label, value, icon, color=None, warning=False):
        bg = color if color else 'rgba(39, 39, 42, 0.5)'
        if warning and value > 0:
            bg = 'rgba(127, 29, 29, 0.4)'
            border = '1px solid rgba(239, 68, 68, 0.4)'
        else:
            border = '1px solid rgba(63, 63, 70, 0.4)'
        return f'''
        <div style="background: {bg}; border: {border}; padding: 10px; border-radius: 10px; flex: 1; min-width: 80px;">
            <div style="font-size: 9px; color: #a1a1aa; font-weight: 700; text-transform: uppercase;">{icon} {label}</div>
            <div style="font-size: 1.1rem; color: #fff; font-weight: 800;">{value}</div>
        </div>
        '''

    html = f'''
    <div style="display: flex; flex-direction: column; gap: 10px;">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
            {metric_pill('Stories', stories, '📋')}
            {metric_pill('Pts', total_pts, '🎯')}
            {metric_pill('Velocity', f'{ratio:.2f}x', '⚡', color=f'{vel_color}33')}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
            {metric_pill('Estimates', unest, '❓', warning=True)}
            {metric_pill('Assigns', unassigned, '👤', warning=True)}
            {metric_pill('Deps', dep_issues, '🔗', warning=True)}
            {metric_pill('Risks', risks, '🚩', warning=True)}
        </div>
    '''
    if overloaded:
        html += f"<div style='background:rgba(127,29,29,0.2); border:1px solid rgba(239,68,68,0.3); padding:8px; border-radius:8px; font-size:10px; color:#fca5a5;'>🔴 <b>Overloaded:</b> {', '.join(overloaded)}</div>"
    if finalized:
        html += "<div style='background:rgba(16,185,129,0.2); border:1px solid #10b98144; padding:10px; border-radius:8px; font-size:12px; color:#34d399; text-align:center; font-weight:800;'>✅ SPRINT FINALIZED</div>"
    html += "</div>"
    return html

def _format_score(obs) -> str:
    meta = getattr(obs, 'metadata', {}) or {}
    cum_reward = meta.get('cumulative_reward', 0.0)
    score = meta.get('grader_score')
    is_resolved = meta.get('is_resolved', False)

    def progress_bar(label, value, weight, color='#8b5cf6'):
        pct = int(value * 100) if value is not None else 0
        return f'''
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #a1a1aa; margin-bottom: 4px;">
                <span>{label}</span>
                <span>{pct}%</span>
            </div>
            <div style="background: #27272a; height: 6px; border-radius: 3px; overflow: hidden;">
                <div style="background: {color}; width: {pct}%; height: 100%; transition: width 0.8s ease-out;"></div>
            </div>
        </div>
        '''

    html = f'''
    <div style="background: rgba(24, 24, 27, 0.8); border-radius: 12px; padding: 20px; border: 1px solid #3f3f46;">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 10px; color: #8b5cf6; font-weight: 900; letter-spacing: 2px; text-transform: uppercase;">Reward</div>
            <div style="font-size: 2.2rem; color: #fff; font-weight: 900;">{cum_reward:.3f}</div>
        </div>
    '''
    if score is not None:
        pct = int(score * 100)
        color = '#10b981' if score >= 0.6 else '#f59e0b'
        html += f'''
        <div style="border-top: 1px solid #3f3f46; padding-top: 20px;">
            <div style="text-align: center; margin-bottom: 20px; color: {color};">
                <div style="font-size: 1.8rem; font-weight: 900;">{pct}%</div>
            </div>
            {progress_bar("Investigation (30%)", score, 30, color)}
            {progress_bar("Planning (50%)", score, 50, color)}
            {progress_bar("Process (20%)", score, 20, color)}
            <div style="text-align: center; margin-top: 12px;">
                <span style="background: {'rgba(16,185,129,0.2)' if is_resolved else '#27272a'}; color: {'#34d399' if is_resolved else '#71717a'}; padding: 4px 12px; border-radius: 999px; font-size: 10px; font-weight: 800;">
                    { "✅ RESOLVED" if is_resolved else "🔄 IN PROGRESS" }
                </span>
            </div>
        </div>
        '''
    else:
        html += f'''<div style="border-top: 1px solid #3f3f46; padding-top: 20px; opacity: 0.4;">
            {progress_bar("Investigation", 0, 30)}
            {progress_bar("Planning", 0, 50)}
            {progress_bar("Process", 0, 20)}
        </div>'''
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
        return _render_terminal(terminal_log), _format_metrics(env.board.metrics), _format_score_initial(), "", "", terminal_log

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

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="zinc",
    neutral_hue="zinc",
)

def build_ui():
    with gr.Blocks(title="SprintBoard — Modern RL Environment", theme=THEME, css=CSS) as demo:
        gr.HTML("""
        <div id="header">
            <h1>⚡ SprintBoard</h1>
            <p>Professional Sprint Planning & Backlog Management · LLM Agent Training Grounds</p>
        </div>
        """)

        env_state = gr.State(_make_env)
        raw_terminal_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with gr.Column(elem_classes=["panel-glass"]):
                    gr.Markdown("### 🎯 Session Control")
                    task_selector = gr.Dropdown(choices=_task_choices(), value="task_1", label="Active Task", interactive=True)
                    btn_start = gr.Button("▶  INITIALIZE ENVIRONMENT", variant="primary", elem_classes=["btn-primary"])
                    
                    gr.Markdown("### 📋 Current Context")
                    alert_display = gr.Textbox(value="Initialize a task to begin.", label="Contextual Alert", lines=6, interactive=False)
                    
                    gr.Markdown("### 📊 Real-time Metrics")
                    metrics_display = gr.HTML(value=_format_metrics(None))

            with gr.Column(scale=3):
                with gr.Column(elem_classes=["terminal-window"]):
                    gr.HTML("""<div class="terminal-header">
                        <div class="dot red"></div><div class="dot amber"></div><div class="dot green"></div>
                        <div class="terminal-title">SPRINTBOARD TERMINAL - v2.0</div>
                    </div>""")
                    terminal_log = gr.HTML(
                        value=_render_terminal("Welcome to SprintBoard Terminal.\nWaiting for initialization...\n"),
                        elem_id="terminal-outer"
                    )

                with gr.Row():
                    step_display = gr.HTML(value="<div style='color:#71717a;font-size:12px;padding:8px;font-family:\"JetBrains Mono\"'>Step 0 / --</div>")

                with gr.Row():
                    cmd_input = gr.Textbox(placeholder="Enter command...", scale=4, interactive=True, elem_id="cmd-input", show_label=False)
                    btn_exec = gr.Button("EXECUTE", scale=1, variant="primary", elem_classes=["btn-primary"])

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("**⚡ Quick Actions:**")
                        with gr.Row():
                            q_btn_backlog = gr.Button("📋 Backlog", size="sm", elem_classes=["btn-secondary"])
                            q_btn_team    = gr.Button("👥 Team", size="sm", elem_classes=["btn-secondary"])
                            q_btn_vel     = gr.Button("📈 Velocity", size="sm", elem_classes=["btn-secondary"])
                            q_btn_sprint  = gr.Button("🔍 Sprint", size="sm", elem_classes=["btn-secondary"])
                    with gr.Column(scale=1):
                        gr.Markdown("**🤖 Automation:**")
                        btn_autosolve = gr.Button("🤖  AUTO-SOLVE", variant="primary", elem_classes=["btn-primary"])

                gr.Markdown("### 📜 Live Sprint Manifest")
                sprint_manifest = gr.HTML(value=_format_sprint_manifest(_make_env().board))

            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 🏆 Performance Tracker")
                score_display = gr.HTML(value=_format_score_initial())
                
                with gr.Accordion("📖 Command Reference", open=False):
                    gr.Markdown("""
**Investigation:**
- LIST_BACKLOG
- VIEW_STORY <id>
- VIEW_TEAM

**Planning:**
- ESTIMATE <id> <pts>
- ASSIGN <id> <name>
- FINALIZE_SPRINT
""")

                btn_reset = gr.Button("↺  RESET EPISODE", variant="secondary", elem_classes=["btn-secondary"])

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
