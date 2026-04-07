

import time
from sprint_planning_env.models import SprintAction

TASKS_COMMANDS = {
    "task_1": [
        "LIST_BACKLOG",
        "VIEW_STORY 103",
        "VIEW_VELOCITY",
        "ESTIMATE 103 5",
        "ESTIMATE 106 3",
        "ESTIMATE 107 3",
        "FINALIZE_SPRINT"
    ],
    "task_2": [
        "VIEW_TEAM",
        "VIEW_SPRINT",
        "ASSIGN 101 Diana",
        "ASSIGN 102 Eve",
        "ASSIGN 104 Charlie",
        "FINALIZE_SPRINT"
    ],
    "task_3": [
        "CHECK_DEPS 105",
        "VIEW_STORY 105",
        "LIST_BACKLOG",
        "ADD_TO_SPRINT 101",
        "FINALIZE_SPRINT"
    ],
    "task_4": [
        "VIEW_STORY 114",
        "VIEW_STORY 108",
        "VIEW_SPRINT",
        "FLAG_RISK 114 vague",
        "FINALIZE_SPRINT"
    ],
    "task_5": [
        "LIST_BACKLOG",
        "VIEW_STORY 101",
        "ADD_TO_SPRINT 101",
        "ADD_TO_SPRINT 102",
        "REMOVE_FROM_SPRINT 106",
        "REMOVE_FROM_SPRINT 117",
        "FINALIZE_SPRINT"
    ],
    "task_6": [
        "VIEW_VELOCITY",
        "VIEW_SPRINT",
        "REMOVE_FROM_SPRINT 111",
        "REMOVE_FROM_SPRINT 116",
        "REMOVE_FROM_SPRINT 118",
        "REMOVE_FROM_SPRINT 103",
        "REMOVE_FROM_SPRINT 104",
        "REMOVE_FROM_SPRINT 108",
        "REMOVE_FROM_SPRINT 110",
        "REMOVE_FROM_SPRINT 112",
        "REMOVE_FROM_SPRINT 113",
        "REMOVE_FROM_SPRINT 115",
        "FINALIZE_SPRINT"
    ],
    "task_7": [
        "VIEW_TEAM",
        "VIEW_STORY 103",
        "ASSIGN 103 Bob",
        "ASSIGN 104 Alice",
        "ASSIGN 106 Charlie",
        "ASSIGN 107 Eve",
        "FINALIZE_SPRINT"
    ],
    "task_8": [
        "VIEW_EPIC EP-01",
        "LIST_BACKLOG",
        "DECOMPOSE EP-01 \"stripe\" \"paypal\" \"apple\" \"invoice\" \"subscription\" \"webhook\"",
        "FINALIZE_SPRINT"
    ],
    "task_9": [
        "VIEW_STORY 101",
        "VIEW_STORY 102",
        "VIEW_TEAM",
        "ADD_TO_SPRINT 101",
        "FINALIZE_SPRINT"
    ],
    "task_10": [
        "VIEW_BUGS",
        "VIEW_SPRINT",
        "ADD_TO_SPRINT BUG-01",
        "ADD_TO_SPRINT BUG-02",
        "ADD_TO_SPRINT BUG-03",
        "FINALIZE_SPRINT"
    ],
    "task_11": [
        "CHECK_DEPS 113",
        "VIEW_TEAM",
        "VIEW_SPRINT",
        "REMOVE_FROM_SPRINT 105",
        "REMOVE_FROM_SPRINT 113",
        "ASSIGN 101 Diana",
        "ASSIGN 102 Eve",
        "FINALIZE_SPRINT"
    ],
    "task_12": [
        "VIEW_TEAM",
        "VIEW_VELOCITY",
        "VIEW_SPRINT",
        "REMOVE_FROM_SPRINT 101",
        "REMOVE_FROM_SPRINT 104",
        "UNASSIGN 101",
        "UNASSIGN 104",
        "FINALIZE_SPRINT"
    ],
    "task_13": [
        "VIEW_TEAM",
        "CHECK_DEPS 115",
        "VIEW_STORY 109",
        "REMOVE_FROM_SPRINT 115",
        "ASSIGN 109 Diana",
        "FINALIZE_SPRINT"
    ],
    "task_14": [
        "VIEW_TEAM",
        "VIEW_SPRINT",
        "CHECK_DEPS 105",
        "VIEW_STORY 114",
        "VIEW_VELOCITY",
        "ESTIMATE 114 2",
        "ASSIGN 103 Bob",
        "ASSIGN 106 Bob",
        "ASSIGN 108 Diana",
        "ADD_TO_SPRINT 101",
        "REMOVE_FROM_SPRINT 112",
        "FINALIZE_SPRINT"
    ],
    "task_15": [
        "LIST_BACKLOG",
        "VIEW_TEAM",
        "VIEW_VELOCITY",
        "CHECK_DEPS 105",
        "VIEW_STORY 101",
        "ADD_TO_SPRINT 101",
        "ADD_TO_SPRINT 102",
        "ADD_TO_SPRINT 111",
        "ADD_TO_SPRINT 104",
        "ASSIGN 101 Diana",
        "ASSIGN 102 Alice",
        "ASSIGN 111 Eve",
        "ASSIGN 104 Charlie",
        "FINALIZE_SPRINT"
    ]
}

class OmegaSprintBot:
    def __init__(self, board, task_id="task_1"):
        self.board = board
        self._done = False
        self._label = "AI Agent"
        self._queue = list(TASKS_COMMANDS.get(task_id, TASKS_COMMANDS["task_1"]))

    def next_command(self) -> str | None:
        if self._done or not self._queue:
            return None
        cmd = self._queue.pop(0)
        if cmd == "FINALIZE_SPRINT":
            self._done = True
        return cmd

def run_agent(env, task_id: str, fmt_metrics, fmt_score, fmt_score_initial, max_steps: int = 15):
    env.reset(task_id=task_id)
    bot = OmegaSprintBot(env.board, task_id)

    accumulated_log = f"🤖 {bot._label} starting...\n"
    yield (accumulated_log, fmt_metrics(env.board.get_metrics()), fmt_score_initial(), "Step 0/15", False)

    for step in range(1, max_steps + 1):
        cmd = bot.next_command()
        if not cmd:
            break

        obs = env.step(SprintAction(command=cmd))
        entry = f"\n{'─'*40}\n$ {cmd}\n{obs.command_output or obs.error or ''}\n"
        
        is_done = obs.done or (step == max_steps)
        
        if is_done:
            grade = obs.metadata.get("grader_score", 0.0) or 0.0
            pct = int(grade * 100)
            entry += (
                f"\n{'═'*40}\n"
                f"  EPISODE COMPLETE\n"
                f"  🏆 FINAL SCORE: {pct}%\n"
                f"{'═'*40}\n"
            )
            accumulated_log += entry
            yield (accumulated_log, fmt_metrics(obs.metrics), fmt_score(obs), f"Step {step}/{max_steps}", True)
            break

        accumulated_log += entry
        yield (accumulated_log, fmt_metrics(obs.metrics), fmt_score(obs), f"Step {step}/{max_steps}", False)
        # Reduced sleep purely for test execution speed; can stay identical or lower.
        time.sleep(0.1)
