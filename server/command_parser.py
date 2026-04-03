"""
SprintBoard — Command parser for free-form planning commands.

Parses agent text into structured board operations. The agent sends
commands like "ASSIGN US-101 Alice" or "VIEW_VELOCITY --last 5",
and this module routes them to the appropriate ProjectBoard method.

Design choice: free-form text commands (like SQL in SQLab) rather than
structured JSON. This creates a genuine compositional reasoning challenge
and mirrors how real SMs interact with project tools.
"""

import logging
import shlex
from typing import Tuple, Optional

from sprint_planning_env.server.board import ProjectBoard

logger = logging.getLogger(__name__)


# Commands that only read state (no mutations)
INVESTIGATION_COMMANDS = {
    "LIST_BACKLOG", "VIEW_STORY", "CHECK_DEPS", "VIEW_TEAM",
    "VIEW_VELOCITY", "VIEW_SPRINT", "SEARCH_BACKLOG", "VIEW_BUGS",
    "VIEW_EPIC", "HELP",
}

# Commands that modify state
PLANNING_COMMANDS = {
    "ESTIMATE", "ASSIGN", "UNASSIGN", "ADD_TO_SPRINT",
    "REMOVE_FROM_SPRINT", "FLAG_RISK", "SET_PRIORITY",
    "DECOMPOSE", "ADD_NOTE", "FINALIZE_SPRINT",
}

# Destructive commands — episode terminates with penalty
DESTRUCTIVE_COMMANDS = {
    "DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER",
    "DROP_ALL", "RESET_BOARD",
}


def parse_and_execute(command: str, board: ProjectBoard) -> Tuple[str, Optional[str]]:
    """Parse a free-form command and execute it against the board.

    Returns:
        (output, error) — output is the formatted result, error is None on success.
    """
    command = command.strip()
    if not command:
        return "", "ERROR: Empty command. Type HELP for available commands."

    # Tokenize, handling quoted strings
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    if not tokens:
        return "", "ERROR: Could not parse command."

    cmd = tokens[0].upper()
    args = tokens[1:]

    # ── Destructive commands ─────────────────────────────────────
    if cmd in DESTRUCTIVE_COMMANDS:
        return "", f"FATAL: Destructive command '{cmd}' detected. Episode terminated."

    # ── Help ─────────────────────────────────────────────────────
    if cmd == "HELP":
        return _help_text(), None

    # ── Investigation commands ───────────────────────────────────
    if cmd == "LIST_BACKLOG":
        sort_by = "priority"
        for i, a in enumerate(args):
            if a == "--sort" and i + 1 < len(args):
                sort_by = args[i + 1]
        return board.list_backlog(sort_by), None

    if cmd == "VIEW_STORY":
        if not args:
            return "", "ERROR: Usage: VIEW_STORY <story_id>"
        return board.view_story(args[0].upper()), None

    if cmd == "CHECK_DEPS":
        if not args:
            return "", "ERROR: Usage: CHECK_DEPS <story_id>"
        return board.check_deps(args[0].upper()), None

    if cmd == "VIEW_TEAM":
        return board.view_team(), None

    if cmd == "VIEW_VELOCITY":
        last_n = 5
        for i, a in enumerate(args):
            if a == "--last" and i + 1 < len(args):
                try:
                    last_n = int(args[i + 1])
                except ValueError:
                    pass
        return board.view_velocity(last_n), None

    if cmd == "VIEW_SPRINT":
        return board.view_sprint(), None

    if cmd == "SEARCH_BACKLOG":
        if not args:
            return "", "ERROR: Usage: SEARCH_BACKLOG <keyword>"
        return board.search_backlog(" ".join(args)), None

    if cmd == "VIEW_BUGS":
        return board.view_bugs(), None

    if cmd == "VIEW_EPIC":
        if not args:
            return "", "ERROR: Usage: VIEW_EPIC <epic_id>"
        return board.view_epic(args[0].upper()), None

    # ── Planning commands ────────────────────────────────────────
    if cmd == "ESTIMATE":
        if len(args) < 2:
            return "", "ERROR: Usage: ESTIMATE <story_id> <points>"
        story_id = args[0].upper()
        try:
            points = int(args[1])
        except ValueError:
            return "", f"ERROR: Invalid points '{args[1]}'. Must be a number."
        return board.estimate(story_id, points), None

    if cmd == "ASSIGN":
        if len(args) < 2:
            return "", "ERROR: Usage: ASSIGN <story_id> <developer_name>"
        story_id = args[0].upper()
        developer = args[1]  # Case-sensitive for names
        # Try to match case-insensitively
        team_names = list(board.team.keys())
        matched = next((n for n in team_names if n.lower() == developer.lower()), None)
        if matched:
            developer = matched
        return board.assign(story_id, developer), None

    if cmd == "UNASSIGN":
        if not args:
            return "", "ERROR: Usage: UNASSIGN <story_id>"
        return board.unassign(args[0].upper()), None

    if cmd == "ADD_TO_SPRINT":
        if not args:
            return "", "ERROR: Usage: ADD_TO_SPRINT <story_id>"
        return board.add_to_sprint(args[0].upper()), None

    if cmd == "REMOVE_FROM_SPRINT":
        if not args:
            return "", "ERROR: Usage: REMOVE_FROM_SPRINT <story_id>"
        return board.remove_from_sprint(args[0].upper()), None

    if cmd == "FLAG_RISK":
        if len(args) < 2:
            return "", "ERROR: Usage: FLAG_RISK <story_id> <reason>"
        story_id = args[0].upper()
        reason = " ".join(args[1:])
        return board.flag_risk(story_id, reason), None

    if cmd == "SET_PRIORITY":
        if len(args) < 2:
            return "", "ERROR: Usage: SET_PRIORITY <story_id> <P0|P1|P2>"
        story_id = args[0].upper()
        priority = args[1].upper()
        return board.set_priority(story_id, priority), None

    if cmd == "DECOMPOSE":
        if len(args) < 3:
            return "", 'ERROR: Usage: DECOMPOSE <epic_id> "subtask 1" "subtask 2" ...'
        epic_id = args[0].upper()
        subtask_titles = args[1:]
        return board.decompose(epic_id, subtask_titles), None

    if cmd == "ADD_NOTE":
        if len(args) < 2:
            return "", "ERROR: Usage: ADD_NOTE <story_id> <note text>"
        story_id = args[0].upper()
        note = " ".join(args[1:])
        return board.add_note(story_id, note), None

    if cmd == "FINALIZE_SPRINT":
        return board.finalize_sprint(), None

    # ── Unknown command ──────────────────────────────────────────
    return "", (
        f"ERROR: Unknown command '{cmd}'. Type HELP for available commands.\n"
        f"Available: {', '.join(sorted(INVESTIGATION_COMMANDS | PLANNING_COMMANDS))}"
    )


def is_destructive(command: str) -> bool:
    """Check if a command is destructive (triggers episode termination)."""
    tokens = command.strip().split()
    if not tokens:
        return False
    return tokens[0].upper() in DESTRUCTIVE_COMMANDS


def get_command_type(command: str) -> str:
    """Return the command type: 'investigation', 'planning', 'destructive', or 'unknown'."""
    tokens = command.strip().split()
    if not tokens:
        return "unknown"
    cmd = tokens[0].upper()
    if cmd in INVESTIGATION_COMMANDS:
        return "investigation"
    if cmd in PLANNING_COMMANDS:
        return "planning"
    if cmd in DESTRUCTIVE_COMMANDS:
        return "destructive"
    return "unknown"


def _help_text() -> str:
    return """SPRINTBOARD — Available Commands
═══════════════════════════════════════════════════════════

INVESTIGATION (read-only):
  LIST_BACKLOG [--sort priority|points]    List all stories
  VIEW_STORY <id>                          Story details + acceptance criteria
  CHECK_DEPS <id>                          Dependency graph for a story
  VIEW_TEAM                                Team members, skills, capacity, load
  VIEW_VELOCITY [--last N]                 Historical sprint velocity
  VIEW_SPRINT                              Current sprint plan summary
  SEARCH_BACKLOG <keyword>                 Search stories by keyword
  VIEW_BUGS                                Show bug backlog
  VIEW_EPIC <id>                           Epic details

PLANNING (modify sprint):
  ESTIMATE <id> <points>                   Set story point estimate (Fibonacci)
  ASSIGN <id> <developer>                  Assign story to developer
  UNASSIGN <id>                            Remove assignment
  ADD_TO_SPRINT <id>                       Add story to sprint
  REMOVE_FROM_SPRINT <id>                  Remove story from sprint
  FLAG_RISK <id> <reason>                  Flag story as risky
  SET_PRIORITY <id> <P0|P1|P2>             Set story priority
  DECOMPOSE <epic_id> "sub1" "sub2" ...    Break epic into subtasks
  ADD_NOTE <id> <text>                     Add coordination note
  FINALIZE_SPRINT                          Submit sprint plan for grading
"""
