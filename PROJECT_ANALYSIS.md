# SprintBoard: Project Analysis & Documentation

## Overview

**SprintBoard** is an OpenEnv-compliant reinforcement learning environment designed for training LLM agents in Agile sprint planning and backlog management. The environment simulates real-world project management scenarios where agents must investigate, diagnose, and fix sprint planning issues.

### Key Features

- **15 Complex Tasks**: Ranging from unestimated story grooming to complex dependency resolution
- **3-Axis Deterministic Grading**: Scores based on Investigation (30%), Planning Quality (50%), and Process compliance (20%)
- **High-Fidelity Simulation**: Real-time state machine for project boards, team capacity, and velocity tracking
- **OpenEnv Compliant**: Standard HTTP/WebSocket API for easy integration with RL training frameworks
- **Interactive UI**: Gradio-based playground for manual testing and demonstration

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio UI (app.py)                       │
│              Interactive playground for HF Spaces            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ HTTP/WebSocket
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server (server/app.py)                 │
│         OpenEnv-compliant HTTP + WebSocket endpoints         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         SprintBoardEnvironment (server/environment.py)      │
│              Main environment orchestrator                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ ProjectBoard │  │   Grader     │  │ Command     │
│ (board.py)   │  │ (grader.py)  │  │ Parser      │
│              │  │              │  │ (command_   │
│ - Stories    │  │ - Determin-   │  │ parser.py)  │
│ - Team       │  │   istic      │  │             │
│ - Velocity   │  │   scoring    │  │ - Parse &   │
│ - Assignments│  │ - 3-axis     │  │   execute   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                  │
        ▼                  ▼
┌──────────────┐  ┌──────────────┐
│ Tasks        │  │ Reward       │
│ (tasks.py)   │  │ (reward.py)  │
│ - 15 scenarios│ │ - Step-wise  │
│ - Fault inject│ │   reward     │
└──────────────┘  └──────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| **Gradio UI** | `app.py` | Interactive web interface for manual testing |
| **FastAPI Server** | `server/app.py` | HTTP/WebSocket endpoints, OpenEnv protocol |
| **Environment** | `server/environment.py` | Episode lifecycle, step/reset/state |
| **Project Board** | `server/board.py` | State machine for stories, team, velocity |
| **Task Registry** | `server/tasks.py` | 15 task definitions with fault injection |
| **Grader** | `server/grader.py` | Deterministic 3-axis scoring |
| **Command Parser** | `server/command_parser.py` | Parse free-form planning commands |
| **Reward Function** | `server/reward.py` | Step-wise reward shaping |
| **Models** | `models.py` | Pydantic models for Action/Observation/State |
| **Client** | `client.py` | WebSocket client for agent connections |
| **Inference** | `inference.py` | LLM-based agent implementation |
| **Agent** | `agent.py` | Heuristic baseline agent |

---

## Project Structure

```
sprint_planning_env/
├── app.py                      # Gradio UI for HF Spaces
├── agent.py                    # Heuristic baseline agent
├── client.py                   # OpenEnv WebSocket client
├── inference.py                # LLM-based agent (for hackathon)
├── models.py                   # Pydantic models (Action, Observation, State)
├── pyproject.toml              # Project configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── README.md                   # Project documentation
├── server/
│   ├── app.py                  # FastAPI server
│   ├── board.py                # Project board state machine
│   ├── command_parser.py       # Command parsing logic
│   ├── environment.py          # Main environment class
│   ├── grader.py               # Deterministic grading
│   ├── reward.py               # Reward shaping
│   └── tasks.py                # 15 task definitions
├── tests/
│   ├── test_commands.py        # Command parsing tests
│   └── test_reward_hacking.py  # Anti-reward-hacking tests
└── sprint_planning_env/        # Package directory
```

---

## Task Catalog

### Difficulty Levels

- **Easy (Tasks 1-5)**: Single-fault diagnosis, solvable in 3-5 steps
- **Medium (Tasks 6-10)**: Multi-step planning with ambiguity
- **Hard (Tasks 11-15)**: Compound faults requiring multi-root-cause analysis

### Task List

| ID | Name | Difficulty | Fault Type |
|----|------|------------|------------|
| task_1 | Unestimated Stories | Easy | `unestimated_stories` |
| task_2 | Developer Overload | Easy | `developer_overload` |
| task_3 | Missing Dependency | Easy | `missing_dependency` |
| task_4 | Scope Creep | Easy | `scope_creep` |
| task_5 | Wrong Priority | Easy | `wrong_priority` |
| task_6 | Velocity Overload | Medium | `velocity_overload` |
| task_7 | Skill Mismatch | Medium | `skill_mismatch` |
| task_8 | Epic Decomposition | Medium | `epic_decomposition` |
| task_9 | Priority Conflict | Medium | `priority_conflict` |
| task_10 | Tech Debt Balance | Medium | `tech_debt_balance` |
| task_11 | Dependency Chain Overload | Hard | `dependency_chain_overload` |
| task_12 | PTO Velocity Drop | Hard | `pto_velocity_drop` |
| task_13 | Cross-Team Dependency | Hard | `cross_team_dependency` |
| task_14 | Sprint Rescue | Hard | `sprint_rescue` |
| task_15 | Full Sprint Planning | Hard | `full_sprint_planning` |

### Fault Categories

1. **Estimation Faults**: Unestimated stories, velocity mismatches
2. **Assignment Faults**: Developer overload, skill mismatches
3. **Dependency Faults**: Missing dependencies, circular chains
4. **Capacity Faults**: Velocity overload, PTO impacts
5. **Process Faults**: Scope creep, priority conflicts, tech debt balance

---

## Command Reference

### Investigation Commands (Read-Only)

| Command | Usage | Description |
|---------|-------|-------------|
| `LIST_BACKLOG` | `LIST_BACKLOG [--sort priority]` | Show all user stories in backlog |
| `VIEW_STORY <id>` | `VIEW_STORY US-101` | View detailed story information |
| `CHECK_DEPS <id>` | `CHECK_DEPS US-101` | Check story dependencies |
| `VIEW_TEAM` | `VIEW_TEAM` | View team members and capacities |
| `VIEW_VELOCITY` | `VIEW_VELOCITY [--last 5]` | View velocity history |
| `VIEW_SPRINT` | `VIEW_SPRINT` | View current sprint plan |
| `VIEW_BUGS` | `VIEW_BUGS` | View bug backlog |
| `VIEW_EPIC <id>` | `VIEW_EPIC EP-01` | View epic details |
| `SEARCH_BACKLOG <kw>` | `SEARCH_BACKLOG auth` | Search stories by keyword |

### Planning Commands (State-Modifying)

| Command | Usage | Description |
|---------|-------|-------------|
| `ESTIMATE <id> <pts>` | `ESTIMATE US-101 5` | Set story points |
| `ASSIGN <id> <name>` | `ASSIGN US-101 Alice` | Assign story to developer |
| `UNASSIGN <id>` | `UNASSIGN US-101` | Remove assignment |
| `ADD_TO_SPRINT <id>` | `ADD_TO_SPRINT US-101` | Add story to sprint |
| `REMOVE_FROM_SPRINT <id>` | `REMOVE_FROM_SPRINT US-101` | Remove from sprint |
| `SET_PRIORITY <id> P0|P1|P2` | `SET_PRIORITY US-101 P0` | Set story priority |
| `FLAG_RISK <id> <reason>` | `FLAG_RISK US-101 vague` | Flag a risk |
| `DECOMPOSE <epic> <stories>` | `DECOMPOSE EP-01 "s1" "s2" "s3"` | Decompose epic into stories |
| `FINALIZE_SPRINT` | `FINALIZE_SPRINT` | Submit final sprint plan |

### Destructive Commands (Fatal)

These commands will terminate the episode with a penalty:
- `DELETE_STORY`, `CLEAR_SPRINT`, `REMOVE_DEVELOPER`, `DROP_ALL`, `RESET_BOARD`

---

## Grading System

### 3-Axis Scoring

Each task is graded across three dimensions:

1. **Investigation (30%)**: Did the agent investigate the right data?
   - Checks for appropriate investigation commands (LIST_BACKLOG, VIEW_TEAM, etc.)
   - Rewards thorough diagnosis before action

2. **Planning Quality (50%)**: Is the resulting sprint plan valid and good?
   - Verifies actual board state (assignments, estimates, dependencies)
   - Checks for capacity compliance, dependency satisfaction
   - Prevents reward hacking by validating state, not commands

3. **Process (20%)**: Did the agent avoid destructive actions?
   - Penalizes destructive commands
   - Rewards error-free execution
   - Considers step efficiency

### Step Thresholds

Each task has an optimal step threshold. Exceeding it applies an efficiency penalty:

| Fault Type | Threshold |
|------------|-----------|
| Unestimated stories | 8 steps |
| Developer overload | 8 steps |
| Missing dependency | 7 steps |
| Scope creep | 8 steps |
| Wrong priority | 8 steps |
| Velocity overload | 10 steps |
| Skill mismatch | 10 steps |
| Epic decomposition | 8 steps |
| Priority conflict | 10 steps |
| Tech debt balance | 10 steps |
| Compound faults | 12-14 steps |

---

## API Reference

### OpenEnv Standard Endpoints

#### POST `/reset`
Reset the environment and start a new episode.

**Request Body:**
```json
{
  "task_id": "task_1",
  "seed": 42,
  "episode_id": "optional-uuid"
}
```

**Response:** `SprintObservation`

#### POST `/step`
Execute an action and get the next observation.

**Request Body:** `SprintAction`
```json
{
  "command": "LIST_BACKLOG"
}
```

**Response:** `SprintObservation`

#### GET `/state`
Get current episode state.

**Response:** `SprintState`

#### WebSocket `/ws`
Real-time bidirectional communication for agent training.

### Custom Endpoints

#### GET `/info`
Environment metadata and available endpoints.

#### GET `/tasks`
List all available tasks with metadata.

**Response:**
```json
{
  "tasks": [
    {
      "id": "task_1",
      "name": "Unestimated Stories",
      "difficulty": "easy",
      "description": "...",
      "fault_type": "unestimated_stories"
    }
  ],
  "action_schema": {"command": "string"},
  "max_steps": 15
}
```

#### GET `/grader`
Retrieve score and feedback for the last completed episode.

---

## Installation & Usage

### Local Installation

```bash
# Install dependencies
pip install -e .

# Run Gradio UI (interactive playground)
python app.py

# Run FastAPI server (for agent training)
python -m sprint_planning_env.server.app
```

### Docker Deployment

```bash
# Build image
docker build -t sprintboard .

# Run container
docker run -p 8000:8000 sprintboard
```

### Hugging Face Spaces

The environment is configured for Hugging Face Spaces deployment:
- **SDK**: Docker
- **App File**: `app.py`
- **Port**: 7860 (default Gradio port)

---

## Training an Agent

### Using the Client

```python
import asyncio
from sprint_planning_env.client import SprintBoardEnv
from sprint_planning_env.models import SprintAction

async def main():
    async with SprintBoardEnv(base_url="http://localhost:8000") as env:
        # Reset with specific task
        obs = await env.reset(task_id="task_1")
        
        # Execute commands
        for _ in range(15):
            action = SprintAction(command="LIST_BACKLOG")
            result = await env.step(action)
            if result.done:
                break

asyncio.run(main())
```

### LLM-Based Agent (Inference)

The `inference.py` script provides a baseline LLM agent:

```bash
# Set environment variables
export API_KEY="your-api-key"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run inference
python inference.py
```

The agent:
1. Uses OpenAI-compatible API for LLM queries
2. Implements heuristic fallback for invalid commands
3. Logs in `[START]/[STEP]/[END]` format for hackathon compliance

### Heuristic Agent

The `agent.py` provides a rule-based baseline with pre-defined command sequences for each task.

---

## Development Guide

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Adding New Tasks

To add a new task:

1. **Define task in `server/tasks.py`**:
   ```python
   TASK_REGISTRY["task_16"] = {
       "name": "New Task",
       "difficulty": "medium",
       "fault_type": "new_fault_type",
       "alert": "SCENARIO: ...",
       "params": {...},
       "fatal_patterns": [],
       "allowed_dangerous": [],
   }
   ```

2. **Implement grader in `server/grader.py`**:
   ```python
   def _grade_new_fault_type(board, params, history, error_history, steps_used):
       # Return (score, breakdown)
       pass
   ```

3. **Add resolution check in `server/environment.py`**:
   ```python
   if ft == "new_fault_type":
       return # resolution condition
   ```

4. **Update step threshold in `server/grader.py`**:
   ```python
   STEP_THRESHOLDS["new_fault_type"] = 10
   ```

### Adding New Commands

To add a new planning command:

1. **Add to command sets in `server/command_parser.py`**
2. **Implement parsing logic in `parse_and_execute()`**
3. **Add corresponding method to `ProjectBoard` class in `server/board.py`**
4. **Update help text in `_help_text()`**

---

## Dependencies

### Core Dependencies

- `openenv-core>=0.2.0` - OpenEnv framework
- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `pydantic>=2.0` - Data validation
- `openai` - LLM client

### UI Dependencies

- `gradio>=6.0.0` - Web UI
- `huggingface_hub>=0.22.0` - HF integration
- `mistralai>=1.0.0` - Alternative LLM provider

### Dev Dependencies

- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `httpx` - HTTP client for testing

---

## Design Philosophy

### Open Action Space

Unlike traditional RL environments with discrete action spaces, SprintBoard uses **free-form text commands**. This design choice:

- Mirrors real Scrum Master workflow (no artificial discretization)
- Tests compositional reasoning capabilities
- Enables frontier models to demonstrate command generation
- Creates a more realistic and challenging environment

### Deterministic Grading

All graders are fully deterministic:
- No LLM judges or stochastic sampling
- Reproducible scores given identical board states
- Critical for RL reward signal stability
- Prevents reward hacking through state validation

### Real-World Scenarios

Each task models a genuine sprint planning failure:
- Based on real Agile/Scrum practices
- Observable symptoms only (no root-cause hints)
- Forces investigation before action
- Covers 5 fault categories for broad domain coverage

---

## Performance Characteristics

### Episode Length

- **Maximum steps**: 15 per episode
- **Typical completion**: 5-12 steps depending on task difficulty
- **Step budget**: Forces efficient planning

### Scoring Range

- **Score range**: 0.001 to 0.999 (clamped to avoid exact 0.0/1.0)
- **Current frontier model performance**: 0.4-0.7 on compound tasks
- **Headroom for improvement**: Significant room for RL fine-tuning

### State Size

- **Stories**: 20 user stories in backlog pool
- **Team**: 5 developers with varying capacities
- **Velocity**: 5-sprint history
- **Dependencies**: Complex dependency graph

---

## Security & Anti-Hacking

### Reward Hacking Prevention

The grader validates **actual board state**, not command patterns:
- Checks assignments, estimates, dependencies directly
- Cannot game the system by typing keywords
- Resolution verified through state inspection

### Fatal Action Detection

Global and task-specific fatal patterns:
- Destructive commands terminate with penalty
- Task-specific dangerous actions can be allowed
- Prevents agents from "clearing the board" to avoid faults

### Error Tracking

- Error history maintained per episode
- High error rates reduce process score
- Encourages valid command generation

---

## Future Enhancements

### Potential Extensions

1. **Additional Tasks**: More fault types and scenarios
2. **Dynamic Difficulty**: Adaptive task selection based on agent performance
3. **Multi-Agent**: Support for team-based planning scenarios
4. **Curriculum Learning**: Built-in curriculum scheduling
5. **Visualization**: Enhanced board state visualization
6. **Export/Import**: Save/load sprint plans for analysis

### Research Applications

- **GRPO Fine-tuning**: Train models on the environment
- **Curriculum Learning**: Study task progression strategies
- **Multi-Modal**: Add story descriptions, mockups, etc.
- **Human-in-the-loop**: Hybrid human-AI planning

---

## License & Attribution

This project is designed for research and educational purposes in the field of AI agent training and reinforcement learning.

---

## Contact & Support

For issues, questions, or contributions, please refer to the project repository or contact the maintainers.

---

*Generated on April 14, 2026*
