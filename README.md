# 🚀 SprintBoard: Agile RL Training Environment

SprintBoard is a production-grade **OpenEnv** environment designed to train LLM agents in **Sprint Planning and Backlog Grooming**. It challenges agents to act as Scrum Masters, investigating project boards and resolving complex planning faults through a free-form command interface.

---

## 🛠️ Key Features
- **15 Realistic Tasks**: Spanning 3 difficulty tiers (Easy, Medium, Hard).
- **Command-Based Action Space**: No dropdowns—agents must compose valid planning commands (ASSIGN, ESTIMATE, etc.).
- **3-Axis Deterministic Grading**: Scoring based on **Investigation (30%)**, **Planning Quality (50%)**, and **Process/Safety (20%)**.
- **Anti-Reward-Hacking**: Hardened with 47 adversarial tests to ensure agents learn genuine reasoning, not pattern-matching.
- **High-Fidelity Simulation**: A stateful project board with dependencies, velocity history, and developer skill matrices.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Navigate to the project
cd sprint_planning_env

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 2. Run the Environment Server
The server exposes the OpenEnv API (HTTP/WebSocket) on port 8000.
```bash
# Run via the entry point defined in pyproject.toml
python -m sprint_planning_env.server.app
```

### 3. Run Verification Tests
Ensure the reward system and graders are functioning perfectly.
```bash
python -m pytest tests/ -v
```

### 4. Run Baseline Inference (LLM Agent)
Watch an LLM (e.g., Qwen-72B) attempt to solve the tasks.
*Note: Requires an OpenAI-compatible API (like Hugging Face Router).*
```bash
export HF_TOKEN="your_token_here"
export IMAGE_NAME="sprintboard"
python -m sprint_planning_env.inference
```

---

## 🕹️ Command Reference
Agents use these commands to interact with the board:

| Type | Commands |
|---|---|
| **Investigation** | `LIST_BACKLOG`, `VIEW_TEAM`, `VIEW_VELOCITY`, `CHECK_DEPS`, `VIEW_STORY` |
| **Planning** | `ESTIMATE`, `ASSIGN`, `ADD_TO_SPRINT`, `REMOVE_FROM_SPRINT`, `SET_PRIORITY` |
| **Control** | `FINALIZE_SPRINT`, `HELP` |

---

## 🏗️ Architecture
SprintBoard follows the **OpenEnv Gold Standard**:
- **`models.py`**: Pydantic contracts for Actions/Observations.
- **`board.py`**: The stateful execution engine.
- **`reward.py`**: Fault-gated, target-aware reward shaping.
- **`grader.py`**: Deterministic state-based grading logic.
