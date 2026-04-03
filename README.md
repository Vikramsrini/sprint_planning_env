---
title: Sprint Planning Env
emoji: ⚡
colorFrom: violet
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# ⚡ SprintBoard: Agile Sprint Planning & Backlog Grooming RL Environment

An OpenEnv-compliant reinforcement learning environment for training LLM agents in Agile project management.

## 🚀 Interactive Playground
You can try the environment manually using the provided Gradio interface or connect an RL agent via the OpenEnv API.

### Features
- **15 Complex Tasks:** Ranging from unestimated story grooming to complex dependency resolution.
- **3-Axis Deterministic Grading:** Scores based on Investigation, Planning Quality, and Process compliance.
- **High-Fidelity Simulation:** Real-time state machine for project boards, team capacity, and velocity tracking.

## 🛠️ Installation (Local)
```bash
pip install -e .
python app.py
```

## 🧠 Training an Agent
Refer to `inference.py` for a baseline implementation using the OpenEnv framework.
