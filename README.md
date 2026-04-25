---
title: Sprint Planning Env
emoji: ⚡
colorFrom: purple
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# ⚡ SprintBoard: Agile Sprint Planning & Backlog Grooming RL Environment

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vikramsrini/sprint_planning_env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/meta-pytorch/OpenEnv)

SprintBoard is an OpenEnv-compliant RL environment for training LLM agents on realistic Agile sprint planning. Agents must investigate board state, diagnose planning faults, and execute multi-step corrections under step limits.

## Quick Links
- Interactive demo (HF Space): https://huggingface.co/spaces/vikramsrini/sprint_planning_env
- OpenEnv manifest: `openenv.yaml`
- Training script (TRL/GRPO): `train_grpo.py`
- Colab run guide: `COLAB_TRAINING.md`
- Baseline vs trained comparison script: `evaluate_training.py`
- LLM inference script: `inference.py`
- Mini-blog draft: `mini-blog.md`

## Theme Fit
Primary: **Theme #3.1 Professional Tasks**

Why this fits:
- Tool-driven, partially observable professional workflow.
- Multi-step causal reasoning with delayed credit assignment.
- Deterministic world model (board, dependencies, velocity, team capacity).

Secondary overlap:
- Theme #2 Long-horizon planning (15-step episodes with sparse completion signal).

## Environment Design
- 15 tasks across easy/medium/hard tiers.
- Free-form command action space (investigation + planning + finalization).
- Deterministic 3-axis grading:
  - Investigation (30%)
  - Planning quality (50%)
  - Process/safety (20%)
- Anti-reward-hacking checks based on board state, not command keywords.

## Minimum Requirements Checklist
- [x] Uses OpenEnv with `reset`/`step`/`state` + WebSocket endpoint.
- [x] Includes OpenEnv manifest: `openenv.yaml`.
- [x] Includes training script with HF TRL GRPO: `train_grpo.py`.
- [x] Includes Colab-friendly run guide: `COLAB_TRAINING.md`.
- [x] Includes baseline/trained comparison script: `evaluate_training.py`.
- [x] Includes mini-blog content: `mini-blog.md`.
- [x] Hosted on HF Space.
- [ ] Add published mini-blog/video URL to this README.
- [ ] Run training and commit generated plots under `runs/` or `artifacts/`.

## Run Locally

### 1) Install environment
```bash
pip install -e .
pip install -r requirements-train.txt
```

### 2) Start OpenEnv server
```bash
python -m sprint_planning_env.server.app
```

### 3) Verify environment endpoints
```bash
python check_routes.py
```

### 4) Run GRPO training (TRL)
```bash
python train_grpo.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir runs/sprintboard-grpo \
  --epochs 1 \
  --max-samples 15
```

Generated artifacts:
- `runs/sprintboard-grpo/artifacts/training_metrics.jsonl`
- `runs/sprintboard-grpo/artifacts/reward_curve.png` (if matplotlib available)
- `runs/sprintboard-grpo/artifacts/loss_curve.png` (if logged)
- `runs/sprintboard-grpo/artifacts/summary.json`

### 5) Run baseline vs trained-like evaluation
```bash
python evaluate_training.py --output-dir runs/eval --task-limit 15
```

Generated artifacts:
- `runs/eval/baseline_vs_trained.json`
- `runs/eval/baseline_vs_trained.png`

## Evidence Section (Fill After Run)
After running the two scripts above, embed these artifacts in this README:

```markdown
![Reward Curve](runs/sprintboard-grpo/artifacts/reward_curve.png)
*Reward over training steps from GRPO run.*

![Baseline vs Trained](runs/eval/baseline_vs_trained.png)
*Comparison of average final score and success rate.*
```

## Judging Alignment
| Criterion | How SprintBoard addresses it |
|-----------|------------------------------|
| Innovation (40%) | Underexplored professional workflow (Agile sprint planning), free-form actions, deterministic grading. |
| Storytelling (30%) | Scenario-driven alerts, Scrum Master framing, structured write-up in `mini-blog.md`. |
| Improvement (20%) | Training script emits reproducible metrics/plots + baseline comparison artifacts. |
| Reward & Pipeline (10%) | Coherent reward shaping + deterministic episode grader + TRL GRPO integration. |

## Submission Notes
- Keep all external links (Space, blog/video, optional W&B run) in this README.
- Do not store large videos directly in repo; link externally.

## License
MIT License. Created for OpenEnv Hackathon 2026.
