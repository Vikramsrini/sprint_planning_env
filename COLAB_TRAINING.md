# SprintBoard Colab Training Guide (HF TRL + OpenEnv)

This project includes a minimal TRL training script (`train_grpo.py`) designed to run in Colab.

## HF Credits Path (recommended if you want to spend HF credits)

If your goal is to use Hugging Face credits, run training on HF-managed compute (not Colab).

### A) Run inside a paid GPU Space
1. Open your Space: `https://huggingface.co/spaces/vikramsrini/sprint_planning_env`
2. In Space settings, select a paid GPU hardware tier.
3. Open the Space terminal (Dev Mode/terminal session) and run:
```bash
git clone https://huggingface.co/spaces/vikramsrini/sprint_planning_env
cd sprint_planning_env
pip install -e .
pip install -r requirements-train.txt
python train_grpo.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir runs/sprintboard-grpo \
  --epochs 3 \
  --max-samples 50 \
  --max-completion-length 64
```
4. Download artifacts from the Space runtime or push results to a dataset/repo.

### B) Keep Colab only if you do not need HF credits
Colab uses Google compute and does not consume HF training credits.

## 1) Open Colab
Create a new Colab notebook and set runtime to GPU.

## 2) Install dependencies
```bash
import os
if os.path.isdir("sprint_planning_env"):
    !git -C sprint_planning_env pull
else:
    !git clone https://huggingface.co/spaces/vikramsrini/sprint_planning_env
%cd sprint_planning_env
!pip install -e .
!pip install -r requirements-train.txt
```

## 3) Run training
```bash
!python train_grpo.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir runs/sprintboard-grpo \
  --epochs 3 \
  --max-samples 50 \
  --max-completion-length 64
```

## 4) Run evaluation comparison
```bash
!python evaluate_training.py \
  --output-dir runs/eval \
  --task-limit 15 \
  --checkpoint-dir runs/sprintboard-grpo/checkpoint-final
```

## 5) Save artifacts
Expected artifacts:
- `runs/sprintboard-grpo/artifacts/training_metrics.jsonl`
- `runs/sprintboard-grpo/artifacts/reward_curve.png`
- `runs/sprintboard-grpo/artifacts/loss_curve.png` (if present)
- `runs/eval/baseline_vs_trained.json`
- `runs/eval/baseline_vs_trained.png`

Upload these to your repo and embed key plots in `README.md`.
