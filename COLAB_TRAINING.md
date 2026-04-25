# SprintBoard Colab Training Guide (HF TRL + OpenEnv)

This project includes a minimal TRL training script (`train_grpo.py`) designed to run in Colab.

## 1) Open Colab
Create a new Colab notebook and set runtime to GPU.

## 2) Install dependencies
```bash
!git clone https://huggingface.co/spaces/vikramsrini/sprint_planning_env
%cd sprint_planning_env
!pip install -e .
!pip install -r requirements-train.txt
```

## 3) Run training
```bash
!python train_grpo.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir runs/sprintboard-grpo \
  --epochs 1 \
  --max-samples 15
```

## 4) Run evaluation comparison
```bash
!python evaluate_training.py --output-dir runs/eval --task-limit 15
```

## 5) Save artifacts
Expected artifacts:
- `runs/sprintboard-grpo/artifacts/training_metrics.jsonl`
- `runs/sprintboard-grpo/artifacts/reward_curve.png`
- `runs/sprintboard-grpo/artifacts/loss_curve.png` (if present)
- `runs/eval/baseline_vs_trained.json`
- `runs/eval/baseline_vs_trained.png`

Upload these to your repo and embed key plots in `README.md`.
