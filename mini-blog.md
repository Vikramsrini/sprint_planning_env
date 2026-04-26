# SprintBoard: Training LLMs for Real Sprint Planning

**Theme:** [OpenEnv Hackathon](https://huggingface.co/openenv) · **#3.1 World Modeling — Professional Tasks**  
**Live environment (HF Space):** <https://huggingface.co/spaces/vikramsrini/sprint_planning_env>  
**Training notebook:** <https://huggingface.co/spaces/vikramsrini/sprint_planning_env/blob/main/colab_train_sprintboard_grpo.ipynb>  
**Trained adapter:** <https://huggingface.co/vikramsrini/sprintboard-qwen25-1.5b-lora>

## Why this environment?

Sprint planning looks simple, but in practice it is a partially observable, multi-step reasoning problem:
- You see symptoms first, not root cause.
- You must inspect team capacity, dependencies, velocity, and priorities.
- One wrong change can break multiple constraints.

This is exactly the kind of workflow where LLMs need a durable internal world model, not shallow one-shot prompting.

## What SprintBoard simulates

SprintBoard is an OpenEnv-compliant environment where the model acts like a Scrum Master:
- **15 scenarios** (easy to hard, including compound failures)
- **18 free-form text commands** (no multiple-choice action list)
- **Deterministic 3-axis grading** on investigation, planning quality, and process
- **Board-state-based scoring** (hard to game with keyword tricks)

## Training approach

We train `Qwen/Qwen2.5-1.5B-Instruct` with LoRA using a practical two-phase recipe:
1. **SFT warm-start** on expert SprintBoard command traces (teaches strict command format and basic policy shape)
2. **GRPO refinement** with a **multi-step trajectory reward**: generate a full plan, execute it in the live environment, and reward by final grader score (+ format shaping)

This setup keeps training grounded in environment outcomes instead of token-level imitation alone.

## Results

From committed run artifacts (`assets/training_summary.json`):
- **Baseline random mean:** `0.3925`
- **After SFT mean:** `0.9862`
- **After GRPO mean:** `0.9866`

Takeaway: SFT gives the major jump; GRPO stabilizes and refines policy quality near ceiling.

## Try it yourself

In the Space, pick any task and run **Auto-Solve** to watch end-to-end command execution against the same environment used for training.  
For reproducibility, run the linked notebook and regenerate the plots/summary artifacts.

---

`#OpenEnv` `#WorldModeling` `#RLHF` `#TRL` `#GRPO` `#PEFT` `#LLM` `#Agile`
