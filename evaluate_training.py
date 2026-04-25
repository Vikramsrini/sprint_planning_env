import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

try:
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.server.environment import SprintBoardEnvironment
    from sprint_planning_env.server.tasks import list_task_ids
    from sprint_planning_env.agent import TASKS_COMMANDS
except ModuleNotFoundError:
    from models import SprintAction
    from server.environment import SprintBoardEnvironment
    from server.tasks import list_task_ids
    from agent import TASKS_COMMANDS

try:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer as HFAutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def _safe_reward(value) -> float:
    if value is None:
        return 0.0
    return float(max(0.0, min(1.0, value)))


def run_policy(command: str, max_steps: int, task_id: str) -> Dict[str, float]:
    env = SprintBoardEnvironment()
    obs = env.reset(task_id=task_id)

    rewards: List[float] = []
    steps = 0
    while not obs.done and steps < max_steps:
        obs = env.step(SprintAction(command=command))
        rewards.append(_safe_reward(obs.reward))
        steps += 1

    final_score = 0.0
    if obs.metadata and obs.metadata.get("grader_score") is not None:
        final_score = _safe_reward(obs.metadata["grader_score"])
    elif rewards:
        final_score = _safe_reward(rewards[-1])

    return {
        "task_id": task_id,
        "steps": float(steps),
        "episode_reward_mean": mean(rewards) if rewards else 0.0,
        "final_score": final_score,
        "success": 1.0 if final_score >= 0.7 else 0.0,
    }


def run_heuristic_policy(max_steps: int, task_id: str) -> Dict[str, float]:
    env = SprintBoardEnvironment()
    obs = env.reset(task_id=task_id)

    command_plan = TASKS_COMMANDS.get(task_id, ["LIST_BACKLOG", "FINALIZE_SPRINT"])
    rewards: List[float] = []
    steps = 0

    for command in command_plan:
        if obs.done or steps >= max_steps:
            break
        obs = env.step(SprintAction(command=command))
        rewards.append(_safe_reward(obs.reward))
        steps += 1

    final_score = 0.0
    if obs.metadata and obs.metadata.get("grader_score") is not None:
        final_score = _safe_reward(obs.metadata["grader_score"])
    elif rewards:
        final_score = _safe_reward(rewards[-1])

    return {
        "task_id": task_id,
        "steps": float(steps),
        "episode_reward_mean": mean(rewards) if rewards else 0.0,
        "final_score": final_score,
        "success": 1.0 if final_score >= 0.7 else 0.0,
    }


_SYSTEM_PROMPT = """You are an expert Agile sprint planning agent.
Return EXACTLY ONE command as plain text (no markdown, no explanation).

Allowed command verbs:
LIST_BACKLOG, VIEW_STORY, CHECK_DEPS, VIEW_TEAM, VIEW_VELOCITY, VIEW_SPRINT,
VIEW_BUGS, VIEW_EPIC, SEARCH_BACKLOG, ESTIMATE, ASSIGN, UNASSIGN,
ADD_TO_SPRINT, REMOVE_FROM_SPRINT, SET_PRIORITY, FLAG_RISK, DECOMPOSE,
FINALIZE_SPRINT.

Goal: maximize SprintBoard reward by investigating first, then executing safe planning actions.
"""


def _extract_first_command(text: str) -> str:
    text = text.strip().strip("`").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _format_obs_prompt(obs, task_id: str) -> str:
    metrics_str = ", ".join(f"{k}={v}" for k, v in obs.metrics.items()) if obs.metrics else "N/A"
    return (
        f"{_SYSTEM_PROMPT}\n"
        f"TASK_ID: {task_id}\n"
        f"Alert: {obs.alert}\n"
        f"Metrics: {metrics_str}\n"
        f"Last Output: {obs.command_output[:400]}\n"
        f"Step: {obs.step_number}/{obs.max_steps}\n"
        "What is your next best command?"
    )


def run_trained_model_policy(model, tokenizer, max_steps: int, task_id: str) -> Dict[str, float]:
    env = SprintBoardEnvironment()
    obs = env.reset(task_id=task_id)

    rewards: List[float] = []
    steps = 0

    while not obs.done and steps < max_steps:
        prompt = _format_obs_prompt(obs, task_id)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        command = _extract_first_command(generated) or "LIST_BACKLOG"
        obs = env.step(SprintAction(command=command))
        rewards.append(_safe_reward(obs.reward))
        steps += 1

    final_score = 0.0
    if obs.metadata and obs.metadata.get("grader_score") is not None:
        final_score = _safe_reward(obs.metadata["grader_score"])
    elif rewards:
        final_score = _safe_reward(rewards[-1])

    return {
        "task_id": task_id,
        "steps": float(steps),
        "episode_reward_mean": mean(rewards) if rewards else 0.0,
        "final_score": final_score,
        "success": 1.0 if final_score >= 0.7 else 0.0,
    }


def evaluate(task_ids: List[str], max_steps: int, model=None, tokenizer=None) -> Dict[str, object]:
    baseline_cmd = "FINALIZE_SPRINT"
    trained_like_name = "heuristic_task_policy"

    baseline = [run_policy(baseline_cmd, max_steps=max_steps, task_id=t) for t in task_ids]
    trained_like = [run_heuristic_policy(max_steps=max_steps, task_id=t) for t in task_ids]

    summary = {
        "baseline": {
            "policy": baseline_cmd,
            "avg_final_score": mean([r["final_score"] for r in baseline]),
            "success_rate": mean([r["success"] for r in baseline]),
            "avg_steps": mean([r["steps"] for r in baseline]),
        },
        "trained_like": {
            "policy": trained_like_name,
            "avg_final_score": mean([r["final_score"] for r in trained_like]),
            "success_rate": mean([r["success"] for r in trained_like]),
            "avg_steps": mean([r["steps"] for r in trained_like]),
        },
        "per_task": {
            "baseline": baseline,
            "trained_like": trained_like,
        },
    }
    if model is not None and tokenizer is not None:
        print("Evaluating trained model checkpoint...")
        trained_model = [
            run_trained_model_policy(model, tokenizer, max_steps=max_steps, task_id=t)
            for t in task_ids
        ]
        summary["trained_model"] = {
            "policy": "grpo_checkpoint",
            "avg_final_score": mean([r["final_score"] for r in trained_model]),
            "success_rate": mean([r["success"] for r in trained_model]),
            "avg_steps": mean([r["steps"] for r in trained_model]),
        }
        summary["per_task"]["trained_model"] = trained_model
    return summary


def save_artifacts(summary: Dict[str, object], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "baseline_vs_trained.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        labels = ["Avg Final Score", "Success Rate"]
        baseline_vals = [summary["baseline"]["avg_final_score"], summary["baseline"]["success_rate"]]
        heuristic_vals = [summary["trained_like"]["avg_final_score"], summary["trained_like"]["success_rate"]]
        trained_model_data = summary.get("trained_model")

        x = list(range(len(labels)))
        n_bars = 3 if trained_model_data else 2
        width = 0.25 if n_bars == 3 else 0.36
        offsets = [-width, 0, width] if n_bars == 3 else [-width / 2, width / 2]

        plt.figure(figsize=(7, 4))
        plt.bar([i + offsets[0] for i in x], baseline_vals, width=width, label="Baseline")
        plt.bar([i + offsets[1] for i in x], heuristic_vals, width=width, label="Heuristic")
        if trained_model_data:
            trained_vals = [trained_model_data["avg_final_score"], trained_model_data["success_rate"]]
            plt.bar([i + offsets[2] for i in x], trained_vals, width=width, label="GRPO Trained")
        plt.xticks(x, labels)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Score")
        plt.title("SprintBoard: Baseline vs Heuristic vs GRPO Trained")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "baseline_vs_trained.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Could not generate baseline comparison plot: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs trained-like performance")
    parser.add_argument("--output-dir", default="runs/eval")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--task-limit", type=int, default=15)
    parser.add_argument("--checkpoint-dir", default=None, help="Path to trained GRPO checkpoint for evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_ids = list_task_ids()[: args.task_limit]

    model = None
    tokenizer = None
    if args.checkpoint_dir:
        if not _TRANSFORMERS_AVAILABLE:
            print("[WARN] transformers not installed; skipping trained model evaluation.")
        else:
            print(f"Loading checkpoint from {args.checkpoint_dir}...")
            tokenizer = HFAutoTokenizer.from_pretrained(args.checkpoint_dir)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model.eval()

    summary = evaluate(task_ids=task_ids, max_steps=args.max_steps, model=model, tokenizer=tokenizer)
    save_artifacts(summary, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
