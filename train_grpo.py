import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List

sys.path.insert(0, os.path.dirname(__file__))

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.server.environment import SprintBoardEnvironment
    from sprint_planning_env.server.tasks import list_task_ids
except ModuleNotFoundError:
    from models import SprintAction
    from server.environment import SprintBoardEnvironment
    from server.tasks import list_task_ids


SYSTEM_PROMPT = """You are an expert Agile sprint planning agent.
Return EXACTLY ONE command as plain text (no markdown, no explanation).

Allowed command verbs:
LIST_BACKLOG, VIEW_STORY, CHECK_DEPS, VIEW_TEAM, VIEW_VELOCITY, VIEW_SPRINT,
VIEW_BUGS, VIEW_EPIC, SEARCH_BACKLOG, ESTIMATE, ASSIGN, UNASSIGN,
ADD_TO_SPRINT, REMOVE_FROM_SPRINT, SET_PRIORITY, FLAG_RISK, DECOMPOSE,
FINALIZE_SPRINT.

Goal: maximize SprintBoard reward by investigating first, then executing safe planning actions.
"""

VALID_COMMAND_PREFIXES = {
    "LIST_BACKLOG",
    "VIEW_STORY",
    "CHECK_DEPS",
    "VIEW_TEAM",
    "VIEW_VELOCITY",
    "VIEW_SPRINT",
    "VIEW_BUGS",
    "VIEW_EPIC",
    "SEARCH_BACKLOG",
    "ESTIMATE",
    "ASSIGN",
    "UNASSIGN",
    "ADD_TO_SPRINT",
    "REMOVE_FROM_SPRINT",
    "SET_PRIORITY",
    "FLAG_RISK",
    "DECOMPOSE",
    "FINALIZE_SPRINT",
}


def _extract_task_id(prompt: str) -> str:
    match = re.search(r"TASK_ID:\s*(task_\d+)", prompt)
    return match.group(1) if match else "task_1"


def _normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("content", "text", "completion"):
            if key in completion and completion[key]:
                return str(completion[key])
        return str(completion)
    if isinstance(completion, list):
        if not completion:
            return ""
        return _normalize_completion(completion[0])
    return str(completion)


def _extract_first_command(raw_completion: Any) -> str:
    text = _normalize_completion(raw_completion)
    text = text.strip().strip("`").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def reward_score_function(prompts, completions, **kwargs) -> List[float]:
    """Reward generated command by executing it in SprintBoard."""
    rewards: List[float] = []
    env = SprintBoardEnvironment()

    for prompt, completion in zip(prompts, completions):
        task_id = _extract_task_id(str(prompt))
        command = _extract_first_command(completion)
        env.reset(task_id=task_id)
        result = env.step(SprintAction(command=command))

        reward = float(result.reward or 0.0)
        if result.done and result.metadata.get("grader_score") is not None:
            reward = max(reward, float(result.metadata["grader_score"]))
        rewards.append(max(0.0, min(1.0, reward)))

    return rewards


def format_reward_function(prompts, completions, **kwargs) -> List[float]:
    """Extra reward for syntactically valid command outputs."""
    rewards: List[float] = []
    for completion in completions:
        command = _extract_first_command(completion).upper()
        command_prefix = command.split()[0] if command else ""
        rewards.append(0.10 if command_prefix in VALID_COMMAND_PREFIXES else 0.0)
    return rewards


def build_training_dataset(task_ids: List[str], max_samples: int) -> Dataset:
    prompts = []
    for task_id in task_ids:
        prompts.append(
            (
                f"{SYSTEM_PROMPT}\n"
                f"TASK_ID: {task_id}\n"
                "You are at the beginning of the episode. "
                "What is your next best command?"
            )
        )
        if len(prompts) >= max_samples:
            break
    return Dataset.from_dict({"prompt": prompts})


def save_training_artifacts(trainer: GRPOTrainer, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history

    metrics_file = artifacts_dir / "training_metrics.jsonl"
    with metrics_file.open("w", encoding="utf-8") as f:
        for row in log_history:
            f.write(json.dumps(row) + "\n")

    reward_series = []
    loss_series = []
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        reward_keys = [k for k in row.keys() if "reward" in k.lower()]
        loss_keys = [k for k in row.keys() if "loss" in k.lower()]

        reward_value = next((row[k] for k in reward_keys if isinstance(row.get(k), (int, float))), None)
        loss_value = next((row[k] for k in loss_keys if isinstance(row.get(k), (int, float))), None)

        if reward_value is not None:
            reward_series.append((step, float(reward_value)))
        if loss_value is not None:
            loss_series.append((step, float(loss_value)))

    summary = {
        "num_log_rows": len(log_history),
        "num_reward_points": len(reward_series),
        "num_loss_points": len(loss_series),
        "last_reward": reward_series[-1][1] if reward_series else None,
        "last_loss": loss_series[-1][1] if loss_series else None,
    }
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        if reward_series:
            x, y = zip(*reward_series)
            plt.figure(figsize=(7, 4))
            plt.plot(x, y, marker="o")
            plt.title("SprintBoard GRPO Reward Curve")
            plt.xlabel("Training Step")
            plt.ylabel("Reward")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(artifacts_dir / "reward_curve.png", dpi=160)
            plt.close()

        if loss_series:
            x, y = zip(*loss_series)
            plt.figure(figsize=(7, 4))
            plt.plot(x, y, marker="o")
            plt.title("SprintBoard GRPO Loss Curve")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(artifacts_dir / "loss_curve.png", dpi=160)
            plt.close()
    except Exception as exc:
        print(f"[WARN] Could not generate plots: {exc}")


def train(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    task_ids = list_task_ids()
    dataset = build_training_dataset(task_ids=task_ids, max_samples=args.max_samples)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        report_to=[],
        dataloader_num_workers=0,  # Disable multiprocessing to fix dill pickling issues
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[reward_score_function, format_reward_function],
        args=training_args,
        train_dataset=dataset,
    )

    print(f"Starting GRPO training for SprintBoard using {len(dataset)} prompts...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))
    save_training_artifacts(trainer, args.output_dir)
    print(f"Done. Artifacts saved under: {args.output_dir}/artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRPO policy on SprintBoard")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="runs/sprintboard-grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-samples", type=int, default=15)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--logging-steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
