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


def evaluate(task_ids: List[str], max_steps: int) -> Dict[str, object]:
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
        trained_vals = [summary["trained_like"]["avg_final_score"], summary["trained_like"]["success_rate"]]

        x = range(len(labels))
        width = 0.36

        plt.figure(figsize=(7, 4))
        plt.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline")
        plt.bar([i + width / 2 for i in x], trained_vals, width=width, label="Trained-like")
        plt.xticks(list(x), labels)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Score")
        plt.title("SprintBoard Baseline vs Trained-like Comparison")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_ids = list_task_ids()[: args.task_limit]
    summary = evaluate(task_ids=task_ids, max_steps=args.max_steps)
    save_artifacts(summary, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
