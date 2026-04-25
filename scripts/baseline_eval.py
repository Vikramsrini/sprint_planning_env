"""SprintBoard — Baseline evaluation harness.

Runs three policies against every task in TASK_REGISTRY and produces:

  - assets/baseline_scores.csv   : per-task scores + mean reward
  - assets/baseline_vs_heuristic.png : side-by-side bar chart

Policies evaluated
------------------
1. random_policy   — picks any valid command verb (no domain knowledge)
2. heuristic_bot   — the canned OmegaSprintBot trajectories shipped in agent.py
3. (post-train)    — column reserved; populated by the Colab notebook

Why this matters
----------------
A submission that only shows a training-loss curve cannot prove the agent learned
anything meaningful about the *task*. By recording the scores of two reference
policies up-front, the post-training reward curve has something to be compared
against. The notebook produces a third bar on the same axes; the visual delta is
the headline result we ship in the README.

The script is deliberately stdlib + matplotlib only so it can be re-run from a
fresh checkout without any extra dependencies (`python -m scripts.baseline_eval`).
"""
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sprint_planning_env.models import SprintAction
from sprint_planning_env.server.environment import SprintBoardEnvironment
from sprint_planning_env.server.tasks import TASK_REGISTRY
from sprint_planning_env.agent import TASKS_COMMANDS

# ── Fixed action vocabulary used by the random-policy baseline. ─────────────
# We sample full commands (verb + plausible argument) so the baseline isn't
# trivially zero — even a random agent occasionally lands on a useful query.
RANDOM_COMMANDS: list[str] = [
    "LIST_BACKLOG", "VIEW_TEAM", "VIEW_VELOCITY", "VIEW_SPRINT", "VIEW_BUGS",
    *(f"VIEW_STORY {sid}" for sid in ("101", "102", "103", "104", "105", "108", "111", "114")),
    *(f"CHECK_DEPS {sid}" for sid in ("101", "102", "105", "110")),
    *(f"ESTIMATE {sid} {pts}" for sid in ("103", "106", "107") for pts in (3, 5, 8)),
    *(f"ASSIGN {sid} {dev}" for sid in ("101", "102", "104", "111")
                              for dev in ("Alice", "Bob", "Diana", "Eve")),
    *(f"ADD_TO_SPRINT {sid}" for sid in ("101", "102", "105", "111")),
    *(f"REMOVE_FROM_SPRINT {sid}" for sid in ("103", "104", "112", "115")),
    "FLAG_RISK 114 vague",
    "FINALIZE_SPRINT",
]


def run_policy(policy_fn, max_steps: int = 15, seed: int = 0) -> dict[str, float]:
    """Run `policy_fn(task_id, step) -> command` once per task."""
    env = SprintBoardEnvironment()
    scores: dict[str, float] = {}
    for tid in TASK_REGISTRY:
        rng = random.Random(seed + hash(tid) % 1024)
        env.reset(task_id=tid, seed=seed)
        cumulative = 0.0
        last_grader = 0.0
        for step in range(1, max_steps + 1):
            cmd = policy_fn(tid, step, rng)
            obs = env.step(SprintAction(command=cmd))
            cumulative += obs.reward or 0.0
            grader = obs.metadata.get("grader_score")
            if grader is not None:
                last_grader = float(grader)
            if obs.done:
                break
        # Use the grader score when available; fall back to cumulative reward.
        scores[tid] = last_grader if last_grader > 0 else cumulative
    return scores


def random_policy(task_id: str, step: int, rng: random.Random) -> str:
    return rng.choice(RANDOM_COMMANDS)


def heuristic_policy(task_id: str, step: int, rng: random.Random) -> str:
    """Replay the curated OmegaSprintBot script for this task."""
    queue = TASKS_COMMANDS.get(task_id, ["FINALIZE_SPRINT"])
    idx = min(step - 1, len(queue) - 1)
    return queue[idx]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_bars(rand_scores, heur_scores, out_path: Path) -> None:
    """Save a clear, judge-ready bar chart at `out_path`."""
    import matplotlib.pyplot as plt
    import numpy as np

    task_ids = list(TASK_REGISTRY.keys())
    rand_vals = [rand_scores[t] for t in task_ids]
    heur_vals = [heur_scores[t] for t in task_ids]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(task_ids))
    width = 0.38
    ax.bar(x - width / 2, rand_vals, width=width, label=f"Random policy (mean={sum(rand_vals)/len(rand_vals):.2f})",
           color="#94a3b8", edgecolor="#475569")
    ax.bar(x + width / 2, heur_vals, width=width, label=f"Heuristic bot (mean={sum(heur_vals)/len(heur_vals):.2f})",
           color="#7c3aed", edgecolor="#5b21b6")

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("task_", "T") for t in task_ids])
    ax.set_ylabel("SprintBoard grader score (0–1)")
    ax.set_xlabel("Task")
    ax.set_ylim(0, 1.0)
    ax.set_title("SprintBoard baselines: random vs heuristic policy across 15 tasks")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    print("Running random policy across 15 tasks ...", flush=True)
    rand_scores = run_policy(random_policy, seed=7)
    print("Running heuristic policy across 15 tasks ...", flush=True)
    heur_scores = run_policy(heuristic_policy, seed=7)

    rows = []
    for tid in TASK_REGISTRY:
        rows.append({
            "task_id": tid,
            "name": TASK_REGISTRY[tid]["name"],
            "difficulty": TASK_REGISTRY[tid]["difficulty"],
            "random_policy": round(rand_scores[tid], 4),
            "heuristic_bot": round(heur_scores[tid], 4),
        })
    out_dir = ROOT / "assets"
    write_csv(out_dir / "baseline_scores.csv", rows)
    plot_bars(rand_scores, heur_scores, out_dir / "baseline_vs_heuristic.png")

    rand_mean = sum(rand_scores.values()) / len(rand_scores)
    heur_mean = sum(heur_scores.values()) / len(heur_scores)
    print(f"Random policy mean score : {rand_mean:.3f}")
    print(f"Heuristic bot mean score : {heur_mean:.3f}")
    print(f"Wrote {out_dir/'baseline_scores.csv'}")
    print(f"Wrote {out_dir/'baseline_vs_heuristic.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
