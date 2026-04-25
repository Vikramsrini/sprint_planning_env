import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List

import torch

sys.path.insert(0, os.path.dirname(__file__))

# Unsloth should be imported before trl/transformers for full patching.
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Unsloth imports for fast training
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("[WARN] Unsloth not installed. Install with: pip install unsloth")

try:
    from sprint_planning_env.models import SprintAction
    from sprint_planning_env.server.environment import SprintBoardEnvironment
    from sprint_planning_env.server.tasks import list_task_ids
except ModuleNotFoundError:
    from models import SprintAction
    from server.environment import SprintBoardEnvironment
    from server.tasks import list_task_ids


SYSTEM_PROMPT = """You are an expert Agile sprint planning agent.
Output a SHORT NUMBERED PLAN of commands, ONE PER LINE, no markdown, no commentary.
Investigate first (LIST_BACKLOG, VIEW_STORY, VIEW_TEAM, CHECK_DEPS, VIEW_VELOCITY, VIEW_SPRINT, VIEW_BUGS, VIEW_EPIC),
then act (ESTIMATE, ASSIGN, UNASSIGN, ADD_TO_SPRINT, REMOVE_FROM_SPRINT, SET_PRIORITY, FLAG_RISK, DECOMPOSE),
and finish with FINALIZE_SPRINT.
Use 4 to 8 steps. STOP IMMEDIATELY after FINALIZE_SPRINT and emit nothing else.

Allowed command verbs:
LIST_BACKLOG, VIEW_STORY, CHECK_DEPS, VIEW_TEAM, VIEW_VELOCITY, VIEW_SPRINT,
VIEW_BUGS, VIEW_EPIC, SEARCH_BACKLOG, ESTIMATE, ASSIGN, UNASSIGN,
ADD_TO_SPRINT, REMOVE_FROM_SPRINT, SET_PRIORITY, FLAG_RISK, DECOMPOSE,
FINALIZE_SPRINT.

Format example:
1. LIST_BACKLOG
2. VIEW_TEAM
3. ESTIMATE US-101 5
4. ASSIGN US-101 Alice
5. FINALIZE_SPRINT
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


_NUMBERED_LINE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+)$")
_MAX_ROLLOUT_STEPS = 15


def _extract_command_sequence(raw_completion: Any) -> List[str]:
    """Parse a numbered/bulleted plan into a clean command list."""
    text = _normalize_completion(raw_completion)
    text = text.strip().strip("`").strip()
    commands: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _NUMBERED_LINE.match(stripped)
        candidate = (match.group(1) if match else stripped).strip()
        if not candidate:
            continue
        first_token = candidate.split()[0].upper()
        if first_token not in VALID_COMMAND_PREFIXES:
            continue
        commands.append(candidate)
        if len(commands) >= _MAX_ROLLOUT_STEPS:
            break
    return commands


def reward_score_function(prompts, completions, **kwargs) -> List[float]:
    """Reward by rolling out the full command sequence in SprintBoard.

    For each completion we execute its commands sequentially in a fresh env
    until the env terminates or we exhaust the plan, then return the final
    grader score plus per-step shaping rewards (clamped to [0,1]).
    """
    rewards: List[float] = []
    env = SprintBoardEnvironment()

    for prompt, completion in zip(prompts, completions):
        task_id = _extract_task_id(str(prompt))
        commands = _extract_command_sequence(completion)
        if not commands:
            rewards.append(0.0)
            continue

        env.reset(task_id=task_id)
        shaped_total = 0.0
        grader_score = None
        steps_taken = 0
        for command in commands:
            result = env.step(SprintAction(command=command))
            steps_taken += 1
            if result.reward is not None:
                shaped_total += max(0.0, float(result.reward))
            if result.done:
                if result.metadata and result.metadata.get("grader_score") is not None:
                    grader_score = float(result.metadata["grader_score"])
                break

        if grader_score is not None:
            reward = grader_score
            if steps_taken <= 12:
                reward += 0.05
        else:
            reward = min(0.5, shaped_total)

        rewards.append(max(0.0, min(1.0, reward)))

    return rewards


def format_reward_function(prompts, completions, **kwargs) -> List[float]:
    """Reward well-formed concise plans that terminate with FINALIZE_SPRINT."""
    rewards: List[float] = []
    for completion in completions:
        text = _normalize_completion(completion)
        commands = _extract_command_sequence(completion)
        if not commands:
            rewards.append(0.0)
            continue
        score = 0.04
        if 3 <= len(commands) <= 10:
            score += 0.04
        if commands[-1].split()[0].upper() == "FINALIZE_SPRINT":
            score += 0.06
        if len(text) <= 600:
            score += 0.04
        unique_verbs = {cmd.split()[0].upper() for cmd in commands}
        if len(unique_verbs) >= max(3, len(commands) // 2):
            score += 0.02
        rewards.append(min(0.20, score))
    return rewards


def build_training_dataset(task_ids: List[str], max_samples: int) -> Dataset:
    prompts: List[str] = []
    while len(prompts) < max_samples:
        for task_id in task_ids:
            prompts.append(
                (
                    f"{SYSTEM_PROMPT}\n"
                    f"TASK_ID: {task_id}\n"
                    "Write the full numbered plan of commands needed to resolve the sprint. "
                    "Stop after FINALIZE_SPRINT."
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
            plt.title("SprintBoard GRPO Reward Curve (Unsloth)")
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
            plt.title("SprintBoard GRPO Loss Curve (Unsloth)")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(artifacts_dir / "loss_curve.png", dpi=160)
            plt.close()
    except Exception as exc:
        print(f"[WARN] Could not generate plots: {exc}")


def load_unsloth_model(model_id: str, args: argparse.Namespace):
    """Load model with Unsloth optimizations."""
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth not installed. Run: pip install unsloth")

    print(f"Loading model with Unsloth: {model_id}")
    print(f"  - 4-bit quantization: {args.load_in_4bit}")
    print(f"  - LoRA rank: {args.lora_rank}")
    print(f"  - Max sequence length: {args.max_prompt_length}")

    use_bf16 = is_bfloat16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Load base model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_prompt_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch_dtype,
    )

    # Build LoRA target modules from the loaded model itself (no blind fallback).
    module_leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    preferred_order = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        # Alternate names seen in some architectures
        "query_key_value", "out_proj", "fc1", "fc2", "dense", "dense_h_to_4h", "dense_4h_to_h",
    ]
    target_modules = [name for name in preferred_order if name in module_leaf_names]
    if not target_modules:
        sample_names = sorted(module_leaf_names)[:80]
        raise RuntimeError(
            "Could not find compatible LoRA target module names in loaded model. "
            f"Sample discovered module names: {sample_names}"
        )

    print(f"Resolved LoRA target_modules={target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,  # Use standard LoRA
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError(
            "Unsloth LoRA adapters were not attached (0 trainable parameters). "
            "Try lowering rank or disabling 4-bit quantization."
        )

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train(args: argparse.Namespace) -> None:
    task_ids = list_task_ids()
    dataset = build_training_dataset(task_ids=task_ids, max_samples=args.max_samples)

    if args.use_unsloth:
        model, tokenizer = load_unsloth_model(args.model_id, args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = args.model_id  # Pass model_id string to GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=args.logging_steps,
        report_to=[],
        # KL constraint to prevent policy drift seen in earlier runs
        beta=args.kl_beta,
        # Save best checkpoint by reward instead of last
        save_strategy="epoch",
        save_total_limit=3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # Unsloth optimizations
        bf16=is_bfloat16_supported() if args.use_unsloth else False,
        fp16=not is_bfloat16_supported() if args.use_unsloth else False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_score_function, format_reward_function],
        args=training_args,
        train_dataset=dataset,
    )

    print(f"Starting GRPO training for SprintBoard using {len(dataset)} prompts...")
    if args.use_unsloth:
        print("Unsloth optimization ENABLED - training will be ~2-5x faster")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))
    save_training_artifacts(trainer, args.output_dir)
    print(f"Done. Artifacts saved under: {args.output_dir}/artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRPO policy on SprintBoard with Unsloth")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="runs/sprintboard-grpo-unsloth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=1)

    # Unsloth-specific arguments
    parser.add_argument("--use-unsloth", action="store_true", default=True,
                        help="Use Unsloth for fast training (default: True)")
    parser.add_argument("--no-unsloth", dest="use_unsloth", action="store_false",
                        help="Disable Unsloth and use standard training")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Use 4-bit quantization with Unsloth (default: True)")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank for Unsloth (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha for Unsloth (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout for Unsloth (default: 0.0 for best compatibility)")
    parser.add_argument("--kl-beta", type=float, default=0.04,
                        help="KL penalty coefficient for GRPO (default: 0.04)")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
