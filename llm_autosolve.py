"""
LLM-driven Auto-Solve: Qwen2.5 base + optional PEFT LoRA, same chat template as training.
Used by the Gradio Space when the user clicks Auto-Solve.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Callable, List, Optional, Tuple, Generator, Any

from sprint_planning_env.models import SprintAction
from sprint_planning_env.server.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)

# — Same strings as colab_train_sprintboard_grpo.ipynb (SFT / eval) —
SYSTEM_PROMPT = """You are an expert Agile Scrum Master operating SprintBoard.
Given a sprint-planning scenario, produce an ordered PLAN of commands that
investigates the board and fixes the issues, then finalises the sprint.
Output **one command per line**, no markdown, no commentary, no numbering.
Always end the plan with FINALIZE_SPRINT.

Allowed commands:
  Investigation: LIST_BACKLOG, VIEW_STORY <id>, CHECK_DEPS <id>, VIEW_TEAM,
                 VIEW_VELOCITY, VIEW_SPRINT, VIEW_BUGS, VIEW_EPIC <id>,
                 SEARCH_BACKLOG <kw>
  Planning:      ESTIMATE <id> <pts>, ASSIGN <id> <Name>, UNASSIGN <id>,
                 ADD_TO_SPRINT <id>, REMOVE_FROM_SPRINT <id>,
                 SET_PRIORITY <id> <P0|P1|P2>, FLAG_RISK <id> <reason>,
                 DECOMPOSE <epic_id> "sub1" "sub2" ...,
                 FINALIZE_SPRINT

Strategy:
  1. Investigate first (LIST_BACKLOG, VIEW_TEAM, VIEW_VELOCITY ...).
  2. Then fix the planning issue revealed by the scenario alert.
  3. End with FINALIZE_SPRINT."""

VALID_VERBS = {
    "LIST_BACKLOG", "VIEW_STORY", "CHECK_DEPS", "VIEW_TEAM", "VIEW_VELOCITY",
    "VIEW_SPRINT", "VIEW_BUGS", "VIEW_EPIC", "SEARCH_BACKLOG", "ESTIMATE",
    "ASSIGN", "UNASSIGN", "ADD_TO_SPRINT", "REMOVE_FROM_SPRINT", "SET_PRIORITY",
    "FLAG_RISK", "DECOMPOSE", "FINALIZE_SPRINT",
}

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = "vikramsrini/sprintboard-qwen25-1.5b-lora"

_solve_cache: dict[str, Any] = {"model": None, "tokenizer": None, "error": None}


def build_user_text(task_id: str) -> str:
    task = TASK_REGISTRY[task_id]
    return (
        f"TASK_ID: {task_id}  (difficulty: {task['difficulty']})\n"
        f"SCENARIO ALERT: {task['alert']}\n\n"
        f"Produce the full plan now (one command per line):"
    )


def build_chat_messages(task_id: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_text(task_id)},
    ]


def parse_plan(text: str, max_cmds: int = 24) -> List[str]:
    cmds: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip().strip("`").strip("-*>• ").strip()
        if not line:
            continue
        verb = line.split()[0].upper()
        if verb in VALID_VERBS:
            cmds.append(line)
        if len(cmds) >= max_cmds:
            break
    return cmds


def _adapter_id() -> str:
    return (os.environ.get("SPRINTBOARD_ADAPTER_ID", DEFAULT_ADAPTER) or "").strip()


def _autosolve_mode() -> str:
    return (os.environ.get("SPRINTBOARD_AUTOSOLVE", "llm") or "llm").lower()


def use_llm_autosolve() -> bool:
    if _autosolve_mode() in ("heuristic", "off", "0", "false"):
        return False
    return bool(_adapter_id())


def get_model() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """Load once; returns (model, tokenizer, err)."""
    if _solve_cache["error"]:
        return _solve_cache["model"], _solve_cache["tokenizer"], _solve_cache["error"]
    if _solve_cache["model"] is not None:
        return _solve_cache["model"], _solve_cache["tokenizer"], None

    if not _adapter_id():
        _solve_cache["error"] = "No SPRINTBOARD_ADAPTER_ID set"
        return None, None, _solve_cache["error"]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        _solve_cache["error"] = f"ML deps missing: {e}"
        return None, None, _solve_cache["error"]

    base = os.environ.get("SPRINTBOARD_BASE_MODEL", DEFAULT_BASE).strip() or DEFAULT_BASE
    adapter = _adapter_id()

    try:
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = __import__("torch").float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            model = model.to("cpu")
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    except Exception as e:
        logger.exception("LLM load failed")
        _solve_cache["error"] = str(e)
        return None, None, str(e)

    _solve_cache["model"] = model
    _solve_cache["tokenizer"] = tokenizer
    _solve_cache["error"] = None
    return model, tokenizer, None


def generate_plan(
    model: Any,
    tokenizer: Any,
    task_id: str,
    max_new_tokens: int = 512,
) -> str:
    import torch
    atxt = tokenizer.apply_chat_template(
        build_chat_messages(task_id),
        tokenize=False,
        add_generation_prompt=True,
    )
    max_prompt = int(os.environ.get("SPRINTBOARD_MAX_PROMPT_TOKENS", "1024"))
    dev = next(model.parameters()).device
    inputs = tokenizer(
        atxt, return_tensors="pt", truncation=True, max_length=max_prompt
    ).to(dev)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_llm_agent(
    env,
    task_id: str,
    fmt_metrics: Callable,
    fmt_score: Callable,
    fmt_score_initial: Callable,
) -> Generator[Tuple[str, str, str, str, bool], None, None]:
    """
    Yields the same 5-tuple as `agent.run_agent` for Gradio.
    """
    from sprint_planning_env.agent import run_agent

    if not use_llm_autosolve():
        yield from run_agent(env, task_id, fmt_metrics, fmt_score, fmt_score_initial)
        return

    model, tokenizer, err = get_model()
    if model is None:
        pre = f"⚠ LLM not available ({err}). Using heuristic expert bot.\n\n"
        first = True
        for log, a, b, c, d in run_agent(
            env, task_id, fmt_metrics, fmt_score, fmt_score_initial
        ):
            if first:
                log = pre + log
                first = False
            yield (log, a, b, c, d)
        return

    obs = env.reset(task_id=task_id)
    max_steps = obs.max_steps
    base_name = os.environ.get("SPRINTBOARD_BASE_MODEL", DEFAULT_BASE)
    ad_name = _adapter_id()
    accumulated_log = (
        f"🤖 LLM Auto-Solve · {base_name} + LoRA ({ad_name})\n"
        f"⏳ Generating plan (greedy)…\n"
    )
    yield (
        accumulated_log,
        fmt_metrics(env.board.get_metrics()),
        fmt_score_initial(),
        f"Step 0/{max_steps}",
        False,
    )

    try:
        completion = generate_plan(model, tokenizer, task_id)
    except Exception as e:
        logger.exception("LLM generate")
        accumulated_log += f"✖ Generation failed: {e!s}\n"
        yield (
            accumulated_log,
            fmt_metrics(env.board.get_metrics()),
            fmt_score_initial(),
            f"Step 0/{max_steps}",
            False,
        )
        return

    plan = parse_plan(completion)
    queue: List[str] = list(plan) if plan else ["FINALIZE_SPRINT"]
    accumulated_log += f"✓ Parsed {len(queue)} command(s) from model.\n\n"

    for step in range(1, max_steps + 1):
        if not queue:
            break
        cmd = queue.pop(0)
        obs = env.step(SprintAction(command=cmd))
        entry = f"\n{'─'*40}\n$ {cmd}\n{obs.command_output or obs.error or ''}\n"
        is_done = obs.done or (step == max_steps)
        if is_done:
            grade = obs.metadata.get("grader_score", 0.0) or 0.0
            entry += (
                f"\n{'═'*40}\n"
                f"  EPISODE COMPLETE\n"
                f"  🏆 FINAL SCORE: {grade:.3f}\n"
                f"{'═'*40}\n"
            )
            accumulated_log += entry
            yield (
                accumulated_log,
                fmt_metrics(obs.metrics),
                fmt_score(obs),
                f"Step {step}/{max_steps}",
                True,
            )
            return
        accumulated_log += entry
        yield (
            accumulated_log,
            fmt_metrics(obs.metrics),
            fmt_score(obs),
            f"Step {step}/{max_steps}",
            False,
        )
        time.sleep(0.1)

    if not env.board.is_finalized:
        obs = env.step(SprintAction(command="FINALIZE_SPRINT"))
        grade = obs.metadata.get("grader_score", 0.0) or 0.0
        entry = f"\n{'─'*40}\n$ FINALIZE_SPRINT\n{obs.command_output or obs.error or ''}\n"
        entry += (
            f"\n{'═'*40}\n"
            f"  EPISODE COMPLETE\n"
            f"  🏆 FINAL SCORE: {grade:.3f}\n"
            f"{'═'*40}\n"
        )
        accumulated_log += entry
        yield (
            accumulated_log,
            fmt_metrics(obs.metrics),
            fmt_score(obs),
            f"Step +/{max_steps}",
            True,
        )
