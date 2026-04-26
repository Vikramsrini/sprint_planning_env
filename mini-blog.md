# Training LLMs to Plan Sprints: SprintBoard on OpenEnv (under 2 min read)

**Theme:** [OpenEnv Hackathon](https://huggingface.co/openenv) · **#3.1 World Modeling — Professional Tasks**  
**Space:** <https://huggingface.co/spaces/vikramsrini/sprint_planning_env>  
**This post (source on Hub):** <https://huggingface.co/spaces/vikramsrini/sprint_planning_env/blob/main/mini-blog.md>  
**Code & notebook:** same repo as the Space, plus [`colab_train_sprintboard_grpo.ipynb`](https://huggingface.co/spaces/vikramsrini/sprint_planning_env/blob/main/colab_train_sprintboard_grpo.ipynb)  
**Fine-tuned adapter:** [`vikramsrini/sprintboard-qwen25-1.5b-lora`](https://huggingface.co/vikramsrini/sprintboard-qwen25-1.5b-lora)

## The gap

Sprint planning is a **real** partially observable workflow: the agent only sees a **symptom** alert, then must **investigate** a board (backlog, velocity, team) and **edit** a plan with many dependent commands. Random policies land near **0.4** mean grader score on our 15 tasks; a hand-crafted **reference policy** scores **~0.99** — huge headroom for learning.

## The environment (SprintBoard)

* **15 scenarios**, easy → hard, each with a deterministic **3-axis grader** (investigation / planning / process) over **true board state**, not string tricks.  
* **18 text commands** — no multiple choice; outputs must be valid `LIST_BACKLOG`, `ADD_TO_SPRINT`, …, `FINALIZE_SPRINT` lines.  
* Built on **OpenEnv** (`reset` / `step` / `state`), HTTP + Gradio, **reward shaping** + terminal grade.

## What we trained

We use **Hugging Face TRL**: **SFT** on expert command traces from the env (`TASKS_COMMANDS`) with the **Qwen2.5 Instruct chat template**, then optional **GRPO** with a **full multi-step rollout** reward (grader on the plan executed in the real env) plus a small format bonus. The base model is **`Qwen/Qwen2.5-1.5B-Instruct`**; we ship a **LoRA** on the Hub.

After SFT, **greedy** eval on the 15 training tasks typically reaches a **~0.98+** mean grader — near the expert ceiling — so GRPO is mostly **stability** at the top, not a second 10× jump.

## Try it

Open the **Space**: pick a task, use **Auto-Solve (Qwen+LoRA)** to watch the model drive the same **live** `step` loop as in training, or use **manual** commands. Judges can re-run the **Colab** to reproduce metrics and `assets/training_summary.json`.

---

`#OpenEnv` `#SprintBoard` `#TRL` `#GRPO` `#PEFT` `#Agile` `#LLM`
