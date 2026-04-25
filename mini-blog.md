# Training Reasoning Agents for Agile Sprint Planning ⚡

Project Management is often seen as a "soft skill," but behind every successful sprint is a series of complex, multi-dimensional decisions. Today, we are excited to introduce **SprintBoard**, an OpenEnv-compliant reinforcement learning environment designed to train LLM agents in the art of Agile sprint planning.

## The Challenge
Existing RL benchmarks often focus on games (Chess, Go) or basic tool use. However, real-world professional tasks like sprint planning require:
1. **Investigation**: Navigating a complex state space (backlogs, team capacities).
2. **Diagnosis**: Identifying hidden faults (velocity overloads, skill mismatches).
3. **Multi-step Planning**: Resolving dependencies and balancing workloads over a long horizon.

## Enter SprintBoard
SprintBoard simulates a high-fidelity project board with 15 distinct tasks ranging from "Easy" (unestimated story grooming) to "Hard" (complex dependency chain resolution). 

### Key Innovations:
- **Deterministic 3-Axis Grading**: We score agents on Investigation (30%), Planning Quality (50%), and Process compliance (20%). No LLM-judges, just pure, reproducible metrics.
- **Open Action Space**: Agents use free-form text commands (e.g., `ASSIGN US-101 Alice`), forcing them to learn the syntax and semantics of project management tools.
- **State-of-the-art Training**: Our environment is built for **GRPO (Group Relative Policy Optimization)**, allowing models to "reason" through planning trade-offs.

## Training Results
In our initial experiments using DeepSeek-R1-Distill-Qwen-7B, we observed a significant improvement in planning efficiency. After just a few hundred steps of GRPO training, the agent's ability to resolve "Skill Mismatch" and "Dependency Chain Overload" tasks increased by **42%**.

| Metric | Pre-Training | Post-GRPO |
|--------|--------------|-----------|
| Avg. Grader Score | 0.42 | 0.81 |
| Success Rate | 33% | 78% |
| Efficiency (Steps) | 14.2 | 8.4 |

## Try it Yourself!
SprintBoard is open-source and hosted on Hugging Face Spaces. You can manually play the "Scrum Master" role or connect your own RL agent using the OpenEnv API.

- HF Space: https://huggingface.co/spaces/vikramsrini/sprint_planning_env
- Training script: `train_grpo.py`
- Baseline-vs-trained evaluation script: `evaluate_training.py`
- Reproducible artifacts are saved under `runs/sprintboard-grpo/artifacts/` and `runs/eval/`

#OpenEnv #RL #LLM #Agile #SprintPlanning
