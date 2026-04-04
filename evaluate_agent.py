import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sprint_planning_env.server.environment import SprintBoardEnvironment
from sprint_planning_env.server.tasks import TASK_REGISTRY
from sprint_planning_env.agent import run_agent

env = SprintBoardEnvironment()

for task_id in TASK_REGISTRY.keys():
    print(f"Evaluating {task_id}")
    steps = list(run_agent(env, task_id, max_steps=40))
    last_log, metrics, score, step, is_done = steps[-1]
    
    # print final score percentage or reward
    print(f"{task_id} completed.")
    # Extract score from the log snippet
    lines = last_log.split('\n')
    for line in lines:
        if "FINAL SCORE:" in line:
            print(line.strip())
    print("-" * 40)
