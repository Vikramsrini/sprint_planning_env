"""
SprintBoard — FastAPI application exposing the OpenEnv-compatible HTTP + WebSocket API.

Serves the complete SprintBoard environment with:
- Standard OpenEnv protocol: /reset, /step, /state (HTTP) and /ws (WebSocket)
  provided by openenv-core's create_app()
- Custom endpoints: /tasks, /grader for hackathon spec compliance

Architecture: create_app() handles per-session environment instances for WebSocket
connections (each EnvClient gets its own SprintBoardEnvironment).
"""

import logging
import os
import sys
import threading
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.responses import JSONResponse

from openenv.core.env_server.http_server import create_app

from sprint_planning_env.models import SprintAction, SprintObservation
from sprint_planning_env.server.environment import SprintBoardEnvironment
from sprint_planning_env.server.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)

# Create the OpenEnv FastAPI app
app = create_app(
    SprintBoardEnvironment,
    SprintAction,
    SprintObservation,
    env_name="sprintboard",
    max_concurrent_envs=1,
)

# Persistent singleton for HTTP endpoints
_env = SprintBoardEnvironment()
_env_lock = threading.Lock()


@app.get("/")
async def root():
    """Return environment metadata and available endpoints."""
    return {
        "project": "SprintBoard",
        "status": "online",
        "version": "0.1.0",
        "description": "Sprint planning and backlog grooming training environment for LLM agents.",
        "endpoints": {
            "/tasks": "List available multi-step planning tasks",
            "/grader": "Retrieve score and feedback for the last completed episode",
            "/reset": "Reset environment state (OpenEnv Standard)",
            "/step": "Execute action and get next observation (OpenEnv Standard)",
            "/ws": "Real-time WebSocket connection (OpenEnv Standard)",
            "/docs": "FastAPI Swagger documentation"
        }
    }


def _serialize_observation(obs: SprintObservation) -> dict:
    """Serialize a SprintObservation to a JSON-friendly dict."""
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()


@app.get("/tasks")
async def list_tasks():
    """Return all available tasks with their metadata."""
    tasks = []
    for tid, task in TASK_REGISTRY.items():
        tasks.append({
            "id": tid,
            "name": task["name"],
            "difficulty": task["difficulty"],
            "description": task["description"],
            "fault_type": task["fault_type"],
        })
    return {
        "tasks": tasks,
        "action_schema": {"command": "string (planning command to execute)"},
        "max_steps": 15,
    }


@app.get("/grader")
async def get_grader_score():
    """Return the grader score for the current/last episode."""
    result = SprintBoardEnvironment.last_grader_result
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No episode has been graded yet. Complete an episode first."},
        )
    return result


def main():
    """Entry point for running the SprintBoard server."""
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("sprint_planning_env.server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
