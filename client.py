"""
SprintBoard — Environment client.

Wraps WebSocket communication with the environment server.
Provides typed step/reset/state methods for the agent.
"""

from typing import Dict, Any
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from sprint_planning_env.models import SprintAction, SprintObservation, SprintState


class SprintBoardEnv(EnvClient[SprintAction, SprintObservation, SprintState]):
    """Client for the SprintBoard environment."""

    def _step_payload(self, action: SprintAction) -> Dict[str, Any]:
        """Convert an Action to the JSON payload expected by the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SprintObservation]:
        """Parse server response into a StepResult with typed observation."""
        obs_data = payload.get("observation", {})
        obs = SprintObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SprintState:
        """Parse server state response into typed State object."""
        return SprintState(**payload)
