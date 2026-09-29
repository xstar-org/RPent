# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RPC client for one RoboDojo Isaac Sim environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.robodojo.robot_spec import (
    ROBODOJO_CAMERA_NAMES,
    ROBODOJO_READ_TIMEOUT_S,
    ROBODOJO_STATE_CHANGE_TIMEOUT_S,
    ROBODOJO_STATUS_KEYS,
)
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class RoboDojoEnvClient(BaseEnvClient):
    """Client for one native RoboDojo ``EvalEnv`` instance."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "default": ROBODOJO_READ_TIMEOUT_S,
        "env.reset": 900.0,
        "env.step": ROBODOJO_STATE_CHANGE_TIMEOUT_S,
        "env.chunk_step": ROBODOJO_STATE_CHANGE_TIMEOUT_S,
        "env.policy_act": ROBODOJO_STATE_CHANGE_TIMEOUT_S,
    }

    def __init__(self, client: RpcClient, *, expected_meta: dict[str, Any]):
        self.terminated = False
        self.truncated = False
        self._expected_seed = int(expected_meta["seed"])
        super().__init__(client, expected_meta=expected_meta)
        self.server_meta = dict(expected_meta)
        execution = self.server_meta.get("execution", {})
        self.execution_capabilities = (
            dict(execution) if isinstance(execution, dict) else {}
        )
        self.policy_available = bool(
            self._client.call("env.policy_available", timeout_s=self._TIMEOUT_S["default"])
        )

    # ---- helpers ----

    @staticmethod
    def _require_result_tuple(result: Any, size: int, method: str) -> tuple:
        if not isinstance(result, (list, tuple)) or len(result) != size:
            raise TypeError(f"{method} must return a {size}-item tuple, got {result!r}")
        return tuple(result)

    @staticmethod
    def _require_episode_status(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            raise TypeError(f"execution info must be a mapping, got {info!r}")
        status = info.get("episode_status")
        if not isinstance(status, dict):
            raise TypeError(f"episode_status must be a mapping, got {status!r}")
        missing = [key for key in ROBODOJO_STATUS_KEYS if key not in status]
        if missing:
            raise ValueError(f"episode_status is missing {missing}: {status!r}")
        return status

    def _require_active(self) -> None:
        if self.terminated or self.truncated:
            raise RuntimeError("RoboDojo episode is terminal; reset is required")

    @staticmethod
    def _validate_action_dict(action: Any) -> dict[str, Any]:
        if not isinstance(action, dict) or not action:
            raise ValueError("RoboDojo action must be a non-empty mapping")
        allowed = {
            "left_arm_joint_state",
            "left_ee_joint_state",
            "left_ee_pose",
            "right_arm_joint_state",
            "right_ee_joint_state",
            "right_ee_pose",
        }
        unexpected = [key for key in action if key not in allowed]
        if unexpected:
            raise ValueError(f"unexpected RoboDojo action keys: {unexpected}")
        return action

    # ---- RPC methods ----

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to the layout seed and validate the native result."""
        kwargs = {} if seed is None else {"seed": int(seed)}
        result = self._client.call(
            "env.reset",
            kwargs=kwargs,
            timeout_s=self._TIMEOUT_S["env.reset"],
        )
        observation, info = self._require_result_tuple(result, 2, "env.reset")
        if not isinstance(observation, dict):
            raise TypeError(
                f"RoboDojo reset observation must be a mapping, got {observation!r}"
            )
        status = self._require_episode_status(info)
        if seed is not None and status["actual_seed"] != int(seed):
            raise ValueError(f"reset did not use the requested seed: {info!r}")
        if seed is None:
            self._expected_seed = int(status["actual_seed"])
        if not isinstance(info.get("instruction"), str):
            raise TypeError("reset instruction must be a string")
        self.last_obs = observation
        self.last_reset_info = dict(info)
        self.last_info = dict(info)
        self.terminated = False
        self.truncated = False
        return observation, info

    def step(self, action: dict[str, Any]) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Execute one RoboDojo action dict and cache the episode state."""
        self._validate_action_dict(action)
        self._require_active()
        result = self._client.call(
            "env.step",
            args=(action,),
            timeout_s=self._TIMEOUT_S["env.step"],
        )
        result = self._require_result_tuple(result, 5, "env.step")
        _, _, terminated, truncated, info = result
        self._require_episode_status(info)
        self.last_obs = result[0]
        self.last_info = info
        self.terminated |= bool(np.asarray(terminated).any())
        self.truncated |= bool(np.asarray(truncated).any())
        return result

    def chunk_step(
        self, actions: list[dict[str, Any]], *, return_all_frames: bool = False
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Execute a RoboDojo action sequence and cache the final state."""
        normalized = [self._validate_action_dict(action) for action in actions]
        self._require_active()
        result = self._client.call(
            "env.chunk_step",
            args=(normalized,),
            kwargs={"return_all_frames": return_all_frames},
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        result = self._require_result_tuple(result, 5, "env.chunk_step")
        _, _, terminated, truncated, info = result
        self._require_episode_status(info)
        n_executed = info.get("executed_actions")
        if (
            isinstance(n_executed, bool)
            or not isinstance(n_executed, int)
            or not 1 <= n_executed <= len(normalized)
        ):
            raise ValueError(f"invalid RoboDojo chunk result: {result!r}")
        self.last_obs = result[0]
        self.last_info = info
        self.terminated |= bool(np.asarray(terminated).any())
        self.truncated |= bool(np.asarray(truncated).any())
        return result

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        if camera_name not in ROBODOJO_CAMERA_NAMES:
            raise ValueError(
                f"unknown RoboDojo camera {camera_name!r}; "
                f"available={list(ROBODOJO_CAMERA_NAMES)}"
            )
        return super().render_camera(camera_name, depth=depth)

    def get_task_language(self) -> str:
        result = super().get_task_language()
        if not isinstance(result, str):
            raise TypeError(f"RoboDojo task language must be a string: {result!r}")
        return result

    def policy_act(self, *, chunks: int = 1, chunk_steps: int | None = None) -> dict[str, Any]:
        """Run the server-side Pi_05 chunk loop and cache the final state."""
        if not self.policy_available:
            raise RuntimeError(
                "pi05 policy proxy is not available on this env server"
            )
        result = self._client.call(
            "env.policy_act",
            kwargs={"chunks": int(chunks), "chunk_steps": chunk_steps},
            timeout_s=self._TIMEOUT_S["env.policy_act"],
        )
        if not isinstance(result, dict):
            raise TypeError(f"policy_act must return a mapping, got {result!r}")
        status = self._require_episode_status(
            {"episode_status": result.get("episode_status")}
        )
        # The server includes the post-chunk observation so robot poses stay
        # fresh without an extra render RPC; strip it from the tool result so
        # camera arrays never reach the planner context.
        final_obs = result.pop("obs", None)
        if isinstance(final_obs, dict):
            self.last_obs = final_obs
        self.last_info = {**self.last_info, "episode_status": status}
        self.terminated |= bool(status["episode_ended"])
        return result
