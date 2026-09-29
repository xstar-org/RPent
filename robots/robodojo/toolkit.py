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

"""RPent tools for the RoboDojo robot."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.robodojo import tools
from robots.robodojo.primitives import RoboDojoPrimitives
from robots.robodojo.robot_spec import ROBODOJO_CAMERA_NAMES
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

# State-advancing RoboDojo primitives eligible for the recipe. Read-only
# tools are excluded so the recipe records only commands that move robots.
_RECIPE_ACTIONS = {
    "pi05_act",
    "move_eef",
    "set_gripper",
    "release",
}


class RoboDojoToolkit(Toolkit):
    """Common RPent tools plus RoboDojo primitives."""

    _SPECS = {spec["name"]: spec for spec in tools.TOOLS_SPEC}

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
    ):
        state = EnvState(get_output_dir())
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._latest_status: dict[str, Any] = {}
        self._primitives = RoboDojoPrimitives(
            env=runtime_kwargs["env"],
            seed=runtime_kwargs["seed"],
            check_cancelled=self.raise_if_cancelled,
        )
        self._primitives.start_recording()
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        reset_result = {
            **self._primitives.env.last_reset_info,
            "success": True,
        }
        self._register_robodojo_tools()
        initial = self.get_env_state(
            command={"action": "reset"},
            result=reset_result,
            elapsed_s=0.0,
        )
        record = self._state.latest_record()
        if record is not None:
            self._publish_step(record)
        initial_state = initial.get("state")
        if isinstance(initial_state, dict):
            self._latest_status = initial_state.get(
                "episode_status", self._latest_status
            )

    def _register_robodojo_tools(self) -> None:
        self._tools.pop("finish", None)
        self.add_tool(
            "view_env_state",
            self._SPECS["view_env_state"],
            partial(tools.view_env_state, state=self._state),
        )
        for name in (
            "render",
            "pi05_act",
            "move_eef",
            "set_gripper",
            "release",
        ):
            self.add_tool(name, self._SPECS[name], partial(self._step, name))
        self.add_tool("finish", self._SPECS["finish"], self._finish)

    @readonly
    def _finish(self, *, status: str, summary: str) -> dict[str, Any]:
        return self._primitives.finish(status=status, summary=summary)

    def _capture_full_observation(self) -> dict[str, Any]:
        """Assemble the agent-visible observation (rgb views + robot state)."""
        env = self._primitives.env
        obs = env.last_obs
        views: dict[str, dict[str, Any]] = {}
        for camera_name in ROBODOJO_CAMERA_NAMES:
            views[camera_name] = {"rgb": np.asarray(obs["vision"][camera_name]["rgb"])}
        return {
            "views": views,
            "robot_state": obs.get("state", {}),
            "task_name": self._primitives.env.server_meta["task_name"],
            "task_language": env.get_task_language(),
        }

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        status = self._primitives.status()
        self._latest_status = status
        observation = self._capture_full_observation()
        record = tools.dump_observation(
            observation,
            env_state=self._state,
            status=status,
            log={
                "command": command,
                "result": result,
                "elapsed_s": elapsed_s,
            },
        )
        if self._dashboard_events.enabled:
            frames = self._primitives.frame_slice(frame_start)
            if frames:
                self._state.save(
                    f"action_{command['action']}.mp4",
                    frames,
                    step=record.step_idx,
                    fps=20,
                )
        return tools.view_env_state(record.step_idx, state=self._state)

    def close(self) -> None:
        """Flush the per-step frame buffer into ``episode.mp4`` (LIBERO parity)."""
        frames = self._primitives.stop_recording()
        if frames:
            self._state.save("episode.mp4", frames, step=None, fps=20)

    def _step(self, name: str, **kwargs) -> dict[str, Any]:
        self.raise_if_cancelled()
        if name == "render":
            return {"success": True}
        return getattr(self._primitives, name)(**kwargs)

    def write_recipe(self, recipe_tag: str) -> str:
        """Export state-advancing RoboDojo primitives with no error and no
        explicit ``success=False`` from ``EnvState.records()``."""
        recipe = [
            record.command
            for record in self._state.records()
            if isinstance(record.command, dict)
            and record.command.get("action") in _RECIPE_ACTIONS
            and not (
                isinstance(record.result, dict)
                and (
                    record.result.get("error") or record.result.get("success") is False
                )
            )
        ]
        name = f"{recipe_tag}_recipe.jsonl"
        saved = self._state.save(name, recipe, step=None)
        if saved is None:
            raise RuntimeError(f"failed to save RoboDojo recipe artifact: {name}")
        return str(self._state.artifact_path(name, step=None))
