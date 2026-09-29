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

"""Agent-facing RoboDojo tool helpers and tool specs."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.session import EnvState, StepRecord
from rpent.tools.toolkit import readonly


def _tool_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, **details},
    }


def _artifact_name(view: str, field: str) -> str:
    suffix = {
        "rgb": ".png",
        "depth": ".npy",
        "world_xyz": ".npy",
        "camera_meta": ".json",
    }[field]
    return f"{view}_{field}{suffix}"


def dump_observation(
    observation: dict[str, Any],
    *,
    env_state: EnvState,
    status: dict[str, Any],
    log: dict[str, Any] | None,
) -> StepRecord:
    """Persist one agent-visible observation without simulator oracle state."""
    step_idx = 0 if env_state.latest_step is None else env_state.latest_step + 1
    paths: dict[str, dict[str, str]] = {}
    view_specs: dict[str, dict[str, Any]] = {}
    for view_name, view in observation["views"].items():
        view_paths: dict[str, str] = {}
        if "rgb" in view:
            name = _artifact_name(view_name, "rgb")
            view_paths["rgb"] = str(env_state.artifact_path(name, step=step_idx))
        paths[view_name] = view_paths
        shape_source = np.asarray(view["rgb"]) if "rgb" in view else None
        if shape_source is not None and shape_source.ndim >= 2:
            view_specs[view_name] = {
                "coordinate_space": view_name,
                "image_shape": [
                    int(shape_source.shape[0]),
                    int(shape_source.shape[1]),
                ],
                "pixel_order": "row_col",
            }

    state = {
        "step_idx": step_idx,
        "task_name": observation["task_name"],
        "task_language": observation["task_language"],
        "robot_state": observation["robot_state"],
        "episode_status": status,
        "artifacts": paths,
        "view_specs": view_specs,
        "log": log,
    }
    eval_success = status.get("eval_success") is True
    with env_state.record_step(
        state=state,
        terminated=eval_success,
        truncated=False,
        command=(log or {}).get("command"),
        result=(log or {}).get("result"),
        elapsed_s=(log or {}).get("elapsed_s"),
        extras={"task_language": observation.get("task_language")},
    ) as recorded_step:
        for view_name, view in observation["views"].items():
            if "rgb" in view:
                env_state.save(
                    _artifact_name(view_name, "rgb"),
                    view["rgb"],
                    step=recorded_step,
                )
    return env_state.get(step_idx)


@readonly
def view_env_state(step: int = -1, *, state: EnvState) -> dict[str, Any]:
    try:
        record = state.get(step)
    except Exception as error:
        return {"error": f"state step not available: {error}"}
    result: dict[str, Any] = {
        "step": record.step_idx,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras.get("task_language"),
    }
    result["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    for slot, views in (
        ("_image_bytes", ("head",)),
        ("_image_cam_bytes", ("left_wrist",)),
        ("_image_wrist_bytes", ("right_wrist",)),
    ):
        name = next(
            (
                _artifact_name(view, "rgb")
                for view in views
                if _artifact_name(view, "rgb") in record.artifacts
            ),
            None,
        )
        if name is not None:
            try:
                result[slot] = state.load_bytes(name, step=record.step_idx)
            except FileNotFoundError:
                pass
    return result


TOOLS_SPEC = [
    {
        "name": "view_env_state",
        "description": (
            "Read one EnvState step and its synchronized RoboDojo observation "
            "artifacts. Step -1 selects the latest entry. Embeds the head, left "
            "wrist, and right wrist RGB images when available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Step number; 0 = initial, -1 = latest.",
                }
            },
        },
    },
    {
        "name": "render",
        "description": "Capture a fresh synchronized RoboDojo observation.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "pi05_act",
        "description": (
            "Run Pi_05 joint-action chunks using the native task instruction. "
            "One call covers `chunks` policy inferences executed next to the "
            "simulator; use it for contact-rich phases such as grasping, "
            "bimanual coordination, and insertion. The optional prompt is "
            "recorded but never sent to the policy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2,
                    "description": (
                        "Number of policy inferences; each executes one action "
                        "chunk (up to ~50 actions). 1 near contact or for small "
                        "corrections; 2 for ordinary progress; 3 only for "
                        "continuity-sensitive phases already moving correctly."
                    ),
                },
                "prompt": {"type": ["string", "null"]},
            },
        },
    },
    {
        "name": "move_eef",
        "description": (
            "Move one arm's end effector to a world-frame xyz (metres) and "
            "optional [qw,qx,qy,qz] orientation. The native IK (CuRobo) solves "
            "the target; multi-step calls interpolate xyz linearly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "quat": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[qw,qx,qy,qz]; defaults to current orientation.",
                },
                "gripper": {
                    "type": ["number", "null"],
                    "description": "Normalized gripper opening 0-1 sent with the move.",
                },
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Interpolation waypoint count for long moves.",
                },
            },
            "required": ["arm", "xyz"],
        },
    },
    {
        "name": "set_gripper",
        "description": (
            "Linearly move one normalized gripper opening to val over several "
            "actions. 0 = closed, 1 = open."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "val": {"type": "number", "minimum": 0, "maximum": 1},
                "steps": {"type": "integer", "minimum": 1, "default": 5},
            },
            "required": ["arm", "val"],
        },
    },
    {
        "name": "release",
        "description": "Open one gripper to 1.0 over several actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "val": {"type": "number", "default": 1.0},
                "steps": {"type": "integer", "minimum": 1, "default": 5},
            },
            "required": ["arm"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Stop the run. A fresh native status query is authoritative; "
            "requesting success cannot override the RoboDojo episode verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
    },
]
