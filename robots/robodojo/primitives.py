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

"""RoboDojo primitives built on the native EvalEnv RPC surface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.robot_spec import ROBODOJO_CAMERA_NAMES

#: Action keys per arm side accepted by the native ``take_action``.
_ARM_KEYS = {
    "left": ("left_ee_pose", "left_arm_joint_state", "left_ee_joint_state"),
    "right": ("right_ee_pose", "right_arm_joint_state", "right_ee_joint_state"),
}


class RoboDojoPrimitives:
    """Compose RoboDojo operations from the env RPC API.

    The Pi_05 VLA is driven through the env server's policy proxy, so —
    unlike RoboTwin — this class owns no model client.
    """

    def __init__(
        self,
        *,
        env: RoboDojoEnvClient,
        seed: int,
        check_cancelled: Callable[[], None],
    ):
        self.env = env
        self.seed = int(seed)
        self._check_cancelled = check_cancelled
        self.policy_actions = 0
        self.native_actions = 0
        self._recording = False
        self._frames: list[np.ndarray] = []

    # ---- recording ----

    def start_recording(self) -> None:
        self._recording = True
        self._frames = []

    def record_frame(self, rgb: Any) -> None:
        self._frames.append(np.ascontiguousarray(np.asarray(rgb)))

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._frames)
        self._recording = False
        self._frames = []
        return frames

    def frame_slice(self, start: int) -> list[np.ndarray]:
        return list(self._frames[int(start) :])

    def _record_head_frame(self) -> None:
        if not self._recording:
            return
        try:
            self.record_frame(self.env.render_camera("head"))
        except RuntimeError:
            # Episode already terminal; the final obs frame is enough.
            pass

    # ---- lifecycle ----

    def reset(self) -> dict[str, Any]:
        """Reset the RoboDojo episode and return the native info."""
        _, info = self.env.reset()
        return {**info, "success": True}

    @staticmethod
    def _completion(
        *, requested: int, executed: int, status: dict[str, Any]
    ) -> dict[str, Any]:
        step_lim = status.get("step_lim")
        budget_exhausted = step_lim is not None and int(
            status.get("take_action_cnt", 0)
        ) >= int(step_lim)
        completed = executed == requested
        if status.get("eval_success") is True:
            stop_reason = "native_success"
        elif status.get("episode_ended") and budget_exhausted:
            stop_reason = "budget_exhausted"
        elif status.get("episode_ended"):
            stop_reason = "native_failure"
        elif completed:
            stop_reason = "completed"
        else:
            stop_reason = "runtime_failure"
        return {
            "completed": completed,
            "requested_steps": requested,
            "executed_steps": executed,
            "stop_reason": stop_reason,
        }

    def status(self) -> dict[str, Any]:
        """Return the native episode status plus action counters."""
        return {
            **self.env.last_info["episode_status"],
            "policy_actions": self.policy_actions,
            "native_actions": self.native_actions,
        }

    def finish(self, *, status: str, summary: str) -> dict[str, Any]:
        """Finish the Planner run and verify success against native status."""
        requested_success = status.lower() == "success"
        try:
            native = self.status()
        except Exception as error:  # The terminal tool must still stop the Planner.
            return {
                "_finish": True,
                "status": "error",
                "summary": summary,
                "requested_status": status,
                "requested_success": requested_success,
                "runtime_error": f"{type(error).__name__}: {error}",
            }
        verified_success = native.get("eval_success") is True
        reported_status = (
            "success"
            if verified_success
            else ("failure" if requested_success else status)
        )
        return {
            "_finish": True,
            "status": reported_status,
            "summary": summary,
            "requested_success": requested_success,
            "success": verified_success,
            "episode_status": native,
        }

    # ---- actions ----

    def pi05_act(
        self,
        *,
        chunks: int = 2,
        chunk_steps: int | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Run Pi_05 joint-action chunks with the native task instruction.

        The chunk loop executes server-side; one RPC covers ``chunks`` policy
        inferences. The optional prompt is recorded but never sent to the
        policy (the native instruction is authoritative).
        """
        del prompt
        if int(chunks) < 1:
            raise ValueError("chunks must be at least 1")
        status = self.env.last_info["episode_status"]
        if status.get("episode_ended"):
            return {
                **self._completion(requested=0, executed=0, status=status),
                "success": True,
                "episode_status": status,
                "note": "episode already terminal before pi05_act",
            }
        result = self.env.policy_act(chunks=int(chunks), chunk_steps=chunk_steps)
        executed = int(result.get("executed_actions", 0))
        self.policy_actions += executed
        self.native_actions += executed
        self._record_head_frame()
        status = self.env.last_info["episode_status"]
        return {
            **self._completion(requested=executed, executed=executed, status=status),
            "success": True,
            "chunks_used": result.get("chunks_used"),
            "episode_status": status,
        }

    def move_eef(
        self,
        *,
        arm: str,
        xyz: list[float],
        quat: list[float] | None = None,
        gripper: float | None = None,
        steps: int = 1,
    ) -> dict[str, Any]:
        """Move one arm's end effector to a world-frame pose.

        ``xyz`` is metres in the env frame and ``quat`` the ``[qw,qx,qy,qz]``
        orientation (defaults to the current orientation). ``gripper`` is the
        normalized 0-1 opening; omit it to keep the current opening.
        """
        if arm not in _ARM_KEYS:
            raise ValueError("arm must be 'left' or 'right'")
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        self._check_cancelled()
        self._require_active()
        state = self.env.last_obs.get("state", {})
        pose_key = f"{arm}_ee_pose"
        gripper_key = f"{arm}_ee_joint_state"
        current = np.asarray(state[pose_key], dtype=np.float64)
        if current.shape != (7,) or not np.isfinite(current).all():
            raise ValueError(
                f"state['{pose_key}'] must be finite with shape (7,), got {current}"
            )
        # Native take_action requires the gripper key alongside every ee-pose
        # action; keep the current opening when the caller does not set one.
        current_gripper = float(
            np.asarray(state[gripper_key]).reshape(-1)[0]
        )
        gripper_value = (
            float(np.clip(gripper, 0.0, 1.0)) if gripper is not None
            else current_gripper
        )
        target_xyz = np.asarray(xyz, dtype=np.float64)
        if target_xyz.shape != (3,) or not np.isfinite(target_xyz).all():
            raise ValueError("xyz must be finite with shape (3,)")
        if quat is None:
            target_quat = current[3:]
        else:
            target_quat = np.asarray(quat, dtype=np.float64)
            if target_quat.shape != (4,) or not np.isfinite(target_quat).all():
                raise ValueError("quat must be finite with shape (4,)")
        target = np.concatenate([target_xyz, target_quat]).tolist()

        actions = []
        for index in range(1, int(steps) + 1):
            alpha = index / int(steps)
            pose = (
                (current[:3] * (1 - alpha) + target_xyz * alpha).tolist()
                + current[3:].tolist()
                if steps > 1 and index < steps
                else target
            )
            actions.append(
                {
                    f"{arm}_ee_pose": pose,
                    f"{arm}_ee_joint_state": [gripper_value],
                }
            )
        execution = self.env.chunk_step(actions)
        executed = int(execution[4].get("executed_actions", 0))
        self.native_actions += executed
        self._record_head_frame()
        self._check_cancelled()
        status = self.env.last_info["episode_status"]
        final_state = self.env.last_obs.get("state", {})
        final_pose = np.asarray(final_state[pose_key], dtype=np.float64)
        return {
            **self._completion(requested=len(actions), executed=executed, status=status),
            "success": True,
            "action_type": "ee_pose",
            "final_eef_xyz": final_pose[:3].tolist(),
            "final_dist_m": float(
                np.linalg.norm(final_pose[:3] - target_xyz)
            ),
            "episode_status": status,
        }

    def set_gripper(
        self,
        *,
        arm: str,
        val: float,
        steps: int = 5,
    ) -> dict[str, Any]:
        """Interpolate one gripper to a normalized opening value.

        RoboDojo infers the action type from the arm keys, so every gripper
        action re-sends the current EE pose as a no-op carrier.
        """
        if arm not in _ARM_KEYS:
            raise ValueError("arm must be 'left' or 'right'")
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        self._check_cancelled()
        self._require_active()
        state = self.env.last_obs.get("state", {})
        pose_key = f"{arm}_ee_pose"
        gripper_key = f"{arm}_ee_joint_state"
        pose = np.asarray(state[pose_key], dtype=np.float64).tolist()
        if f"{gripper_key}" not in state:
            raise ValueError(f"state is missing {gripper_key}")
        current = float(np.asarray(state[gripper_key]).reshape(-1)[0])
        target = float(np.clip(val, 0.0, 1.0))
        actions = []
        for index in range(1, int(steps) + 1):
            value = current + (target - current) * index / int(steps)
            actions.append(
                {
                    f"{arm}_ee_pose": pose,
                    f"{arm}_ee_joint_state": [float(value)],
                }
            )
        execution = self.env.chunk_step(actions)
        executed = int(execution[4].get("executed_actions", 0))
        self.native_actions += executed
        self._record_head_frame()
        self._check_cancelled()
        status = self.env.last_info["episode_status"]
        final_state = self.env.last_obs.get("state", {})
        return {
            **self._completion(requested=len(actions), executed=executed, status=status),
            "success": True,
            "gripper_val": float(
                np.asarray(final_state[gripper_key]).reshape(-1)[0]
            ),
            "episode_status": status,
        }

    def release(self, *, arm: str, val: float = 1.0, steps: int = 5) -> dict[str, Any]:
        """Open one gripper to the requested release value."""
        return self.set_gripper(arm=arm, val=val, steps=steps)

    # ---- helpers ----

    def _require_active(self) -> None:
        if self.env.terminated or self.env.truncated:
            raise RuntimeError("RoboDojo episode is terminal; reset is required")
