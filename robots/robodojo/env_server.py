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

"""RPC server owning one RoboDojo Isaac Sim environment.

This script runs inside the RoboDojo shared-storage venv (Isaac Sim 5.1 /
IsaacLab / XPolicyLab). It mirrors ``src/eval_client/main.py`` for the app
bootstrap, then serves the standard RPent env RPC surface plus a
server-side Pi_05 policy proxy: the planner's ``pi05_act`` becomes ONE RPC
that executes the whole update_obs/get_action/take_action chunk loop next
to the simulator, instead of shipping camera frames per sub-action.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Support direct execution from an RPent checkout before package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("robodojo_env_server")

# RPent camera names (artifact space) -> RoboDojo obs vision keys.
_CAMERA_OBS_KEYS = {
    "head": "cam_head",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--robodojo-root", required=True)
    parser.add_argument(
        "--policy-port",
        type=int,
        default=0,
        help=(
            "Pi_05 policy server port on localhost. 0 disables the policy "
            "proxy (env-only smoke runs)."
        ),
    )
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")

    # Isaac Sim app bootstrap (same flags as src/eval_client/main.py).
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    args.device_id = 0
    # Camera capture annotators live in these kit extensions; without them the
    # obs vision dicts come back empty (eval_policy.sh KIT_ARGS parity).
    if not getattr(args, "kit_args", None):
        args.kit_args = (
            "--enable isaacsim.replicator.behavior "
            "--enable isaacsim.sensors.camera"
        )
    return args


def _prepare_robodojo_import_root(robodojo_root: Path) -> None:
    """Make ``env.*``/``task.*``/``utils.*``/``src.*`` importable and stable.

    Mirrors the native ``eval_policy.sh`` PYTHONPATH
    (``PROJECT_ROOT:PROJECT_ROOT/XPolicyLab``). RoboDojo resolves ``env_cfg``
    ymls and writes ``eval_result`` relative to the checkout root, so the
    server process must also run there.
    """
    # Insert in reverse so the final order is [robodojo_root, XPolicyLab_root,
    # ...] — identical to the native eval_policy.sh PYTHONPATH. XPolicyLab
    # ships its own top-level `utils`, which must NOT shadow RoboDojo's.
    for path in (str(robodojo_root / "XPolicyLab"), str(robodojo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(str(robodojo_root))


def _install_noop_policy_client() -> None:
    """Replace WsModelClient with a no-op for env-only smoke runs.

    The native client connects eagerly and would block EvalEnv construction
    for ~15 minutes against a dead endpoint; robots that never call the
    policy do not need a connection at all.
    """
    from client_server.ws import model_client as model_client_module

    class _NoopPolicyClient(model_client_module.WsModelClient):
        def __init__(self, *, url: str, **kwargs: Any):  # noqa: ARG002
            self.url = url
            self._disabled = True

        def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
            if func_name == "reset":
                return None
            raise RuntimeError(
                "pi05 policy proxy is disabled on this env server "
                "(--policy-port 0)"
            )

        def close(self) -> None:
            return None

    model_client_module.WsModelClient = _NoopPolicyClient


def _build_env_cfg(args: argparse.Namespace, run_id: str) -> Any:
    """Assemble the OmegaConf env config exactly like the native eval client."""
    from omegaconf import OmegaConf

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args.env_cfg + ".yml"))
    eval_cfg["task_name"] = args.task_name
    eval_cfg["num_envs"] = 1
    eval_cfg["device_id"] = 0
    eval_cfg["eval_batch"] = False
    eval_cfg["policy_name"] = "Pi_05"
    eval_cfg["additional_info"] = f"rpent_seed={args.seed}"
    eval_cfg["seed"] = int(args.seed)
    eval_cfg["physx_monitor_enabled"] = False

    deploy_cfg = {
        "policy_name": "Pi_05",
        "port": int(args.policy_port),
        "host": "localhost",
        "protocol": "ws",
        "policy_server_url": f"ws://localhost:{int(args.policy_port)}",
        "evaluation_id": run_id,
        "trial_id": f"{args.task_name}-{run_id}",
        "action_case_id": f"{args.task_name}_case",
        "repeat_index": None,
    }

    task_registry = __import__(
        f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"]
    )
    benchmark_path = os.path.join(
        os.environ.get("ROBODOJO_ROOT", "."), "task", str(BENCHMARK)
    )
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml"
                )
            ),
            "scene": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml"
                )
            ),
            "camera": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml"
                )
            ),
            "robot": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml"
                )
            ),
            "task_env": load_yaml(
                task_registry.task_config_path(
                    os.path.join(benchmark_path, "config"), args.task_name
                )
            ),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
    # The native client seeds the sim explicitly (layout seeds arrive via
    # reset()); task code reads sim.seed during scene construction.
    OmegaConf.update(env_cfg, "sim.seed", [int(args.seed)], force_add=True)
    # Same config pipeline as src/eval_client/main.py: randomization first,
    # then task-driven config resolution (step limits, robot config paths).
    env_cfg = process_randomization(env_cfg)
    env_cfg, _eval_num = process_config(env_cfg, task_name=args.task_name)
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        eval_cfg["observation"].get("collect_freq", 0),
        force_add=True,
    )
    return env_cfg


class RoboDojoEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Expose the native RoboDojo EvalEnv over the RPent env RPC contract.

    Isaac Sim is single-threaded: USD/PhysX work must stay on the thread that
    booted the app, so every dispatch (including the task env's lazy scene
    construction on first reset) is executed on the main thread via
    :class:`MainThreadServeMixin` — same pattern as the robocasa facade.
    """

    def __init__(self, env: Any, *, metadata: dict[str, Any]):
        self._env = env
        self._metadata = dict(metadata)
        self._initial_seed = int(metadata["seed"])
        self._last_state: dict[str, Any] = {}
        super().__init__()

    # ---- helpers ----

    def _episode_status(self) -> dict[str, Any]:
        env = self._env
        env_idx = 0
        ended = bool(env.end_flag[env_idx])
        succeeded = ended and bool(env.success[env_idx])
        seeds = getattr(env, "env_seeds", None)
        actual = int(seeds[env_idx]) if seeds else self._initial_seed
        return {
            "eval_success": succeeded,
            "episode_ended": ended,
            "take_action_cnt": int(env.take_action_cnt[env_idx]),
            "step_lim": int(env.step_lim),
            "actual_seed": actual,
        }

    def _obs_payload(self, obs: dict[str, Any] | None) -> dict[str, Any]:
        obs = obs if obs is not None else self._env.get_obs()
        self._last_state = obs.get("state", dict(self._last_state))
        vision = {}
        for obs_key, cam in obs.get("vision", {}).items():
            # RoboDojo names cameras "cam_head"/"cam_left_wrist"/... in the
            # obs and stores the rgb annotator under "color"; RPent artifact
            # names drop the "cam_" prefix and use "rgb".
            view_name = obs_key[4:] if obs_key.startswith("cam_") else obs_key
            entry = {}
            raw_rgb = cam.get("color", cam.get("rgb"))
            if raw_rgb is not None:
                entry["rgb"] = np.asarray(raw_rgb)[:, :, :3]
            if "shape" in cam:
                entry["shape"] = list(cam["shape"])
            vision[view_name] = entry
        return {
            "vision": vision,
            "state": obs.get("state", {}),
            "last_action": obs.get("action", {}),
            "instruction": obs.get("instruction"),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "episode_status": self._episode_status(),
            "instruction": self._env.obs_manager.instruction[0],
        }

    # ---- RPC surface ----

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.policy_act"] = self.policy_act
        self._rpc["env.policy_available"] = self.policy_available

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self._metadata)

    def policy_available(self) -> bool:
        return int(getattr(self._env, "port", 0) or 0) > 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset an episode and return (obs, info).

        Without an explicit layout id the next seed comes from the native
        SeedManager (same order as the official evaluation); an explicit id
        replays that exact layout.
        """
        if seed is None:
            seeds = self._env.seed_manager.get_seeds(max_count=1)
            effective = int(seeds[0]) if seeds else self._initial_seed
        else:
            effective = int(seed)
        self._env.reset(seed=[effective])
        # Native run_eval() evaluates the task reward once at episode start;
        # without this the reward manager returns its default and the first
        # is_episode_end() would report a false success.
        self._env.run_reward()
        obs = self._env.get_obs()
        return self._obs_payload(obs), self._info()

    def step(self, action: dict[str, Any]) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        """Execute one native action dict; returns the gym-style tuple."""
        action = self._validate_action(action)
        if not self._episode_status()["episode_ended"]:
            self._env.take_action(action)
        obs = self._env.get_obs()
        info = self._info()
        status = info["episode_status"]
        return (
            self._obs_payload(obs),
            None,
            bool(status["episode_ended"]),
            False,
            info,
        )

    def chunk_step(
        self,
        actions: list[dict[str, Any]],
        *,
        return_all_frames: bool = False,
    ) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        """Execute action dicts in order, stopping at episode end.

        ``return_all_frames`` is accepted for contract parity but ignored:
        per-frame video is captured at toolkit cadence, not per sub-action.
        """
        del return_all_frames
        normalized = [self._validate_action(action) for action in actions]
        executed = 0
        status = self._episode_status()
        for action in normalized:
            if status["episode_ended"]:
                break
            self._env.take_action(action)
            executed += 1
            status = self._episode_status()
        obs = self._env.get_obs()
        info = {
            **self._info(),
            "executed_actions": executed,
        }
        return (
            self._obs_payload(obs),
            None,
            bool(status["episode_ended"]),
            False,
            info,
        )

    def get_task_language(self) -> str:
        return str(self._env.obs_manager.instruction[0])

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        obs = self._env.get_obs()
        cam = obs.get("vision", {}).get(_CAMERA_OBS_KEYS.get(camera_name, ""), {})
        return {
            "camera_name": camera_name,
            "shape": list(cam.get("shape", [])),
            "intrinsics": "not-collected",
        }

    def render_camera(self, camera_name: str, depth: bool = False) -> Any:
        if depth:
            raise ValueError(
                "RoboDojo verified config collects RGB only; depth is disabled"
            )
        obs_key = _CAMERA_OBS_KEYS.get(camera_name)
        if obs_key is None:
            raise ValueError(
                f"unknown RoboDojo camera {camera_name!r}; "
                f"available={list(_CAMERA_OBS_KEYS)}"
            )
        obs = self._env.get_obs()
        return np.asarray(obs["vision"][obs_key]["rgb"])[:, :, :3]

    def policy_act(self, *, chunks: int = 1, chunk_steps: int | None = None) -> dict[str, Any]:
        """Run the native Pi_05 chunk loop server-side.

        Mirrors ``XPolicyLab/policy/Pi_05/deploy.py::eval_one_episode`` for
        ``chunks`` policy inferences, each executing one action chunk.
        """
        if not self.policy_available():
            raise RuntimeError(
                "pi05 policy proxy is disabled (env server started with "
                "--policy-port 0); pi05_act is unavailable"
            )
        if int(chunks) < 1:
            raise ValueError("chunks must be at least 1")
        client = self._env.model_client
        executed = 0
        chunks_used = 0
        status = self._episode_status()
        for _ in range(int(chunks)):
            if status["episode_ended"]:
                break
            obs = self._env.get_obs()
            client.call(func_name="update_obs", obs=obs)
            actions = client.call(func_name="get_action")
            chunks_used += 1
            for action in actions:
                if chunk_steps is not None and executed >= int(chunk_steps):
                    break
                self._env.take_action(action)
                executed += 1
                status = self._episode_status()
                if status["episode_ended"]:
                    break
            if status["episode_ended"]:
                break
        return {
            "chunks_used": chunks_used,
            "executed_actions": executed,
            "episode_status": self._episode_status(),
            "obs": self._obs_payload(None),
        }

    # ---- validation ----

    _ACTION_KEYS = {
        "joint": ("arm_joint_state", "ee_joint_state"),
        "ee": ("ee_pose", "ee_joint_state"),
    }

    def _validate_action(self, action: Any) -> dict[str, Any]:
        """Validate and complete one action dict for the native take_action.

        Native ``take_action_batch`` reads BOTH arms' keys unconditionally, so
        a partial action (e.g. left-arm-only) is completed from the last
        observed state. Joint and ee-pose keys must not be mixed.
        """
        if not isinstance(action, dict):
            raise TypeError(f"RoboDojo action must be a mapping, got {type(action)}")
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
        if not action:
            raise ValueError("RoboDojo action must set at least one key")
        modes = {
            "joint" if "_arm_joint_state" in key else "ee"
            if key.endswith("_ee_pose")
            else None
            for key in action
        }
        modes.discard(None)
        if len(modes) > 1:
            raise ValueError(
                f"joint and ee-pose keys cannot be mixed in one action: {sorted(action)}"
            )
        mode = modes.pop() if modes else "ee"

        completed = dict(action)
        for arm in ("left", "right"):
            for suffix in self._ACTION_KEYS[mode]:
                key = f"{arm}_{suffix}"
                if key not in completed:
                    if key not in self._last_state:
                        raise ValueError(
                            f"cannot complete action key {key!r}: no observation yet"
                        )
                    completed[key] = np.asarray(
                        self._last_state[key], dtype=np.float64
                    ).tolist()
        for key, value in completed.items():
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(
                    f"action['{key}'] must be a finite 1-D array, got {value!r}"
                )
            completed[key] = array.tolist()
        return completed


def main() -> None:
    args = _parse_args()

    robodojo_root = Path(args.robodojo_root).expanduser().resolve()
    _prepare_robodojo_import_root(robodojo_root)
    os.environ.setdefault("ROBODOJO_ROOT", str(robodojo_root))
    from datetime import datetime

    run_id = os.environ.get("ROBODOJO_RUN_ID") or (
        "rpent_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.environ["ROBODOJO_RUN_ID"] = run_id

    # Camera frames must not lag physics: same fix as the native client.
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env_cfg = _build_env_cfg(args, run_id)
    if int(args.policy_port) == 0:
        # Must run BEFORE eval_env is imported (it binds WsModelClient at
        # import time).
        _install_noop_policy_client()
    from src.eval_client.eval_env import create_eval_env

    print("[rpent] creating robodojo eval env ...", flush=True)
    env = create_eval_env(env_cfg, simulation_app)
    print("[rpent] eval env created", flush=True)

    from robots.robodojo.robot_spec import env_runtime_contract

    facade = RoboDojoEnvFacade(
        env,
        metadata=env_runtime_contract(
            task_name=args.task_name,
            env_cfg=args.env_cfg,
            seed=args.seed,
        ),
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
