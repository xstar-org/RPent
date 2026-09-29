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

"""RoboDojo robot extension — runtime contracts and runner hooks.

RoboDojo (Isaac Sim 5.1 / IsaacLab) runs inside its own shared-storage
venv; the RPent planner process only talks HTTP RPC to
``robots/robodojo/env_server.py``, which is launched with that venv's
interpreter and owns the Isaac app, the task env, and the XPolicyLab
Pi_05 policy client.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robodojo.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.dashboard.spec import DashboardSpec
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import (
    stop_owned_daemons,
    try_spawn_server,
    try_wait_server,
)
from rpent.utils.config import get_memory_dir, get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon

#: Default RoboDojo embodiment config (ARX X5 dual-arm).
ROBODOJO_ENV_CFG = "arx_x5"

#: Default gate-verified representative task (native 25-episode gate passed
#: on the deployment cluster with the pinned pi0.5 checkpoint).
ROBODOJO_DEFAULT_TASK = "stack_blocks"

#: Camera views exposed by the RoboDojo env server, in fixed order.
#: RoboDojo obs keys are ``cam_<name>``; the RPent artifact names drop the
#: ``cam_`` prefix.
ROBODOJO_CAMERA_NAMES = (
    "head",
    "left_wrist",
    "right_wrist",
)

#: Episode-status keys every RoboDojo status mapping must carry.
ROBODOJO_STATUS_KEYS = (
    "eval_success",
    "episode_ended",
    "take_action_cnt",
    "step_lim",
    "actual_seed",
)

#: Per-call RPC read timeout (seconds); Isaac renders are slow.
ROBODOJO_READ_TIMEOUT_S = 300.0

#: Per-call RPC timeout (seconds) for state-changing calls. A policy chunk
#: executes many interpolated physics steps server-side.
ROBODOJO_STATE_CHANGE_TIMEOUT_S = 3600.0


def env_runtime_contract(
    *,
    task_name: str,
    env_cfg: str,
    seed: int,
) -> dict[str, object]:
    """Return the identity required from a RoboDojo env server."""
    return {
        "runtime": "rpent_robodojo_env",
        "benchmark": "RoboDojo",
        "task_name": task_name,
        "env_cfg": env_cfg,
        "seed": int(seed),
        "action_layouts": ["joint_dict", "ee_pose_dict"],
        "execution": {
            "reset": True,
            "step": True,
            "chunk_step": True,
            "policy_proxy": True,
            "step_lim": "native-per-task",
        },
    }


def policy_runtime_contract() -> dict[str, object]:
    """Return the identity of the server-side Pi_05 policy proxy."""
    return {
        "runtime": "xpolicylab_pi05",
        "protocol": "ws",
        "action_type": "joint",
        "camera_order": list(ROBODOJO_CAMERA_NAMES),
    }


ROBODOJO_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_name> <seed>",
        "fields": (
            {"name": "task_name"},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / seed {seed}",
        "output_slug": "{task_name}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "policy", "label": "PI05", "scope": "shared"},
    ),
    "primitives": (
        "pi05_act",
        "move_eef",
        "set_gripper",
        "release",
    ),
}


def get_robot_spec() -> RobotSpec:
    return RobotSpec(
        name="robodojo",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=ROBODOJO_DASHBOARD_SPEC,
        supports_exploration=False,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the RoboDojo toolkit for the current session."""
    from robots.robodojo.toolkit import RoboDojoToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("robodojo"),
    )
    return RoboDojoToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.add_argument("--task-name", default=ROBODOJO_DEFAULT_TASK)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        required=required,
        help=(
            "RoboDojo eval-layout seed (layout id). The native protocol uses "
            "seed 0 for the standard evaluation episode."
        ),
    )
    parser.add_argument(
        "--env-cfg",
        default=ROBODOJO_ENV_CFG,
        help="RoboDojo embodiment config name under env_cfg/.",
    )
    parser.add_argument(
        "--robodojo-root",
        default=os.environ.get("ROBODOJO_ROOT"),
        help="RoboDojo checkout root. Defaults to ROBODOJO_ROOT.",
    )
    parser.add_argument(
        "--sim-env",
        default=os.environ.get("ROBODOJO_SIM_ENV"),
        help=(
            "Isaac Sim / IsaacLab venv directory that owns the RoboDojo "
            "runtime. Defaults to ROBODOJO_SIM_ENV."
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=os.environ.get("ROBODOJO_CKPT"),
        help="Pi_05 checkpoint directory. Defaults to ROBODOJO_CKPT.",
    )
    parser.add_argument(
        "--policy-root",
        default=os.environ.get("ROBODOJO_POLICY_ROOT"),
        help=(
            "XPolicyLab Pi_05 openpi deployment root (owns .venv). "
            "Defaults to ROBODOJO_POLICY_ROOT."
        ),
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--policy-endpoint", default=None)
    parser.add_argument(
        "--skip-policy",
        action="store_true",
        help=(
            "Do not spawn the Pi_05 policy server (env-only smoke runs; "
            "pi05_act will be unavailable)."
        ),
    )
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument(
        "--env-cuda-device",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for the Isaac env server.",
    )
    parser.add_argument(
        "--policy-cuda-device",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for the Pi_05 policy server.",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if not args.task_name:
        raise ValueError("--task-name is required")
    if args.env_endpoint is None and not getattr(args, "robodojo_root", None):
        raise ValueError(
            "--robodojo-root is required when launching the local env "
            "server; set ROBODOJO_ROOT or pass the option explicitly"
        )
    if args.env_endpoint is None and not getattr(args, "sim_env", None):
        raise ValueError(
            "--sim-env is required when launching the local env server; "
            "set ROBODOJO_SIM_ENV or pass the option explicitly"
        )
    if not args.skip_policy and args.policy_endpoint is None:
        missing = [
            name
            for name in ("ckpt", "policy_root")
            if not getattr(args, name, None)
        ]
        if missing:
            raise ValueError(
                f"--{missing[0].replace('_', '-')} is required when "
                "launching the Pi_05 policy server; set the corresponding "
                "environment variable or pass the option explicitly (or use "
                "--skip-policy for env-only smoke runs)"
            )
    env_cfg = getattr(args, "env_cfg", ROBODOJO_ENV_CFG) or ROBODOJO_ENV_CFG
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root()
            / "logs"
            / f"{timestamp}_robodojo_{args.task_name}_s{args.seed}"
        )
    output_dir = Path(output_dir)
    recipe_tag = f"robodojo_{args.task_name}_s{args.seed}"
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else get_memory_dir("robodojo")
    )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars={
            "task_name": args.task_name,
            "seed": args.seed,
            "env_cfg": env_cfg,
            "instruction": "<native instruction from state_00>",
            "memory_dir": str(memory_dir),
            "reference_tag": f"{args.task_name}_s0",
        },
        task_desc={
            "env": "robodojo",
            "task_name": args.task_name,
            "requested_seed": args.seed,
            "env_cfg": env_cfg,
            "instruction": None,
        },
    )


def _wait_for_tcp(
    host: str,
    port: int,
    daemon: ProcessDaemon | None,
    timeout_s: float = 1800.0,
) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if daemon is not None and daemon.poll() is not None:
            raise RuntimeError(
                f"{daemon.name} exited before listening; inspect its log"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as error:
            last_error = error
            time.sleep(0.5)
    raise TimeoutError(f"RoboDojo policy server not ready: {last_error}")


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    value = endpoint.split("://", 1)[-1]
    host, separator, port_text = value.rpartition(":")
    if not separator or not host or not port_text:
        raise ValueError("--policy-endpoint must be [ws://]host:port")
    return host, int(port_text)


def _resolve_cuda_devices(
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    shared = getattr(args, "cuda_device", None)
    env_device = getattr(args, "env_cuda_device", None)
    policy_device = getattr(args, "policy_cuda_device", None)
    if shared is not None and (env_device is not None or policy_device is not None):
        raise ValueError(
            "--cuda-device cannot be combined with --env-cuda-device or "
            "--policy-cuda-device"
        )
    if shared is not None:
        return str(shared), str(shared)
    return (
        str(env_device) if env_device is not None else None,
        str(policy_device) if policy_device is not None else None,
    )


def _isaac_env_overrides(env_cuda_device: str | None) -> dict[str, str]:
    """Graphics/cache variables the Isaac env server needs on every boot."""
    base = "/n105_ssd/user/baige/robot-benchmark-eval"
    overrides = {
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "LD_LIBRARY_PATH": "/.singularity.d/libs"
        + ("" if not os.environ.get("LD_LIBRARY_PATH")
           else ":" + os.environ["LD_LIBRARY_PATH"]),
        "XDG_CACHE_HOME": f"{base}/.cache/robodojo-xdg",
        "MESA_SHADER_CACHE_DIR": f"{base}/.cache/robodojo-mesa",
    }
    if env_cuda_device is not None:
        overrides["CUDA_VISIBLE_DEVICES"] = str(env_cuda_device)
    return overrides


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list["ProcessDaemon"], dict[str, Any]]:
    """Initialize every RoboDojo component, or only ``components`` when given."""
    available = {"env", "policy"}
    selected = available if components is None else components
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown RoboDojo runtime components: {sorted(unknown)}")
    if args.skip_policy:
        selected = selected.difference({"policy"})

    owned_daemons: dict[str, ProcessDaemon] = {}
    env_pending: tuple[ProcessDaemon | None, Any] | None = None
    policy_pending: tuple[ProcessDaemon | None, tuple[str, int]] | None = None

    # The policy server is spawned first: the env server embeds its endpoint
    # (a lazy WebSocket client) at construction, and both cold boots (~2-4 min
    # each: 42 GB JAX checkpoint / Isaac app) then overlap.
    if "policy" in selected:
        dashboard_events.emit(RuntimeStatusEvent("policy", "starting"))
        try:
            policy_pending = _spawn_policy_server(args, output_dir)
            policy_daemon, _ = policy_pending
            if policy_daemon is not None:
                owned_daemons["policy"] = policy_daemon
        except Exception as exc:
            stop_owned_daemons(owned_daemons, dashboard_events)
            dashboard_events.emit(RuntimeStatusEvent("policy", "failed", error=exc))
            raise RuntimeError(f"[policy] spawn failed: {exc}") from exc

    if "env" in selected:
        env_pending = try_spawn_server(
            owned_daemons,
            dashboard_events,
            "env",
            lambda: _spawn_env_server(args, output_dir),
        )

    runtime_kwargs: dict[str, Any] = {}

    if env_pending is not None:
        env_daemon, env_rpc = env_pending
        env_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            "env",
            env_rpc,
            env_daemon,
            2400.0 if env_daemon is not None else 600.0,
            post_fn=lambda: _build_env_runtime_kwargs(args, env_rpc),
        )
        runtime_kwargs.update(env_kwargs)

    if policy_pending is not None:
        policy_daemon, endpoint = policy_pending
        host, port = endpoint
        try:
            _wait_for_tcp(
                host,
                port,
                policy_daemon,
                timeout_s=1800.0 if policy_daemon is not None else 600.0,
            )
        except Exception as exc:
            stop_owned_daemons(owned_daemons, dashboard_events)
            dashboard_events.emit(RuntimeStatusEvent("policy", "failed", error=exc))
            raise RuntimeError(f"[policy] wait failed: {exc}") from exc
        dashboard_events.emit(RuntimeStatusEvent("policy", "ready"))
        runtime_kwargs["policy_endpoint"] = f"ws://{host}:{port}"

    return list(owned_daemons.values()), runtime_kwargs


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple["ProcessDaemon | None", Any]:
    from rpent.utils.daemon import ProcessDaemon, pick_free_port

    env_cuda_device, _ = _resolve_cuda_devices(args)

    if args.env_endpoint is not None:
        from rpent.utils.rpc import make_rpc_client

        return None, make_rpc_client(args.env_endpoint)

    robodojo_root = Path(args.robodojo_root).expanduser().resolve()
    sim_python = Path(args.sim_env).expanduser().resolve() / "bin" / "python"
    if not sim_python.is_file():
        raise ValueError(f"RoboDojo sim-venv python not found: {sim_python}")
    policy_endpoint = args.policy_endpoint or _running_policy_endpoint(args)
    _, policy_port = (
        _parse_endpoint(policy_endpoint) if policy_endpoint else (None, 0)
    )
    host, env_port = "127.0.0.1", pick_free_port()
    env_daemon = ProcessDaemon(
        "robodojo_env_server",
        [
            str(sim_python),
            str(get_repo_root() / "robots" / "robodojo" / "env_server.py"),
            "--task-name",
            args.task_name,
            "--env-cfg",
            args.env_cfg,
            "--seed",
            str(int(args.seed)),
            "--robodojo-root",
            str(robodojo_root),
            "--policy-port",
            str(int(policy_port or 0)),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(env_port),
            "--parent-watch",
        ],
        env_overrides=_isaac_env_overrides(env_cuda_device),
        log_path=str(output_dir / "robodojo_env_server.log"),
    )
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    env_rpc = HttpRpcClient(f"http://{host}:{env_port}")
    env_daemon.start()
    return env_daemon, env_rpc


def _running_policy_endpoint(args: argparse.Namespace) -> str | None:
    """Best-effort endpoint of the policy server spawned by this runtime."""
    policy = getattr(args, "_policy_endpoint", None)
    return str(policy) if policy else None


def _spawn_policy_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, tuple[str, int]]:
    from rpent.utils.daemon import ProcessDaemon, pick_free_port

    if args.policy_endpoint is not None:
        return None, _parse_endpoint(args.policy_endpoint)

    _, policy_cuda_device = _resolve_cuda_devices(args)
    robodojo_root = Path(args.robodojo_root).expanduser().resolve()
    ckpt = Path(args.ckpt).expanduser().resolve()
    policy_root = Path(args.policy_root).expanduser().resolve()
    setup_script = (
        robodojo_root / "XPolicyLab" / "policy" / "Pi_05"
        / "setup_eval_policy_server.sh"
    )
    if not setup_script.is_file():
        raise ValueError(f"Pi_05 server script not found: {setup_script}")
    host, policy_port = "127.0.0.1", pick_free_port()
    overrides = {
        "CUDA_VISIBLE_DEVICES": str(policy_cuda_device)
        if policy_cuda_device is not None
        else "0",
    }
    daemon = ProcessDaemon(
        "robodojo_pi05_policy_server",
        [
            "bash",
            str(setup_script),
            "RoboDojo",
            args.task_name,
            str(ckpt),
            args.env_cfg,
            "joint",
            overrides["CUDA_VISIBLE_DEVICES"],
            "0",
            str(policy_root),
            str(policy_port),
            host,
        ],
        env_overrides=overrides,
        log_path=str(output_dir / "robodojo_policy_server.log"),
    )
    daemon.start()
    args._policy_endpoint = f"ws://{host}:{policy_port}"
    return daemon, (host, policy_port)


def _build_env_runtime_kwargs(
    args: argparse.Namespace,
    env_rpc: Any,
) -> dict[str, Any]:
    from robots.robodojo.env_client import RoboDojoEnvClient

    return {
        "env": RoboDojoEnvClient(
            env_rpc,
            expected_meta=env_runtime_contract(
                task_name=args.task_name,
                env_cfg=args.env_cfg,
                seed=int(args.seed),
            ),
        ),
        "seed": int(args.seed),
    }
