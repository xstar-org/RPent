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

"""RPC server wrapping the Pi0.5 VLA.

Embodiment-specific settings (openpi config name, action dim, …) are
selected by the ``--embodiment`` CLI flag and looked up in
``PI05_EMBODIMENTS``.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
import threading
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.config import get_pi05_checkpoint_path
from rpent.utils.logging import get_logger

logger = get_logger("vla_server")

# ---------------------------------------------------------------------------
# Embodiment registry
# ---------------------------------------------------------------------------

# NOTE: an embodiment added here must also be registered in the client's
# ``_ENCODE_OBS`` (obs encoding); the two registries are kept in sync manually.
PI05_EMBODIMENTS: dict[str, dict] = {
    "libero": {
        "num_action_chunks": 5,
        "action_dim": 7,
        "use_proprio": True,
        "num_steps": 5,
        "add_value_head": False,
        "openpi": {
            "config_name": "pi05_libero",
            "num_images_in_input": 2,
            "action_chunk": 5,
            "num_steps": 5,
            "action_env_dim": 7,
            "add_value_head": False,
        },
    },
}

PI05_ROBOT_PLATFORMS: dict[str, str] = {
    "libero": "LIBERO",
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_model_cfg(model_path: str, emb_cfg: dict) -> Any:
    """OmegaConf for ``rlinf.models.embodiment.openpi.get_model``.

    Two-level merge ``emb_cfg`` into a default config template.  ``emb_cfg``
    mirrors the OmegaConf structure (top-level keys + ``openpi`` sub-dict),
    so adding a new key to an embodiment preset automatically flows into
    the model config.  ``model_path`` is set at runtime, not from the
    embodiment preset.
    """
    cfg = {
        "model_type": "openpi",
        "model_path": model_path,
        "precision": None,
        "is_lora": False,
        "lora_rank": 32,
        "openpi": {
            "noise_level": 0.5,
            "train_expert_only": True,
            "noise_method": "flow_sde",
            "value_after_vlm": False,
            "value_vlm_mode": "mean_token",
            "detach_critic_input": None,
            "use_dsrl": False,
        },
    }
    # Deep merge: top-level keys override, openpi sub-dict merges into cfg.openpi
    for k, v in emb_cfg.items():
        if k == "openpi":
            cfg["openpi"].update(v)
        else:
            cfg[k] = v

    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# VLA facade
# ---------------------------------------------------------------------------


def _pi05_worker(model_path: str, embodiment: str, connection) -> None:
    """Own one Pi0.5 CUDA model until the RPC parent requests shutdown."""
    try:
        import torch
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        cfg = build_model_cfg(
            model_path=model_path,
            emb_cfg=PI05_EMBODIMENTS[embodiment],
        )
        started = time.time()
        model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
        logger.info("Pi0.5 model ready in %.1fs", time.time() - started)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "stop":
                return
            if command != "predict":
                raise ValueError(f"unknown Pi0.5 worker command: {command}")
            obs, mode = payload
            with torch.no_grad():
                actions, _ = model.predict_action_batch(obs, mode=mode)
            array = (
                actions.detach().cpu().numpy()
                if all(hasattr(actions, name) for name in ("detach", "cpu", "numpy"))
                else np.asarray(actions)
            ).astype(np.float32)
            connection.send(("ok", array))
    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class Pi05VLAFacade(BaseVLAFacade):
    """Pi0.5 RPC facade with a disposable CUDA worker process."""

    _WORKER_TIMEOUT_S = 115.0

    def __init__(self, *, model_path: str, embodiment: str):
        if embodiment not in PI05_EMBODIMENTS:
            raise ValueError(
                f"unknown pi05 server embodiment: {embodiment!r}; "
                f"registered={list(PI05_EMBODIMENTS)}"
            )
        self._model_path = model_path
        self._embodiment = embodiment
        self._worker = None
        self._connection = None
        self._worker_lock = threading.Lock()
        self._process_context = multiprocessing.get_context("spawn")
        platform = PI05_ROBOT_PLATFORMS.get(embodiment)
        if platform is not None:
            os.environ.setdefault("ROBOT_PLATFORM", platform)
        logger.info("Pi0.5 worker configured (embodiment=%s, model_path=%s)", embodiment, model_path)
        super().__init__()

    def _stop_worker_locked(self) -> None:
        process, connection = self._worker, self._connection
        self._worker = None
        self._connection = None
        if connection is not None:
            try:
                connection.send(("stop", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        if process is not None:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_worker_locked()
        parent, child = self._process_context.Pipe(duplex=True)
        process = self._process_context.Process(
            target=_pi05_worker,
            args=(self._model_path, self._embodiment, child),
            daemon=True,
        )
        process.start()
        child.close()
        self._worker, self._connection = process, parent
        if not parent.poll(self._WORKER_TIMEOUT_S):
            self._stop_worker_locked()
            raise TimeoutError("Pi0.5 model worker startup timed out")
        status, payload = parent.recv()
        if status != "ready":
            self._stop_worker_locked()
            raise RuntimeError(f"Pi0.5 model worker failed: {payload}")

    def unload(self) -> dict[str, bool]:
        with self._worker_lock:
            self._stop_worker_locked()
        return {"loaded": False}

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc["vla.unload"] = self.unload

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        mode = (options or {}).get("mode", "eval")
        with self._worker_lock:
            self._ensure_worker_locked()
            assert self._connection is not None
            self._connection.send(("predict", (obs, mode)))
            if not self._connection.poll(self._WORKER_TIMEOUT_S):
                self._stop_worker_locked()
                raise TimeoutError("Pi0.5 prediction timed out")
            status, payload = self._connection.recv()
            if status != "ok":
                self._stop_worker_locked()
                raise RuntimeError(f"Pi0.5 prediction worker failed: {payload}")
            return np.asarray(payload, dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embodiment",
        required=True,
        help="Embodiment preset name (e.g. 'libero'); see PI05_EMBODIMENTS",
    )
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Pi0.5 checkpoint (defaults to PI05_CHECKPOINT_PATH env)",
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target

    model_path = args.model_path or get_pi05_checkpoint_path()
    if not model_path:
        raise RuntimeError(
            "PI05_CHECKPOINT_PATH is not set; provide the Pi0.5 checkpoint "
            "path via --model-path or the environment."
        )

    facade = Pi05VLAFacade(model_path=model_path, embodiment=args.embodiment)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
