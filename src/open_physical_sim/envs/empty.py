"""EmptyRoomEnv — headless MuJoCo empty room + placeholder robot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, SupportsFloat

import numpy as np
import yaml

import gymnasium as gym
from gymnasium import spaces

from open_physical_sim.scene.loader import (
    _PKG_ROOT,
    load_scene_package,
    resolve_mjcf,
)

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    mujoco = None  # type: ignore
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        default = _PKG_ROOT / "configs" / "empty_placeholder.yaml"
        config_path = default if default.is_file() else None
    if config_path is None:
        return {}
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    # Resolve relative asset paths against package root (and config dir).
    cfg_dir = path.parent
    for key in ("robot", "scene", "env"):
        block = data.get(key)
        if isinstance(block, dict) and "path" in block:
            p = Path(block["path"])
            if not p.is_absolute():
                for base in (cfg_dir, _PKG_ROOT, cfg_dir.parent):
                    cand = (base / p).resolve()
                    if cand.exists():
                        block["path"] = str(cand)
                        break
                else:
                    block["path"] = str((_PKG_ROOT / p).resolve())
    return data


class EmptyRoomEnv(gym.Env):
    """Gymnasium env: empty room collision MJCF + simple free/2-link robot.

    Observation dict
    ----------------
    proprio : float32 vector — concat(qpos, qvel)
    <camera_key> : float32 zeros of configured shape (no renderer in v0)

    Action
    ------
    Box matching ``nu`` actuators (or free-joint ctrl dims).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        if mujoco is None:
            raise ImportError(
                "mujoco is required for EmptyRoomEnv. "
                "Install with: pip install mujoco"
            ) from _MUJOCO_IMPORT_ERROR

        self.config = _load_config(config_path)
        self.render_mode = render_mode
        self.max_episode_steps = int(
            self.config.get("episode", {}).get("max_steps", max_episode_steps)
        )
        self._step_count = 0

        robot_path = None
        scene_path = None
        if "robot" in self.config:
            robot_path = self.config["robot"].get("path")
        if "scene" in self.config:
            scene_path = self.config["scene"].get("path")
        elif "env" in self.config:
            scene_path = self.config["env"].get("path")

        self.scene = load_scene_package(scene_path, robot_path=robot_path)
        mjcf = resolve_mjcf(self.scene, include_robot=True)

        # Headless: load XML directly — no EGL / viewer required.
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)

        nq = self.model.nq
        nv = self.model.nv
        nu = max(int(self.model.nu), 1)
        proprio_dim = nq + nv

        cam_cfg = self.config.get("cameras", {})
        if isinstance(cam_cfg, list):
            # list of {name, shape}
            self._camera_specs = {
                c["name"]: tuple(c.get("shape", [3, 64, 64])) for c in cam_cfg
            }
        elif isinstance(cam_cfg, dict):
            keys = cam_cfg.get("keys", [])
            default_shape = tuple(cam_cfg.get("shape", [3, 64, 64]))
            self._camera_specs = {k: default_shape for k in keys}
        else:
            self._camera_specs = {}

        obs_spaces: dict[str, spaces.Space] = {
            "proprio": spaces.Box(
                low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32
            )
        }
        for name, shape in self._camera_specs.items():
            obs_spaces[name] = spaces.Box(
                low=0.0, high=255.0, shape=tuple(shape), dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_spaces)

        ctrl_range = self.model.actuator_ctrlrange
        if self.model.nu > 0 and np.isfinite(ctrl_range).all():
            low = ctrl_range[:, 0].astype(np.float32)
            high = ctrl_range[:, 1].astype(np.float32)
        else:
            low = -np.ones(nu, dtype=np.float32)
            high = np.ones(nu, dtype=np.float32)
            if self.model.nu == 0:
                # No actuators: expose a dummy 1-d action so policies can call step.
                pass
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._nu = int(self.model.nu)

        self._frame_skip = int(self.config.get("sim", {}).get("frame_skip", 10))

    def _get_obs(self) -> dict[str, np.ndarray]:
        proprio = np.concatenate(
            [self.data.qpos.astype(np.float32), self.data.qvel.astype(np.float32)]
        )
        obs: dict[str, np.ndarray] = {"proprio": proprio}
        for name, shape in self._camera_specs.items():
            obs[name] = np.zeros(shape, dtype=np.float32)
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Small random proprio nudge for diversity (disabled if DR empty).
        dr = self.config.get("domain_randomization") or self.config.get("dr") or {}
        if dr.get("enabled") and self.np_random is not None:
            noise = float(dr.get("qpos_noise", 0.01))
            self.data.qpos[:] += self.np_random.uniform(
                -noise, noise, size=self.model.nq
            )
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        info = {
            "scene_source": self.scene.source,
            "mjcf": str(self.scene.collision_mjcf),
            "units": self.scene.units,
        }
        return self._get_obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._nu > 0:
            ctrl = np.clip(action[: self._nu], self.action_space.low, self.action_space.high)
            self.data.ctrl[:] = ctrl
        for _ in range(self._frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        info: dict[str, Any] = {"step": self._step_count}
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        # v0: no offscreen renderer (avoids EGL). Return None / zeros.
        if self.render_mode == "rgb_array":
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        self.model = None  # type: ignore
        self.data = None  # type: ignore


__all__ = ["EmptyRoomEnv"]
