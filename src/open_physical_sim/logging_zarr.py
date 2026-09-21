"""Episodic Zarr trajectory writer for Policy Eng BC/eval.

Canonical layout (``open_physical_sim.trajectory.v1``)::

    <root>/
      episodes/<ep_id>/
        timestamps   (T,) float64
        actions      (T, …)
        rewards      (T,) float32   optional
        terminated   (T,) bool
        truncated    (T,) bool
        obs/<key>    (T, …)
        info/…       optional scalar/array attrs or arrays

Requires ``zarr>=2.16,<3`` (zarr 3 breaks ``Group.array``).
Install: ``pip install open_physical_sim[zarr]``.

If zarr is missing the API is a no-op so training loops can call it always.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

try:
    import zarr  # type: ignore

    _HAS_ZARR = True
except ImportError:  # pragma: no cover
    zarr = None  # type: ignore
    _HAS_ZARR = False

_FORMAT = "open_physical_sim.trajectory.v1"


class TrajectoryWriter:
    """Buffer steps per episode, flush as ``(T, …)`` arrays under ``episodes/``."""

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: Optional[bool] = None,
        episode_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.enabled = (_HAS_ZARR if enabled is None else bool(enabled)) and _HAS_ZARR
        self._root: Any = None
        self._ep_id: str | None = None
        self._buf: dict[str, list[Any]] = {}
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Use DirectoryStore-friendly path; mode=a so multiple episodes can append.
            self._root = zarr.open_group(str(self.path), mode="a")
            self._root.attrs["format"] = _FORMAT
            self._root.require_group("episodes")
            if episode_id is not None:
                self.start_episode(episode_id)

    @property
    def available(self) -> bool:
        return _HAS_ZARR

    def start_episode(self, episode_id: str) -> None:
        """Begin buffering a new episode (flushes any open buffer first)."""
        if self._ep_id is not None and self._buf.get("actions"):
            self.end_episode()
        self._ep_id = str(episode_id)
        self._buf = {
            "timestamps": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "obs": {},  # key -> list
            "info": {},
        }

    def append(
        self,
        *,
        obs: Mapping[str, np.ndarray] | None = None,
        action: np.ndarray | None = None,
        reward: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        info: Mapping[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        if not self.enabled or self._root is None:
            return
        if self._ep_id is None:
            self.start_episode("0")

        t = float(timestamp) if timestamp is not None else float(len(self._buf["actions"]))
        self._buf["timestamps"].append(t)
        self._buf["actions"].append(
            np.asarray(action if action is not None else 0.0, dtype=np.float32)
        )
        self._buf["rewards"].append(0.0 if reward is None else float(reward))
        self._buf["terminated"].append(bool(terminated) if terminated is not None else False)
        self._buf["truncated"].append(bool(truncated) if truncated is not None else False)

        if obs:
            for k, v in obs.items():
                self._buf["obs"].setdefault(k, []).append(np.asarray(v))
        if info:
            for k, v in info.items():
                if isinstance(v, (bool, int, float, np.generic)):
                    self._buf["info"].setdefault(k, []).append(
                        v.item() if isinstance(v, np.generic) else v
                    )

        if terminated or truncated:
            self.end_episode()

    def end_episode(self) -> None:
        """Flush the current episode buffer to ``episodes/<ep_id>/``."""
        if not self.enabled or self._root is None or self._ep_id is None:
            return
        actions = self._buf.get("actions") or []
        if not actions:
            self._ep_id = None
            self._buf = {}
            return

        ep = self._root.require_group("episodes").require_group(self._ep_id)
        # zarr 2.x API
        ep.array("timestamps", np.asarray(self._buf["timestamps"], dtype=np.float64), overwrite=True)
        ep.array("actions", np.stack(actions), overwrite=True)
        ep.array("rewards", np.asarray(self._buf["rewards"], dtype=np.float32), overwrite=True)
        ep.array(
            "terminated",
            np.asarray(self._buf["terminated"], dtype=bool),
            overwrite=True,
        )
        ep.array(
            "truncated",
            np.asarray(self._buf["truncated"], dtype=bool),
            overwrite=True,
        )
        obs_grp = ep.require_group("obs")
        for k, series in self._buf["obs"].items():
            obs_grp.array(k, np.stack(series), overwrite=True)
        if self._buf["info"]:
            info_grp = ep.require_group("info")
            for k, series in self._buf["info"].items():
                info_grp.array(k, np.asarray(series), overwrite=True)

        n_eps = int(self._root.attrs.get("n_episodes", 0)) + 1
        self._root.attrs["n_episodes"] = n_eps
        self._ep_id = None
        self._buf = {}

    def close(self) -> None:
        if self._ep_id is not None and self._buf.get("actions"):
            self.end_episode()
        self._root = None


def write_noop(*_args: Any, **_kwargs: Any) -> None:
    """Convenience no-op for call sites that ignore writers."""
    return None


__all__ = ["TrajectoryWriter", "write_noop", "_HAS_ZARR", "_FORMAT"]
