"""Action scaling helpers for BC / policies.

MuJoCo H1 actuators use large torque ranges; raw MSE on torques is meaningless.
Normalize to ``[-1, 1]`` (or a custom box) for training, denormalize for ``env.step``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray


def ctrl_range_from_env(env: Any) -> tuple[Array, Array]:
    """Return ``(low, high)`` float32 arrays from a Gymnasium / MuJoCo env."""
    space = env.action_space
    low = np.asarray(space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(space.high, dtype=np.float64).reshape(-1)
    # Replace non-finite with ±1
    low = np.where(np.isfinite(low), low, -1.0)
    high = np.where(np.isfinite(high), high, 1.0)
    # Avoid zero-span
    span = high - low
    bad = span < 1e-8
    low = low.copy()
    high = high.copy()
    low[bad] = -1.0
    high[bad] = 1.0
    return low.astype(np.float32), high.astype(np.float32)


def normalize_action(action: Array, low: Array, high: Array) -> Array:
    """Map action in ``[low, high]`` → ``[-1, 1]``."""
    a = np.asarray(action, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    return (2.0 * (a - low) / (high - low) - 1.0).astype(np.float32)


def denormalize_action(action_norm: Array, low: Array, high: Array) -> Array:
    """Map action in ``[-1, 1]`` → ``[low, high]``."""
    a = np.asarray(action_norm, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    return (0.5 * (a + 1.0) * (high - low) + low).astype(np.float32)


class ActionScaler:
    """Stateful helper bound to an env's action box."""

    def __init__(self, env: Any) -> None:
        self.low, self.high = ctrl_range_from_env(env)

    def normalize(self, action: Array) -> Array:
        return normalize_action(action, self.low, self.high)

    def denormalize(self, action_norm: Array) -> Array:
        return denormalize_action(action_norm, self.low, self.high)

    def meta(self) -> dict[str, Any]:
        return {
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "convention": "normalize maps [low,high] -> [-1,1]",
        }


__all__ = [
    "ActionScaler",
    "ctrl_range_from_env",
    "normalize_action",
    "denormalize_action",
]
