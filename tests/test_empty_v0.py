"""Smoke test for PhysicalSim/empty-v0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PKG_ROOT / "configs" / "empty_placeholder.yaml"


@pytest.fixture(scope="module")
def env():
    mujoco = pytest.importorskip("mujoco")
    import open_physical_sim  # noqa: F401
    import gymnasium as gym

    e = gym.make("PhysicalSim/empty-v0", config_path=str(CONFIG))
    yield e
    e.close()


def test_make_reset_step(env):
    obs, info = env.reset(seed=42)
    assert "proprio" in obs
    assert obs["proprio"].dtype == np.float32
    assert obs["proprio"].ndim == 1
    assert "rgb" in obs
    assert obs["rgb"].shape == (3, 64, 64)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert "proprio" in obs2
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.action_space.shape[0] >= 1


def test_registered():
    import open_physical_sim  # noqa: F401
    import gymnasium as gym
    from gymnasium.envs.registration import registry

    assert "PhysicalSim/empty-v0" in registry
