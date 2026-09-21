"""Smoke test for PhysicalSim/h1-hello-twin-v0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
H1_DIR = PKG_ROOT / "assets" / "robots" / "unitree_h1"
H1_CONFIG = PKG_ROOT / "configs" / "h1_hello_twin.yaml"


def _meshes_present() -> bool:
    assets = H1_DIR / "assets"
    return (
        (H1_DIR / "h1.xml").is_file()
        and (H1_DIR / "scene.xml").is_file()
        and assets.is_dir()
        and any(assets.glob("*.stl"))
    )


pytestmark = pytest.mark.skipif(
    not _meshes_present(),
    reason="Unitree H1 MJCF/meshes not vendored under assets/robots/unitree_h1/",
)


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("mujoco")
    import open_physical_sim  # noqa: F401
    import gymnasium as gym

    e = gym.make("PhysicalSim/h1-hello-twin-v0")
    yield e
    e.close()


def test_registered():
    import open_physical_sim  # noqa: F401
    from gymnasium.envs.registration import registry

    assert "PhysicalSim/h1-hello-twin-v0" in registry
    assert "PhysicalSim/empty-v0" in registry


def test_make_reset_step(env):
    obs, info = env.reset(seed=42)
    assert "proprio" in obs
    assert obs["proprio"].dtype == np.float32
    assert obs["proprio"].ndim == 1
    assert obs["proprio"].shape[0] >= 20
    assert "rgb" in obs
    assert obs["rgb"].shape == (3, 64, 64)
    assert "ego" in obs
    assert "unitree_h1" in str(info.get("mjcf", ""))

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert "proprio" in obs2
    assert obs2["proprio"].shape == obs["proprio"].shape
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert env.action_space.shape[0] >= 10


def test_explicit_config_path():
    pytest.importorskip("mujoco")
    import open_physical_sim  # noqa: F401
    import gymnasium as gym

    e = gym.make(
        "PhysicalSim/h1-hello-twin-v0",
        config_path=str(H1_CONFIG),
    )
    obs, info = e.reset(seed=0)
    assert "unitree_h1" in str(info.get("mjcf", ""))
    e.step(e.action_space.sample())
    e.close()
