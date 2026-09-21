"""Smoke: Unitree H1 MJCF loads headless in MuJoCo."""

from __future__ import annotations

from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
H1_DIR = PKG_ROOT / "assets" / "robots" / "unitree_h1"
H1_XML = H1_DIR / "h1.xml"
H1_SCENE = H1_DIR / "scene.xml"
H1_CONFIG = PKG_ROOT / "configs" / "h1_hello_twin.yaml"


def _meshes_present() -> bool:
    assets = H1_DIR / "assets"
    return (
        H1_XML.is_file()
        and H1_SCENE.is_file()
        and assets.is_dir()
        and any(assets.glob("*.stl"))
    )


pytestmark = pytest.mark.skipif(
    not _meshes_present(),
    reason="Unitree H1 MJCF/meshes not vendored under assets/robots/unitree_h1/",
)


@pytest.fixture(scope="module")
def mujoco_mod():
    return pytest.importorskip("mujoco")


def test_h1_xml_loads(mujoco_mod):
    model = mujoco_mod.MjModel.from_xml_path(str(H1_XML))
    data = mujoco_mod.MjData(model)
    mujoco_mod.mj_forward(model, data)
    assert model.nq >= 20
    assert model.nu >= 10
    assert model.nbody >= 10


def test_h1_scene_loads(mujoco_mod):
    model = mujoco_mod.MjModel.from_xml_path(str(H1_SCENE))
    assert model.ngeom >= 1


def test_h1_config_env_loads(mujoco_mod):
    import open_physical_sim  # noqa: F401
    from open_physical_sim.envs.empty import EmptyRoomEnv

    env = EmptyRoomEnv(config_path=str(H1_CONFIG))
    obs, info = env.reset(seed=1)
    assert "proprio" in obs
    assert obs["proprio"].shape[0] == env.model.nq + env.model.nv
    assert env.action_space.shape[0] == env.model.nu
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2["proprio"].shape == obs["proprio"].shape
    # Same-package scene.xml shortcut — not the empty_room compose build.
    assert "unitree_h1" in str(info.get("mjcf", ""))
    env.close()
